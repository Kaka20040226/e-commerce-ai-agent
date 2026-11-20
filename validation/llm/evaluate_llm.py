"""Evaluate local Llama reranking on fixed, training-derived ItemCF candidates."""

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
from math import log2
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "demo"
sys.path.insert(0, str(SOURCE))

from agents.product_rec_agent import ProductRecAgent
from models.schemas import Product, UserProfile


HOST = "http://127.0.0.1:11434"
MODEL = "llama3.2:latest"
K = 10
SEED = 114
TEMPERATURE = 0.3
OPTIONS = {"seed": SEED, "temperature": TEMPERATURE, "num_predict": 512, "num_ctx": 4096}


class Ollama:
    def __init__(self):
        self.opener = build_opener(ProxyHandler({}))
        self.reply = None

    def request(self, path, body=None):
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = Request(HOST + path, headers=headers,
                          data=json.dumps(body).encode() if body is not None else None)
        with self.opener.open(request, timeout=90) as response:
            return json.load(response)

    def installed_model(self):
        for model in self.request("/api/tags")["models"]:
            if model["name"] == MODEL:
                return model
        raise RuntimeError(f"Preinstalled Ollama model {MODEL} was not found.")

    async def ainvoke(self, messages):
        self.reply = self.request("/api/chat", {
            "model": MODEL,
            "messages": [
                {"role": {"system": "system", "human": "user"}[message.type],
                 "content": message.content}
                for message in messages
            ],
            "stream": False,
            "options": OPTIONS,
        })
        return SimpleNamespace(content=self.reply["message"]["content"])


def parse_status(raw, candidates, expected):
    text = raw.strip()
    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
    except (IndexError, json.JSONDecodeError):
        return "malformed_json"
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return "invalid_schema"
    if len(set(value)) != len(value):
        return "duplicate_ids"
    if not set(value).issubset(candidates):
        return "unknown_ids"
    if len(value) != expected:
        return "wrong_length"
    return "valid"


def metrics(ranked, truth, k):
    hits = [pid in truth for pid in ranked[:k]]
    discount = [1 / log2(index + 2) for index in range(k)]
    ideal = sum(discount[:min(k, len(truth))])
    return (
        sum(hits) / len(truth),
        sum(hit * w for hit, w in zip(hits, discount)) / ideal,
        float(any(hits)),
    )


async def evaluate(cases, backend):
    agent = ProductRecAgent.__new__(ProductRecAgent)
    agent.llm = backend
    results = []
    for case in cases:
        candidates = [
            Product(product_id=item["id"], name=item["name"], category=item["category"],
                    price=item["price"], brand=item.get("brand", ""), tags=item.get("tags", []))
            for item in case["candidates"]
        ]
        baseline = [item.product_id for item in candidates]
        truth = set(case["positives"])
        if len(baseline) < K or len(set(baseline)) != len(baseline) or not truth:
            raise ValueError(f"Invalid evaluation input for case {case['case_id']}.")
        profile = UserProfile(user_id=str(case["case_id"]), **case["profile"])
        backend.reply = None
        started = time.perf_counter()
        try:
            proposed = await agent._rerank(profile, candidates, K)
            status = parse_status(backend.reply["message"]["content"], baseline, K)
        except (URLError, TimeoutError, KeyError, TypeError, AttributeError):
            proposed, status = baseline, "runtime_error"
        elapsed = (time.perf_counter() - started) * 1000
        ranked = proposed if status == "valid" else baseline
        candidate_recall = len(set(baseline) & truth) / len(truth)
        reply = backend.reply or {}
        results.append({
            "status": status, "candidate_count": len(baseline), "candidate_recall": candidate_recall,
            "baseline": metrics(baseline, truth, K), "rerank": metrics(ranked, truth, K),
            "elapsed_ms": elapsed, "ollama_load_ms": reply.get("load_duration", 0) / 1_000_000,
            "prompt_tokens": reply.get("prompt_eval_count"), "completion_tokens": reply.get("eval_count"),
            "changed_top10": ranked[:K] != baseline[:K],
        })
        print(f"{len(results)}/{len(cases)} {status} {elapsed / 1000:.2f}s", flush=True)
    return results


def summarize(rows):
    baseline = np.asarray([row["baseline"] for row in rows])
    rerank = np.asarray([row["rerank"] for row in rows])
    delta = rerank - baseline
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(rows), size=(5000, len(rows)))
    ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975], axis=0)
    names = ("recall_at_10", "ndcg_at_10", "hit_rate_at_10")
    latency = np.asarray([row["elapsed_ms"] for row in rows])

    def mean_tokens(key):
        counts = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(counts)) if counts else None

    return {
        "users": len(rows),
        "metrics": {
            name: {"itemcf": float(baseline[:, i].mean()), "llm_rerank": float(rerank[:, i].mean()),
                   "delta": float(delta[:, i].mean()), "paired_user_bootstrap_95_ci": ci[:, i].tolist()}
            for i, name in enumerate(names)
        },
        "same_candidate_mean_purchase_recall_ceiling": float(np.mean([row["candidate_recall"] for row in rows])),
        "candidate_count_range": [min(row["candidate_count"] for row in rows),
                                  max(row["candidate_count"] for row in rows)],
        "response_status_counts": dict(Counter(row["status"] for row in rows)),
        "baseline_fallback_count": sum(row["status"] != "valid" for row in rows),
        "changed_top_10_count": sum(row["changed_top10"] for row in rows),
        "wall_latency_ms_p50_p95": np.quantile(latency, [0.5, 0.95]).tolist(),
        "first_call_wall_latency_ms": float(latency[0]),
        "mean_prompt_tokens": mean_tokens("prompt_tokens"),
        "mean_completion_tokens": mean_tokens("completion_tokens"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    content = args.cases.read_bytes()
    cases = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Expected at least one case and unique case IDs.")
    backend = Ollama()
    model = backend.installed_model()
    rows = asyncio.run(evaluate(cases, backend))
    summary = summarize(rows)
    if all(row["status"] == "runtime_error" for row in rows):
        raise RuntimeError("Local inference failed for every case. No comparison was recorded.")
    output = {
        "model": MODEL, "model_digest": model["digest"],
        "runner": "local Ollama via upstream ProductRecAgent._rerank and fixed ItemCF candidate list",
        "options": OPTIONS, "input_sha256": sha256(content).hexdigest(),
        "policy": "Select the top 10 from fixed candidates. Keep original ItemCF order on invalid response or runtime error.",
        "paid_api_calls": 0,
        **summary,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
