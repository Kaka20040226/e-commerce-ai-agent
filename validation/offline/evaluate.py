#!/usr/bin/env python3
"""Evaluate purchase ranking on a UTC time split of Kaggle event logs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import file_digest
import json
from math import log2, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy


WEIGHTS = {"view": 1, "cart": 2, "purchase": 3}
CUTOFFS = (5, 10, 20, 50, 100)
MAX_K = 100
MAX_SEEDS = 20
NEIGHBORS = 100
SHRINKAGE = 10
PROFILE_WEIGHT = 0.03
PROFILE_OFFSET = 20
DISCOUNTS = np.array([1 / log2(i + 2) for i in range(MAX_K)])
IDEAL = np.insert(DISCOUNTS.cumsum(), 0, 0)
METHODS = ("popularity", "recent_items", "itemcf", "itemcf_category_rerank")


def load_events(paths, train_start, train_end, eval_end, with_metadata):
    types = {"event_type": "category", "user_id": "int64", "product_id": "int64", "category_id": "int64"}
    if with_metadata:
        types.update(category_code="string", brand="string", price="float64")
    columns = ["event_time", *types]
    start, cutoff, end = [time.strftime("%Y-%m-%d %H:%M:%S UTC") for time in (train_start, train_end, eval_end)]
    history, holdout = [], []
    counts = Counter()
    for path in paths:
        for chunk in pd.read_csv(path, usecols=columns, dtype=types, chunksize=250_000):
            if not chunk.event_time.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC").all():
                raise ValueError(f"Expected the Kaggle UTC timestamp format in {path}")
            counts["input_events"] += len(chunk)
            train = chunk.loc[(chunk.event_time >= start) & (chunk.event_time < cutoff)].copy()
            counts.update({f"train_{kind}": n for kind, n in train.event_type.value_counts().items()})
            train["strength"] = train.event_type.map(WEIGHTS)
            train = train.loc[train.strength.notna()].drop(columns="event_type")
            train["strength"] = train.strength.astype("uint8")
            history.append(train)
            future = chunk.loc[(chunk.event_time >= cutoff) & (chunk.event_time < end)]
            counts.update({f"holdout_{kind}": n for kind, n in future.event_type.value_counts().items()})
            strength = future.event_type.map(WEIGHTS)
            labels = future.loc[strength.notna(), ["user_id", "product_id"]].copy()
            labels["strength"] = strength.loc[labels.index].astype("uint8")
            holdout.append(labels)
    if not history or not holdout:
        raise ValueError("No input rows")
    history = pd.concat(history, ignore_index=True)
    holdout = pd.concat(holdout, ignore_index=True)
    if history.empty or holdout.empty:
        raise ValueError("Training events and held-out purchases are both required")
    return history, holdout, dict(counts)


def build_training_data(history):
    history.sort_values("event_time", kind="stable", inplace=True)
    columns = ["product_id", "category_id", *[name for name in ("category_code", "brand", "price") if name in history]]
    catalog = history[columns].drop_duplicates("product_id", keep="last").set_index("product_id")
    profiles = history.groupby(["user_id", "product_id"], sort=False, as_index=False).agg(
        last_time=("event_time", "last"), strength=("strength", "max")
    )
    user_codes, users = pd.factorize(profiles.user_id, sort=True)
    item_codes, items = pd.factorize(profiles.product_id, sort=True)
    profiles["item"] = item_codes
    interaction = csr_matrix((np.ones(len(profiles), dtype=np.int32), (user_codes, item_codes)), shape=(len(users), len(items)))
    support = np.asarray(interaction.sum(axis=0)).ravel()
    buyers = history.loc[history.strength == WEIGHTS["purchase"]].groupby("product_id").user_id.nunique()
    purchase_support = buyers.reindex(items, fill_value=0).to_numpy()
    # Primary sort is distinct training buyers, then distinct interacting users and item ID.
    popular = np.lexsort((items, -support, -purchase_support))[:MAX_K]
    return profiles, interaction, support, popular, users, items, catalog.loc[items].rename_axis("product_id").reset_index()


def fit_neighbors(interaction, support, all_seeds):
    by_item = interaction.T.tocsr()
    neighbors = {}
    for start in range(0, len(all_seeds), 96):
        seeds = all_seeds[start:start + 96]
        counts = (by_item[seeds] @ interaction).tocsr()
        for offset, item in enumerate(seeds):
            row = slice(counts.indptr[offset], counts.indptr[offset + 1])
            candidate, common = counts.indices[row], counts.data[row]
            valid = (candidate != item) & (common >= 2)
            candidate, common = candidate[valid], common[valid].astype(np.float64)
            similarity = common / np.sqrt(support[item] * support[candidate])
            similarity *= common / (common + SHRINKAGE)
            order = np.lexsort((candidate, -similarity))[:NEIGHBORS]
            neighbors[int(item)] = (candidate[order], similarity[order])
    return neighbors


def fill_with_popularity(primary, popular):
    return list(dict.fromkeys([*primary, *popular]))[:MAX_K]


def rank(seeds, strengths, popular, neighbors, catalog_categories):
    scores = Counter()
    for position, (seed, strength) in enumerate(zip(seeds[:MAX_SEEDS], strengths[:MAX_SEEDS], strict=True), 1):
        candidate, similarity = neighbors[seed]
        mass = float(strength) / sqrt(position)
        for item, signal in zip(candidate, similarity * mass, strict=True):
            scores[item] += signal
    itemcf = fill_with_popularity(sorted(scores, key=lambda item: (-scores[item], item))[:MAX_K], popular)
    affinity = Counter()
    for item, strength in zip(seeds, strengths, strict=True):
        affinity[catalog_categories[item]] += int(strength)
    total = affinity.total()
    rerank = sorted(
        enumerate(itemcf, 1),
        key=lambda position_item: -(
            1 / (PROFILE_OFFSET + position_item[0])
            + PROFILE_WEIGHT * affinity[catalog_categories[position_item[1]]] / total
        ),
    )
    return {
        "popularity": popular,
        "recent_items": fill_with_popularity(seeds, popular),
        "itemcf": itemcf,
        "itemcf_category_rerank": [item for _, item in rerank],
    }, min(MAX_K, len(scores))


def binary_metrics(recommendations, truth):
    # Ranks are already unique. Repeated purchases are deduplicated in the label set.
    hits = np.fromiter((item in truth for item in recommendations), dtype=bool)
    result = {}
    for k in CUTOFFS:
        matched = int(hits[:k].sum())
        result[f"recall@{k}"] = matched / len(truth)
        if k <= 20:
            result[f"precision@{k}"] = matched / k
            result[f"hit@{k}"] = float(matched > 0)
            result[f"ndcg@{k}"] = float(hits[:k] @ DISCOUNTS[:min(k, len(hits))] / IDEAL[min(len(truth), k)])
    return result


def graded_ndcg_at_10(recommendations, truth):
    actual = [truth.get(item, 0) for item in recommendations[:10]]
    ideal = sorted(truth.values(), reverse=True)[:10]
    discount = DISCOUNTS[:10]
    return float((2 ** np.array(actual) - 1) @ discount[:len(actual)] / ((2 ** np.array(ideal) - 1) @ discount[:len(ideal)]))


def mean_metrics(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}


def paired_intervals(metrics, seed, draws):
    pairs = [("itemcf", "popularity"), ("itemcf", "recent_items"), ("itemcf_category_rerank", "itemcf")]
    columns, keys = [], []
    for model, baseline in pairs:
        for key in ("recall@10", "ndcg@10"):
            keys.append((f"{model} minus {baseline}", key))
            columns.append([a[key] - b[key] for a, b in zip(metrics[model], metrics[baseline], strict=True)])
    scores = np.asarray(columns).T
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 64):
        indices = rng.integers(0, len(scores), size=(min(64, draws - start), len(scores)))
        samples.extend(scores[indices].mean(axis=1))
    low, high = np.quantile(samples, [0.025, 0.975], axis=0)
    result = {}
    for index, (pair, key) in enumerate(keys):
        result.setdefault(pair, {})[key] = {
            "mean_delta": float(scores[:, index].mean()),
            "ci95": [float(low[index]), float(high[index])],
        }
    return result


def llm_case(case_number, candidate_items, seeds, strengths, truth_products, catalog):
    def category_name(row):
        return str(row.category_code) if pd.notna(row.category_code) else f"category_{row.category_id}"

    def product_info(row):
        category = category_name(row)
        brand = str(row.brand) if pd.notna(row.brand) else ""
        label = " ".join(x for x in (brand, category, str(row.product_id)) if x)
        return {
            "id": str(row.product_id), "name": label, "category": category,
            "price": float(row.price) if np.isfinite(row.price) else None,
            "brand": brand, "tags": [],
        }

    affinity = Counter()
    observed = catalog.iloc[seeds]
    for row, strength in zip(observed.itertuples(index=False), strengths, strict=True):
        affinity[category_name(row)] += int(strength)
    price = observed.price.to_numpy()
    valid = price[np.isfinite(price)]
    return {
        "case_id": f"case-{case_number:03}",
        "candidates": [product_info(row) for row in catalog.iloc[candidate_items[:20]].itertuples(index=False)],
        "positives": [str(item) for item in sorted(truth_products)],
        "profile": {
            "segments": [],
            "preferred_categories": [label for label, _ in affinity.most_common(3)],
            "price_range": np.quantile(valid, [0.25, 0.75]).tolist() if len(valid) else [],
        },
    }


def evaluate(args):
    start = perf_counter()
    history, holdout_labels, counts = load_events(args.events, args.train_start, args.train_end, args.eval_end, args.llm_cases is not None)
    holdout = holdout_labels.loc[holdout_labels.strength == WEIGHTS["purchase"], ["user_id", "product_id"]].drop_duplicates()
    if holdout.empty:
        raise ValueError("No purchases in the holdout interval")
    profiles, interactions, support, popular, users, items, catalog = build_training_data(history)
    categories = catalog.category_id.to_numpy()
    training_pairs = len(profiles)
    history_events = len(history)
    observed_range = [history.event_time.iloc[0], history.event_time.iloc[-1]]
    del history
    eligible = np.intersect1d(holdout.user_id.unique(), users)
    train_known = holdout.product_id.isin(items).to_numpy()
    warm_user = holdout.user_id.isin(users).to_numpy()
    if not len(eligible):
        raise ValueError("No held-out buyers have training history")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(eligible, size=min(len(eligible), args.max_users), replace=False))
    case_users = set(rng.choice(selected, size=min(len(selected), args.case_count), replace=False)) if args.llm_cases else set()
    profiles = profiles.loc[profiles.user_id.isin(selected)].sort_values(
        ["user_id", "last_time", "item"], ascending=[True, False, True]
    )
    all_seeds = profiles.groupby("user_id", sort=False).head(MAX_SEEDS).item.unique()
    neighbors = fit_neighbors(interactions, support, all_seeds)
    item_index = {int(item): i for i, item in enumerate(items)}
    actual = {
        uid: set(group.product_id)
        for uid, group in holdout.loc[holdout.user_id.isin(selected)].groupby("user_id", sort=False)
    }
    strongest = holdout_labels.loc[holdout_labels.user_id.isin(selected)].groupby(["user_id", "product_id"], sort=False).strength.max()
    graded_truth = {
        uid: {item_index.get(product, -index): int(strength) for index, ((_, product), strength) in enumerate(labels.items(), 1)}
        for uid, labels in strongest.groupby(level=0, sort=False)
    }
    del holdout_labels
    slices = ("all_purchases", "known_catalog", "novel_to_user")
    per_user = {section: {method: [] for method in METHODS} for section in slices}
    recommended = {method: set() for method in METHODS}
    intra_list_diversity = {method: [] for method in METHODS}
    graded_scores = {method: [] for method in METHODS}
    sampled_positives = Counter()
    cold_retrieval = Counter()
    cases = []

    for uid, group in profiles.groupby("user_id", sort=False):
        seeds, strengths = group.item.to_numpy(), group.strength.to_numpy()
        known = {item_index[item] for item in actual[uid] if item in item_index}
        truth = known | set(range(len(known) - len(actual[uid]), 0))
        novel = known - set(seeds)
        truths = (truth, known, novel)
        lists, personalized_count = rank(seeds, strengths, popular, neighbors, categories)
        if uid in case_users:
            cases.append(llm_case(len(cases) + 1, lists["itemcf"], seeds, strengths, actual[uid], catalog))
        if not personalized_count:
            cold_retrieval["entirely_popularity_fallback"] += 1
        elif personalized_count < MAX_K:
            cold_retrieval["partly_popularity_fallback"] += 1
        for section, ground_truth in zip(slices, truths, strict=True):
            sampled_positives[f"{section}_pairs"] += len(ground_truth)
            if not ground_truth:
                continue
            for method, listing in lists.items():
                per_user[section][method].append(binary_metrics(listing, ground_truth))
        for method, listing in lists.items():
            graded_scores[method].append(graded_ndcg_at_10(listing, graded_truth[uid]))
            head = listing[:10]
            recommended[method].update(head)
            if len(head) > 1:
                counts_by_category = Counter(categories[head])
                same_pairs = sum(n * (n - 1) for n in counts_by_category.values())
                intra_list_diversity[method].append(1 - same_pairs / (len(head) * (len(head) - 1)))

    if args.llm_cases:
        args.llm_cases.write_text("".join(json.dumps(case) + "\n" for case in cases))

    result = {
        "utc_intervals_half_open": {"train": [args.train_start, args.train_end], "holdout": [args.train_end, args.eval_end]},
        "first_last_training_event_utc": observed_range,
        "counts": counts | {
            "training_model_events": history_events,
            "training_distinct_user_item_pairs": training_pairs,
            "training_users": len(users),
            "training_catalog_items": len(items),
            "holdout_buyers": holdout.user_id.nunique(),
            "holdout_distinct_purchase_pairs": len(holdout),
            "holdout_purchase_pairs_unseen_catalog": int((~train_known).sum()),
            "holdout_buyers_without_training_history": holdout.user_id.nunique() - len(eligible),
            "holdout_purchase_pairs_without_training_history": int((~warm_user).sum()),
            "sampled_distinct_graded_holdout_pairs": len(strongest),
            "eligible_buyers_with_training_history": len(eligible),
            "sampled_eligible_buyers": len(selected),
            "exported_llm_cases": len(cases),
            "itemcf_entirely_popularity_fallback_users": cold_retrieval["entirely_popularity_fallback"],
            "itemcf_partly_popularity_fallback_users": cold_retrieval["partly_popularity_fallback"],
        } | dict(sampled_positives),
        "slices": {
            section: {"users": len(records["popularity"]), "metrics": {method: mean_metrics(rows) for method, rows in records.items()}}
            for section, records in per_user.items()
        },
        "coverage_and_diversity_at_10": {
            method: {
                "unique_recommended_items": len(recommended[method]),
                "catalog_coverage": len(recommended[method]) / len(items),
                "category_ild": float(np.mean(intra_list_diversity[method])) if intra_list_diversity[method] else None,
            }
            for method in METHODS
        },
        "graded_ndcg_at_10_on_same_buyer_cohort": {method: float(np.mean(values)) for method, values in graded_scores.items()},
        "all_purchases_paired_user_bootstrap": paired_intervals(per_user["all_purchases"], args.seed, args.bootstrap),
        "config": {
            "seed": args.seed, "max_users": args.max_users, "bootstrap_draws": args.bootstrap,
            "itemcf_max_seeds": MAX_SEEDS, "itemcf_neighbors": NEIGHBORS,
            "itemcf_shrinkage": SHRINKAGE, "itemcf_min_common_users": 2,
            "profile_weight": PROFILE_WEIGHT, "profile_rank_offset": PROFILE_OFFSET, "event_weights": WEIGHTS,
        },
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "inputs": [{"path": p.name, "bytes": p.stat().st_size, "sha256": checksum(p)} for p in args.events],
        "elapsed_seconds": round(perf_counter() - start, 3),
    }
    return result


def checksum(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def utc_boundary(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.microsecond:
        raise ValueError("UTC boundary must be a whole second")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    for boundary in ("train-start", "train-end", "eval-end"):
        parser.add_argument(f"--{boundary}", type=utc_boundary, required=True, help="UTC date or ISO timestamp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--llm-cases", type=Path, help="Write paired anonymous ItemCF top-20 cases and separate purchase labels")
    parser.add_argument("--case-count", type=int, default=48)
    args = parser.parse_args()
    if not args.train_start < args.train_end < args.eval_end or min(args.max_users, args.bootstrap, args.case_count) < 1:
        parser.error("Require increasing UTC boundaries and positive sample/bootstrap sizes")
    result = evaluate(args)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["slices"]["all_purchases"], indent=2))


if __name__ == "__main__":
    main()
