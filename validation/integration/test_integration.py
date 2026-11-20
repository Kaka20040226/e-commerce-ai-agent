"""Offline integration tests for the unchanged upstream Python implementation."""

import asyncio
import csv
import gzip
import importlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI


@dataclass
class LLMStub:
    profile: dict = field(default_factory=lambda: {
        "segments": ["high_value"],
        "price_range": [45, 350],
    })
    ranked_ids: list = field(default_factory=lambda: ["SKU-2", "SKU-1"])
    copies: list | None = None
    malformed: set = field(default_factory=set)
    timeout: set = field(default_factory=set)
    delays: dict = field(default_factory=dict)
    calls: dict = field(default_factory=lambda: {"profile": [], "rerank": [], "copy": []})

    async def reply(self, messages):
        human = messages[-1].content
        if human.startswith("用户ID: "):
            stage = "profile"
            content = self.profile
        elif human.startswith("你是电商推荐排序专家"):
            stage = "rerank"
            content = self.ranked_ids
        else:
            assert human.startswith("商品列表:\n"), human
            stage = "copy"
            product_ids = re.findall(r"^- ID:(\S+)", human, re.MULTILINE)
            content = self.copies if self.copies is not None else [
                {"product_id": pid, "copy": "最好，适合日常使用。"} for pid in product_ids
            ]
        self.calls[stage].append([message.content for message in messages])
        if stage in self.timeout:
            raise TimeoutError("simulated model request timeout")
        if stage in self.delays:
            await asyncio.sleep(self.delays[stage])
        if stage in self.malformed:
            return SimpleNamespace(content="not JSON")
        return SimpleNamespace(content=json.dumps(content, ensure_ascii=False))


@pytest.fixture
def llm(monkeypatch):
    # Set a fake credential and a loopback URL before importing clients or the app.
    # Inference is intercepted earlier at ChatOpenAI.ainvoke in every test.
    monkeypatch.setenv("ECOM_LLM_API_KEY", "offline-integration-placeholder")
    monkeypatch.setenv("ECOM_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    stub = LLMStub()

    async def mock_ainvoke(client, messages, **kwargs):
        return await stub.reply(messages)

    monkeypatch.setattr(ChatOpenAI, "ainvoke", mock_ainvoke)
    return stub


@pytest.fixture
def catalogue(monkeypatch, llm):
    from agents import product_rec_agent
    from models.schemas import Product

    items = [
        Product(
            product_id=f"SKU-{index}", name=f"Fixture {index}",
            category="音频" if index <= 2 else "配件", price=90 + index,
            stock=stock, tags=["新品"] if index == 4 else [],
        )
        for index, stock in enumerate([150, 32, 200, 80, 200, 200], start=1)
    ]
    # Replace fixture inputs at the catalogue boundary only. Inventory still uses
    # the unchanged Agent's fallback_stock, alert and limit logic.
    monkeypatch.setattr(product_rec_agent, "MOCK_PRODUCTS", items)
    return items


def recommend_direct(user_id="offline-user", *, supervisor=None):
    from models.schemas import RecommendationRequest
    from orchestrator.supervisor import SupervisorOrchestrator

    request = RecommendationRequest(
        user_id=user_id, num_items=2, context={"recent_views": ["音频"], "view_count_7d": 17},
    )
    return asyncio.run((supervisor or SupervisorOrchestrator()).recommend(request))


def product_ids(products):
    return [product.product_id for product in products]


def test_supervisor_passes_context_profile_stock_and_products(llm, catalogue):
    llm.profile["preferred_categories"] = ["音频"]
    result = recommend_direct()

    assert product_ids(result.products) == ["SKU-2", "SKU-1"]
    assert result.agent_results["user_profile"].profile.preferred_categories == ["音频"]
    assert result.agent_results["product_rec"].data == {"candidate_count": 6, "reranked": 2}
    inventory = result.agent_results["inventory"]
    assert inventory.available_products == ["SKU-1", "SKU-2", "SKU-3", "SKU-4"]
    assert inventory.purchase_limits == {"SKU-2": 1, "SKU-4": 2}
    assert {alert["level"] for alert in inventory.low_stock_alerts} == {"critical", "warning"}
    assert result.agent_results["marketing_copy"].prompt_template_used == "high_value"
    assert result.marketing_copies == [
        {"product_id": pid, "copy": "***，适合日常使用。"} for pid in ["SKU-2", "SKU-1"]
    ]
    assert {stage: len(calls) for stage, calls in llm.calls.items()} == {
        "profile": 1, "rerank": 1, "copy": 1,
    }
    assert '"view_count_7d": 17' in llm.calls["profile"][0][-1]
    assert '"preferred_categories": ["音频"]' in llm.calls["rerank"][0][-1]
    assert "尊享感" in llm.calls["copy"][0][0]
    assert "SKU-3" not in llm.calls["copy"][0][-1]


def test_partial_stock_filters_unavailable_and_does_not_refill(llm, catalogue):
    catalogue[0].stock = 0
    llm.ranked_ids = ["SKU-1", "SKU-2"]
    result = recommend_direct()

    assert result.agent_results["inventory"].available_products == ["SKU-2", "SKU-3", "SKU-4"]
    assert product_ids(result.products) == ["SKU-2"]
    assert [item["product_id"] for item in result.marketing_copies] == ["SKU-2"]
    assert "SKU-1" not in llm.calls["copy"][0][-1]


def test_all_stock_empty_characterizes_unsafe_ranked_fallback(llm, catalogue):
    for item in catalogue:
        item.stock = 0
    result = recommend_direct()

    assert result.agent_results["inventory"].available_products == []
    assert [(product.product_id, product.stock) for product in result.products] == [
        ("SKU-2", 0), ("SKU-1", 0),
    ]
    assert [item["product_id"] for item in result.marketing_copies] == ["SKU-2", "SKU-1"]


def test_disjoint_ranking_characterizes_inventory_coverage_gap(llm, catalogue):
    llm.ranked_ids = ["SKU-5", "SKU-6"]
    result = recommend_direct()
    assert result.agent_results["inventory"].available_products == ["SKU-1", "SKU-2", "SKU-3", "SKU-4"]
    assert product_ids(result.products) == ["SKU-5", "SKU-6"]
    assert [item["product_id"] for item in result.marketing_copies] == ["SKU-5", "SKU-6"]


def test_marketing_accepts_unknown_product_and_invented_price(llm, catalogue):
    llm.copies = [
        {"product_id": "NOT-IN-CATALOGUE", "copy": "这款商品今天 ¥1。"},
        {"product_id": "SKU-2", "copy": "这款商品今天 ¥1。"},
    ]
    app = importlib.import_module("main").app
    with TestClient(app) as client:
        response = client.post("/api/v1/recommend", json={"user_id": "offline-marketing", "num_items": 2})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert {item["product_id"]: item["price"] for item in payload["products"]} == {
        "SKU-2": 92.0, "SKU-1": 91.0,
    }
    assert "NOT-IN-CATALOGUE" not in llm.calls["copy"][0][-1]
    assert "价格:¥92.0" in llm.calls["copy"][0][-1]
    assert payload["agent_results"]["marketing_copy"]["success"]
    assert payload["marketing_copies"] == llm.copies


def test_rerank_valid_json_allows_duplicate_product_ids(llm, catalogue):
    llm.ranked_ids = ["SKU-2", "SKU-2"]
    result = recommend_direct()

    assert result.agent_results["product_rec"].success
    assert product_ids(result.products) == ["SKU-2", "SKU-2"]
    assert [item["product_id"] for item in result.marketing_copies] == ["SKU-2", "SKU-2"]


def test_model_timeout_error_retries_and_falls_back_to_raw_products(llm, catalogue):
    llm.timeout.add("rerank")
    result = recommend_direct()

    rerank = result.agent_results["product_rec"]
    assert not rerank.success
    assert rerank.error == "simulated model request timeout"
    assert len(llm.calls["rerank"]) == 2
    assert product_ids(result.products) == ["SKU-1", "SKU-2"]
    assert [item["product_id"] for item in result.marketing_copies] == ["SKU-1", "SKU-2"]


def test_agent_timeout_setting_does_not_bound_slow_model(llm, catalogue):
    from orchestrator.supervisor import SupervisorOrchestrator

    llm.delays["rerank"] = 0.04
    supervisor = SupervisorOrchestrator()
    supervisor.product_rec_agent.timeout = 0.005
    result = recommend_direct(supervisor=supervisor)

    rerank = result.agent_results["product_rec"]
    assert rerank.success
    assert rerank.latency_ms > supervisor.product_rec_agent.timeout * 1000
    assert len(llm.calls["rerank"]) == 1
    assert product_ids(result.products) == ["SKU-2", "SKU-1"]


def test_malformed_json_uses_stage_specific_fallbacks(llm, catalogue):
    llm.malformed.update(["profile", "rerank", "copy"])
    result = recommend_direct()

    assert result.agent_results["user_profile"].profile.model_dump()["segments"] == ["active"]
    assert all(agent.success for agent in result.agent_results.values())
    assert product_ids(result.products) == ["SKU-1", "SKU-2"]
    assert result.marketing_copies == []
    assert result.agent_results["marketing_copy"].prompt_template_used == "active"
    assert {stage: len(calls) for stage, calls in llm.calls.items()} == {
        "profile": 1, "rerank": 1, "copy": 1,
    }


def test_experiment_labels_do_not_dispatch_different_strategies(llm, catalogue):
    from services.ab_test import ABTestEngine

    engine = ABTestEngine()
    users = {}
    for index in range(100):
        user = f"offline-bucket-{index}"
        assignment = engine.assign(user)
        users.setdefault(assignment["group"], (user, assignment["config"]))
    assert users["control"][1]["rerank"] == "rule_based"
    assert users["treatment_llm"][1]["rerank"] == "llm"
    control = recommend_direct(users["control"][0])
    treatment = recommend_direct(users["treatment_llm"][0])

    assert [control.experiment_group, treatment.experiment_group] == ["control", "treatment_llm"]
    assert len(llm.calls["rerank"]) == 2
    assert llm.calls["rerank"][0] == llm.calls["rerank"][1]
    assert product_ids(control.products) == product_ids(treatment.products) == ["SKU-2", "SKU-1"]


@pytest.mark.parametrize("route", ["/api/v1/recommend", "/api/v1/recommend/graph"])
def test_fastapi_recommendation_routes_use_real_entry_points(llm, catalogue, route):
    app = importlib.import_module("main").app
    catalogue[0].stock = 0
    llm.ranked_ids = ["SKU-1", "SKU-2"]
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post(route, json={"user_id": "offline-api", "num_items": 2})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["user_id"] == "offline-api"
    assert payload["request_id"]
    assert payload["total_latency_ms"] >= 0
    assert [item["product_id"] for item in payload["products"]] == ["SKU-2"]
    assert payload["marketing_copies"] == [{"product_id": "SKU-2", "copy": "***，适合日常使用。"}]
    if route == "/api/v1/recommend":
        assert set(payload["agent_results"]) == {"user_profile", "product_rec", "inventory", "marketing_copy"}


@pytest.fixture
def marketplace_products(llm, monkeypatch):
    from agents import product_rec_agent
    from models.schemas import Product

    sample_path = os.environ.get("ECOM_VALIDATION_SAMPLE")
    if not sample_path:
        pytest.skip("Set ECOM_VALIDATION_SAMPLE to run with a local marketplace CSV or CSV.GZ")
    path = Path(sample_path)
    items = {}
    open_sample = gzip.open if path.suffix == ".gz" else open
    with open_sample(path, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            product_id = row["product_id"]
            category = row["category_code"]
            price = float(row["price"])
            if not category or price <= 0 or product_id in items:
                continue
            brand = row["brand"]
            items[product_id] = Product(
                product_id=product_id, name=f"{brand or category} {product_id}",
                category=category, brand=brand, price=price, stock=150,
            )
            if len(items) == 6:
                break
    assert len(items) == 6
    products = list(items.values())
    monkeypatch.setattr(product_rec_agent, "MOCK_PRODUCTS", products)
    return products


@pytest.mark.parametrize("all_out_of_stock", [False, True], ids=["partial-stock", "all-empty"])
def test_marketplace_catalogue_with_simulated_inventory(llm, marketplace_products, all_out_of_stock):
    marketplace_products[0].stock = 0
    if all_out_of_stock:
        for product in marketplace_products:
            product.stock = 0
    else:
        marketplace_products[1].stock = 35
    llm.ranked_ids = product_ids(marketplace_products[:2])
    llm.profile["preferred_categories"] = [marketplace_products[1].category]
    app = importlib.import_module("main").app

    with TestClient(app) as client:
        response = client.post("/api/v1/recommend", json={
            "user_id": "offline-marketplace-example",
            "num_items": 2,
            "context": {"recent_views": [marketplace_products[1].category]},
        })
    assert response.status_code == 200, response.text
    payload = response.json()
    expected = llm.ranked_ids if all_out_of_stock else llm.ranked_ids[1:]
    assert [item["product_id"] for item in payload["products"]] == expected
    assert [item["product_id"] for item in payload["marketing_copies"]] == expected
    assert [item["stock"] for item in payload["products"]] == ([0, 0] if all_out_of_stock else [35])
    assert json.dumps(marketplace_products[1].category, ensure_ascii=False) in llm.calls["rerank"][0][-1]
