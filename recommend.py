"""Deterministic e-commerce recommendation pipeline for the portfolio demo."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json


@dataclass(frozen=True)
class Profile:
    label: str
    interest: str
    budget: int
    copy: str


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    category: str
    price: int
    popularity: int
    stock: int


PROFILES = {
    "desk": Profile("远程办公", "桌面效率", 399, "给日常办公桌留一点余地"),
    "travel": Profile("通勤差旅", "便携出行", 299, "把常用装备轻装带走"),
    "budget": Profile("预算优先", "桌面效率", 119, "用小预算补齐常用装备"),
}
CATALOG = (
    Product("stand", "铝合金笔记本支架", "桌面效率", 169, 9, 12),
    Product("light", "桌面屏幕挂灯", "桌面效率", 299, 8, 7),
    Product("mat", "软木桌垫", "桌面效率", 89, 7, 20),
    Product("charger", "65W 折叠充电器", "便携出行", 229, 9, 14),
    Product("pouch", "防泼水数码收纳包", "便携出行", 99, 8, 10),
    Product("cable", "短款编织数据线", "便携出行", 49, 6, 16),
)
STOCK_SCENARIOS = {
    "normal": ("正常供应", frozenset()),
    "shortage": ("两款热门商品缺货", frozenset({"stand", "charger"})),
    "empty": ("全场缺货", frozenset(product.id for product in CATALOG)),
}


def rank_products(profile, candidates):
    def score(product):
        return product.popularity + (10 if product.category == profile.interest else 0)

    eligible = [product for product in candidates if product.price <= profile.budget]
    return sorted(eligible, key=lambda product: (-score(product), product.id))


def check_inventory(candidates, out_of_stock):
    return {product.id: product.stock for product in candidates if product.id not in out_of_stock}


def write_marketing_copy(profile, selected):
    if selected:
        return f"{profile.copy}，可以先看看{selected[0].name}。"
    return None


def recommend(profile_id, scenario_id):
    """Use one candidate set for ranking and inventory checks, then select Top-K."""
    profile = PROFILES[profile_id]
    _, out_of_stock = STOCK_SCENARIOS[scenario_id]
    ranked = rank_products(profile, CATALOG)
    stock = check_inventory(ranked, out_of_stock)
    selected = [product for product in ranked if stock.get(product.id, 0) > 0][:3]
    items = [
        {
            "name": product.name,
            "category": product.category,
            "price": product.price,
            "stock": stock[product.id],
            "reason": "匹配常购品类" if product.category == profile.interest else "预算内的补充选项",
        }
        for product in selected
    ]
    return {
        "trace": {
            "interest": profile.interest,
            "budget": profile.budget,
            "candidateCount": len(CATALOG),
            "budgetExcluded": len(CATALOG) - len(ranked),
            "outOfStock": [product.name for product in ranked if product.id in out_of_stock],
        },
        "items": items,
        "copy": write_marketing_copy(profile, selected),
    }


def build_site():
    root = Path(__file__).resolve().parent
    data = {
        "profiles": {key: profile.label for key, profile in PROFILES.items()},
        "scenarios": {key: label for key, (label, _) in STOCK_SCENARIOS.items()},
        "runs": {
            profile: {scenario: recommend(profile, scenario) for scenario in STOCK_SCENARIOS}
            for profile in PROFILES
        },
    }
    source = (root / "page.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(source.replace("<!-- DEMO_DATA -->", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="desk")
    parser.add_argument("--stock", choices=STOCK_SCENARIOS, default="normal")
    parser.add_argument("--build", action="store_true", help="Generate the standalone portfolio page.")
    args = parser.parse_args()
    if args.build:
        build_site()
    else:
        print(json.dumps(recommend(args.profile, args.stock), ensure_ascii=False, indent=2))
