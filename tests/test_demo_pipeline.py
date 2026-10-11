"""Network-independent checks for the demo's data contracts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def test_index_demo_articles_only_use_configured_index_tickers():
    articles = json.loads((DEMO / "index_articles.json").read_text(encoding="utf-8"))
    portfolio = json.loads((DEMO / "index_portfolio.json").read_text(encoding="utf-8"))
    portfolio_tickers = {asset["ticker"] for asset in portfolio["assets"]}
    assert len(articles) == 12
    assert len(portfolio_tickers) == 12
    assert all(set(article.get("tickers", [])) <= portfolio_tickers for article in articles)
    assert any("JPM" in article.get("tickers", []) for article in articles)


def test_synthetic_bank_assets_include_derivative_and_reconcile_to_usd_total():
    portfolio = json.loads((DEMO / "banking_portfolio.json").read_text(encoding="utf-8"))
    assert portfolio["currency"] == "USD"
    assert any(position["asset_class"] == "derivatives" for position in portfolio["positions"])
    assert abs(sum(position["market_value"] for position in portfolio["positions"]) - portfolio["total_assets"]) < 1.0
    assert len(portfolio["positions"]) == 9


def test_every_downside_scenario_declares_derivative_shock_proxy():
    scenarios = json.loads((DEMO / "bank_stress_scenarios.json").read_text(encoding="utf-8"))
    assert scenarios
    assert all("derivative_returns" in scenario["shocks"] for scenario in scenarios)
    assert all(scenario["shocks"]["derivative_returns"] for scenario in scenarios)
