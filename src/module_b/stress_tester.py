"""Scenario-based stress testing for a synthetic wholesale banking portfolio."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.engine.schemas import FinancialEvent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORTFOLIO = PROJECT_ROOT / "data" / "demo" / "banking_portfolio.json"
DEFAULT_SCENARIOS = PROJECT_ROOT / "data" / "demo" / "bank_stress_scenarios.json"


@dataclass(frozen=True)
class BankingPosition:
    position_id: str
    name: str
    asset_class: str
    sector: str
    market_value: float
    modified_duration: float = 0.0
    spread_duration: float = 0.0
    base_default_rate: float = 0.0


@dataclass(frozen=True)
class StressScenarioRule:
    scenario_id: str
    name: str
    description: str
    event_types: tuple[str, ...]
    minimum_impact: float
    shocks: dict[str, Any]


@dataclass(frozen=True)
class PositionStressResult:
    position_id: str
    name: str
    asset_class: str
    sector: str
    market_value_before: float
    return_shock: float
    loss: float
    market_value_after: float
    contribution_to_net_loss: float


@dataclass(frozen=True)
class StressTestResult:
    run_id: str
    scenario_id: str
    scenario_name: str
    trigger_event_id: str
    trigger_event_type: str
    trigger_impact: float
    trigger_reason: str
    portfolio_value_before: float
    portfolio_value_after: float
    net_loss: float
    net_loss_pct: float
    positions: list[PositionStressResult]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_banking_portfolio(path: str | Path = DEFAULT_PORTFOLIO) -> tuple[str, str, list[BankingPosition]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    positions = [
        BankingPosition(
            position_id=str(item["position_id"]),
            name=str(item["name"]),
            asset_class=str(item["asset_class"]).lower(),
            sector=str(item["sector"]),
            market_value=float(item["market_value"]),
            modified_duration=float(item.get("modified_duration", 0.0)),
            spread_duration=float(item.get("spread_duration", 0.0)),
            base_default_rate=float(item.get("base_default_rate", 0.0)),
        )
        for item in payload["positions"]
    ]
    if not positions or any(position.market_value < 0 for position in positions):
        raise ValueError("Banking portfolio requires non-negative positions.")
    total_assets = sum(position.market_value for position in positions)
    declared_total = float(payload.get("total_assets", total_assets))
    if not np.isclose(total_assets, declared_total, rtol=1e-6, atol=1.0):
        raise ValueError(
            f"Position values sum to {total_assets:,.2f}, but total_assets is {declared_total:,.2f}."
        )
    return str(payload["portfolio_id"]), str(payload["name"]), positions


def load_stress_scenarios(path: str | Path = DEFAULT_SCENARIOS) -> list[StressScenarioRule]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = []
    for item in payload:
        rules.append(
            StressScenarioRule(
                scenario_id=str(item["scenario_id"]),
                name=str(item["name"]),
                description=str(item["description"]),
                event_types=tuple(str(x) for x in item.get("event_types", [])),
                minimum_impact=float(item.get("minimum_impact", 8.0)),
                shocks=dict(item.get("shocks", {})),
            )
        )
    return rules


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class PortfolioStressTester:
    """Apply transparent balance-sheet shocks when high-impact events qualify."""

    def __init__(
        self,
        portfolio_path: str | Path = DEFAULT_PORTFOLIO,
        scenario_path: str | Path = DEFAULT_SCENARIOS,
    ):
        self.portfolio_id, self.portfolio_name, self.positions = load_banking_portfolio(portfolio_path)
        self.scenarios = load_stress_scenarios(scenario_path)
        self.initial_assets = sum(position.market_value for position in self.positions)

    @staticmethod
    def _position_return(position: BankingPosition, shocks: dict[str, Any]) -> tuple[float, float]:
        """Return (mark-to-market return, incremental credit loss rate)."""
        asset_class = position.asset_class
        asset_returns = shocks.get("asset_returns", {})
        sector_returns = shocks.get("sector_returns", {})
        incremental_default_by_sector = shocks.get("default_rate_increase_by_sector", {})
        rate_change = float(shocks.get("interest_rate_change", 0.0))
        spread_change = float(shocks.get("credit_spread_change", 0.0))

        if asset_class in {"equities", "equity", "stocks"}:
            broad = float(asset_returns.get("equities", 0.0))
            sector = float(sector_returns.get(position.sector, 0.0))
            return (1.0 + broad) * (1.0 + sector) - 1.0, 0.0

        if asset_class in {"government_bonds", "gov_bonds", "treasuries"}:
            return max(-0.99, -position.modified_duration * rate_change), 0.0

        if asset_class in {"corporate_bonds", "corp_bonds", "bonds"}:
            bond_return = (
                -position.modified_duration * rate_change
                - position.spread_duration * spread_change
            )
            return max(-0.99, bond_return), 0.0

        if asset_class in {"loans", "loan", "credit"}:
            default_increase = float(
                incremental_default_by_sector.get(
                    position.sector,
                    shocks.get("default_rate_increase", 0.0),
                )
            )
            incremental_loss_rate = float(
                np.clip(default_increase, 0.0, max(0.0, 1.0 - position.base_default_rate))
            )
            return 0.0, incremental_loss_rate

        if asset_class == "cash":
            return float(asset_returns.get("cash", 0.0)), 0.0

        # Unknown asset classes get only an explicitly configured class shock.
        return float(asset_returns.get(asset_class, 0.0)), 0.0

    def apply_scenario(
        self,
        event: FinancialEvent,
        scenario: StressScenarioRule,
    ) -> StressTestResult:
        rows: list[PositionStressResult] = []
        before = sum(position.market_value for position in self.positions)
        after = 0.0

        for position in self.positions:
            return_shock, credit_loss_rate = self._position_return(position, scenario.shocks)
            market_value_loss = position.market_value * credit_loss_rate
            mark_to_market_pnl = position.market_value * return_shock
            net_loss = market_value_loss - mark_to_market_pnl
            post_value = max(0.0, position.market_value - net_loss)
            after += post_value
            rows.append(
                PositionStressResult(
                    position_id=position.position_id,
                    name=position.name,
                    asset_class=position.asset_class,
                    sector=position.sector,
                    market_value_before=position.market_value,
                    return_shock=return_shock,
                    loss=net_loss,
                    market_value_after=post_value,
                    contribution_to_net_loss=0.0,
                )
            )

        total_loss = before - after
        rows = [
            PositionStressResult(
                **{
                    **asdict(row),
                    "contribution_to_net_loss": (
                        row.loss / total_loss if total_loss > 0 else 0.0
                    ),
                }
            )
            for row in rows
        ]
        timestamp = _utc(event.timestamp).strftime("%Y%m%dT%H%M%SZ")
        return StressTestResult(
            run_id=f"{scenario.scenario_id}_{event.event_id}_{timestamp}",
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            trigger_event_id=event.event_id,
            trigger_event_type=event.event_type.label,
            trigger_impact=float(event.impact.score),
            trigger_reason=(
                f"{event.event_type.label} event impact {event.impact.score:.2f} "
                f">= configured threshold {scenario.minimum_impact:.2f}"
            ),
            portfolio_value_before=before,
            portfolio_value_after=after,
            net_loss=total_loss,
            net_loss_pct=(total_loss / before if before > 0 else 0.0),
            positions=rows,
        )

    def evaluate_events(
        self,
        events: Iterable[FinancialEvent | dict[str, Any]],
    ) -> list[StressTestResult]:
        parsed = [
            event if isinstance(event, FinancialEvent) else FinancialEvent.model_validate(event)
            for event in events
        ]
        rules = [
            rule for rule in self.scenarios
            if rule.event_types and rule.minimum_impact >= 1.0
        ]

        # A news cluster can contain several stories about one event. Trigger at
        # most once per cluster and scenario, selecting the highest-impact record.
        selected: dict[tuple[str, str], tuple[float, FinancialEvent, StressScenarioRule]] = {}
        for event in parsed:
            event_label = event.event_type.label.strip().lower()
            for rule in rules:
                allowed = {value.strip().lower() for value in rule.event_types}
                if event_label not in allowed or event.impact.score < rule.minimum_impact:
                    continue
                cluster_key = event.cluster_id or event.event_id
                key = (cluster_key, rule.scenario_id)
                previous = selected.get(key)
                if previous is None or event.impact.score > previous[0]:
                    selected[key] = (event.impact.score, event, rule)

        results = [
            self.apply_scenario(event, rule)
            for _, event, rule in selected.values()
        ]
        results.sort(key=lambda result: result.trigger_impact, reverse=True)
        return results

    @staticmethod
    def save_results(results: Iterable[StressTestResult], path: str | Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = list(results)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        return len(rows)
