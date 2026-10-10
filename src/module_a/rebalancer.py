"""Constrained sentiment-responsive rebalancer for a mock S&P Global 100 subset."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.engine.schemas import FinancialEvent, Portfolio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORTFOLIO = PROJECT_ROOT / "data" / "demo" / "index_portfolio.json"


@dataclass(frozen=True)
class RebalanceResult:
    timestamp: str
    portfolio_id: str
    before_weights: dict[str, float]
    after_weights: dict[str, float]
    weight_changes: dict[str, float]
    event_signals: dict[str, float]
    one_way_turnover: float
    processed_events: int
    unmatched_tickers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_index_portfolio(path: str | Path = DEFAULT_PORTFOLIO) -> Portfolio:
    """Load and validate the mock, real-ticker index portfolio."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return Portfolio.model_validate(record)


def project_bounded_simplex(
    values: Iterable[float],
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Project values onto a capped simplex whose weights sum to one."""
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one weight is required.")
    if not (0 <= lower_bound <= upper_bound <= 1):
        raise ValueError("Weight bounds must satisfy 0 <= lower <= upper <= 1.")
    if values.size * lower_bound > 1 + 1e-12:
        raise ValueError("Lower bound is infeasible for the number of assets.")
    if values.size * upper_bound < 1 - 1e-12:
        raise ValueError("Upper bound is infeasible for the number of assets.")
    if not np.all(np.isfinite(values)):
        raise ValueError("All candidate weights must be finite.")

    left = float(np.min(values) - upper_bound - 1.0)
    right = float(np.max(values) - lower_bound + 1.0)
    for _ in range(200):
        shift = (left + right) / 2.0
        projected = np.clip(values - shift, lower_bound, upper_bound)
        total = float(projected.sum())
        if total > 1.0:
            left = shift
        else:
            right = shift

    projected = np.clip(values - ((left + right) / 2.0), lower_bound, upper_bound)
    # Distribute tiny floating-point residual over uncapped entries.
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-10:
        free = (projected > lower_bound + 1e-9) & (projected < upper_bound - 1e-9)
        if not free.any():
            free = projected < upper_bound - 1e-9 if residual > 0 else projected > lower_bound + 1e-9
        indices = np.flatnonzero(free)
        for index in indices:
            adjustment = residual / max(1, len(indices))
            next_value = np.clip(projected[index] + adjustment, lower_bound, upper_bound)
            residual -= float(next_value - projected[index])
            projected[index] = next_value
            if abs(residual) <= 1e-10:
                break
    return projected


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class SentimentRebalancer:
    """
    Long-only, turnover-capped rebalancer.

    Positive sentiment increases the corresponding ticker's target weight;
    negative sentiment decreases it. Impact/confidence/novelty/decay scale the
    signal. Because linguistic sentiment is not a return forecast, this module
    is a constrained simulation, not an investment recommendation.
    """

    def __init__(
        self,
        portfolio: Portfolio | str | Path | None = None,
        min_weight: float = 0.02,
        max_weight: float = 0.20,
        max_one_way_turnover: float = 0.10,
        sensitivity: float = 0.8,
        lookback_hours: float = 72.0,
    ):
        if portfolio is None:
            portfolio_obj = load_index_portfolio()
        elif isinstance(portfolio, (str, Path)):
            portfolio_obj = load_index_portfolio(portfolio)
        elif isinstance(portfolio, Portfolio):
            portfolio_obj = portfolio
        else:
            portfolio_obj = Portfolio.model_validate(portfolio)

        self.portfolio = portfolio_obj
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.max_one_way_turnover = float(max_one_way_turnover)
        self.sensitivity = float(sensitivity)
        self.lookback_hours = float(lookback_hours)

        if not (0 <= self.max_one_way_turnover <= 1):
            raise ValueError("max_one_way_turnover must be in [0, 1].")
        if self.sensitivity < 0 or self.lookback_hours <= 0:
            raise ValueError("sensitivity must be non-negative and lookback positive.")

        tickers = [asset.ticker.upper() for asset in portfolio_obj.assets]
        initial = [asset.weight for asset in portfolio_obj.assets]
        projected = project_bounded_simplex(initial, self.min_weight, self.max_weight)
        self.weights = dict(zip(tickers, projected.tolist()))
        self.asset_names = {asset.ticker.upper(): asset.name for asset in portfolio_obj.assets}
        self.history: list[RebalanceResult] = []

    def rebalance(
        self,
        events: Iterable[FinancialEvent | dict[str, Any]],
        as_of: datetime | None = None,
    ) -> RebalanceResult:
        events = [
            event if isinstance(event, FinancialEvent) else FinancialEvent.model_validate(event)
            for event in events
        ]
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        as_of = _as_utc(as_of)

        before = self.weights.copy()
        holdings = set(before)
        per_cluster_ticker: dict[tuple[str, str], tuple[float, FinancialEvent]] = {}
        unmatched: set[str] = set()
        cutoff_seconds = self.lookback_hours * 3600.0

        for event in events:
            timestamp = _as_utc(event.timestamp)
            age_seconds = (as_of - timestamp).total_seconds()
            if age_seconds < 0 or age_seconds > cutoff_seconds:
                continue

            symbols = {str(t).strip().upper() for t in (event.tickers or event.affected_assets) if str(t).strip()}
            if not symbols:
                continue

            age_decay = math.exp(-math.log(2.0) * age_seconds / (24.0 * 3600.0))
            sign = float(np.clip(event.sentiment.score, -1.0, 1.0))
            confidence_scale = 0.5 + 0.5 * min(
                1.0,
                max(0.0, (event.sentiment.confidence + event.event_type.confidence) / 2.0),
            )
            impact_scale = float(np.clip(event.impact.score / 10.0, 0.0, 1.0))
            evidence_scale = 0.5 + 0.5 * float(np.clip(event.signal.novelty, 0.0, 1.0))
            decay_scale = float(np.clip(event.signal.decay, 0.0, 1.0))
            event_strength = sign * impact_scale * confidence_scale * evidence_scale * decay_scale * age_decay

            for ticker in symbols:
                if ticker not in holdings:
                    unmatched.add(ticker)
                    continue
                cluster_key = event.cluster_id or event.event_id
                key = (cluster_key, ticker)
                previous = per_cluster_ticker.get(key)
                if previous is None or abs(event_strength) > abs(previous[0]):
                    per_cluster_ticker[key] = (event_strength, event)

        signal_sums = {ticker: 0.0 for ticker in before}
        used_event_ids: set[str] = set()
        for (_, ticker), (strength, event) in per_cluster_ticker.items():
            signal_sums[ticker] += strength
            used_event_ids.add(event.event_id)

        signals = {ticker: float(np.tanh(value)) for ticker, value in signal_sums.items()}
        current = np.array([before[ticker] for ticker in before], dtype=float)
        signal_vector = np.array([signals[ticker] for ticker in before], dtype=float)

        logits = np.log(np.maximum(current, 1e-12)) + self.sensitivity * signal_vector
        logits -= float(logits.max())
        target = np.exp(logits)
        target /= target.sum()
        target = project_bounded_simplex(target, self.min_weight, self.max_weight)

        turnover = float(np.abs(target - current).sum() / 2.0)
        if turnover > self.max_one_way_turnover + 1e-12:
            scale = self.max_one_way_turnover / turnover if turnover else 0.0
            target = current + scale * (target - current)

        tickers_in_order = list(before)
        after = {ticker: float(weight) for ticker, weight in zip(tickers_in_order, target)}
        # A final normalization only corrects floating-point drift; convex
        # interpolation keeps all projected bounds and the turnover cap.
        residual = 1.0 - sum(after.values())
        if abs(residual) > 1e-12:
            after[tickers_in_order[0]] += residual
        turnover = float(sum(abs(after[t] - before[t]) for t in before) / 2.0)
        changes = {ticker: after[ticker] - before[ticker] for ticker in before}

        self.weights = after.copy()
        result = RebalanceResult(
            timestamp=as_of.isoformat(),
            portfolio_id=self.portfolio.portfolio_id,
            before_weights=before,
            after_weights=after,
            weight_changes=changes,
            event_signals=signals,
            one_way_turnover=turnover,
            processed_events=len(used_event_ids),
            unmatched_tickers=sorted(unmatched),
        )
        self.history.append(result)
        return result

    def save_history(self, path: str | Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for snapshot in self.history:
                handle.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")
        return len(self.history)
