"""
Leakage-aware historical impact dataset and model.

The model learns the magnitude of next-session abnormal return from the
structured event features available to EventPulse. With daily close-only
prices, signals are treated as end-of-day observations: an event is aligned
to the latest close on or before its publication date, and the target is the
next available close-to-close return. Weekend news therefore anchors to the
last prior close. This is a proxy for market reaction, not causal attribution.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "sentiment_score",
    "sentiment_confidence",
    "event_confidence",
    "cluster_size",
    "corroboration_count",
    "novelty",
    "decay",
    "volatility_20d",
]
CATEGORICAL_FEATURES = ["ticker", "event_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "abs_abnormal_return"


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _select_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_key = {_key(column): column for column in columns}
    for alias in aliases:
        if _key(alias) in by_key:
            return by_key[_key(alias)]
    return None


def _read_price_file(
    path: str | Path,
    inferred_ticker: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col = _select_column(
        frame.columns, ["date", "datetime", "timestamp", "trading_date"]
    )
    ticker_col = _select_column(
        frame.columns, ["ticker", "symbol", "stock_symbol", "stock"]
    )
    close_col = _select_column(
        frame.columns,
        ["adj close", "adjusted close", "adj_close", "adjclose", "close", "closing price"],
    )
    if date_col is None or close_col is None:
        raise ValueError(
            f"Price file {path} must contain a date and close/adjusted-close column; "
            f"columns found: {list(frame.columns)}"
        )

    result = pd.DataFrame()
    result["date"] = pd.to_datetime(
        frame[date_col], errors="coerce", utc=True
    ).dt.tz_localize(None).dt.normalize()
    result["close"] = pd.to_numeric(frame[close_col], errors="coerce")
    if ticker_col:
        result["ticker"] = frame[ticker_col].astype(str).str.strip().str.upper()
    elif inferred_ticker:
        result["ticker"] = inferred_ticker.upper()
    else:
        raise ValueError(
            f"Price file {path} has no ticker/symbol column and no ticker could "
            "be inferred from its filename."
        )
    return result.dropna(subset=["date", "close", "ticker"]).query("close > 0")


def load_price_panel(
    path: str | Path,
    tickers: set[str] | None = None,
) -> pd.DataFrame:
    """
    Load price history from a long CSV, a directory of per-ticker CSV files,
    or a ZIP containing per-ticker CSVs (such as FNSPID full_history.zip).
    """
    path = Path(path)
    desired = {ticker.upper() for ticker in tickers} if tickers else None
    pieces: list[pd.DataFrame] = []

    if path.is_dir():
        files = sorted(path.rglob("*.csv"))
        if desired:
            files = [p for p in files if p.stem.upper() in desired]
        if not files:
            raise FileNotFoundError(
                f"No price CSV files found for the requested tickers in {path}."
            )
        for file_path in files:
            pieces.append(_read_price_file(file_path, inferred_ticker=file_path.stem))
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if desired:
                csv_names = [
                    name for name in csv_names
                    if Path(name).stem.upper() in desired
                ]
            if not csv_names:
                raise FileNotFoundError(
                    f"No matching per-ticker price CSV files found in {path}."
                )
            for name in csv_names:
                with archive.open(name) as handle:
                    frame = pd.read_csv(handle)
                date_col = _select_column(
                    frame.columns, ["date", "datetime", "timestamp", "trading_date"]
                )
                ticker_col = _select_column(
                    frame.columns, ["ticker", "symbol", "stock_symbol", "stock"]
                )
                close_col = _select_column(
                    frame.columns,
                    ["adj close", "adjusted close", "adj_close", "adjclose", "close", "closing price"],
                )
                if date_col is None or close_col is None:
                    continue
                piece = pd.DataFrame({
                    "date": pd.to_datetime(
                        frame[date_col], errors="coerce", utc=True
                    ).dt.tz_localize(None).dt.normalize(),
                    "close": pd.to_numeric(frame[close_col], errors="coerce"),
                    "ticker": (
                        frame[ticker_col].astype(str).str.strip().str.upper()
                        if ticker_col else Path(name).stem.upper()
                    ),
                })
                pieces.append(piece.dropna(subset=["date", "close", "ticker"]).query("close > 0"))
    elif path.is_file():
        pieces.append(_read_price_file(path))
    else:
        raise FileNotFoundError(f"Price data path does not exist: {path}")

    if not pieces:
        raise ValueError("No usable rows were read from the supplied price data.")
    prices = pd.concat(pieces, ignore_index=True)
    prices["ticker"] = prices["ticker"].astype(str).str.upper().str.strip()
    if desired:
        prices = prices[prices["ticker"].isin(desired)]
    prices = (
        prices.dropna(subset=["date", "close"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )
    if prices.empty:
        raise ValueError("No prices remained after ticker and data-quality filtering.")
    return prices


def load_event_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file containing canonical FinancialEvent objects."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def _field(record: dict[str, Any], parent: str, key: str, default: Any = None) -> Any:
    value = record.get(parent)
    return value.get(key, default) if isinstance(value, dict) else default


def _event_features(record: dict[str, Any], ticker: str) -> dict[str, Any]:
    signal = record.get("signal") if isinstance(record.get("signal"), dict) else {}
    return {
        "event_id": str(record.get("event_id", "")),
        "sentiment_score": _field(record, "sentiment", "score", 0.0),
        "sentiment_confidence": _field(record, "sentiment", "confidence", 0.0),
        "event_confidence": _field(record, "event_type", "confidence", 0.0),
        "cluster_size": record.get("cluster_size", 1),
        "corroboration_count": signal.get("corroboration_count", 1),
        "novelty": signal.get("novelty", 1.0),
        "decay": signal.get("decay", 1.0),
        "ticker": ticker.upper(),
        "event_type": _field(record, "event_type", "label", "Unknown") or "Unknown",
    }


def build_market_reaction_dataset(
    events_path: str | Path,
    prices_path: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Expand each structured event into one row per matched ticker, label it with
    the next close-to-close abnormal return, and write a leakage-aware dataset.
    """
    records = load_event_records(events_path)
    event_tickers = {
        str(ticker).strip().upper()
        for record in records
        for ticker in (record.get("tickers") or record.get("affected_assets") or [])
        if str(ticker).strip()
    }
    if not records:
        raise ValueError(f"No event records found in {events_path}.")
    if not event_tickers:
        raise ValueError("The event JSONL contains no tickers/affected_assets.")

    prices = load_price_panel(prices_path, tickers=event_tickers)
    ticker_frames: dict[str, pd.DataFrame] = {}
    for ticker, group in prices.groupby("ticker", sort=False):
        group = group.sort_values("date").copy()
        group["daily_return"] = group["close"].pct_change(fill_method=None)
        group["forward_return"] = group["close"].shift(-1) / group["close"] - 1.0
        group["next_date"] = group["date"].shift(-1)
        group["volatility_20d"] = group["daily_return"].rolling(20, min_periods=5).std()
        ticker_frames[ticker] = group.reset_index(drop=True)

    # Equal-weight benchmark formed from the available requested tickers.
    forward = prices.copy()
    forward = forward.sort_values(["ticker", "date"])
    forward["forward_return"] = forward.groupby("ticker")["close"].shift(-1) / forward["close"] - 1.0
    market_by_date = (
        forward.dropna(subset=["forward_return"])
        .groupby("date")["forward_return"]
        .agg(["sum", "count"])
    )

    rows: list[dict[str, Any]] = []
    skipped_no_price = 0
    skipped_no_target = 0

    for record in records:
        timestamp_raw = record.get("timestamp")
        timestamp = pd.to_datetime(timestamp_raw, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        event_date = timestamp.tz_localize(None).normalize()
        tickers = record.get("tickers") or record.get("affected_assets") or []
        for raw_ticker in tickers:
            ticker = str(raw_ticker).strip().upper()
            history = ticker_frames.get(ticker)
            if history is None or history.empty:
                skipped_no_price += 1
                continue

            # Map weekends/holidays to latest available close on or before event date.
            anchor_index = int(history["date"].searchsorted(event_date, side="right")) - 1
            if anchor_index < 0:
                skipped_no_price += 1
                continue
            anchor = history.iloc[anchor_index]
            if pd.isna(anchor["forward_return"]) or pd.isna(anchor["next_date"]):
                skipped_no_target += 1
                continue

            anchor_date = anchor["date"]
            market = market_by_date.loc[anchor_date] if anchor_date in market_by_date.index else None
            if market is None:
                benchmark_return = 0.0
            else:
                peers = int(market["count"]) - (1 if int(market["count"]) > 1 else 0)
                if int(market["count"]) > 1:
                    benchmark_return = (
                        float(market["sum"]) - float(anchor["forward_return"])
                    ) / peers
                else:
                    benchmark_return = 0.0

            abnormal_return = float(anchor["forward_return"]) - benchmark_return
            features = _event_features(record, ticker)
            rows.append({
                **features,
                "date": anchor_date.strftime("%Y-%m-%d"),
                "next_date": pd.Timestamp(anchor["next_date"]).strftime("%Y-%m-%d"),
                "stock_forward_return": float(anchor["forward_return"]),
                "benchmark_forward_return": float(benchmark_return),
                "abnormal_return": abnormal_return,
                "abs_abnormal_return": abs(abnormal_return),
                "volatility_20d": (
                    float(anchor["volatility_20d"])
                    if pd.notna(anchor["volatility_20d"]) else np.nan
                ),
            })

    if not rows:
        raise ValueError(
            "No event/price pairs could be labeled. Check ticker symbols, date formats, "
            "and the overlap between the event and price histories."
        )

    frame = pd.DataFrame(rows)
    counts = frame.groupby(["ticker", "date"])["event_id"].transform("count").clip(lower=1)
    frame["sample_weight"] = 1.0 / counts
    frame = frame.sort_values(["date", "ticker", "event_id"]).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(
        f"Built {len(frame):,} event-ticker labels; {frame['date'].nunique():,} trading-date anchors. "
        f"Skipped missing prices={skipped_no_price:,}, missing future returns={skipped_no_target:,}. "
        f"Output: {output_path.resolve()}"
    )
    return frame


def _make_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    regressor = ExtraTreesRegressor(
        n_estimators=350,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def train_market_impact_model(
    dataset_csv: str | Path,
    artifact_path: str | Path = "models/impact_model.joblib",
    metrics_path: str | Path = "data/processed/impact_model_metrics.json",
) -> dict[str, Any]:
    """
    Train and chronologically evaluate a market-reaction magnitude model.

    Model acceptance is based on validation MAE versus a training-median
    baseline. The final time block is kept untouched until final evaluation.
    """
    frame = pd.read_csv(dataset_csv)
    required = {
        "date", "ticker", "event_type", "abnormal_return", TARGET_COLUMN,
        *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training dataset is missing columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in NUMERIC_FEATURES + [TARGET_COLUMN, "abnormal_return"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", TARGET_COLUMN, "abnormal_return"])
    frame["ticker"] = frame["ticker"].fillna("UNKNOWN").astype(str).str.upper()
    frame["event_type"] = frame["event_type"].fillna("Unknown").astype(str)
    frame = frame.sort_values("date").reset_index(drop=True)

    unique_dates = sorted(frame["date"].dt.normalize().unique())
    if len(frame) < 100 or len(unique_dates) < 30:
        raise ValueError(
            "For a minimally meaningful chronological evaluation, provide at least "
            "100 event-ticker rows spanning 30 distinct price dates. A larger history "
            "is strongly preferred."
        )

    train_cut = max(1, int(len(unique_dates) * 0.70))
    valid_cut = max(train_cut + 1, int(len(unique_dates) * 0.85))
    train_dates = set(unique_dates[:train_cut])
    valid_dates = set(unique_dates[train_cut:valid_cut])
    test_dates = set(unique_dates[valid_cut:])
    if not valid_dates or not test_dates:
        raise ValueError("Not enough distinct dates for train/validation/test splitting.")

    normalized_dates = frame["date"].dt.normalize()
    train_mask = normalized_dates.isin(train_dates)
    valid_mask = normalized_dates.isin(valid_dates)
    test_mask = normalized_dates.isin(test_dates)
    if train_mask.sum() < 20 or valid_mask.sum() < 5 or test_mask.sum() < 5:
        raise ValueError("Temporal split produced too few rows in at least one partition.")

    x_train = frame.loc[train_mask, FEATURE_COLUMNS]
    y_train = frame.loc[train_mask, TARGET_COLUMN].to_numpy(dtype=float)
    x_valid = frame.loc[valid_mask, FEATURE_COLUMNS]
    y_valid = frame.loc[valid_mask, TARGET_COLUMN].to_numpy(dtype=float)
    x_test = frame.loc[test_mask, FEATURE_COLUMNS]
    y_test = frame.loc[test_mask, TARGET_COLUMN].to_numpy(dtype=float)

    pipeline = _make_pipeline()
    train_weight = pd.to_numeric(
        frame.loc[train_mask].get("sample_weight", pd.Series(1.0, index=frame.index[train_mask])),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=float)
    pipeline.fit(x_train, y_train, regressor__sample_weight=train_weight)

    validation_predictions = np.clip(pipeline.predict(x_valid), 0.0, None)
    baseline_valid = np.full_like(y_valid, max(0.0, float(np.median(y_train))))
    validation_model_metrics = _regression_metrics(y_valid, validation_predictions)
    validation_baseline_metrics = _regression_metrics(y_valid, baseline_valid)
    accepted = validation_model_metrics["mae"] < validation_baseline_metrics["mae"]

    # Refit on train+validation only if the model beats the baseline there.
    training_valid_mask = train_mask | valid_mask
    if accepted:
        final_pipeline = _make_pipeline()
        combined_weight = pd.to_numeric(
            frame.loc[training_valid_mask].get(
                "sample_weight",
                pd.Series(1.0, index=frame.index[training_valid_mask]),
            ),
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=float)
        final_pipeline.fit(
            frame.loc[training_valid_mask, FEATURE_COLUMNS],
            frame.loc[training_valid_mask, TARGET_COLUMN].to_numpy(dtype=float),
            regressor__sample_weight=combined_weight,
        )
    else:
        final_pipeline = pipeline

    test_predictions = np.clip(final_pipeline.predict(x_test), 0.0, None)
    y_train_valid = frame.loc[training_valid_mask, TARGET_COLUMN].to_numpy(dtype=float)
    test_baseline = np.full_like(y_test, max(0.0, float(np.median(y_train_valid))))
    report = {
        "dataset_rows": int(len(frame)),
        "distinct_anchor_dates": int(len(unique_dates)),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(valid_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "target": "absolute next-session abnormal return",
        "return_frequency": "daily close-to-close",
        "benchmark": "equal-weight average of the loaded universe excluding the target ticker",
        "validation_model": validation_model_metrics,
        "validation_median_baseline": validation_baseline_metrics,
        "test_model": _regression_metrics(y_test, test_predictions),
        "test_median_baseline": _regression_metrics(y_test, test_baseline),
        "accepted_for_live_inference": bool(accepted),
        "acceptance_rule": "validation MAE must beat the training-median baseline",
        "limitations": [
            "Market reaction is a proxy, not proof of causal effect.",
            "Daily close-only prices do not identify intraday reaction timing.",
            "Validation/test are chronological; no random event-level split was used.",
        ],
    }

    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_values = np.sort(y_train_valid)
    artifact = {
        "pipeline": final_pipeline,
        "calibration_abs_returns": calibration_values,
        "accepted_for_live_inference": bool(accepted),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "report": report,
        "target": "abs_abnormal_return",
    }
    joblib.dump(artifact, artifact_path)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Artifact: {artifact_path.resolve()}")
    print(f"Metrics:  {metrics_path.resolve()}")
    if not accepted:
        print(
            "MODEL NOT ACCEPTED: validation MAE did not beat the median baseline. "
            "The saved artifact will not be used by live EventPulse inference."
        )
    return report


class MarketReactionImpactModel:
    """Load an accepted trained artifact and score EventPulse event features."""

    def __init__(self, artifact: dict[str, Any]):
        if not artifact.get("accepted_for_live_inference", False):
            raise ValueError("Impact model did not pass validation baseline acceptance.")
        self.pipeline: Pipeline = artifact["pipeline"]
        self.calibration_abs_returns = np.asarray(
            artifact["calibration_abs_returns"], dtype=float
        )
        if self.calibration_abs_returns.size == 0:
            raise ValueError("Impact model artifact has no calibration targets.")

    @classmethod
    def load_if_accepted(
        cls,
        path: str | Path = "models/impact_model.joblib",
    ) -> "MarketReactionImpactModel | None":
        path = Path(path)
        if not path.exists():
            return None
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or not artifact.get(
            "accepted_for_live_inference", False
        ):
            return None
        try:
            return cls(artifact)
        except (KeyError, TypeError, ValueError):
            return None

    def predict_event(
        self,
        *,
        sentiment_score: float,
        sentiment_confidence: float,
        event_confidence: float,
        event_type: str,
        tickers: list[str],
        cluster_size: int = 1,
        corroboration_count: int = 1,
        novelty: float = 1.0,
        decay: float = 1.0,
        volatility_20d: float | None = None,
    ) -> dict[str, float]:
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tickers = tickers or ["UNKNOWN"]
        records = []
        for ticker in tickers:
            records.append({
                "sentiment_score": sentiment_score,
                "sentiment_confidence": sentiment_confidence,
                "event_confidence": event_confidence,
                "cluster_size": cluster_size,
                "corroboration_count": corroboration_count,
                "novelty": novelty,
                "decay": decay,
                "volatility_20d": (
                    float(volatility_20d) if volatility_20d is not None else np.nan
                ),
                "ticker": ticker,
                "event_type": event_type or "Unknown",
            })
        features = pd.DataFrame(records, columns=FEATURE_COLUMNS)
        raw_predictions = np.clip(self.pipeline.predict(features), 0.0, None)
        predicted_abs_return = float(np.mean(raw_predictions))
        percentile = float(
            np.searchsorted(
                self.calibration_abs_returns,
                predicted_abs_return,
                side="right",
            ) / len(self.calibration_abs_returns)
        )
        impact_score = float(np.clip(1.0 + 9.0 * percentile, 1.0, 10.0))
        return {
            "predicted_abs_abnormal_return": predicted_abs_return,
            "impact_score": impact_score,
        }
