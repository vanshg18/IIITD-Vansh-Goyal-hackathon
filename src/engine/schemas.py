"""
Canonical data contracts for EventPulse.

All ingestion, NLP, signal generation, and downstream modules
communicate through these validated structures.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# SOURCE
# ---------------------------------------------------------------------------

class SourceInfo(BaseModel):
    """Information about where an article/event originated."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_type: Literal["news", "social", "official", "demo"]
    url: str | None = None


# ---------------------------------------------------------------------------
# ENTITY
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """Named entity extracted from an article."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    entity_type: Literal[
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "PRODUCT",
        "OTHER",
    ]


# ---------------------------------------------------------------------------
# SENTIMENT
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    """
    Financial sentiment output.

    score is normalized to [-1, 1].
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=-1.0, le=1.0)
    label: Literal["negative", "neutral", "positive"]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# EVENT CLASSIFICATION
# ---------------------------------------------------------------------------
class EventClassification(BaseModel):
    """Event category predicted by the NLP layer."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence_terms: list[str] = Field(
        default_factory=list
    )

    # Name/version of the model producing this result.
    model_name: str = "unknown"

    # Full candidate-label score distribution when available.
    scores: dict[str, float] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# IMPACT
# ---------------------------------------------------------------------------

class ImpactResult(BaseModel):
    """
    Estimated potential financial impact.

    score is on a 1-10 scale.
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=1.0, le=10.0)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# SIGNAL METADATA
# ---------------------------------------------------------------------------

class SignalMetadata(BaseModel):
    """
    Additional information used by the signal-generation layer.
    """

    model_config = ConfigDict(extra="forbid")

    novelty: float = Field(default=1.0, ge=0.0, le=1.0)

    # Number of independent/corroborating sources.
    corroboration_count: int = Field(default=1, ge=0)

    # Time-decay factor.
    decay: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# ARTICLE
# ---------------------------------------------------------------------------

class Article(BaseModel):
    """
    Canonical representation of an ingested text item.

    This is produced by the ingestion/normalization layer before NLP.
    """

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1)

    timestamp: datetime

    source: SourceInfo

    title: str = Field(min_length=1)

    raw_text: str = Field(min_length=1)

    entities: list[Entity] = Field(default_factory=list)

    tickers: list[str] = Field(default_factory=list)

    url: str | None = None


# ---------------------------------------------------------------------------
# FINANCIAL EVENT
# ---------------------------------------------------------------------------

class FinancialEvent(BaseModel):
    """
    Unified structured event produced by the AI/NLP Risk Engine.

    All downstream modules consume this object rather than raw text.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)

    timestamp: datetime

    source: SourceInfo

    title: str = Field(min_length=1)

    raw_text: str = Field(min_length=1)

    entities: list[Entity] = Field(default_factory=list)

    tickers: list[str] = Field(default_factory=list)

    event_type: EventClassification

    sentiment: SentimentResult

    impact: ImpactResult

    signal: SignalMetadata = Field(default_factory=SignalMetadata)

    affected_assets: list[str] = Field(default_factory=list)
    
    cluster_id: str | None = None
    cluster_size: int = Field(
        default=1,
        ge=1,
    )


# ---------------------------------------------------------------------------
# PORTFOLIO
# ---------------------------------------------------------------------------

class PortfolioAsset(BaseModel):
    """One asset in a synthetic portfolio/index."""

    model_config = ConfigDict(extra="forbid")

    ticker: str = Field(min_length=1)
    name: str = Field(min_length=1)
    sector: str = Field(min_length=1)

    weight: float = Field(gt=0.0, le=1.0)


class Portfolio(BaseModel):
    """Synthetic portfolio/index."""

    model_config = ConfigDict(extra="forbid")

    portfolio_id: str = Field(min_length=1)
    name: str = Field(min_length=1)

    assets: list[PortfolioAsset] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_total_weight(self):
        total_weight = sum(asset.weight for asset in self.assets)

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Portfolio weights must sum to 1.0, got {total_weight:.6f}"
            )

        return self

# ---------------------------------------------------------------------------
# STRESS SCENARIO
# ---------------------------------------------------------------------------

class StressScenario(BaseModel):
    """
    Synthetic stress scenario used by Module B.

    shocks maps a sector/asset-class identifier to a percentage change.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    shocks: dict[str, float]