"""Factories for selecting configured live data sources."""

from __future__ import annotations

import os

from src.ingestion.alphavantage import AlphaVantageSource
from src.ingestion.gdelt import GDELTSource
from src.ingestion.sec_edgar import SECEdgarSource
from src.ingestion.base import BaseSource


def build_live_sources() -> list[BaseSource]:
    """
    Return public sources plus Alpha Vantage when an API key is configured.

    SEC EDGAR requires SEC_USER_AGENT with a contact email. Its failure is
    reported by IngestionPipeline instead of being silently replaced by demo data.
    """
    sources: list[BaseSource] = [
        GDELTSource(),
        SECEdgarSource(),
    ]
    if os.getenv("ALPHAVANTAGE_API_KEY"):
        sources.append(AlphaVantageSource())
    return sources
