"""
Financial entity and ticker resolution.

The resolver combines:

1. Source-provided ticker symbols
2. Exact company-name/alias matching
3. Normalized name matching
4. Conservative fuzzy matching

The goal is to resolve entities that matter to our portfolio without
blindly interpreting arbitrary uppercase words as ticker symbols.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from src.engine.schemas import Article, Entity, Portfolio


@dataclass(frozen=True)
class AssetEntity:
    """Internal representation of a portfolio entity."""

    ticker: str
    name: str
    sector: str
    aliases: tuple[str, ...]


def normalize_text(text: str) -> str:
    """
    Convert text into a comparison-friendly representation.

    Example:
        "Nova-Bank, Inc."
        -> "nova bank inc"
    """

    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_company_name(name: str) -> str:
    """
    Normalize company/entity names and remove common legal suffixes.
    """

    text = normalize_text(name)

    suffixes = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "limited",
        "ltd",
        "plc",
        "llc",
    }

    tokens = [
        token
        for token in text.split()
        if token not in suffixes
    ]

    return " ".join(tokens)


class FinancialEntityResolver:
    """
    Resolve text mentions to entities in a known portfolio universe.

    This prevents arbitrary capitalized text from becoming a financial
    ticker and provides deterministic mappings for downstream modules.
    """

    def __init__(self, portfolio: Portfolio):

        self.portfolio = portfolio

        self.assets: list[AssetEntity] = []

        for asset in portfolio.assets:

            name_normalized = normalize_company_name(
                asset.name
            )

            # Generate useful aliases from the canonical asset name.
            aliases = {
                name_normalized,
                normalize_text(asset.name),
                normalize_text(asset.ticker),
            }

            # Add compact aliases where the company name contains spaces.
            compact = name_normalized.replace(" ", "")

            if len(compact) >= 5:
                aliases.add(compact)

            self.assets.append(
                AssetEntity(
                    ticker=asset.ticker.upper(),
                    name=asset.name,
                    sector=asset.sector,
                    aliases=tuple(
                        alias
                        for alias in aliases
                        if alias
                    ),
                )
            )

        self.alias_to_asset: dict[str, AssetEntity] = {}
        self.name_to_ticker: dict[str, str] = {}

        for asset in self.assets:
            self.name_to_ticker[normalize_company_name(asset.name)] = asset.ticker
            for alias in asset.aliases:
                self.alias_to_asset[alias] = asset

    def extract_tickers(
        self,
        text: str,
        existing_tickers: list[str] | None = None,
    ) -> list[str]:
        """
        Extract ticker symbols conservatively.

        Only ticker-like strings belonging to our known universe are
        accepted from arbitrary text. Source-provided tickers are retained.
        """

        resolved: list[str] = []

        # Preserve trusted tickers coming from source metadata.
        for ticker in existing_tickers or []:

            ticker = ticker.strip().upper()

            if ticker and ticker not in resolved:
                resolved.append(ticker)

        # Detect possible ticker tokens such as:
        # $AAPL
        # AAPL
        candidate_pattern = r"(?<![A-Za-z0-9])\$?([A-Z]{2,5})(?![A-Za-z0-9])"

        candidates = re.findall(
            candidate_pattern,
            text,
        )

        known_tickers = {
            asset.ticker
            for asset in self.assets
        }

        for candidate in candidates:

            candidate = candidate.upper()

            if candidate in known_tickers:
                if candidate not in resolved:
                    resolved.append(candidate)

        return resolved

    def extract_entities(
        self,
        text: str,
    ) -> list[Entity]:
        """
        Find portfolio entities mentioned in the text.
        """

        if not text:
            return []

        normalized = normalize_text(text)

        found: dict[str, Entity] = {}

        # ---------------------------------------------------------------
        # Stage 1: exact normalized matching
        # ---------------------------------------------------------------

        for asset in self.assets:

            for alias in asset.aliases:

                if not alias:
                    continue

                # Word-boundary matching prevents partial word matches.
                pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"

                if re.search(pattern, normalized):

                    found[asset.ticker] = Entity(
                        name=asset.name,
                        entity_type="ORGANIZATION",
                    )

                    break

        # ---------------------------------------------------------------
        # Stage 2: conservative fuzzy matching
        # ---------------------------------------------------------------

        # We only use fuzzy matching for reasonably long entity names.
        #
        # This is intentionally conservative. A false financial entity
        # match can be more harmful than a missed match.
        words = normalized.split()

        for asset in self.assets:

            if asset.ticker in found:
                continue

            canonical = normalize_company_name(
                asset.name
            )

            canonical_tokens = canonical.split()

            if not canonical_tokens:
                continue

            window_size = len(canonical_tokens)

            candidate_windows: list[str] = []

            if window_size == 1:

                candidate_windows = [
                    word
                    for word in words
                    if len(word) >= 5
                ]

            else:

                for i in range(
                    0,
                    max(0, len(words) - window_size + 1),
                ):
                    candidate_windows.append(
                        " ".join(
                            words[
                                i:i + window_size
                            ]
                        )
                    )

            best_score = 0

            for window in candidate_windows:

                score = fuzz.ratio(
                    canonical,
                    window,
                )

                best_score = max(
                    best_score,
                    score,
                )

            # Very conservative threshold.
            if best_score >= 92:

                found[asset.ticker] = Entity(
                    name=asset.name,
                    entity_type="ORGANIZATION",
                )

        return list(found.values())

    def enrich_article(
        self,
        article: Article,
    ) -> Article:
        """
        Enrich an Article with portfolio-aware entity/ticker resolution.

        Existing model-detected entities are preserved and combined with
        portfolio-aware matching.
        """

        combined_text = (
            f"{article.title} "
            f"{article.raw_text}"
        )

        tickers = self.extract_tickers(
            text=combined_text,
            existing_tickers=article.tickers,
        )

        resolved_entities = self.extract_entities(
            text=combined_text
        )

        # Company-name matches must also produce ticker links. Previously,
        # "Microsoft shares fell" could yield an ORGANIZATION entity but no
        # MSFT ticker because extract_tickers only recognizes ticker tokens.
        for entity in resolved_entities:
            resolved_ticker = self.name_to_ticker.get(
                normalize_company_name(entity.name)
            )
            if resolved_ticker and resolved_ticker not in tickers:
                tickers.append(resolved_ticker)

        # Preserve entities already detected by a pretrained NER model.
        entity_map: dict[str, Entity] = {}

        for entity in article.entities:
            key = normalize_company_name(
                entity.name
            )

            if key:
                entity_map[key] = entity

        # Portfolio-aware resolution gets precedence because it gives
        # us a canonical company name.
        for entity in resolved_entities:
            key = normalize_company_name(
                entity.name
            )

            if key:
                entity_map[key] = entity

        return article.model_copy(
            update={
                "tickers": tickers,
                "entities": list(
                    entity_map.values()
                ),
            }
        )

    
    def enrich_articles(
        self,
        articles: list[Article],
    ) -> list[Article]:

        return [
            self.enrich_article(article)
            for article in articles
        ]