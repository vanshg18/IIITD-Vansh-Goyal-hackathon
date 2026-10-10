"""
Event Intelligence Engine.

Combines:
    - financial sentiment
    - entity/ticker resolution
    - event classification
    - event clustering
    - novelty
    - corroboration
    - time decay
    - impact scoring

The output is a list of canonical FinancialEvent objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
import warnings

from src.engine.corroboration import CorroborationEngine
from src.engine.decay import TimeDecayEngine
from src.engine.event_clustering import EventClusterer
from src.engine.impact import ImpactEngine
from src.engine.entity_merge import merge_entities
from src.engine.schemas import (
    FinancialEvent,
    ImpactResult,
    SentimentResult,
    SignalMetadata,
    SourceInfo,
)
from src.ingestion.demo import load_portfolio
from src.nlp.entity_extractor import FinancialEntityResolver
from src.nlp.event_classifier import (
    FinancialEventClassifier,
)
from src.nlp.sentiment import (
    FinancialSentimentAnalyzer,
)
from src.nlp.financial_ner import FinancialNER
from src.nlp.event_router import (
    EventClassifierRouter,
)
from src.nlp.semantic_clustering import (
    SemanticEventClusterer,
)
from src.ml.historical_impact import MarketReactionImpactModel


class EventIntelligenceEngine:
    """
    Main intelligence layer of EventPulse.
    """

    def __init__(
        self,
        portfolio=None,
        sentiment_analyzer=None,
        event_classifier=None,
        entity_resolver=None,
        use_pretrained=True,
    ):

        self.portfolio = (
            portfolio
            if portfolio is not None
            else load_portfolio()
        )

        self.sentiment_analyzer = (
            sentiment_analyzer
            or FinancialSentimentAnalyzer()
        )

        if event_classifier is not None:

            self.event_classifier = (
                event_classifier
            )

        elif use_pretrained:

            if event_classifier is not None:

                self.event_classifier = (event_classifier)

            else:

                self.event_classifier = (EventClassifierRouter())

        else:

            self.event_classifier = (
                FinancialEventClassifier()
            )

        self.ner = FinancialNER()

        self.entity_resolver = (
            entity_resolver
            or FinancialEntityResolver(
                self.portfolio
            )
        )
        
        self.clusterer = SemanticEventClusterer(
            event_classifier=self.event_classifier
        )

        self.impact_engine = ImpactEngine()
        model_path = (
            Path(__file__).resolve().parents[2]
            / "models"
            / "impact_model.joblib"
        )
        try:
            self.market_impact_model = (
                MarketReactionImpactModel.load_if_accepted(model_path)
            )
        except Exception as exc:
            # A missing or incompatible optional artifact must never prevent
            # the baseline engine from running.
            self.market_impact_model = None
            warnings.warn(
                f"Historical impact model could not be loaded; using heuristic baseline: {exc}",
                RuntimeWarning,
            )

        self.corroboration_engine = (
            CorroborationEngine()
        )

        self.decay_engine = (
            TimeDecayEngine()
        )

    def process(
        self,
        articles,
        reference_time: datetime | None = None,
    ) -> list[FinancialEvent]:

        if not articles:
            return []

        if reference_time is None:
            reference_time = datetime.now(
                timezone.utc
            )

        # ---------------------------------------------------------------
        # STEP 1 — Transformer NER + ticker resolution
        # ---------------------------------------------------------------

        ner_enriched_articles = []

        for article in articles:

            text = (
                f"{article.title} "
                f"{article.raw_text}"
            )

            model_entities = self.ner.extract(text)
            merged_entities = merge_entities(
                article.entities,
                model_entities,
            )

            article_with_ner = article.model_copy(
                update={"entities": merged_entities}
            )

            enriched = (
                self.entity_resolver.enrich_article(
                    article_with_ner
                )
            )

            ner_enriched_articles.append(
                enriched
            )

        enriched_articles = (
            ner_enriched_articles
        )

        # ---------------------------------------------------------------
        # STEP 2 — Cluster related articles
        # ---------------------------------------------------------------

        clusters = self.clusterer.cluster(
            enriched_articles
        )

        article_by_id = {
            article.article_id: article
            for article in enriched_articles
        }

        # ---------------------------------------------------------------
        # STEP 3 — Create FinancialEvent for each article
        # ---------------------------------------------------------------

        results: list[FinancialEvent] = []

        for cluster in clusters:

            cluster_articles = [
                article_by_id[article_id]
                for article_id in cluster.article_ids
            ]

            cluster_articles.sort(
                key=lambda article: article.timestamp
            )

            for article in cluster_articles:
                # Use only reports available at this article's publication
                # time. Future reports must not inflate this event's novelty,
                # corroboration, cluster-size feature, or impact prediction.
                observed_cluster_articles = (
                    self.corroboration_engine.articles_available_as_of(
                        cluster_articles,
                        article.timestamp,
                    )
                )
                observed_cluster_size = max(
                    1,
                    len(observed_cluster_articles),
                )

                text = (
                    f"{article.title} "
                    f"{article.raw_text}"
                )

                # -------------------------------------------------------
                # Sentiment
                # -------------------------------------------------------

                sentiment_prediction = (
                    self.sentiment_analyzer.predict(
                        text
                    )
                )

                sentiment = SentimentResult(
                    score=sentiment_prediction.score,
                    label=sentiment_prediction.label,
                    confidence=sentiment_prediction.confidence,
                )

                # -------------------------------------------------------
                # Event type
                # -------------------------------------------------------

                event_prediction = (
                    self.event_classifier.classify(
                        text
                    )
                )

                # -------------------------------------------------------
                # Novelty
                # -------------------------------------------------------

                novelty = 1.0 / observed_cluster_size

                # -------------------------------------------------------
                # Corroboration
                # -------------------------------------------------------

                corroboration = (
                    self.corroboration_engine.calculate(
                        articles=observed_cluster_articles,
                        base_confidence=(
                            sentiment.confidence
                            + event_prediction.confidence
                        ) / 2.0,
                    )
                )

                # -------------------------------------------------------
                # Time decay
                # -------------------------------------------------------

                decay = self.decay_engine.decay(
                    event_type=event_prediction.label,
                    event_timestamp=article.timestamp,
                    reference_time=reference_time,
                )

                # -------------------------------------------------------
                # Impact
                # -------------------------------------------------------

                impact_prediction = (
                    self.impact_engine.calculate(
                        event_type=event_prediction.label,
                        sentiment_score=sentiment.score,
                        sentiment_confidence=sentiment.confidence,
                        event_confidence=(
                            event_prediction.confidence
                        ),
                        affected_count=len(
                            article.tickers
                        ),
                        novelty=novelty,
                    )
                )

                # Replace only the impact score when a historically trained
                # model passed the temporal validation baseline. Keep the
                # transparent heuristic's confidence until confidence is
                # independently calibrated.
                if self.market_impact_model is not None:
                    try:
                        market_estimate = (
                            self.market_impact_model.predict_event(
                                sentiment_score=sentiment.score,
                                sentiment_confidence=sentiment.confidence,
                                event_confidence=event_prediction.confidence,
                                event_type=event_prediction.label,
                                tickers=article.tickers,
                                cluster_size=observed_cluster_size,
                                corroboration_count=corroboration.unique_source_count,
                                novelty=novelty,
                                decay=decay,
                            )
                        )
                        impact_prediction = replace(
                            impact_prediction,
                            score=market_estimate["impact_score"],
                        )
                    except Exception as exc:
                        warnings.warn(
                            "Trained impact inference failed for an event; "
                            f"using heuristic baseline: {exc}",
                            RuntimeWarning,
                        )

                # -------------------------------------------------------
                # Final adjusted confidence
                # -------------------------------------------------------

                adjusted_confidence = min(
                    0.99,
                    corroboration.adjusted_confidence
                    * (
                        0.85
                        + 0.15 * decay
                    ),
                )

                # -------------------------------------------------------
                # Create canonical FinancialEvent
                # -------------------------------------------------------

                event_id = (
                    f"{cluster.cluster_id}_"
                    f"{article.article_id}"
                )

                result = FinancialEvent(
                    event_id=event_id,
                    timestamp=article.timestamp,
                    source=SourceInfo(
                        name=article.source.name,
                        source_type=article.source.source_type,
                        url=article.source.url,
                    ),
                    title=article.title,
                    raw_text=article.raw_text,
                    entities=article.entities,
                    tickers=article.tickers,
                    event_type=event_prediction,
                    sentiment=sentiment,
                    impact=ImpactResult(
                        score=impact_prediction.score,
                        confidence=impact_prediction.confidence,
                    ),
                    signal=SignalMetadata(
                        novelty=novelty,
                        corroboration_count=corroboration.unique_source_count,
                        decay=decay,
                    ),
                    affected_assets=article.tickers,
                    cluster_id=cluster.cluster_id,
                    cluster_size=observed_cluster_size,
                )

                results.append(result)

        results.sort(
            key=lambda event: event.timestamp
        )

        return results