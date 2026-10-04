from src.engine.intelligence import (
    EventIntelligenceEngine,
)
from src.ingestion.demo import load_articles


def main():

    articles = load_articles()

    engine = EventIntelligenceEngine()

    # Fixed reference time makes our demo reproducible.
    from datetime import datetime

    reference_time = datetime.fromisoformat(
        "2026-10-03T10:30:00+00:00"
    )

    events = engine.process(
        articles,
        reference_time=reference_time,
    )

    print()
    print("=" * 110)
    print("EVENTPULSE — EVENT INTELLIGENCE ENGINE")
    print("=" * 110)

    for event in events:

        print()
        print("-" * 110)

        print(
            f"EVENT ID:          {event.event_id}"
        )

        print(
            f"CLUSTER:            {event.cluster_id}"
        )

        print(
            f"CLUSTER SIZE:       {event.cluster_size}"
        )

        print(
            f"TITLE:              {event.title}"
        )

        print(
            f"ENTITY:             "
            f"{[e.name for e in event.entities]}"
        )

        print(
            f"TICKERS:            {event.tickers}"
        )

        print(
            f"EVENT TYPE:         "
            f"{event.event_type.label}"
        )

        print(
            f"EVENT CONFIDENCE:   "
            f"{event.event_type.confidence:.3f}"
        )

        print(
            f"EVENT EVIDENCE:     "
            f"{event.event_type.evidence_terms}"
        )

        print(
            f"SENTIMENT:          "
            f"{event.sentiment.score:+.3f}"
        )

        print(
            f"SENTIMENT LABEL:    "
            f"{event.sentiment.label}"
        )

        print(
            f"IMPACT:             "
            f"{event.impact.score:.2f}/10"
        )

        print(
            f"IMPACT CONFIDENCE:  "
            f"{event.impact.confidence:.3f}"
        )

        print(
            f"NOVELTY:            "
            f"{event.signal.novelty:.3f}"
        )

        print(
            f"CORROBORATION:      "
            f"{event.signal.corroboration_count} sources"
        )

        print(
            f"TIME DECAY:         "
            f"{event.signal.decay:.3f}"
        )


if __name__ == "__main__":
    main()