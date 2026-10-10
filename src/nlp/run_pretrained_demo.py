from src.ingestion.demo import load_articles, load_portfolio
from src.nlp.financial_ner import FinancialNER
from src.nlp.event_router import (
    EventClassifierRouter,
)
from src.nlp.finance_embeddings import (
    FinanceEmbedder,
)

def main():

    articles = load_articles()

    print()
    print("=" * 100)
    print("EVENTPULSE — PRETRAINED MODEL SMOKE TEST")
    print("=" * 100)

    ner = FinancialNER()

    for article in articles[:5]:

        text = (
            f"{article.title} "
            f"{article.raw_text}"
        )

        entities = ner.extract(text)

        print()
        print(
            f"[NER] {article.title}"
        )

        for entity in entities:
            print(
                f"   {entity.entity_type}: "
                f"{entity.name}"
            )

    # ---------------------------------------------------------------
    # 2. Zero-shot event classification
    # ---------------------------------------------------------------

    classifier = EventClassifierRouter()

    for article in articles:

        text = (
            f"{article.title} "
            f"{article.raw_text}"
        )

        result = classifier.classify(text)

        print()
        print(
            f"[EVENT] {article.title}"
        )

        print(
            f"   Label:       {result.label}"
        )

        print(
            f"   Confidence:  "
            f"{result.confidence:.4f}"
        )

        print(
            f"   Model:        "
            f"{result.model_name}"
        )

        print(
            f"   Candidates:   "
            f"{result.scores}"
        )

    # ---------------------------------------------------------------
    # 3. Finance embeddings
    # ---------------------------------------------------------------

    embedder = FinanceEmbedder()

    texts = [
        (
            f"{article.title} "
            f"{article.raw_text}"
        )
        for article in articles
    ]

    embeddings = embedder.encode(texts)

    print()
    print(
        "=" * 100
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Embedding norms:",
        (
            embeddings ** 2
        ).sum(axis=1)[:5]
    )

if __name__ == "__main__":
    main()