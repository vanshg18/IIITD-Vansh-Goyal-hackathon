"""Compare EventPulse event-classification baselines.

This script evaluates four approaches on the same labelled demo examples:
1. Rule-based classifier
2. Finance-topic classifier
3. ModernBERT zero-shot classifier
4. Confidence-aware router

The synthetic demo results are regression metrics, not estimates of
performance on real-world financial news.
"""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.ingestion.demo import load_articles
from src.nlp.event_classifier import FinancialEventClassifier
from src.nlp.financial_topic_classifier import FinancialTopicClassifier
from src.nlp.pretrained_event import PretrainedEventClassifier
from src.nlp.event_router import EventClassifierRouter


ROOT = Path(__file__).resolve().parents[2]
LABEL_FILE = ROOT / "data" / "demo" / "event_eval.json"


def main():
    with LABEL_FILE.open("r", encoding="utf-8") as f:
        gold_records = json.load(f)

    articles = {
        article.article_id: article
        for article in load_articles()
    }

    expected = []
    texts = []

    for record in gold_records:
        article_id = record["article_id"]

        if article_id not in articles:
            raise ValueError(
                f"Gold file references unknown article: {article_id}"
            )

        article = articles[article_id]

        expected.append(record["expected_label"])
        texts.append(f"{article.title}. {article.raw_text}")

    rule_model = FinancialEventClassifier()
    topic_model = FinancialTopicClassifier()
    zero_shot_model = PretrainedEventClassifier()

    router = EventClassifierRouter(
        topic_classifier=topic_model,
        fallback_classifier=zero_shot_model,
    )

    predictions = {
        "Rule baseline": [],
        "Finance-topic model": [],
        "Zero-shot model": [],
        "Event router": [],
    }

    for text in texts:
        predictions["Rule baseline"].append(
            rule_model.classify(text).label
        )

        topic_result = topic_model.predict(text)
        predictions["Finance-topic model"].append(
            topic_result.mapped_event or "Other"
        )

        predictions["Zero-shot model"].append(
            zero_shot_model.classify(text).label
        )

        predictions["Event router"].append(
            router.classify(text).label
        )

    # Restrict macro-F1 to categories represented in the gold set.
    # This makes the metric interpretable for this small smoke test.
    evaluated_labels = sorted(set(expected))

    print("\nEVENTPULSE CLASSIFIER BENCHMARK")
    print("=" * 72)
    print(f"Examples: {len(expected)}")
    print(f"Gold categories: {len(evaluated_labels)}")
    print("Dataset: synthetic demo regression set")
    print("-" * 72)

    for model_name, predicted in predictions.items():
        accuracy = accuracy_score(expected, predicted)

        macro_f1 = f1_score(
            expected,
            predicted,
            labels=evaluated_labels,
            average="macro",
            zero_division=0,
        )

        print(
            f"{model_name:24s} "
            f"Accuracy={accuracy:.3f}  "
            f"Macro-F1={macro_f1:.3f}"
        )

    print("\nIncorrect router predictions:")

    errors_found = False

    for article_id, text, gold, predicted in zip(
        [record["article_id"] for record in gold_records],
        texts,
        expected,
        predictions["Event router"],
    ):
        if gold != predicted:
            errors_found = True
            print(f"\n{article_id}")
            print(f"Text:      {text}")
            print(f"Expected:  {gold}")
            print(f"Predicted: {predicted}")

    if not errors_found:
        print("None on this synthetic regression set.")


if __name__ == "__main__":
    main()