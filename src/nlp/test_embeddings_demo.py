from src.nlp.finance_embeddings import FinanceEmbedder


def main():

    sentences = [
        "The central bank unexpectedly increased interest rates.",
        "The monetary authority announced a surprise rate hike.",
        "The company launched a new processor.",
        "The technology firm released a new chip.",
        "A ratings agency downgraded the bank.",
    ]

    model = FinanceEmbedder()

    embeddings = model.encode(
        sentences
    )

    import numpy as np

    similarity = np.matmul(
        embeddings,
        embeddings.T,
    )

    print()
    print("=" * 100)
    print("FINANCE EMBEDDING SIMILARITY")
    print("=" * 100)

    for i in range(
        len(sentences)
    ):

        for j in range(
            i + 1,
            len(sentences)
        ):

            print(
                f"{i} ↔ {j}: "
                f"{similarity[i, j]:.4f}"
            )


if __name__ == "__main__":
    main()