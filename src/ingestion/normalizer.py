"""
Common article normalization and lightweight deduplication utilities.
"""

import re
from hashlib import sha256

from src.engine.schemas import Article


def clean_text(text: str | None) -> str:
    """
    Normalize whitespace and remove obvious control characters.
    """
    if not text:
        return ""

    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_title(title: str) -> str:
    """
    Normalize a title for duplicate detection.
    """
    title = clean_text(title).lower()

    # Keep alphanumeric characters and spaces.
    title = re.sub(r"[^a-z0-9\s]", "", title)

    title = re.sub(r"\s+", " ", title)

    return title.strip()


def article_fingerprint(article: Article) -> str:
    """
    Create a deterministic fingerprint for an article.

    URL is preferred when available; otherwise source + normalized title
    are used.
    """

    if article.url:
        base = article.url.strip().lower()
    else:
        base = f"{article.source.name}|{normalize_title(article.title)}"

    return sha256(base.encode("utf-8")).hexdigest()


def deduplicate_articles(
    articles: list[Article],
) -> list[Article]:
    """
    Remove duplicate articles while preserving original order.
    """

    seen: set[str] = set()
    unique: list[Article] = []

    for article in articles:
        fingerprint = article_fingerprint(article)

        if fingerprint in seen:
            continue

        seen.add(fingerprint)
        unique.append(article)

    return unique