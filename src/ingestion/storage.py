"""
Small JSONL persistence helpers for ingested articles and processed events.

Raw public data is appended with URL/title deduplication. Processed event
snapshots can be overwritten to represent the latest processing run.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Any

from src.engine.schemas import Article
from src.ingestion.normalizer import article_fingerprint, normalize_title


def _record_fingerprint(record: dict[str, Any]) -> str:
    url = str(record.get("url") or "").strip().lower()
    if url:
        base = url
    else:
        source = record.get("source") or {}
        source_name = source.get("name", "") if isinstance(source, dict) else ""
        base = f"{source_name}|{normalize_title(str(record.get('title') or ''))}"
    return sha256(base.encode("utf-8")).hexdigest()


def append_articles_jsonl(
    articles: Iterable[Article],
    path: str | Path,
) -> int:
    """Append articles not already present by URL/title fingerprint; return count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}"
                    ) from exc
                if isinstance(record, dict):
                    seen.add(_record_fingerprint(record))

    added = 0
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for article in articles:
            fingerprint = article_fingerprint(article)
            if fingerprint in seen:
                continue
            handle.write(article.model_dump_json() + "\n")
            seen.add(fingerprint)
            added += 1
    return added


def load_articles_jsonl(path: str | Path) -> list[Article]:
    """Load and validate articles from a JSON Lines file."""
    path = Path(path)
    if not path.exists():
        return []

    articles: list[Article] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                articles.append(Article.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid Article record in {path} at line {line_number}: {exc}"
                ) from exc
    return articles


def write_models_jsonl(records: Iterable[Any], path: str | Path) -> int:
    """Write a fresh JSONL snapshot from Pydantic models or dictionaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if hasattr(record, "model_dump_json"):
                serialized = record.model_dump_json()
            elif isinstance(record, dict):
                serialized = json.dumps(record, ensure_ascii=False, default=str)
            else:
                raise TypeError(
                    "write_models_jsonl accepts Pydantic models or dictionaries."
                )
            handle.write(serialized + "\n")
            count += 1
    return count
