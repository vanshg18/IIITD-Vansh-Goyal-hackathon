"""Lightweight entity-list merge utility shared by the engine and tests."""

from __future__ import annotations


def merge_entities(existing_entities, extracted_entities):
    """Merge source-provided and model-extracted entities without data loss.

    Source adapters may know an issuer even when generic financial NER misses
    it. Entity identity is the case-insensitive (name, entity type) pair.
    """
    merged = []
    seen = set()
    for entity in list(existing_entities or []) + list(extracted_entities or []):
        name = str(getattr(entity, "name", "") or "").strip()
        entity_type = str(getattr(entity, "entity_type", "OTHER") or "OTHER")
        if not name:
            continue
        key = (name.casefold(), entity_type)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entity)
    return merged
