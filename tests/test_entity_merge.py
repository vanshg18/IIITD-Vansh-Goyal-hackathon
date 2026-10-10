from src.engine.intelligence import EventIntelligenceEngine
from src.engine.schemas import Entity


def test_source_entities_survive_empty_ner_output():
    official = [Entity(name="Apple Inc.", entity_type="ORGANIZATION")]
    merged = EventIntelligenceEngine._merge_entities(official, [])
    assert merged == official


def test_entity_merge_deduplicates_case_insensitive_exact_matches():
    official = [Entity(name="Apple Inc.", entity_type="ORGANIZATION")]
    ner = [
        Entity(name="Apple Inc.", entity_type="ORGANIZATION"),
        Entity(name="Tim Cook", entity_type="PERSON"),
    ]
    merged = EventIntelligenceEngine._merge_entities(official, ner)
    assert [(entity.name, entity.entity_type) for entity in merged] == [
        ("Apple Inc.", "ORGANIZATION"),
        ("Tim Cook", "PERSON"),
    ]
