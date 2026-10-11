# EventPulse Architecture

EventPulse converts public financial headlines and corporate filings into structured event signals, then demonstrates two downstream risk workflows. The Mermaid diagram below is the source-of-truth architecture view; `docs/architecture.png` is its presentation-ready companion image.

```mermaid
flowchart TB
    subgraph SOURCES["1. Public / Optional Inputs"]
        GDELT["GDELT DOC API\nPublic news metadata"]
        SEC["SEC EDGAR\nCurrent 8-K filings"]
        AV["Alpha Vantage\nOptional API-key feed"]
    end

    subgraph INGEST["2. Ingestion, normalization and state"]
        ADAPTERS["Source adapters + status reporting"]
        NORMALIZE["Normalize to Article schema\nIDs, timestamps, source, URL, text"]
        ARCHIVE["Raw article archive\nJSONL + fingerprint deduplication"]
        REGISTRY["Processed fingerprint registry\nSeparate from raw archive"]
    end

    subgraph NLP["3. NLP and event intelligence"]
        NER["Financial NER\nwhataboutyou-ai/financial_bert"]
        RESOLVE["Portfolio-aware entity / ticker resolution\nMerge source + model entities"]
        SENT["Financial sentiment\nProsusAI/finbert"]
        TOPIC["Financial topic classifier\nleonas5555/finnews-topic-single-classify"]
        ZERO["Zero-shot event classifier\nMoritzLaurer/ModernBERT-base-zeroshot-v2.0"]
        ROUTER["Confidence-aware router\nAgreement, score margin, thresholds"]
        EMBED["Text embeddings\nBAAI/bge-small-en-v1.5"]
        CLUSTER["Semantic clustering\nCategory + similarity + ticker + time"]
        SIGNALS["Point-in-time signal metadata\nNovelty, corroboration, time decay"]
        IMPACT["Impact score 1-10\nCurrent active path: heuristic baseline"]
    end

    EVENT["4. Canonical FinancialEvent schema\nSentiment, category, impact, tickers, provenance, cluster"]

    subgraph ACTIONS["5. Downstream simulations"]
        A["Module A\nLong-only mock-index rebalancer\nWeight bounds + 10% one-way turnover cap"]
        B["Module B\nSynthetic banking stress tester\nNegative sentiment + category + impact gates\nLoans, bonds, equities, cash + derivative shock proxy"]
    end

    subgraph PRESENT["6. Outputs and presentation"]
        FILES["JSON / JSONL snapshots + histories\nCycle + per-source status"]
        API["FastAPI read-only endpoints"]
        UI["Streamlit dashboard\nReads synthetic replay outputs"]
    end

    GDELT --> ADAPTERS
    SEC --> ADAPTERS
    AV -. "optional" .-> ADAPTERS
    ADAPTERS --> NORMALIZE --> ARCHIVE
    ARCHIVE --> NER --> RESOLVE
    NORMALIZE --> SENT
    NORMALIZE --> TOPIC --> ROUTER
    NORMALIZE --> ZERO --> ROUTER
    NORMALIZE --> EMBED --> CLUSTER
    RESOLVE --> CLUSTER
    ROUTER --> CLUSTER
    CLUSTER --> SIGNALS --> IMPACT --> EVENT
    SENT --> EVENT
    ROUTER --> EVENT
    RESOLVE --> EVENT
    EVENT --> A
    EVENT --> B
    A --> FILES
    B --> FILES
    EVENT --> FILES
    FILES --> API
    FILES --> UI
    REGISTRY -. "filters previously processed records" .-> NER

    classDef source fill:#13263b,stroke:#4f8dd6,color:#eef5ff,stroke-width:1.5px;
    classDef nlp fill:#123832,stroke:#35c9a6,color:#e8fff8,stroke-width:1.5px;
    classDef output fill:#30284a,stroke:#a995f5,color:#f6f2ff,stroke-width:1.5px;
    classDef caution fill:#392a1a,stroke:#e9b35a,color:#fff5df,stroke-width:1.5px;
    class GDELT,SEC,AV,ADAPTERS,NORMALIZE,ARCHIVE,REGISTRY source;
    class NER,RESOLVE,SENT,TOPIC,ZERO,ROUTER,EMBED,CLUSTER,SIGNALS nlp;
    class EVENT,A,B,FILES,API,UI output;
    class IMPACT caution;
```

## Core design choices

- **One canonical event contract:** downstream modules consume `FinancialEvent` objects rather than parsing raw headlines independently.
- **Distinct source and processed state:** archiving a record does not imply that the NLP and downstream cycle completed.
- **Explicit source provenance:** timestamps, source names, URLs and ticker metadata remain attached where supplied.
- **Point-in-time clustering signals:** the engine considers only articles available as of an article's publication time when calculating novelty and corroboration, reducing future-report leakage.
- **Constrained simulations:** Module A enforces long-only weight bounds and a one-way turnover cap, applies the event-specific decay once, and emits progressive replay snapshots; Module B requires negative sentiment for downside scenario triggers and uses declared synthetic scenario assumptions including a simplified derivative shock proxy.
- **Honest model status:** the currently active impact score is an interpretable heuristic. The historical training workflow exists, but a trained/accepted impact-model artifact is not included in the repository version reviewed.

## Model inventory

| Model / component | Role | Status in this repository |
|---|---|---|
| `ProsusAI/finbert` | Financial sentiment | Pretrained inference |
| `whataboutyou-ai/financial_bert` | Financial named-entity extraction | Pretrained inference |
| `leonas5555/finnews-topic-single-classify` | Financial-news topic classification | Pretrained inference |
| `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` | Event labels from candidate descriptions | Pretrained inference, routed with topic model |
| `BAAI/bge-small-en-v1.5` | Text embeddings for semantic similarity | Pretrained inference; general-purpose English embedding model, not a custom finance-fine-tuned model |
| `src/engine/impact.py` | Explainable 1-10 impact score | Heuristic baseline currently active |

## Execution paths

- **Live:** `python -m src.engine.run_live_pipeline --limit 10 --timespan 1day`. Uses public feeds and the live 12-stock mock-index universe. It does not silently substitute synthetic data when a live source fails.
- **Synthetic replay:** `python -m src.engine.run_demo_pipeline`. Uses `data/demo/index_articles.json` (synthetic headlines assigned to the configured 12-stock ticker universe), `data/demo/index_portfolio.json`, and the USD-denominated synthetic banking portfolio and scenarios. Outputs are written under `data/processed/demo_replay/` (ignored by Git) and read by `src/dashboard/app.py`.
- **Dashboard:** `python -m streamlit run src/dashboard/app.py`. The Streamlit page displays the saved synthetic replay; it does not run transformer inference on each refresh.
- **API:** `uvicorn src.api.main:app --reload`. A read-only API over the persisted outputs.
