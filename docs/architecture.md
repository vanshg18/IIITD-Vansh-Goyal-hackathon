# EventPulse Architecture

The full architecture description and Mermaid source are maintained in the repository-root [`architecture.md`](../architecture.md). This file is a convenient docs-folder entry point for reviewers.

The standalone Mermaid source is [`architecture.mmd`](architecture.mmd). The rendered high-resolution architecture diagram is [`architecture.png`](architecture.png).

Key separation of concerns:

1. Ingestion adapters normalize public GDELT and SEC EDGAR content (with optional Alpha Vantage).
2. Pretrained NLP components extract entities, sentiment, event topic and semantic embeddings.
3. The router, clustering and point-in-time metadata logic produce structured `FinancialEvent` objects.
4. Module A performs constrained mock-index rebalancing; Module B runs rule-triggered stress tests on a synthetic banking portfolio.
5. JSON/JSONL outputs feed the read-only API and Streamlit dashboard.

The current impact score is a heuristic baseline, not a validated return forecast. The demo bank portfolio and stress losses are synthetic.
