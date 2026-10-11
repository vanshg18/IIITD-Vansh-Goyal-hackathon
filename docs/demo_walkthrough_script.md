# EventPulse - 10-minute demo walkthrough script

**Required recording:** capture the screen walkthrough and upload to YouTube as **Unlisted**. Add the public watch URL to README and test playback in a private/incognito browser.

## Before recording
1. Activate the venv; confirm dependencies are installed.
2. Run `python -m pytest -q`; observed result on the reviewed branch: 89 passed, 1 dependency deprecation warning.
3. Run `python -m src.engine.run_demo_pipeline`; confirm 12 articles/events, 12 real-ticker mock-index holdings, 11 cluster contributions across 3 snapshots, latest turnover 5.90%, and 3 independent stress results.
4. Run `python -m streamlit run src/dashboard/app.py` and verify all three tabs load.
5. Do not show `.env`, tokens, personal keys, or confidential data. Preload the models before recording so downloads do not consume the walkthrough.

## Timed script

### 0:00-0:40 - Introduction
"EventPulse turns public financial news and corporate disclosures into structured event signals, then demonstrates constrained mock-index rebalancing and scenario-based stress testing. The dashboard shown here is a reproducible synthetic replay, not a live-market forecast."

### 0:40-1:30 - Architecture
Open `architecture.md`. Explain GDELT and SEC EDGAR ingestion, schema normalization, FinBERT sentiment, entity/ticker resolution, finance-topic plus zero-shot routing, embeddings/clustering, point-in-time metadata, and the shared `FinancialEvent` contract. Mention the 1-10 impact score is currently a heuristic.

### 1:30-2:00 - Run the demo
Show:
```cmd
python -m src.engine.run_demo_pipeline
python -m streamlit run src/dashboard/app.py
```
Point out that `data/demo/index_articles.json` contains synthetic headlines assigned to the 12-stock real-ticker universe; it is not live news. Live ingestion is separate.

### 2:00-3:30 - Risk Feed
Show headline, class, sentiment, impact, ticker and cluster size. Example: synthetic JPMorgan Chase credit-outlook downgrade; ticker JPM; class Credit; negative sentiment; impact 8.88/10. Stress that impact is a heuristic, not calibrated probability or return prediction.

### 3:30-5:15 - Module A
Show before/after weights, then the weights-over-time chart. This synthetic replay uses 12 real ticker symbols with equal initial weights for demo purposes, not official index weights. Point out 11 cluster contributions across three progressive snapshots and latest one-way turnover of 5.90%, below the 10% cap.

### 5:15-7:30 - Module B
Select Broad Credit Deterioration. A negative Credit event with impact 8.88 meets the 8.0 threshold. Show USD 1.000B before, USD 923.056M after, and USD 76.944M simulated loss (7.6944%). Show Supply Chain Stress: impact 8.14 and USD 73.880M loss. A separate Regulatory event also triggers the Credit scenario. These are independent runs from the same USD 1B starting portfolio; do not add them. The derivative position uses a simplified configured shock proxy, not options-pricing/Greeks calculations.

### 7:30-8:30 - Live source path
Show the live pipeline command in README. State that a recent smoke run fetched 10 SEC EDGAR records while GDELT returned HTTP 429; that batch had no eligible mock-index ticker match or scenario trigger. Streamlit is currently the synthetic replay dashboard, not the live API dashboard.

### 8:30-9:20 - Tests and engineering
Show `python -m pytest -q` and the result of 89 passed. These validate software behavior; they are not independently measured real-news classification metrics.

### 9:20-10:00 - Close
Summarize the value: source-aware NLP, semantic clustering, reusable structured events, a constrained rebalancer, and transparent banking stress scenarios. Acknowledge limitations: no independent real-headline precision/recall/F1, no accepted trained impact model, synthetic banking shocks, and provider rate limits.

## Recording checklist
- Screen text is readable at 720p or better.
- No secrets or private data are shown.
- Synthetic-demo disclaimer remains visible.
- Video is about 10 minutes and uploaded as Unlisted.
- Playback is tested from a private/incognito window.
- README's placeholder is replaced with the final video URL.
