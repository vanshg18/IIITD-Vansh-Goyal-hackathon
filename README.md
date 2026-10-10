# EventPulse — S&P Global & Crisil Campus Hackathon

> **From Headlines → Events → Impact → Action**

EventPulse converts unstructured financial news and corporate disclosures into structured, explainable event signals. Signals can feed a constrained mock-index rebalancer (Module A) and a synthetic banking portfolio stress tester (Module B).

## Current implementation

- **Ingestion:** GDELT DOC public news API, SEC EDGAR current 8-K filings, and optional Alpha Vantage news.
- **Traceability:** normalized article IDs, source names, source URLs, timestamps, ticker metadata when available, and JSONL persistence.
- **NLP:** FinBERT sentiment, a finance-topic/zero-shot event classifier router, entity/ticker resolution, and semantic clustering.
- **Impact:** the current 1–10 impact score is an interpretable heuristic baseline. It is **not yet trained or calibrated against historical market reactions**.
- **Reproducibility:** synthetic fixtures remain available for unit tests; live mode never silently replaces missing live data with synthetic data.

## Setup

Use Python 3.11 or 3.12 in a virtual environment:

~~~bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

Copy .env.example to .env. For SEC EDGAR, replace SEC_USER_AGENT with a descriptive application name and a real contact email, for example:

~~~dotenv
SEC_USER_AGENT=EventPulse/1.0 Your Name your.email@example.com
~~~

The optional Alpha Vantage feed is only enabled when ALPHAVANTAGE_API_KEY is set. The GDELT adapter needs no API key. SEC requests identify the application through the configured User-Agent.

## Run ingestion

Fetch real news and public SEC filings, persist new records to data/processed/live_articles.jsonl, and report every source's status:

~~~bash
python -m src.ingestion.run_ingestion --mode live --limit 20 --timespan 1day
~~~

Try a shorter GDELT window or custom query:

~~~bash
python -m src.ingestion.run_ingestion --mode live --limit 10 --timespan 6h --query '("stock market" OR earnings OR "central bank")'
~~~

Run offline synthetic fixtures explicitly:

~~~bash
python -m src.ingestion.run_ingestion --mode demo --limit 5 --output data/processed/demo_articles.jsonl
~~~

To require every configured source to work, add --require-all-sources. Without it, a failed source is reported and other working sources can continue. If no live articles are fetched, the command exits non-zero; it does not fall back to demo inputs.

## Run the complete live risk-engine cycle

This is the primary real-data command. It fetches from GDELT and SEC EDGAR (plus Alpha Vantage when configured), runs the pretrained intelligence pipeline on not-yet-processed articles, then updates Module A and triggers Module B stress scenarios where the signal qualifies:

~~~bash
python -m src.engine.run_live_pipeline --limit 10 --timespan 1day
~~~

Outputs are stored under `data/processed/` (ignored by Git):

- `live_articles.jsonl`: deduplicated raw article archive.
- `processed_articles.json`: fingerprints of items whose NLP and downstream cycle completed. This is separate from raw archival so items previously saved by the ingestion-only command can still be processed.
- `latest_events.jsonl` and `events_history.jsonl`: canonical financial event signals.
- `latest_rebalance.json`, `rebalance_history.jsonl`, and `index_portfolio_state.json`: Module A outputs and persistent simulated weights.
- `latest_stress_results.jsonl` and `stress_results_history.jsonl`: Module B outcomes and history.
- `latest_cycle_status.json`: completion status and per-source success/errors for the most recent run.

Overlapping news windows do not reapply already processed articles, and the index simulation state persists across runs. If you intentionally want to start the mock index from equal weights, back up and remove `index_portfolio_state.json`; do not delete the processed registry unless you also intend to reprocess old stories.

Transformer weights are downloaded by the NLP libraries on the first full run, so that run can take longer. Live provider connectivity is still dependent on network access and valid source configuration.

The lower-level NLP-only command remains available when you only want to emit event signals:

~~~bash
python -m src.engine.run_live_intelligence --limit 10 --timespan 1day
~~~

## Read signals over the API

After running the live cycle at least once, start the file-backed read-only API:

~~~bash
uvicorn src.api.main:app --reload
~~~

FastAPI docs: `http://127.0.0.1:8000/docs`

Endpoints include:

- `GET /api/v1/health` and `GET /api/v1/cycle/latest`: service/artifact status and per-source outcomes.
- `GET /api/v1/events/latest` and `GET /api/v1/events/history?limit=100`: structured event signals.
- `GET /api/v1/module-a/latest` and `GET /api/v1/module-a/history?limit=100`: weights, signal changes and turnover.
- `GET /api/v1/module-b/latest` and `GET /api/v1/module-b/history?limit=100`: triggered synthetic portfolio stress results.
- `GET /api/v1/summary`: event-type counts, average sentiment/impact, and stress-test counts.

The API reads persisted files and does not load transformer models per request. Refresh the signals by running the live-cycle command separately.

## Downstream modules

### Module A — sentiment-responsive mock-index rebalancing

The separate Module A portfolio contains 12 real large-cap tickers associated with the S&P Global 100 constituent universe. Starting weights are equal-weighted for the simulation and **are not official index weights**. The rebalancer applies sentiment/impact signals, deduplicates repeat reports by cluster, normalizes weights, enforces configurable minimum/maximum asset weights, and caps one-way turnover. Unmatched tickers are reported rather than silently included.

### Module B — synthetic wholesale-bank stress testing

Module B uses the explicitly synthetic portfolio in data/demo/banking_portfolio.json. High-impact event classes trigger named scenario rules from data/demo/bank_stress_scenarios.json. The model applies sector/equity shocks, duration-based bond repricing, spread widening, and incremental loan default-rate assumptions, then calculates before/after value and position-level loss attribution. Scenario inputs are assumptions for stress testing, not estimates of actual bank holdings or forecasts.

## Canonical signal output

Each structured event includes its event timestamp and source provenance, title/text, tickers, event classification and confidence, sentiment in [-1, 1], impact score in [1, 10], novelty/corroboration/time-decay metadata, and cluster identifiers.

Example commands for the existing synthetic NLP path:

~~~bash
python -m src.engine.run_intelligence_demo
python -m src.nlp.run_pretrained_demo
~~~

## Testing

The ingestion tests are network-independent and mock GDELT/SEC API responses:

~~~bash
pytest -q tests/test_ingestion.py tests/test_live_ingestion.py
~~~

Run the complete suite after installing all project dependencies:

~~~bash
pytest -q
~~~

## Train an impact model from historical data

The historical model uses a market-observable target: the magnitude of a ticker's next-session return relative to the equal-weight return of the loaded comparison universe. It trains on EventPulse outputs, not hand-written synthetic impact labels. The resulting score is a relative 1–10 rank; it is not a probability or causal claim.

For a practical first experiment, use a small ticker universe and a few thousand news records. FNSPID is one possible public starting point; fetch data from its upstream repository (https://github.com/Zdong104/FNSPID_Financial_News_Dataset) and verify the dataset license/terms before use. Keep raw downloads outside Git, e.g. under data/raw/.

1. Score historical news with the same FinBERT and event router used in EventPulse:

~~~bash
python -m src.ml.score_historical_news --input data/raw/nasdaq_exteral_data.csv --tickers AAPL,MSFT,NVDA,AMZN,JPM --limit 2000 --device 0
~~~

2. Label those structured events against historical prices. The price input may be a long CSV, a directory of per-ticker CSVs, or a ZIP of per-ticker CSVs:

~~~bash
python -m src.ml.train_market_impact --prices data/raw/full_history.zip --benchmark-tickers AAPL,MSFT,NVDA,AMZN,JPM
~~~

3. Inspect data/processed/impact_model_metrics.json. The artifact is only enabled for live inference if its validation MAE beats the training-median baseline. The final test block is chronological and is not used to choose acceptance.

The current implementation uses daily close-to-close returns. It treats features as end-of-day information and aligns weekend/holiday news to the latest earlier close. Daily data cannot precisely identify intraday market reaction, so this is a proxy, not causal attribution. For a larger or serious financial evaluation, prefer timestamped intraday prices and a longer date range. If the model does not beat the baseline, EventPulse deliberately continues to use the transparent heuristic impact score.

## Important modeling limitations

- FinBERT linguistic sentiment is not the same as a forecast of price direction.
- The current impact score is a severity/confidence heuristic; do not interpret it as a calibrated probability or expected loss.
- Real-data classification quality must be measured on independently labeled real headlines. The small synthetic fixture benchmark is a smoke test, not evidence of real-world accuracy.
- A historical impact model should be trained with time-ordered splits and a documented news-availability cutoff to prevent look-ahead leakage.
- Module B uses a synthetic banking portfolio and explicit scenario assumptions. It is a stress-test simulation, not a valuation of a real bank.

## Data and responsible use

Do not commit API keys, private customer data, model checkpoints, or large downloaded datasets. Follow each data provider's terms, rate limits, and attribution requirements. This repository does not contain proprietary S&P Global or Crisil data.
