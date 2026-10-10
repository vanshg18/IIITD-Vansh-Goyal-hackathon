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

## Run end-to-end NLP on live records

This command fetches live articles, appends de-duplicated raw articles to JSONL, runs the pretrained intelligence pipeline, and writes the most recent structured events:

~~~bash
python -m src.engine.run_live_intelligence --limit 10 --timespan 1day
~~~

Outputs are stored locally under data/processed/ (ignored by Git). Transformer weights are downloaded by the NLP libraries when required. A complete first run can take longer than later runs.

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
