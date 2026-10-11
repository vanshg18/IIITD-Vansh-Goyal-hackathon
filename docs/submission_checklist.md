# EventPulse submission checklist

Checked against the provided external submission guideline and the current review branch.

## Implemented / verified
- [x] Public GitHub repository and MIT license.
- [x] Python source under `src/`, dependency file, and safe `.env.example`.
- [x] `architecture.md`, standalone Mermaid source, and SVG diagram.
- [x] Streamlit dashboard and local run command documented in README.
- [x] Synthetic article/portfolio/scenario fixtures are included under `data/demo/`.
- [x] Module A uses 12 real ticker symbols in the synthetic replay, with 3 progressive snapshots and weights-over-time visualization.
- [x] Module B downside trigger requires a matching event category, minimum impact, and negative sentiment.
- [x] Synthetic banking portfolio is USD 1,000,000,000 with nine positions, including a simplified derivative exposure shock proxy.
- [x] Updated local test run: 89 passed, 1 dependency deprecation warning.
- [x] Updated synthetic replay: 12 articles, 12 structured events, 11 cluster contributions across 3 snapshots, latest turnover 5.90% (10% cap), 3 independent stress results.
- [x] Seven-slide presentation PDF and editable PowerPoint prepared locally.

## Still needs candidate action
- [ ] Copy `docs/presentation.pdf` and `docs/presentation.pptx` from the local submission package into the repository and commit/push them.
- [ ] Replace the college-email placeholder in README with the college email address.
- [ ] Record the 10-minute walkthrough, upload to YouTube as Unlisted, and add the final URL to README.
- [ ] If the jury slot is five minutes, use the five-minute live demo script for that format.
- [ ] Test repository visibility and all final links in a private/incognito browser.
- [ ] Re-run `python -m src.engine.run_demo_pipeline` on the final merged branch and keep the demo output consistent with the deck.

## Honesty / limitations
- The impact score is a heuristic, not a calibrated probability or return forecast.
- No independent real-news accuracy/precision/recall/F1 or economic backtest is claimed.
- No accepted historical impact-model artifact is included.
- Demo headline inputs and banking portfolio shocks are synthetic; the derivative handling is a configured shock proxy, not a derivatives valuation model.
- The Streamlit dashboard reads the local synthetic replay outputs; it is not a deployed cloud service or a live-feed dashboard.
- The live smoke test fetched 10 SEC records but GDELT was rate-limited (HTTP 429).
