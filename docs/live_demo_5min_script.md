# EventPulse - 5-minute live demo / jury pitch

This script is for a live jury slot capped at five minutes. The external submission guideline separately calls for a 10-minute unlisted walkthrough recording; use `demo_walkthrough_script.md` for that recording.

## 0:00-0:35 - Problem and product
"Financial news arrives as unstructured text, repeated reports and mixed source formats. EventPulse converts this into structured events - category, sentiment, tickers, impact and provenance - then uses the same event contract for two distinct risk simulations."

## 0:35-1:10 - Show the replay result
Show the completed output from:
```cmd
python -m src.engine.run_demo_pipeline
```
Point out: 12 synthetic articles, 12 structured events, 12 real-ticker mock-index holdings, 11 cluster contributions across three progressive rebalance snapshots, latest turnover 5.90% under the 10% cap, and three independent Module B trigger runs. State clearly that the headlines and financial outcomes are synthetic.

## 1:10-2:00 - Risk Feed
Use the synthetic JPMorgan Chase credit-outlook downgrade. Explain that the router classifies it as Credit, the sentiment is negative, and the active heuristic impact is 8.88/10. This is a heuristic estimate, not a calibrated probability or price-return forecast.

## 2:00-3:00 - Module A
Show before/after weights and the weights-over-time chart. The replay uses a 12-stock real-ticker universe with equal initial weights for demonstration, not official index weights. There are three progressive snapshots and 11 cluster contributions across the replay. Latest one-way turnover is 5.90%, below the 10% cap. There are no real trades.

## 3:00-4:15 - Module B
Show the negative Credit trigger (impact 8.88, threshold 8.0): the synthetic USD 1.000B portfolio falls to USD 923.056M, a simulated USD 76.944M loss (7.6944%). Also show Supply Chain stress (impact 8.14; simulated USD 73.880M loss). A separate Regulatory event triggers the Credit scenario again. All three results are independent, not cumulative.

## 4:15-5:00 - Result and limitations
"The full replay is reproducible and the software test suite passed 89 tests. The impact score remains a heuristic; the banking portfolio and headlines are synthetic; and we do not claim independent real-news accuracy or investment-performance metrics. Live-source availability can vary, including provider rate limits."

## Before presenting
- Run `python -m src.engine.run_demo_pipeline`.
- Run `python -m streamlit run src/dashboard/app.py`.
- Open all three tabs, especially Module A's timeline chart.
- Keep credentials and `.env` off-screen.
