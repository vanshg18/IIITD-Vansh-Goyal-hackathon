
"""EventPulse dashboard for the synthetic end-to-end replay."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "demo_replay"

st.set_page_config(
    page_title="EventPulse Risk Engine",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(ttl=10)
def read_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=10)
def read_jsonl(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return []

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


st.title("EventPulse")
st.caption(
    "AI Financial Risk Engine | Synthetic Demo Replay"
)

st.info(
    "This dashboard displays synthetic demo results. "
    "Portfolio changes and stress losses are simulations, "
    "not live-market predictions."
)

if st.button("Refresh saved results"):
    st.cache_data.clear()
    st.rerun()

events = read_jsonl("events.jsonl")
rebalance = read_json("rebalance.json")
stress_results = read_jsonl("stress_results.jsonl")

if not events or not rebalance:
    st.error(
        "Demo outputs not found. Run "
        "`python -m src.engine.run_demo_pipeline` first."
    )
    st.stop()

# Summary metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Structured Events", len(events))
c2.metric(
    "Events Used by Module A",
    rebalance.get("processed_events", 0),
)
c3.metric(
    "One-Way Turnover",
    f"{rebalance.get('one_way_turnover', 0):.2%}",
)
c4.metric("Module B Stress Scenarios", len(stress_results))

tab_events, tab_a, tab_b = st.tabs(
    ["Risk Feed", "Module A — Rebalancer", "Module B — Stress Test"]
)

# ---------------------------------------------------------
# Risk feed
# ---------------------------------------------------------
with tab_events:
    st.subheader("Structured Financial Events")

    rows = []
    for event in events:
        event_type = event.get("event_type") or {}
        sentiment = event.get("sentiment") or {}
        impact = event.get("impact") or {}

        rows.append({
            "Timestamp": event.get("timestamp", ""),
            "Headline": event.get("title", ""),
            "Category": event_type.get("label", "Unknown"),
            "Sentiment": sentiment.get("score", 0),
            "Impact / 10": impact.get("score", 0),
            "Tickers": ", ".join(
                event.get("tickers")
                or event.get("affected_assets")
                or []
            ),
            "Cluster Size": event.get("cluster_size", 1),
        })

    event_df = pd.DataFrame(rows)

    if not event_df.empty:
        event_df = event_df.sort_values(
            "Impact / 10", ascending=False
        )
        st.dataframe(
            event_df,
            use_container_width=True,
            hide_index=True,
        )

        category_counts = (
            event_df["Category"].value_counts()
            .rename_axis("Category")
            .reset_index(name="Events")
        )

        fig = px.bar(
            category_counts,
            x="Category",
            y="Events",
            title="Events by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Module A — mock index rebalancing
# ---------------------------------------------------------
with tab_a:
    st.subheader("Mock Index Rebalancing")

    st.write(
        "Compare portfolio weights before and after "
        "the synthetic event signals were applied."
    )

    before = rebalance.get("before_weights", {})
    after = rebalance.get("after_weights", {})

    tickers = list(before.keys())

    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Before": [before[t] for t in tickers],
        "After": [after.get(t, before[t]) for t in tickers],
    })

    weights_df["Change (percentage points)"] = (
        weights_df["After"] - weights_df["Before"]
    ) * 100

    chart_df = weights_df.melt(
        id_vars="Ticker",
        value_vars=["Before", "After"],
        var_name="Snapshot",
        value_name="Weight",
    )

    fig = px.bar(
        chart_df,
        x="Ticker",
        y="Weight",
        color="Snapshot",
        barmode="group",
        title="Portfolio Weights Before vs After",
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        weights_df.style.format({
            "Before": "{:.2%}",
            "After": "{:.2%}",
            "Change (percentage points)": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    unmatched = rebalance.get("unmatched_tickers", [])
    if unmatched:
        st.caption(
            "Unmatched event tickers: " + ", ".join(unmatched)
        )

# ---------------------------------------------------------
# Module B — synthetic banking stress testing
# ---------------------------------------------------------
with tab_b:
    st.subheader("Synthetic Banking Portfolio Stress Test")

    if not stress_results:
        st.warning(
            "No stress scenario was triggered in this saved replay."
        )
    else:
        labels = [
            f"{i + 1}. {result.get('scenario_name', 'Scenario')}"
            for i, result in enumerate(stress_results)
        ]

        selected_index = st.selectbox(
            "Select a stress scenario",
            range(len(stress_results)),
            format_func=lambda i: labels[i],
        )

        result = stress_results[selected_index]

        before_value = result.get("portfolio_value_before", 0)
        after_value = result.get("portfolio_value_after", 0)
        loss = result.get("net_loss", 0)
        loss_pct = result.get("net_loss_pct", 0)

        st.markdown(f"**Scenario:** {result['scenario_name']}")
        st.markdown(
            f"**Trigger:** {result.get('trigger_event_type', 'Unknown')} "
            f"| Impact: {result.get('trigger_impact', 0):.2f}/10"
        )

        a, b, c = st.columns(3)
        a.metric("Portfolio Before", f"{before_value:,.0f}")
        b.metric("Portfolio After", f"{after_value:,.0f}")
        c.metric(
            "Simulated Net Loss",
            f"{loss:,.0f}",
            delta=f"{loss_pct:.2%}",
            delta_color="inverse",
        )

        positions = result.get("positions", [])

        if positions:
            positions_df = pd.DataFrame(positions)

            st.subheader("Position-Level Impact")

            if "loss" in positions_df and "name" in positions_df:
                fig = px.bar(
                    positions_df.sort_values("loss"),
                    x="name",
                    y="loss",
                    color="asset_class",
                    title="Position-Level Loss or Gain",
                    labels={
                        "name": "Position",
                        "loss": "Simulated loss",
                    },
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.dataframe(
                positions_df,
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.caption(
    "EventPulse | Synthetic demonstration only | "
    "Not an investment recommendation"
)
