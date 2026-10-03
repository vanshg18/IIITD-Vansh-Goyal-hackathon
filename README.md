# EventPulse - S&P Global & Crisil Campus Hackathon

> From Headlines → Events → Impact → Action

## Candidate

**Candidate Name:** Vansh Goyal  
**College / Campus:** IIIT Delhi

## Project Overview

EventPulse is an AI/NLP-driven financial risk engine that converts
unstructured financial information into structured, explainable financial
events and downstream portfolio signals.

Instead of independently reacting to every headline, the system identifies
underlying events, extracts sentiment and event type, estimates financial
impact, evaluates confidence and corroboration, applies novelty and time
decay, and maps the resulting signal to portfolio exposures.

The engine supports two downstream applications:

1. Dynamic Index Rebalancing
2. Strategic Portfolio Stress Testing

## Architecture

```text
Data Sources
     ↓
Ingestion
     ↓
Preprocessing
     ↓
NLP / Event Extraction
     ↓
Financial Event Engine
     ↓
Signal Generation
     ├── Module A: Index Rebalancing
     └── Module B: Portfolio Stress Testing
     ↓
Dashboard / API
=======
