# Risk-Aware Credit Card Growth Decision Engine

A Capital One-style customer strategy case project that recommends credit card growth actions using customer segmentation, risk-adjusted profitability, responsible-lending guardrails, scenario simulation, and A/B test planning.

> This project uses public and/or synthetic data only. It does not use Capital One internal, confidential, or customer data.

## Business Problem

A credit card issuer wants to grow customer engagement and revenue, but growth offers should not be sent equally to all customers. Some customers are high-value and low-risk, some are under-engaged, and some may increase credit exposure if targeted aggressively.

The goal is to answer:

**Which customers should receive which offer, what is the expected business impact, and which offers should be scaled, tested, or blocked because of risk?**

## Planned Deliverables

- SQL feature layer using DuckDB
- Python customer segmentation and risk scoring
- Risk-adjusted profitability and ROI model
- Next-best-action recommendation engine
- Responsible-lending guardrails
- Scenario simulator
- A/B test launch planner
- Streamlit dashboard
- Executive decision memo
- Final public GitHub README and report

## Core Tools

- Python
- SQL
- DuckDB
- pandas
- scikit-learn
- Streamlit
- Plotly / Matplotlib
- Optional: dbt tests, MLflow, SHAP

## Project Structure

```text
risk-aware-credit-card-growth-engine/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic_case_data/
├── sql/
├── notebooks/
├── app/
├── reports/
├── visuals/
├── src/
├── docs/
├── README.md
└── requirements.txt
```

## Project Status

Planning locked. Build in progress.
