import streamlit as st

st.set_page_config(page_title="Risk-Aware Credit Card Growth Engine", layout="wide")

st.title("Risk-Aware Credit Card Growth Decision Engine")
st.caption("Capital One-style customer strategy simulator using synthetic/public data")

st.info("Dashboard build placeholder. We will add portfolio overview, next-best-action, scenario simulator, and A/B test planner tabs after the analytics layer is complete.")

tabs = st.tabs(["Portfolio Overview", "Next Best Action", "Scenario Simulator", "A/B Test Planner"])

with tabs[0]:
    st.subheader("Portfolio Overview")
    st.write("Segment distribution, spend, utilization, default risk, and profitability will appear here.")

with tabs[1]:
    st.subheader("Next Best Action")
    st.write("Customer/segment-level offer recommendations will appear here.")

with tabs[2]:
    st.subheader("Scenario Simulator")
    st.write("Offer assumption sliders and ROI outputs will appear here.")

with tabs[3]:
    st.subheader("A/B Test Planner")
    st.write("Hypothesis, treatment/control design, success metrics, and guardrails will appear here.")
