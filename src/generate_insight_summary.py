from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

CUSTOMER_FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
SEGMENT_SUMMARY_PATH = BASE_DIR / "data" / "processed" / "segment_summary.csv"
OFFER_SUMMARY_PATH = BASE_DIR / "data" / "processed" / "offer_eligibility_summary.csv"
PORTFOLIO_KPIS_PATH = BASE_DIR / "data" / "processed" / "portfolio_kpis.csv"

OUTPUT_PATH = BASE_DIR / "reports" / "insight_summary.md"


def money(x):
    return f"${x:,.0f}"


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    customer_features = pd.read_csv(CUSTOMER_FEATURES_PATH)
    segment_summary = pd.read_csv(SEGMENT_SUMMARY_PATH)
    offer_summary = pd.read_csv(OFFER_SUMMARY_PATH)
    portfolio_kpis = pd.read_csv(PORTFOLIO_KPIS_PATH).iloc[0]

    total_customers = int(portfolio_kpis["total_customers"])
    avg_credit_score = portfolio_kpis["avg_credit_score"]
    avg_utilization = portfolio_kpis["avg_utilization_rate"]
    avg_default_probability = portfolio_kpis["avg_default_probability"]
    total_monthly_spend = portfolio_kpis["total_monthly_spend"]
    total_profit = portfolio_kpis["total_monthly_risk_adjusted_profit"]
    avg_profit = portfolio_kpis["avg_risk_adjusted_profit"]
    campaign_eligible_rate = portfolio_kpis["campaign_eligible_rate"]
    block_rate = portfolio_kpis["block_rate"]

    # Prioritized segments by total risk-adjusted profit and campaign eligibility.
    priority_segments = (
        segment_summary
        .sort_values(["total_risk_adjusted_profit", "campaign_eligible_rate"], ascending=False)
        .head(3)
    )

    # Segments with the highest guardrail/risk concern.
    risk_segments = (
        segment_summary
        .sort_values(["block_count", "avg_utilization_rate", "avg_default_probability"], ascending=False)
        .head(3)
    )

    # Best scale opportunities.
    scale_customers = customer_features[customer_features["decision_status"] == "Scale"]
    scale_by_segment = (
        scale_customers
        .groupby("customer_segment")
        .agg(
            scale_customers=("customer_id", "count"),
            avg_roi=("expected_roi", "mean"),
            avg_profit=("risk_adjusted_profit", "mean"),
            avg_default_probability=("default_probability", "mean"),
        )
        .reset_index()
        .sort_values("scale_customers", ascending=False)
    )

    # Test opportunities.
    test_customers = customer_features[customer_features["decision_status"] == "Test"]
    test_by_segment = (
        test_customers
        .groupby("customer_segment")
        .agg(
            test_customers=("customer_id", "count"),
            avg_roi=("expected_roi", "mean"),
            avg_profit=("risk_adjusted_profit", "mean"),
            avg_default_probability=("default_probability", "mean"),
        )
        .reset_index()
        .sort_values("test_customers", ascending=False)
    )

    # Offer summaries.
    offer_rollup = (
        offer_summary
        .groupby(["recommended_action", "offer_type"])
        .agg(
            customer_count=("customer_count", "sum"),
            avg_expected_roi=("avg_expected_roi", "mean"),
            avg_risk_adjusted_profit=("avg_risk_adjusted_profit", "mean"),
            avg_default_probability=("avg_default_probability", "mean"),
        )
        .reset_index()
        .sort_values("customer_count", ascending=False)
    )

    # Responsible lending check.
    high_util = customer_features[customer_features["customer_segment"] == "High-Utilization Revolver"]
    high_util_decisions = high_util["decision_status"].value_counts()

    risk_watch = customer_features[customer_features["customer_segment"] == "Risk Watch"]
    risk_watch_decisions = risk_watch["decision_status"].value_counts()

    lines = []

    lines.append("# Insight Summary: Risk-Aware Credit Card Growth Decision Engine\n")

    lines.append("## 1. Executive Snapshot\n")
    lines.append(
        f"The synthetic portfolio contains **{total_customers:,} existing credit card customers**. "
        f"The average credit score is **{avg_credit_score:.0f}**, average utilization is **{pct(avg_utilization)}**, "
        f"and average default probability is **{pct(avg_default_probability)}**."
    )
    lines.append(
        f"\nThe portfolio generates about **{money(total_monthly_spend)} in monthly spend** and "
        f"approximately **{money(total_profit)} in monthly risk-adjusted profit**, or "
        f"about **{money(avg_profit)} per customer**."
    )
    lines.append(
        f"\nAbout **{pct(campaign_eligible_rate)}** of customers are eligible for either a Scale or Test action, "
        f"while **{pct(block_rate)}** are blocked by risk guardrails."
    )

    lines.append("\n## 2. Portfolio Decision Mix\n")
    decision_mix = customer_features["decision_status"].value_counts().reset_index()
    decision_mix.columns = ["decision_status", "customer_count"]
    decision_mix["share_of_portfolio"] = decision_mix["customer_count"] / total_customers
    lines.append(decision_mix.to_markdown(index=False))

    lines.append("\n\n## 3. Segment-Level Findings\n")
    lines.append(
        "The highest-volume segments are Core Customer and Loyal High-Value Customer. "
        "These groups drive much of the scale opportunity because they combine large population size, "
        "positive profitability, and manageable risk."
    )
    lines.append("\n")
    lines.append(
        segment_summary[
            [
                "customer_segment",
                "customer_count",
                "avg_credit_score",
                "avg_utilization_rate",
                "avg_default_probability",
                "avg_risk_adjusted_profit",
                "avg_expected_roi",
                "scale_count",
                "test_count",
                "do_not_launch_count",
                "block_count",
                "campaign_eligible_rate",
            ]
        ].to_markdown(index=False)
    )

    lines.append("\n\n## 4. Priority Segments\n")
    lines.append(
        "Based on total risk-adjusted profit and campaign eligibility, the strongest segments to prioritize are:"
    )
    lines.append("\n")
    lines.append(
        priority_segments[
            [
                "customer_segment",
                "customer_count",
                "total_risk_adjusted_profit",
                "avg_expected_roi",
                "scale_count",
                "test_count",
                "campaign_eligible_rate",
            ]
        ].to_markdown(index=False)
    )
    lines.append(
        "\n\nBusiness interpretation: Core Customer and Loyal High-Value Customer are the strongest broad-scale "
        "growth priorities because they combine large customer counts, positive profitability, and manageable risk. "
        "High-Utilization Revolver is financially meaningful, but it should be treated as a controlled test and "
        "guardrail segment rather than a broad growth campaign."
    )

    lines.append("\n\n## 5. Scale Opportunities\n")
    lines.append(
        "Customers marked as Scale have stronger economics and acceptable risk. "
        "These customers are the best candidates for broad rollout after standard business review."
    )
    lines.append("\n")
    lines.append(scale_by_segment.to_markdown(index=False))

    lines.append("\n\n## 6. Test Opportunities\n")
    lines.append(
        "Customers marked as Test are not weak customers. They are customers where the strategy is promising, "
        "but the business should validate lift and risk behavior before scaling."
    )
    lines.append("\n")
    lines.append(test_by_segment.to_markdown(index=False))

    lines.append("\n\n## 7. Offer and Action Insights\n")
    lines.append(
        "The decision engine does not recommend the same offer to everyone. "
        "The action depends on customer segment, risk, profitability, and expected ROI."
    )
    lines.append("\n")
    lines.append(offer_rollup.to_markdown(index=False))

    lines.append("\n\n## 8. Responsible-Lending Guardrail Check\n")
    lines.append(
        "The most important guardrail is that High-Utilization Revolver customers are not scaled aggressively. "
        "These customers may generate interest revenue, but pushing additional spend can increase credit exposure."
    )
    lines.append("\n")
    lines.append("High-Utilization Revolver decision mix:\n")
    lines.append(high_util_decisions.to_markdown())

    lines.append("\n\nRisk Watch decision mix:\n")
    lines.append(risk_watch_decisions.to_markdown())

    lines.append(
        "\n\nBusiness interpretation: the model separates revenue potential from responsible growth. "
        "This is important because a profitable customer is not automatically a safe customer to target."
    )

    lines.append("\n\n## 9. Recommended Strategy\n")
    lines.append(
        "**Recommendation:** Scale growth campaigns for Core Customer and Loyal High-Value Customer segments, "
        "run controlled A/B tests for High-Utilization Revolvers and other borderline groups, "
        "and block growth campaigns for Risk Watch customers."
    )
    lines.append(
        "\n\nThe strongest business move is not to maximize campaign reach. "
        "The stronger move is to maximize risk-adjusted growth by scaling safe profitable customers, "
        "testing uncertain groups, and protecting customers where credit risk is elevated."
    )

    lines.append("\n\n## 10. How This Feeds the App\n")
    lines.append(
        "The Dash app should show the portfolio overview, segment strategy, offer decision logic, "
        "scenario simulator, and A/B test planner. "
        "The Gradio demo should focus on the upload-and-run decision engine, where a user can upload a customer file "
        "and receive recommended actions."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))

    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()