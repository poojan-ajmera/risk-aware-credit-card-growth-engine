from pathlib import Path

import numpy as np
import pandas as pd


CUSTOMER_PATH = Path("data/processed/customer_features.csv")
PROFILE_PATH = Path("data/synthetic_case_data/synthetic_customer_profiles.csv")
CAMPAIGN_PATH = Path("data/campaigns/campaign_library.csv")
OUTPUT_PATH = Path("data/campaigns/campaign_recommendations.csv")


def split_segments(value: str) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def campaign_horizon_months(campaign_type: str) -> int:
    mapping = {
        "Category Cashback": 3,
        "Lifestyle Rewards": 3,
        "Travel Rewards": 4,
        "Merchant-Funded Offer": 3,
        "Digital Engagement": 6,
        "Reactivation": 4,
        "Balance Transfer / Revolver": 6,
        "Credit Line Review": 6,
        "Protective Engagement": 6,
        "Retention": 6,
        "Servicing Engagement": 6,
        "Early Relationship": 6,
    }
    return mapping.get(campaign_type, 3)


def get_category_spend(row: pd.Series, category_focus: str) -> float:
    category = str(category_focus).lower()

    if "grocery" in category:
        return row.get("grocery_spend", row["monthly_spend"] * 0.24)
    if "dining" in category:
        return row.get("dining_spend", row["monthly_spend"] * 0.20)
    if "travel" in category:
        return row.get("travel_spend", row["monthly_spend"] * 0.14)
    if "gas" in category or "transportation" in category:
        return row.get("gas_spend", row["monthly_spend"] * 0.11)
    if "online" in category or "retail" in category:
        return row.get("online_spend", row["monthly_spend"] * 0.22)
    if "utilities" in category:
        return row["monthly_spend"] * 0.10
    if "digital" in category or "servicing" in category or "education" in category:
        return row["monthly_spend"] * 0.06
    if "balance" in category or "purchase apr" in category or "installment" in category:
        return max(row.get("revolving_balance", 0), row["monthly_spend"] * 0.40)
    if "credit line" in category:
        return row["monthly_spend"] * 0.45
    if "retention" in category:
        return row["monthly_spend"] * 0.30

    return row["monthly_spend"] * 0.18


def customer_passes_campaign_guardrails(customer: pd.Series, campaign: pd.Series) -> bool:
    if customer["customer_segment"] in split_segments(campaign["excluded_segments"]):
        return False

    if customer["default_probability"] > campaign["max_default_probability"]:
        return False

    if customer["utilization_rate"] > campaign["max_utilization"]:
        return False

    if customer["credit_score"] < campaign["min_credit_score"]:
        return False

    if customer["late_payments_12m"] > 0 and campaign["risk_sensitivity"] in ["High", "Very High"]:
        return False

    if customer["risk_adjusted_profit"] < -10 and campaign["risk_sensitivity"] != "Protective":
        return False

    return True


def estimate_customer_campaign_value(customer: pd.Series, campaign: pd.Series) -> dict:
    target_segments = split_segments(campaign["target_segments"])
    excluded_segments = split_segments(campaign["excluded_segments"])

    is_target_segment = customer["customer_segment"] in target_segments if target_segments else True
    is_excluded_segment = customer["customer_segment"] in excluded_segments if excluded_segments else False
    passes_guardrails = customer_passes_campaign_guardrails(customer, campaign)

    is_candidate = is_target_segment and passes_guardrails and not is_excluded_segment

    category_spend = get_category_spend(customer, campaign["spend_category_focus"])
    horizon = campaign_horizon_months(campaign["campaign_type"])

    digital_score = customer.get("digital_engagement_score", 55)
    if pd.isna(digital_score):
        digital_score = 55

    response_rate = float(campaign["response_rate_assumption"])

    if is_target_segment:
        response_rate *= 1.25
    else:
        response_rate *= 0.50

    if digital_score >= 75:
        response_rate *= 1.12
    elif digital_score < 40:
        response_rate *= 0.85

    if customer["risk_band"] in ["High Risk", "Very High Risk"]:
        response_rate *= 0.70

    if customer["recommended_action"] == "Scale":
        response_rate *= 1.10
    elif customer["recommended_action"] == "Test":
        response_rate *= 0.95
    elif customer["recommended_action"] == "Block":
        response_rate *= 0.40

    response_rate = float(np.clip(response_rate, 0.002, 0.40))

    expected_lift = float(campaign["expected_lift_pct"])

    monthly_responder_incremental_spend = category_spend * expected_lift
    expected_incremental_spend = monthly_responder_incremental_spend * response_rate * horizon

    campaign_type = str(campaign["campaign_type"])

    interchange_margin = 0.020
    rewards_cost_rate = 0.009

    expected_campaign_cost = float(campaign["cost_per_customer"]) * max(response_rate, 0.02)

    if campaign_type == "Merchant-Funded Offer":
        expected_campaign_cost *= 0.30
        rewards_cost_rate = 0.004

    if campaign_type in ["Digital Engagement", "Servicing Engagement", "Protective Engagement"]:
        expected_campaign_cost *= 0.45
        rewards_cost_rate = 0.003

    if campaign_type == "Balance Transfer / Revolver":
        finance_margin = 0.245 / 12
        risk_cost_rate = customer["default_probability"] * 1.15
        expected_revenue = expected_incremental_spend * (interchange_margin + finance_margin)
    elif campaign_type == "Credit Line Review":
        finance_margin = 0.245 / 12
        risk_cost_rate = customer["default_probability"] * 0.95
        expected_revenue = expected_incremental_spend * (interchange_margin + finance_margin * 0.50)
    elif campaign_type == "Protective Engagement":
        expected_loss_base = customer["default_probability"] * max(customer["current_balance"], 0) * 0.72
        expected_risk_savings = expected_loss_base * 0.040 * response_rate
        expected_revenue = expected_incremental_spend * interchange_margin + expected_risk_savings
        risk_cost_rate = customer["default_probability"] * 0.25
    elif campaign_type == "Servicing Engagement":
        expected_loss_base = customer["default_probability"] * max(customer["current_balance"], 0) * 0.72
        expected_risk_savings = expected_loss_base * 0.020 * response_rate
        expected_revenue = expected_incremental_spend * interchange_margin + expected_risk_savings
        risk_cost_rate = customer["default_probability"] * 0.35
    elif campaign_type in ["Retention", "Travel Rewards", "Lifestyle Rewards"]:
        risk_cost_rate = customer["default_probability"] * 0.50
        expected_revenue = expected_incremental_spend * (interchange_margin + 0.004)
    else:
        risk_cost_rate = customer["default_probability"] * 0.45
        expected_revenue = expected_incremental_spend * interchange_margin

    expected_rewards_cost = expected_incremental_spend * rewards_cost_rate
    expected_risk_cost = expected_incremental_spend * risk_cost_rate

    expected_profit = (
        expected_revenue
        - expected_rewards_cost
        - expected_risk_cost
        - expected_campaign_cost
    )

    expected_roi = expected_profit / expected_campaign_cost if expected_campaign_cost > 0 else 0

    if not passes_guardrails:
        decision = "Block"
    elif not is_target_segment:
        decision = "Do Not Launch"
    elif expected_profit >= 0.20 and expected_roi >= 0.20:
        decision = "Scale"
    elif expected_profit >= -0.20:
        decision = "Test"
    else:
        decision = "Do Not Launch"

    return {
        "is_target_segment": is_target_segment,
        "passes_guardrails": passes_guardrails,
        "is_candidate": is_candidate,
        "response_rate": response_rate,
        "expected_incremental_spend": expected_incremental_spend if is_candidate else 0,
        "expected_profit": expected_profit if is_candidate else 0,
        "expected_cost": expected_campaign_cost if is_candidate else 0,
        "expected_roi": expected_roi if is_candidate else 0,
        "campaign_decision": decision,
    }


def score_campaign(customers: pd.DataFrame, campaign: pd.Series) -> dict:
    scored = pd.DataFrame(
        [estimate_customer_campaign_value(customer, campaign) for _, customer in customers.iterrows()]
    )

    customer_count = len(scored)
    target_segment_customers = int(scored["is_target_segment"].sum())
    eligible_customers = int(scored["is_candidate"].sum())

    scale_customers = int((scored["campaign_decision"] == "Scale").sum())
    test_customers = int((scored["campaign_decision"] == "Test").sum())
    do_not_launch_customers = int((scored["campaign_decision"] == "Do Not Launch").sum())
    blocked_customers = int((scored["campaign_decision"] == "Block").sum())

    expected_profit = float(scored["expected_profit"].sum())
    expected_incremental_spend = float(scored["expected_incremental_spend"].sum())
    expected_cost = float(scored["expected_cost"].sum())
    avg_response_rate = float(scored.loc[scored["is_candidate"], "response_rate"].mean()) if eligible_customers else 0

    expected_roi = expected_profit / expected_cost if expected_cost > 0 else 0

    eligible_rate = eligible_customers / customer_count if customer_count else 0
    block_rate = blocked_customers / customer_count if customer_count else 0
    scale_rate = scale_customers / customer_count if customer_count else 0
    test_rate = test_customers / customer_count if customer_count else 0

    profit_score = np.clip(expected_profit / 8000, -25, 45)
    roi_score = np.clip(expected_roi * 9, -25, 35)
    coverage_score = eligible_rate * 18
    scale_score = scale_rate * 20
    test_score = test_rate * 8
    response_score = avg_response_rate * 30
    risk_penalty = block_rate * 28

    campaign_score = (
        profit_score
        + roi_score
        + coverage_score
        + scale_score
        + test_score
        + response_score
        - risk_penalty
    )

    if campaign["risk_level"] == "Protective":
        campaign_score += 2

    if expected_profit > 0 and expected_roi >= 0.20 and scale_customers >= 300 and block_rate <= 0.35:
        rollout = "Scale"
    elif expected_profit > 0 and eligible_customers >= 250 and block_rate <= 0.55:
        rollout = "Test"
    elif block_rate > 0.55:
        rollout = "Constrain"
    else:
        rollout = "Do Not Launch"

    return {
        "campaign_id": campaign["campaign_id"],
        "campaign_name": campaign["campaign_name"],
        "campaign_family": campaign["campaign_family"],
        "campaign_type": campaign["campaign_type"],
        "risk_level": campaign["risk_level"],
        "risk_sensitivity": campaign["risk_sensitivity"],
        "business_goal": campaign["business_goal"],
        "offer_description": campaign["offer_description"],
        "target_segments": campaign["target_segments"],
        "excluded_segments": campaign["excluded_segments"],
        "primary_success_metric": campaign["primary_success_metric"],
        "guardrail_notes": campaign["guardrail_notes"],
        "cost_per_customer": campaign["cost_per_customer"],
        "expected_lift_pct": campaign["expected_lift_pct"],
        "response_rate_assumption": campaign["response_rate_assumption"],
        "campaign_horizon_months": campaign_horizon_months(campaign["campaign_type"]),
        "customer_count": customer_count,
        "target_segment_customers": target_segment_customers,
        "eligible_customers": eligible_customers,
        "scale_customers": scale_customers,
        "test_customers": test_customers,
        "do_not_launch_customers": do_not_launch_customers,
        "blocked_customers": blocked_customers,
        "eligible_rate": eligible_rate,
        "scale_rate": scale_rate,
        "test_rate": test_rate,
        "block_rate": block_rate,
        "avg_predicted_response_rate": avg_response_rate,
        "expected_incremental_spend": expected_incremental_spend,
        "expected_campaign_cost": expected_cost,
        "expected_campaign_profit": expected_profit,
        "expected_campaign_roi": expected_roi,
        "campaign_score": campaign_score,
        "recommended_rollout_decision": rollout,
    }


def assign_diversified_dashboard_rank(recommendations: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    score_order = recommendations.sort_values(
        ["campaign_score", "expected_campaign_profit"],
        ascending=[False, False],
    ).copy()
    score_order["model_score_rank"] = range(1, len(score_order) + 1)

    viable = score_order[
        (score_order["expected_campaign_profit"] > 0)
        & (score_order["recommended_rollout_decision"].isin(["Scale", "Test", "Constrain"]))
    ].copy()

    ranked = viable.sort_values(
        ["campaign_score", "expected_campaign_profit"],
        ascending=[False, False],
    ).copy()

    selected_indices = []
    family_counts = {}

    # First pass: max 2 per family, only viable campaigns.
    for idx, row in ranked.iterrows():
        family = row["campaign_family"]
        if family_counts.get(family, 0) >= 2:
            continue

        selected_indices.append(idx)
        family_counts[family] = family_counts.get(family, 0) + 1

        if len(selected_indices) == top_n:
            break

    # Second pass: if we still need more, use remaining viable campaigns.
    if len(selected_indices) < top_n:
        for idx, _ in ranked.iterrows():
            if idx not in selected_indices:
                selected_indices.append(idx)

            if len(selected_indices) == top_n:
                break

    # Final fallback: append the rest by model score, but they will rank after the viable dashboard set.
    selected_campaign_ids = set(ranked.loc[selected_indices, "campaign_id"]) if selected_indices else set()

    dashboard_selected = ranked.loc[selected_indices].copy() if selected_indices else ranked.head(0).copy()
    remaining = score_order[~score_order["campaign_id"].isin(selected_campaign_ids)].copy()

    final_ranked = pd.concat([dashboard_selected, remaining], ignore_index=True)
    final_ranked["dashboard_recommendation_rank"] = range(1, len(final_ranked) + 1)

    return final_ranked


def main() -> None:
    if not CUSTOMER_PATH.exists():
        raise FileNotFoundError(f"Missing customer features: {CUSTOMER_PATH}")

    if not CAMPAIGN_PATH.exists():
        raise FileNotFoundError(f"Missing campaign library: {CAMPAIGN_PATH}")

    customers = pd.read_csv(CUSTOMER_PATH)
    campaigns = pd.read_csv(CAMPAIGN_PATH)

    if PROFILE_PATH.exists():
        profiles = pd.read_csv(PROFILE_PATH)
        profile_cols = [
            "customer_id",
            "digital_engagement_score",
            "autopay_enrolled",
            "paperless_enrolled",
            "preferred_channel",
        ]
        customers = customers.merge(profiles[profile_cols], on="customer_id", how="left")

    recommendations = [score_campaign(customers, campaign) for _, campaign in campaigns.iterrows()]
    recommendations_df = pd.DataFrame(recommendations)

    recommendations_df = assign_diversified_dashboard_rank(recommendations_df, top_n=10)

    ordered_cols = [
        "dashboard_recommendation_rank",
        "model_score_rank",
        "campaign_id",
        "campaign_name",
        "campaign_family",
        "campaign_type",
        "risk_level",
        "recommended_rollout_decision",
        "campaign_score",
        "eligible_customers",
        "scale_customers",
        "test_customers",
        "blocked_customers",
        "eligible_rate",
        "scale_rate",
        "test_rate",
        "block_rate",
        "avg_predicted_response_rate",
        "expected_incremental_spend",
        "expected_campaign_cost",
        "expected_campaign_profit",
        "expected_campaign_roi",
        "campaign_horizon_months",
        "cost_per_customer",
        "expected_lift_pct",
        "response_rate_assumption",
        "business_goal",
        "offer_description",
        "target_segments",
        "excluded_segments",
        "primary_success_metric",
        "guardrail_notes",
        "risk_sensitivity",
        "customer_count",
        "target_segment_customers",
        "do_not_launch_customers",
    ]

    recommendations_df = recommendations_df[ordered_cols]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommendations_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Created: {OUTPUT_PATH}")
    print("Shape:", recommendations_df.shape)

    print("\nDashboard Top 10 recommended campaigns:")
    print(
        recommendations_df[
            [
                "dashboard_recommendation_rank",
                "model_score_rank",
                "campaign_id",
                "campaign_name",
                "campaign_family",
                "risk_level",
                "recommended_rollout_decision",
                "campaign_score",
                "eligible_customers",
                "scale_customers",
                "test_customers",
                "blocked_customers",
                "expected_campaign_profit",
                "expected_campaign_roi",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nModel-score top 12 campaigns:")
    print(
        recommendations_df.sort_values("model_score_rank")[
            [
                "model_score_rank",
                "campaign_id",
                "campaign_name",
                "campaign_family",
                "risk_level",
                "recommended_rollout_decision",
                "campaign_score",
                "expected_campaign_profit",
                "expected_campaign_roi",
            ]
        ].head(12).to_string(index=False)
    )

    print("\nRollout decision distribution:")
    print(recommendations_df["recommended_rollout_decision"].value_counts().to_string())

    print("\nCampaign family average score:")
    print(
        recommendations_df.groupby("campaign_family")["campaign_score"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
