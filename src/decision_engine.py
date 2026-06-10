from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "income",
    "credit_score",
    "credit_limit",
    "current_balance",
    "monthly_spend",
    "transactions_count",
    "customer_tenure_months",
    "late_payments_12m",
    "revolving_balance",
]


def validate_input_schema(df: pd.DataFrame) -> None:
    """
    Validate whether an uploaded customer portfolio has the minimum columns
    needed to run the decision engine.

    This keeps the app reusable. A bank does not need to use our exact synthetic
    dataset, but their file must include the core customer, credit, and spend fields.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "Uploaded file is missing required columns: "
            + ", ".join(missing_columns)
        )


def assign_risk_band(default_probability: float) -> str:
    if default_probability < 0.030:
        return "Low Risk"
    if default_probability < 0.080:
        return "Moderate Risk"
    if default_probability < 0.160:
        return "High Risk"
    return "Very High Risk"


def assign_customer_segment(row: pd.Series) -> str:
    if row.get("inactive_flag", 0) == 1 and row["risk_band"] in ["Low Risk", "Moderate Risk"]:
        return "Dormant but Recoverable"

    if (
        row["income"] >= 100000
        and row["credit_score"] >= 720
        and row["utilization_rate"] < 0.35
        and row["monthly_spend"] >= 2500
    ):
        return "Premium Growth Candidate"

    if (
        row["utilization_rate"] >= 0.60
        and row["risk_band"] in ["Moderate Risk", "High Risk", "Very High Risk"]
    ):
        return "High-Utilization Revolver"

    if row["risk_band"] in ["High Risk", "Very High Risk"]:
        return "Risk Watch"

    if (
        row["monthly_spend"] < 1200
        and row["utilization_rate"] < 0.30
        and row["risk_band"] == "Low Risk"
    ):
        return "Underused Low-Risk Customer"

    if (
        row["customer_tenure_months"] >= 48
        and row["monthly_spend"] >= 2200
        and row["risk_band"] in ["Low Risk", "Moderate Risk"]
    ):
        return "Loyal High-Value Customer"

    return "Core Customer"


def assign_recommended_action(row: pd.Series) -> str:
    segment = row["customer_segment"]

    if segment == "Premium Growth Candidate":
        return "Premium Upgrade Offer"
    if segment == "Underused Low-Risk Customer":
        return "Category Cashback Accelerator"
    if segment == "Dormant but Recoverable":
        return "Reactivation Bonus"
    if segment == "High-Utilization Revolver":
        return "Payment Health Messaging"
    if segment == "Risk Watch":
        return "No Growth Offer"
    if segment == "Loyal High-Value Customer":
        return "Retention Loyalty Benefit"

    return "Standard Engagement Offer"


def assign_offer_type(row: pd.Series) -> str:
    action = row["recommended_action"]

    mapping = {
        "Premium Upgrade Offer": "Travel Rewards Upgrade",
        "Category Cashback Accelerator": "3% Grocery/Dining Cashback",
        "Reactivation Bonus": "$50 Spend Reactivation Bonus",
        "Payment Health Messaging": "Financial Health Nudge",
        "No Growth Offer": "No Offer",
        "Retention Loyalty Benefit": "Statement Credit / Loyalty Perk",
        "Standard Engagement Offer": "Standard Cashback Reminder",
    }

    return mapping.get(action, "Standard Cashback Reminder")


def assign_decision_status(row: pd.Series) -> str:
    if row["recommended_action"] == "No Growth Offer":
        return "Block"

    if row["risk_band"] == "Very High Risk":
        return "Block"

    if row["customer_segment"] == "High-Utilization Revolver":
        if row["expected_roi"] >= 0 and row["risk_band"] in ["Low Risk", "Moderate Risk", "High Risk"]:
            return "Test"
        return "Do Not Launch"

    if row["expected_roi"] >= 5.00 and row["risk_band"] in ["Low Risk", "Moderate Risk"]:
        return "Scale"

    if row["expected_roi"] >= 0 and row["risk_band"] in ["Low Risk", "Moderate Risk", "High Risk"]:
        return "Test"

    return "Do Not Launch"


def score_customer_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a customer portfolio and return recommended actions.

    This function is the core reusable engine that Dash and Gradio will use later.
    """
    validate_input_schema(df)

    scored = df.copy()

    scored["utilization_rate"] = (
        scored["current_balance"] / scored["credit_limit"].replace(0, np.nan)
    ).clip(0, 0.95).fillna(0)

    scored["revolver_flag"] = np.where(scored["revolving_balance"] > 0, 1, 0)

    if "inactive_flag" not in scored.columns:
        scored["inactive_flag"] = np.where(scored["monthly_spend"] < 100, 1, 0)

    if "payment_ratio" not in scored.columns:
        scored["payment_ratio"] = np.clip(
            0.72
            + (scored["credit_score"] - 680) / 500
            - scored["utilization_rate"] * 0.45,
            0.05,
            1.00,
        )

    risk_logit = (
        -5.9
        + scored["utilization_rate"] * 3.1
        + scored["late_payments_12m"] * 0.34
        - (scored["credit_score"] - 680) / 115
        - (scored["income"] - 75000) / 150000
        + scored["revolver_flag"] * 0.35
    )

    scored["default_probability"] = np.clip(
        1 / (1 + np.exp(-risk_logit)),
        0.002,
        0.45,
    )

    scored["risk_band"] = scored["default_probability"].apply(assign_risk_band)

    interchange_rate = 0.018
    apr = 0.245
    monthly_interest_rate = apr / 12
    loss_given_default = 0.72

    scored["interchange_revenue"] = scored["monthly_spend"] * interchange_rate
    scored["interest_revenue"] = scored["revolving_balance"] * monthly_interest_rate

    if "rewards_cost" not in scored.columns:
        scored["rewards_cost"] = scored["monthly_spend"] * 0.015

    scored["expected_credit_loss"] = (
        scored["default_probability"]
        * scored["current_balance"]
        * loss_given_default
        / 12
    )

    scored["risk_adjusted_profit"] = (
        scored["interchange_revenue"]
        + scored["interest_revenue"]
        - scored["rewards_cost"]
        - scored["expected_credit_loss"]
    )

    if "expected_spend_lift" not in scored.columns:
        scored["expected_spend_lift"] = np.clip(
            0.08
            + (scored["credit_score"] >= 720) * 0.015
            + scored["inactive_flag"] * 0.025
            - (scored["risk_band"] == "Very High Risk") * 0.04,
            -0.02,
            0.18,
        )

    if "marketing_cost" not in scored.columns:
        scored["marketing_cost"] = 5.00

    campaign_horizon_months = 12

    incremental_revenue = scored["monthly_spend"] * scored["expected_spend_lift"] * interchange_rate
    incremental_interest_revenue = scored["revolving_balance"] * scored["expected_spend_lift"] * monthly_interest_rate
    incremental_rewards_cost = scored["monthly_spend"] * scored["expected_spend_lift"] * 0.015
    incremental_expected_loss = (
        scored["default_probability"]
        * scored["monthly_spend"]
        * scored["expected_spend_lift"]
        * loss_given_default
    )

    scored["expected_incremental_profit"] = (
        campaign_horizon_months
        * (
            incremental_revenue
            + incremental_interest_revenue
            - incremental_rewards_cost
            - incremental_expected_loss
        )
        - scored["marketing_cost"]
    )

    scored["expected_roi"] = scored["expected_incremental_profit"] / scored["marketing_cost"].replace(0, np.nan)
    scored["expected_roi"] = scored["expected_roi"].fillna(0)

    scored["customer_segment"] = scored.apply(assign_customer_segment, axis=1)
    scored["recommended_action"] = scored.apply(assign_recommended_action, axis=1)
    scored["offer_type"] = scored.apply(assign_offer_type, axis=1)
    scored["decision_status"] = scored.apply(assign_decision_status, axis=1)

    scored["campaign_eligible_flag"] = np.where(
        scored["decision_status"].isin(["Scale", "Test"]),
        1,
        0,
    )

    scored["risk_guardrail_flag"] = np.where(
        scored["decision_status"] == "Block",
        1,
        0,
    )

    return scored