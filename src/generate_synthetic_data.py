from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_CUSTOMERS = 10000

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "synthetic_case_data"
OUTPUT_PATH = OUTPUT_DIR / "synthetic_credit_card_customers.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def assign_risk_band(default_probability):
    if default_probability < 0.030:
        return "Low Risk"
    elif default_probability < 0.080:
        return "Moderate Risk"
    elif default_probability < 0.160:
        return "High Risk"
    return "Very High Risk"


def assign_customer_segment(row):
    if row["inactive_flag"] == 1 and row["risk_band"] in ["Low Risk", "Moderate Risk"]:
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


def assign_recommended_action(row):
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


def assign_offer_type(row):
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


def assign_decision_status(row):
    # Hard guardrail: customers flagged for no growth or very high credit risk should not receive growth offers.
    if row["recommended_action"] == "No Growth Offer":
        return "Block"

    if row["risk_band"] == "Very High Risk":
        return "Block"

    # Responsible-lending guardrail:
    # High-utilization revolvers may generate revenue, but scaling offers to them can increase credit exposure.
    # Even when ROI looks good, they should be tested carefully rather than scaled broadly.
    if row["customer_segment"] == "High-Utilization Revolver":
        if row["expected_roi"] >= 0 and row["risk_band"] in ["Low Risk", "Moderate Risk", "High Risk"]:
            return "Test"
        return "Do Not Launch"

    # Scale only strong low/moderate-risk customers with clearly positive economics.
    if row["expected_roi"] >= 5.00 and row["risk_band"] in ["Low Risk", "Moderate Risk"]:
        return "Scale"

    # Test borderline or promising customers before broad rollout.
    if row["expected_roi"] >= 0 and row["risk_band"] in ["Low Risk", "Moderate Risk", "High Risk"]:
        return "Test"

    return "Do Not Launch"


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    customer_id = np.arange(100000, 100000 + N_CUSTOMERS)

    age = rng.integers(21, 70, N_CUSTOMERS)

    income = rng.lognormal(mean=11.10, sigma=0.42, size=N_CUSTOMERS)
    income = np.clip(income, 22000, 250000).round(2)

    employment_status = rng.choice(
        ["Full-Time", "Part-Time", "Self-Employed", "Student", "Retired"],
        size=N_CUSTOMERS,
        p=[0.66, 0.10, 0.10, 0.06, 0.08],
    )

    state = rng.choice(
        ["NY", "CA", "TX", "FL", "VA", "NJ", "MA", "IL", "GA", "NC"],
        size=N_CUSTOMERS,
        p=[0.13, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06],
    )

    customer_tenure_months = rng.integers(3, 121, N_CUSTOMERS)

    # Credit score distribution is centered around an existing-cardholder portfolio.
    # This keeps the synthetic data more realistic for a credit card strategy case.
    credit_score_base = (
        675
        + (income - income.mean()) / income.std() * 35
        + rng.normal(0, 45, N_CUSTOMERS)
    )
    credit_score = np.clip(credit_score_base, 500, 850).round().astype(int)

    card_type = []
    for score, inc in zip(credit_score, income):
        if score >= 740 and inc >= 95000:
            card_type.append(rng.choice(["Venture", "Savor", "Quicksilver"], p=[0.45, 0.30, 0.25]))
        elif score >= 670:
            card_type.append(rng.choice(["Quicksilver", "Savor", "Platinum"], p=[0.45, 0.25, 0.30]))
        else:
            card_type.append(rng.choice(["Platinum", "Secured"], p=[0.75, 0.25]))
    card_type = np.array(card_type)

    rewards_preference = rng.choice(
        ["Cashback", "Travel", "Dining", "Grocery", "Low APR"],
        size=N_CUSTOMERS,
        p=[0.38, 0.20, 0.17, 0.15, 0.10],
    )

    credit_limit = (
        2500
        + (credit_score - 500) * 65
        + income * 0.08
        + rng.normal(0, 2500, N_CUSTOMERS)
    )
    credit_limit = np.clip(credit_limit, 1000, 50000).round(2)

    utilization_raw = (
        0.34
        - (credit_score - 680) / 900
        - (income - 75000) / 900000
        + rng.normal(0, 0.14, N_CUSTOMERS)
    )
    utilization_rate = np.clip(utilization_raw, 0.01, 0.95).round(4)

    current_balance = (credit_limit * utilization_rate).round(2)

    revolver_probability = np.clip(0.25 + utilization_rate * 0.65 - (credit_score - 680) / 1000, 0.05, 0.90)
    revolver_flag = rng.binomial(1, revolver_probability)
    revolving_balance = (current_balance * rng.uniform(0.45, 0.95, N_CUSTOMERS) * revolver_flag).round(2)

    monthly_spend = (
        income * rng.uniform(0.018, 0.055, N_CUSTOMERS)
        + credit_limit * rng.uniform(0.015, 0.050, N_CUSTOMERS)
        + rng.normal(0, 500, N_CUSTOMERS)
    )
    monthly_spend = np.clip(monthly_spend, 50, 12000).round(2)

    inactive_probability = np.clip(0.08 + (monthly_spend < 700) * 0.20 + (customer_tenure_months < 8) * 0.05, 0.02, 0.45)
    inactive_flag = rng.binomial(1, inactive_probability)

    monthly_spend = np.where(inactive_flag == 1, monthly_spend * rng.uniform(0.05, 0.35, N_CUSTOMERS), monthly_spend)
    monthly_spend = np.round(monthly_spend, 2)

    transactions_count = np.clip((monthly_spend / rng.uniform(45, 95, N_CUSTOMERS)).round(), 1, 180).astype(int)
    avg_transaction_amount = np.round(monthly_spend / transactions_count, 2)

    category_weights = rng.dirichlet(alpha=[2.2, 1.8, 1.1, 1.5, 2.0], size=N_CUSTOMERS)
    grocery_spend = np.round(monthly_spend * category_weights[:, 0], 2)
    dining_spend = np.round(monthly_spend * category_weights[:, 1], 2)
    travel_spend = np.round(monthly_spend * category_weights[:, 2], 2)
    gas_spend = np.round(monthly_spend * category_weights[:, 3], 2)
    online_spend = np.round(monthly_spend * category_weights[:, 4], 2)

    payment_ratio = np.clip(
        0.72
        + (credit_score - 680) / 500
        - utilization_rate * 0.45
        + rng.normal(0, 0.12, N_CUSTOMERS),
        0.05,
        1.00,
    ).round(4)

    late_lambda = np.clip(0.10 + utilization_rate * 1.35 - (credit_score - 650) / 250, 0.02, 3.00)
    late_payments_12m = rng.poisson(late_lambda)
    late_payments_12m = np.clip(late_payments_12m, 0, 8)

    risk_logit = (
        -5.9
        + utilization_rate * 3.1
        + late_payments_12m * 0.34
        - (credit_score - 680) / 115
        - (income - 75000) / 150000
        + revolver_flag * 0.35
    )
    default_probability = np.clip(sigmoid(risk_logit), 0.002, 0.45).round(4)

    delinquency_probability = np.clip(default_probability * 1.6 + late_payments_12m * 0.015, 0.005, 0.55)
    delinquency_flag = rng.binomial(1, delinquency_probability)

    risk_band = np.array([assign_risk_band(x) for x in default_probability])

    interchange_rate = 0.018
    apr = 0.245
    monthly_interest_rate = apr / 12
    rewards_rate = np.select(
        [
            card_type == "Venture",
            card_type == "Savor",
            card_type == "Quicksilver",
            card_type == "Platinum",
            card_type == "Secured",
        ],
        [0.022, 0.024, 0.017, 0.010, 0.006],
        default=0.015,
    )

    interchange_revenue = np.round(monthly_spend * interchange_rate, 2)
    interest_revenue = np.round(revolving_balance * monthly_interest_rate, 2)
    rewards_cost = np.round(monthly_spend * rewards_rate, 2)

    marketing_cost = np.round(
        np.select(
            [
                card_type == "Venture",
                card_type == "Savor",
                card_type == "Quicksilver",
                card_type == "Platinum",
                card_type == "Secured",
            ],
            [7, 6, 5, 4, 3],
            default=5,
        )
        + rng.normal(0, 1.2, N_CUSTOMERS),
        2,
    )
    marketing_cost = np.clip(marketing_cost, 1.5, 12)

    loss_given_default = 0.72
    # Monthly expected credit loss.
    # Default probability is treated as a forward-looking annual risk estimate,
    # so we divide by 12 to align it with monthly revenue and cost fields.
    expected_credit_loss = np.round(
        (default_probability * current_balance * loss_given_default) / 12,
        2,
    )

    # Monthly current customer profitability before any new campaign cost.
    # Marketing cost is handled separately in the campaign ROI model.
    risk_adjusted_profit = np.round(
        interchange_revenue + interest_revenue - rewards_cost - expected_credit_loss,
        2,
    )

    base_offer_lift = np.select(
        [
            rewards_preference == "Travel",
            rewards_preference == "Dining",
            rewards_preference == "Grocery",
            rewards_preference == "Cashback",
            rewards_preference == "Low APR",
        ],
        [0.14, 0.13, 0.12, 0.105, 0.075],
        default=0.09,
    )

    expected_spend_lift = np.clip(
        base_offer_lift
        + (inactive_flag == 1) * 0.025
        + (credit_score >= 720) * 0.015
        - (risk_band == "Very High Risk") * 0.04
        + rng.normal(0, 0.015, N_CUSTOMERS),
        -0.02,
        0.18,
    ).round(4)

    campaign_horizon_months = 12

    incremental_revenue = np.round(monthly_spend * expected_spend_lift * interchange_rate, 2)
    incremental_rewards_cost = np.round(monthly_spend * expected_spend_lift * rewards_rate, 2)
    # Monthly incremental expected credit loss from the extra spend created by the offer.
    # We do NOT multiply by campaign_horizon_months here because the final profit
    # formula already annualizes the monthly economics over the campaign horizon.
    incremental_expected_loss = np.round(
        default_probability
        * monthly_spend
        * expected_spend_lift
        * loss_given_default,
        2,
    )

    incremental_interest_revenue = np.round(
        revolving_balance * expected_spend_lift * monthly_interest_rate,
        2,
    )

    expected_incremental_profit = np.round(
        campaign_horizon_months
        * (
            incremental_revenue
            + incremental_interest_revenue
            - incremental_rewards_cost
            - incremental_expected_loss
        )
        - marketing_cost,
        2,
    )

    expected_roi = np.round(expected_incremental_profit / np.maximum(marketing_cost, 1), 4)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income,
            "employment_status": employment_status,
            "state": state,
            "customer_tenure_months": customer_tenure_months,
            "card_type": card_type,
            "rewards_preference": rewards_preference,
            "credit_score": credit_score,
            "credit_limit": credit_limit,
            "current_balance": current_balance,
            "utilization_rate": utilization_rate,
            "revolver_flag": revolver_flag,
            "revolving_balance": revolving_balance,
            "payment_ratio": payment_ratio,
            "late_payments_12m": late_payments_12m,
            "delinquency_flag": delinquency_flag,
            "default_probability": default_probability,
            "risk_band": risk_band,
            "monthly_spend": monthly_spend,
            "transactions_count": transactions_count,
            "avg_transaction_amount": avg_transaction_amount,
            "grocery_spend": grocery_spend,
            "dining_spend": dining_spend,
            "travel_spend": travel_spend,
            "gas_spend": gas_spend,
            "online_spend": online_spend,
            "inactive_flag": inactive_flag,
            "interchange_revenue": interchange_revenue,
            "interest_revenue": interest_revenue,
            "rewards_cost": rewards_cost,
            "marketing_cost": marketing_cost,
            "expected_credit_loss": expected_credit_loss,
            "risk_adjusted_profit": risk_adjusted_profit,
            "expected_spend_lift": expected_spend_lift,
            "expected_incremental_profit": expected_incremental_profit,
            "expected_roi": expected_roi,
        }
    )

    df["customer_segment"] = df.apply(assign_customer_segment, axis=1)
    df["recommended_action"] = df.apply(assign_recommended_action, axis=1)
    df["offer_type"] = df.apply(assign_offer_type, axis=1)
    df["decision_status"] = df.apply(assign_decision_status, axis=1)

    eligible_for_test = df["decision_status"].isin(["Scale", "Test"])
    df["ab_test_group"] = "Not Eligible"
    df.loc[eligible_for_test, "ab_test_group"] = rng.choice(
        ["Treatment", "Control"],
        size=eligible_for_test.sum(),
        p=[0.50, 0.50],
    )

    column_order = [
        "customer_id",
        "age",
        "income",
        "employment_status",
        "state",
        "customer_tenure_months",
        "card_type",
        "rewards_preference",
        "credit_score",
        "credit_limit",
        "current_balance",
        "utilization_rate",
        "revolver_flag",
        "revolving_balance",
        "payment_ratio",
        "late_payments_12m",
        "delinquency_flag",
        "default_probability",
        "risk_band",
        "monthly_spend",
        "transactions_count",
        "avg_transaction_amount",
        "grocery_spend",
        "dining_spend",
        "travel_spend",
        "gas_spend",
        "online_spend",
        "inactive_flag",
        "interchange_revenue",
        "interest_revenue",
        "rewards_cost",
        "marketing_cost",
        "expected_credit_loss",
        "risk_adjusted_profit",
        "expected_spend_lift",
        "expected_incremental_profit",
        "expected_roi",
        "customer_segment",
        "recommended_action",
        "offer_type",
        "decision_status",
        "ab_test_group",
    ]

    df = df[column_order]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Created {OUTPUT_PATH} with {len(df):,} rows and {len(df.columns)} columns")
    print("\nSegment distribution:")
    print(df["customer_segment"].value_counts())
    print("\nRisk band distribution:")
    print(df["risk_band"].value_counts())
    print("\nDecision status distribution:")
    print(df["decision_status"].value_counts())


if __name__ == "__main__":
    main()