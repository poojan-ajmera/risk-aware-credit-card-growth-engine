from pathlib import Path
import duckdb


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "synthetic_case_data" / "synthetic_credit_card_customers.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CUSTOMER_FEATURES_PATH = PROCESSED_DIR / "customer_features.csv"
SEGMENT_SUMMARY_PATH = PROCESSED_DIR / "segment_summary.csv"
OFFER_ELIGIBILITY_PATH = PROCESSED_DIR / "offer_eligibility_summary.csv"
PORTFOLIO_KPIS_PATH = PROCESSED_DIR / "portfolio_kpis.csv"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute(
        f"""
        CREATE OR REPLACE VIEW customers AS
        SELECT *
        FROM read_csv_auto('{RAW_DATA_PATH}')
        """
    )

    customer_features = con.execute(
        """
        SELECT
            customer_id,
            age,
            income,
            employment_status,
            state,
            customer_tenure_months,
            card_type,
            rewards_preference,

            credit_score,
            credit_limit,
            current_balance,
            utilization_rate,
            revolver_flag,
            revolving_balance,
            payment_ratio,
            late_payments_12m,
            delinquency_flag,
            default_probability,
            risk_band,

            monthly_spend,
            transactions_count,
            avg_transaction_amount,
            grocery_spend,
            dining_spend,
            travel_spend,
            gas_spend,
            online_spend,
            inactive_flag,

            interchange_revenue,
            interest_revenue,
            rewards_cost,
            expected_credit_loss,
            risk_adjusted_profit,
            expected_spend_lift,
            expected_incremental_profit,
            expected_roi,

            customer_segment,
            recommended_action,
            offer_type,
            decision_status,
            ab_test_group,

            ROUND(monthly_spend / NULLIF(income, 0), 4) AS spend_to_income_ratio,
            ROUND(current_balance / NULLIF(income, 0), 4) AS balance_to_income_ratio,
            ROUND(risk_adjusted_profit / NULLIF(monthly_spend, 0), 4) AS risk_adjusted_margin,

            CASE
                WHEN utilization_rate < 0.30 THEN 'Low Utilization'
                WHEN utilization_rate < 0.60 THEN 'Moderate Utilization'
                ELSE 'High Utilization'
            END AS utilization_band,

            CASE
                WHEN risk_adjusted_profit >= 100 THEN 'High Profit'
                WHEN risk_adjusted_profit >= 25 THEN 'Moderate Profit'
                WHEN risk_adjusted_profit >= 0 THEN 'Low Profit'
                ELSE 'Negative Profit'
            END AS profitability_band,

            CASE
                WHEN expected_roi >= 20 THEN 'High ROI'
                WHEN expected_roi >= 5 THEN 'Moderate ROI'
                WHEN expected_roi >= 0 THEN 'Low ROI'
                ELSE 'Negative ROI'
            END AS roi_band,

            CASE
                WHEN decision_status IN ('Scale', 'Test') THEN 1
                ELSE 0
            END AS campaign_eligible_flag,

            CASE
                WHEN decision_status = 'Block' THEN 1
                ELSE 0
            END AS risk_guardrail_flag

        FROM customers
        """
    ).df()

    customer_features.to_csv(CUSTOMER_FEATURES_PATH, index=False)

    segment_summary = con.execute(
        """
        SELECT
            customer_segment,
            COUNT(*) AS customer_count,

            ROUND(AVG(income), 2) AS avg_income,
            ROUND(AVG(credit_score), 2) AS avg_credit_score,
            ROUND(AVG(credit_limit), 2) AS avg_credit_limit,
            ROUND(AVG(current_balance), 2) AS avg_current_balance,
            ROUND(AVG(utilization_rate), 4) AS avg_utilization_rate,
            ROUND(AVG(default_probability), 4) AS avg_default_probability,

            ROUND(AVG(monthly_spend), 2) AS avg_monthly_spend,
            ROUND(SUM(monthly_spend), 2) AS total_monthly_spend,

            ROUND(AVG(risk_adjusted_profit), 2) AS avg_risk_adjusted_profit,
            ROUND(SUM(risk_adjusted_profit), 2) AS total_risk_adjusted_profit,

            ROUND(AVG(expected_roi), 4) AS avg_expected_roi,

            SUM(CASE WHEN decision_status = 'Scale' THEN 1 ELSE 0 END) AS scale_count,
            SUM(CASE WHEN decision_status = 'Test' THEN 1 ELSE 0 END) AS test_count,
            SUM(CASE WHEN decision_status = 'Do Not Launch' THEN 1 ELSE 0 END) AS do_not_launch_count,
            SUM(CASE WHEN decision_status = 'Block' THEN 1 ELSE 0 END) AS block_count,

            ROUND(
                SUM(CASE WHEN decision_status IN ('Scale', 'Test') THEN 1 ELSE 0 END) * 1.0 / COUNT(*),
                4
            ) AS campaign_eligible_rate

        FROM customer_features
        GROUP BY customer_segment
        ORDER BY customer_count DESC
        """
    ).df()

    segment_summary.to_csv(SEGMENT_SUMMARY_PATH, index=False)

    offer_eligibility = con.execute(
        """
        SELECT
            recommended_action,
            offer_type,
            decision_status,
            COUNT(*) AS customer_count,

            ROUND(AVG(monthly_spend), 2) AS avg_monthly_spend,
            ROUND(AVG(utilization_rate), 4) AS avg_utilization_rate,
            ROUND(AVG(default_probability), 4) AS avg_default_probability,
            ROUND(AVG(risk_adjusted_profit), 2) AS avg_risk_adjusted_profit,
            ROUND(AVG(expected_incremental_profit), 2) AS avg_expected_incremental_profit,
            ROUND(AVG(expected_roi), 4) AS avg_expected_roi

        FROM customer_features
        GROUP BY
            recommended_action,
            offer_type,
            decision_status
        ORDER BY
            recommended_action,
            customer_count DESC
        """
    ).df()

    offer_eligibility.to_csv(OFFER_ELIGIBILITY_PATH, index=False)

    portfolio_kpis = con.execute(
        """
        SELECT
            COUNT(*) AS total_customers,

            ROUND(AVG(credit_score), 2) AS avg_credit_score,
            ROUND(AVG(utilization_rate), 4) AS avg_utilization_rate,
            ROUND(AVG(default_probability), 4) AS avg_default_probability,

            ROUND(SUM(monthly_spend), 2) AS total_monthly_spend,
            ROUND(AVG(monthly_spend), 2) AS avg_monthly_spend,

            ROUND(SUM(risk_adjusted_profit), 2) AS total_monthly_risk_adjusted_profit,
            ROUND(AVG(risk_adjusted_profit), 2) AS avg_risk_adjusted_profit,

            ROUND(AVG(expected_roi), 4) AS avg_expected_roi,

            SUM(CASE WHEN decision_status = 'Scale' THEN 1 ELSE 0 END) AS scale_customers,
            SUM(CASE WHEN decision_status = 'Test' THEN 1 ELSE 0 END) AS test_customers,
            SUM(CASE WHEN decision_status = 'Do Not Launch' THEN 1 ELSE 0 END) AS do_not_launch_customers,
            SUM(CASE WHEN decision_status = 'Block' THEN 1 ELSE 0 END) AS block_customers,

            ROUND(
                SUM(CASE WHEN decision_status IN ('Scale', 'Test') THEN 1 ELSE 0 END) * 1.0 / COUNT(*),
                4
            ) AS campaign_eligible_rate,

            ROUND(
                SUM(CASE WHEN decision_status = 'Block' THEN 1 ELSE 0 END) * 1.0 / COUNT(*),
                4
            ) AS block_rate

        FROM customer_features
        """
    ).df()

    portfolio_kpis.to_csv(PORTFOLIO_KPIS_PATH, index=False)

    print(f"Created {CUSTOMER_FEATURES_PATH}")
    print(f"Created {SEGMENT_SUMMARY_PATH}")
    print(f"Created {OFFER_ELIGIBILITY_PATH}")
    print(f"Created {PORTFOLIO_KPIS_PATH}")

    print("\nPortfolio KPIs:")
    print(portfolio_kpis.T)

    print("\nSegment Summary Preview:")
    print(segment_summary.head())


if __name__ == "__main__":
    main()