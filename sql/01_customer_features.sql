-- Customer feature layer placeholder
-- Goal: create customer-level features for segmentation, risk scoring, and profitability.

CREATE OR REPLACE TABLE customer_features AS
SELECT
    customer_id,
    age,
    income,
    credit_score,
    credit_limit,
    balance,
    monthly_spend,
    transactions_count,
    months_on_book,
    delinquency_flag,
    balance / NULLIF(credit_limit, 0) AS utilization_rate
FROM customer_base;
