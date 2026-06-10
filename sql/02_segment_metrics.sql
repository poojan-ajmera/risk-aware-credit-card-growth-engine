-- Segment-level metrics placeholder

CREATE OR REPLACE TABLE segment_metrics AS
SELECT
    segment,
    COUNT(*) AS customer_count,
    AVG(monthly_spend) AS avg_monthly_spend,
    AVG(utilization_rate) AS avg_utilization,
    AVG(default_probability) AS avg_default_probability,
    AVG(risk_adjusted_profit) AS avg_risk_adjusted_profit
FROM scored_customers
GROUP BY segment;
