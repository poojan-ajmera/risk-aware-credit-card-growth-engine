-- Offer eligibility placeholder

CREATE OR REPLACE TABLE offer_eligibility AS
SELECT
    customer_id,
    segment,
    risk_level,
    risk_adjusted_profit,
    CASE
        WHEN risk_level = 'High' THEN 'Block Growth Offer'
        WHEN segment = 'Underused Low-Risk' THEN 'Cashback Accelerator'
        WHEN segment = 'Premium Growth Candidate' THEN 'Premium Upgrade'
        WHEN segment = 'Dormant Recoverable' THEN 'Reactivation Bonus'
        WHEN segment = 'High-Utilization Revolver' THEN 'Payment Health Nudge'
        ELSE 'Standard Retention Offer'
    END AS recommended_action
FROM scored_customers;
