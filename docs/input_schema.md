# Input Schema for Reusable Decision Engine

This project is designed as a reusable prototype for credit card portfolio decisioning. A bank or credit card team can adapt the engine by mapping their customer portfolio fields to the required input schema below.

The uploaded file should be a CSV where each row represents one existing credit card customer.

## Required Columns

| Column | Type | Description |
|---|---|---|
| customer_id | string or integer | Unique customer identifier |
| age | numeric | Customer age |
| income | numeric | Estimated or reported annual income |
| credit_score | numeric | Customer credit score |
| credit_limit | numeric | Current credit limit on the card |
| current_balance | numeric | Current outstanding balance |
| monthly_spend | numeric | Recent average monthly card spend |
| transactions_count | numeric | Recent average monthly transaction count |
| customer_tenure_months | numeric | Number of months the customer has had the card |
| late_payments_12m | numeric | Number of late payments in the last 12 months |
| revolving_balance | numeric | Balance amount carried month to month |

## What the Engine Generates

After the required fields are provided, the engine creates additional decisioning fields:

| Output Field | Meaning |
|---|---|
| utilization_rate | Current balance divided by credit limit |
| revolver_flag | Whether the customer is carrying revolving balance |
| inactive_flag | Whether the customer has very low recent spend |
| payment_ratio | Estimated payment behavior proxy |
| default_probability | Estimated probability of default |
| risk_band | Low, Moderate, High, or Very High Risk |
| interchange_revenue | Estimated monthly interchange revenue |
| interest_revenue | Estimated monthly interest revenue |
| rewards_cost | Estimated monthly rewards cost |
| expected_credit_loss | Estimated monthly expected credit loss |
| risk_adjusted_profit | Monthly revenue after rewards cost and expected credit loss |
| expected_spend_lift | Estimated spend lift from an offer |
| expected_incremental_profit | Estimated campaign profit over the campaign horizon |
| expected_roi | Expected return on campaign marketing cost |
| customer_segment | Business segment assigned by the engine |
| recommended_action | Next-best-action recommendation |
| offer_type | Example offer/treatment type |
| decision_status | Scale, Test, Do Not Launch, or Block |
| campaign_eligible_flag | Whether the customer is eligible for Scale/Test campaign action |
| risk_guardrail_flag | Whether a risk guardrail blocks the customer |

## Notes

This is a portfolio analytics prototype, not a production credit decisioning system. It uses simplified assumptions for risk, profitability, and campaign economics.

The model is designed for existing credit card customers, not new credit applicants.

A real implementation would require institution-specific calibration, governance review, compliance review, model validation, and monitoring.
