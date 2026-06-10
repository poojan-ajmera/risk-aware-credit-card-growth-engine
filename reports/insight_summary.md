# Insight Summary: Risk-Aware Credit Card Growth Decision Engine

## 1. Executive Snapshot

The synthetic portfolio contains **10,000 existing credit card customers**. The average credit score is **674**, average utilization is **35.1%**, and average default probability is **2.2%**.

The portfolio generates about **$30,606,820 in monthly spend** and approximately **$489,641 in monthly risk-adjusted profit**, or about **$49 per customer**.

About **52.5%** of customers are eligible for either a Scale or Test action, while **2.5%** are blocked by risk guardrails.

## 2. Portfolio Decision Mix

| decision_status   |   customer_count |   share_of_portfolio |
|:------------------|-----------------:|---------------------:|
| Do Not Launch     |             4505 |               0.4505 |
| Scale             |             4085 |               0.4085 |
| Test              |             1161 |               0.1161 |
| Block             |              249 |               0.0249 |


## 3. Segment-Level Findings

The highest-volume segments are Core Customer and Loyal High-Value Customer. These groups drive much of the scale opportunity because they combine large population size, positive profitability, and manageable risk.


| customer_segment            |   customer_count |   avg_credit_score |   avg_utilization_rate |   avg_default_probability |   avg_risk_adjusted_profit |   avg_expected_roi |   scale_count |   test_count |   do_not_launch_count |   block_count |   campaign_eligible_rate |
|:----------------------------|-----------------:|-------------------:|-----------------------:|--------------------------:|---------------------------:|-------------------:|--------------:|-------------:|----------------------:|--------------:|-------------------------:|
| Core Customer               |             4213 |             664.31 |                 0.3566 |                    0.0198 |                      50.46 |             8.9879 |          1944 |          441 |                  1828 |             0 |                   0.5661 |
| Loyal High-Value Customer   |             3399 |             677.35 |                 0.3347 |                    0.0151 |                      55.63 |             9.2334 |          1573 |          385 |                  1441 |             0 |                   0.5761 |
| Dormant but Recoverable     |              804 |             675.25 |                 0.3456 |                    0.0188 |                      39.31 |            16.3819 |           368 |           47 |                   389 |             0 |                   0.5162 |
| Premium Growth Candidate    |              784 |             769.67 |                 0.1459 |                    0.0023 |                       3.65 |            -1.6411 |           174 |           21 |                   589 |             0 |                   0.2487 |
| High-Utilization Revolver   |              531 |             619.54 |                 0.6764 |                    0.0941 |                      79.98 |            -2.9214 |             0 |          254 |                   213 |            64 |                   0.4783 |
| Risk Watch                  |              185 |             579.7  |                 0.5208 |                    0.113  |                      50.41 |           -22.0911 |             0 |            0 |                     0 |           185 |                   0      |
| Underused Low-Risk Customer |               84 |             658.45 |                 0.2237 |                    0.0121 |                      20.24 |             5.371  |            26 |           13 |                    45 |             0 |                   0.4643 |


## 4. Priority Segments

Based on total risk-adjusted profit and campaign eligibility, the strongest segments to prioritize are:


| customer_segment          |   customer_count |   total_risk_adjusted_profit |   avg_expected_roi |   scale_count |   test_count |   campaign_eligible_rate |
|:--------------------------|-----------------:|-----------------------------:|-------------------:|--------------:|-------------:|-------------------------:|
| Core Customer             |             4213 |                       212585 |             8.9879 |          1944 |          441 |                   0.5661 |
| Loyal High-Value Customer |             3399 |                       189096 |             9.2334 |          1573 |          385 |                   0.5761 |
| High-Utilization Revolver |              531 |                        42470 |            -2.9214 |             0 |          254 |                   0.4783 |


Business interpretation: Core Customer and Loyal High-Value Customer are the strongest broad-scale growth priorities because they combine large customer counts, positive profitability, and manageable risk. High-Utilization Revolver is financially meaningful, but it should be treated as a controlled test and guardrail segment rather than a broad growth campaign.


## 5. Scale Opportunities

Customers marked as Scale have stronger economics and acceptable risk. These customers are the best candidates for broad rollout after standard business review.


| customer_segment            |   scale_customers |   avg_roi |   avg_profit |   avg_default_probability |
|:----------------------------|------------------:|----------:|-------------:|--------------------------:|
| Core Customer               |              1944 |   26.5148 |      97.5127 |                0.0215335  |
| Loyal High-Value Customer   |              1573 |   27.1178 |     104.415  |                0.0155189  |
| Dormant but Recoverable     |               368 |   38.5103 |      89.7592 |                0.0258543  |
| Premium Growth Candidate    |               174 |   19.3521 |      76.7199 |                0.00290057 |
| Underused Low-Risk Customer |                26 |   19.7312 |      55.7612 |                0.0161538  |


## 6. Test Opportunities

Customers marked as Test are not weak customers. They are customers where the strategy is promising, but the business should validate lift and risk behavior before scaling.


| customer_segment            |   test_customers |   avg_roi |   avg_profit |   avg_default_probability |
|:----------------------------|-----------------:|----------:|-------------:|--------------------------:|
| Core Customer               |              441 |  2.21806  |     31.2541  |                0.0143358  |
| Loyal High-Value Customer   |              385 |  2.60141  |     37.1469  |                0.0124984  |
| High-Utilization Revolver   |              254 | 31.1644   |    129.975   |                0.0697957  |
| Dormant but Recoverable     |               47 |  1.19315  |      7.29766 |                0.00509787 |
| Premium Growth Candidate    |               21 |  3.00476  |     28.1671  |                0.00250952 |
| Underused Low-Risk Customer |               13 |  0.916185 |      8.76692 |                0.00988462 |


## 7. Offer and Action Insights

The decision engine does not recommend the same offer to everyone. The action depends on customer segment, risk, profitability, and expected ROI.


| recommended_action            | offer_type                      |   customer_count |   avg_expected_roi |   avg_risk_adjusted_profit |   avg_default_probability |
|:------------------------------|:--------------------------------|-----------------:|-------------------:|---------------------------:|--------------------------:|
| Standard Engagement Offer     | Standard Cashback Reminder      |             4213 |            6.905   |                    44.6033 |                0.0183667  |
| Retention Loyalty Benefit     | Statement Credit / Loyalty Perk |             3399 |            7.0673  |                    49.6267 |                0.0144333  |
| Reactivation Bonus            | $50 Spend Reactivation Bonus    |              804 |           12.3289  |                    30.8367 |                0.0149667  |
| Premium Upgrade Offer         | Travel Rewards Upgrade          |              784 |            4.78283 |                    28.6933 |                0.00253333 |
| Payment Health Messaging      | Financial Health Nudge          |              531 |          -17.1109  |                    62.4133 |                0.125      |
| No Growth Offer               | No Offer                        |              185 |          -22.0911  |                    50.41   |                0.113      |
| Category Cashback Accelerator | 3% Grocery/Dining Cashback      |               84 |            6.3361  |                    22.5167 |                0.0121667  |


## 8. Responsible-Lending Guardrail Check

The most important guardrail is that High-Utilization Revolver customers are not scaled aggressively. These customers may generate interest revenue, but pushing additional spend can increase credit exposure.


High-Utilization Revolver decision mix:

| decision_status   |   count |
|:------------------|--------:|
| Test              |     254 |
| Do Not Launch     |     213 |
| Block             |      64 |


Risk Watch decision mix:

| decision_status   |   count |
|:------------------|--------:|
| Block             |     185 |


Business interpretation: the model separates revenue potential from responsible growth. This is important because a profitable customer is not automatically a safe customer to target.


## 9. Recommended Strategy

**Recommendation:** Scale growth campaigns for Core Customer and Loyal High-Value Customer segments, run controlled A/B tests for High-Utilization Revolvers and other borderline groups, and block growth campaigns for Risk Watch customers.


The strongest business move is not to maximize campaign reach. The stronger move is to maximize risk-adjusted growth by scaling safe profitable customers, testing uncertain groups, and protecting customers where credit risk is elevated.


## 10. How This Feeds the App

The Dash app should show the portfolio overview, segment strategy, offer decision logic, scenario simulator, and A/B test planner. The Gradio demo should focus on the upload-and-run decision engine, where a user can upload a customer file and receive recommended actions.