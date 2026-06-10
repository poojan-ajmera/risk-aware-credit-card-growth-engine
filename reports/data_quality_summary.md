# Synthetic Dataset Quality Check

Rows: 10,000

Columns: 42


## Customer Segment Distribution

| customer_segment            |   count |
|:----------------------------|--------:|
| Core Customer               |    4213 |
| Loyal High-Value Customer   |    3399 |
| Dormant but Recoverable     |     804 |
| Premium Growth Candidate    |     784 |
| High-Utilization Revolver   |     531 |
| Risk Watch                  |     185 |
| Underused Low-Risk Customer |      84 |


## Risk Band Distribution

| risk_band      |   count |
|:---------------|--------:|
| Low Risk       |    7804 |
| Moderate Risk  |    1754 |
| High Risk      |     362 |
| Very High Risk |      80 |


## Decision Status Distribution

| decision_status   |   count |
|:------------------|--------:|
| Do Not Launch     |    4505 |
| Scale             |    4085 |
| Test              |    1161 |
| Block             |     249 |


## Segment x Decision Status

| customer_segment            |   Block |   Do Not Launch |   Scale |   Test |
|:----------------------------|--------:|----------------:|--------:|-------:|
| Core Customer               |       0 |            1828 |    1944 |    441 |
| Dormant but Recoverable     |       0 |             389 |     368 |     47 |
| High-Utilization Revolver   |      64 |             213 |       0 |    254 |
| Loyal High-Value Customer   |       0 |            1441 |    1573 |    385 |
| Premium Growth Candidate    |       0 |             589 |     174 |     21 |
| Risk Watch                  |     185 |               0 |       0 |      0 |
| Underused Low-Risk Customer |       0 |              45 |      26 |     13 |


## High-Utilization Revolver Decision Check

| decision_status   |   count |
|:------------------|--------:|
| Test              |     254 |
| Do Not Launch     |     213 |
| Block             |      64 |


## Key Metrics

|       |   income |   credit_score |   credit_limit |   current_balance |   utilization_rate |   monthly_spend |   default_probability |   expected_credit_loss |   risk_adjusted_profit |   expected_roi |
|:------|---------:|---------------:|---------------:|------------------:|-------------------:|----------------:|----------------------:|-----------------------:|-----------------------:|---------------:|
| count |  10000   |      10000     |        10000   |          10000    |          10000     |        10000    |             10000     |              10000     |              10000     |      10000     |
| mean  |  72251.8 |        673.89  |        19546.9 |           6376.61 |              0.351 |         3060.68 |                 0.022 |                  9.641 |                 48.964 |          7.595 |
| std   |  31932.4 |         56.322 |         6116.7 |           3001.34 |              0.163 |         1764.04 |                 0.03  |                 14.786 |                 58.849 |         25.486 |
| min   |  22000   |        500     |         1000   |            100.24 |              0.01  |            5.74 |                 0.002 |                  0.02  |               -152.88  |       -254.091 |
| 25%   |  49949   |        635     |        15370.7 |           4354.17 |              0.241 |         1878.1  |                 0.006 |                  1.78  |                  2.64  |         -5.193 |
| 50%   |  65609.7 |        671     |        18988.4 |           6225.36 |              0.353 |         2809.24 |                 0.012 |                  4.785 |                 31.435 |          1.274 |
| 75%   |  87570.6 |        710     |        23085.3 |           8233.18 |              0.463 |         3955.91 |                 0.027 |                 11.492 |                 90.94  |         18.449 |
| max   | 250000   |        850     |        50000   |          21443    |              0.903 |        12000    |                 0.437 |                279.62  |                319.02  |        207.8   |


## Final Dataset Note

This synthetic dataset is designed for a credit card growth strategy case. It does not use Capital One internal data or real customer data. The dataset is structured to support segmentation, risk-adjusted profitability, next-best-action recommendations, responsible-lending guardrails, and A/B test planning.