from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("data/campaigns/campaign_library.csv")


SEGMENTS = {
    "core": "Core Customer",
    "loyal": "Loyal High-Value Customer",
    "premium": "Premium Growth Candidate",
    "high_util": "High-Utilization Revolver",
    "underused": "Underused Low-Risk Customer",
    "dormant": "Dormant but Recoverable",
    "risk": "Risk Watch",
}


CAMPAIGN_FAMILIES = [
    {
        "family": "Everyday Spend",
        "campaign_type": "Category Cashback",
        "base_goal": "Increase everyday card spend in routine categories.",
        "risk_level": "Low",
        "risk_sensitivity": "Low",
        "target_segments": [SEGMENTS["core"], SEGMENTS["underused"], SEGMENTS["loyal"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 5.00,
        "base_lift": 0.06,
        "base_response": 0.09,
        "success_metric": "Incremental category spend",
        "guardrail_notes": "Exclude very high risk customers and customers with recent delinquency.",
        "offers": [
            ("Grocery Cashback Booster", "Earn bonus cashback on grocery purchases for 30 days.", "Grocery"),
            ("Gas Spend Booster", "Earn bonus cashback on gas and fuel purchases for 30 days.", "Gas"),
            ("Pharmacy Spend Booster", "Earn bonus cashback on pharmacy purchases for 30 days.", "Pharmacy"),
            ("Online Shopping Booster", "Earn bonus cashback on online shopping purchases for 30 days.", "Online"),
            ("Utility Bill Pay Booster", "Earn bonus rewards for paying utilities with the card.", "Utilities"),
            ("Everyday Essentials Booster", "Earn bonus rewards on daily essentials purchases.", "Essentials"),
            ("Weekend Spend Booster", "Earn bonus rewards on weekend purchases.", "General Spend"),
            ("Local Merchant Booster", "Earn cashback at selected local merchants.", "Local Merchants"),
        ],
    },
    {
        "family": "Dining and Entertainment",
        "campaign_type": "Lifestyle Rewards",
        "base_goal": "Increase dining, entertainment, and lifestyle spend.",
        "risk_level": "Low-Medium",
        "risk_sensitivity": "Medium",
        "target_segments": [SEGMENTS["core"], SEGMENTS["loyal"], SEGMENTS["premium"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 8.00,
        "base_lift": 0.08,
        "base_response": 0.07,
        "success_metric": "Incremental lifestyle spend",
        "guardrail_notes": "Prioritize positive ROI customers with acceptable utilization.",
        "offers": [
            ("Dining Rewards Accelerator", "Earn bonus rewards after meeting a dining spend threshold.", "Dining"),
            ("Coffee Shop Rewards", "Earn cashback at coffee shops and cafes.", "Dining"),
            ("Food Delivery Booster", "Earn rewards on food delivery purchases.", "Dining"),
            ("Movie and Events Offer", "Earn rewards on movies, events, and entertainment.", "Entertainment"),
            ("Weekend Dining Challenge", "Spend a set amount on dining over weekends to earn a bonus.", "Dining"),
            ("Restaurant Partner Offer", "Earn merchant-funded rewards at selected restaurants.", "Dining"),
            ("Family Dining Bonus", "Earn rewards on family dining and casual restaurants.", "Dining"),
            ("Entertainment Night Out", "Earn rewards on entertainment and dining in the same period.", "Entertainment"),
        ],
    },
    {
        "family": "Travel and Premium",
        "campaign_type": "Travel Rewards",
        "base_goal": "Increase travel spend and deepen premium customer engagement.",
        "risk_level": "Medium",
        "risk_sensitivity": "Medium",
        "target_segments": [SEGMENTS["loyal"], SEGMENTS["premium"]],
        "excluded_segments": [SEGMENTS["risk"], SEGMENTS["dormant"]],
        "base_cost": 12.00,
        "base_lift": 0.10,
        "base_response": 0.05,
        "success_metric": "Incremental travel spend",
        "guardrail_notes": "Avoid high default probability and high-utilization customers unless risk-adjusted profit is strong.",
        "offers": [
            ("Travel Spend Accelerator", "Earn bonus points on flights, hotels, and travel bookings.", "Travel"),
            ("Hotel Rewards Booster", "Earn bonus rewards on hotel purchases.", "Travel"),
            ("Airfare Bonus Offer", "Earn rewards on airline purchases.", "Travel"),
            ("Rental Car Rewards", "Earn rewards on rental car purchases.", "Travel"),
            ("Travel Portal Bonus", "Earn extra rewards when booking through a travel portal.", "Travel"),
            ("Premium Trip Planning Offer", "Earn travel credits after meeting a trip spend threshold.", "Travel"),
            ("International Travel Booster", "Earn rewards on international travel purchases.", "Travel"),
            ("Weekend Getaway Rewards", "Earn bonus rewards on short-stay travel purchases.", "Travel"),
        ],
    },
    {
        "family": "Merchant Offers",
        "campaign_type": "Merchant-Funded Offer",
        "base_goal": "Drive spend at selected merchants through targeted card-linked offers.",
        "risk_level": "Low",
        "risk_sensitivity": "Low",
        "target_segments": [SEGMENTS["core"], SEGMENTS["loyal"], SEGMENTS["underused"], SEGMENTS["premium"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 4.00,
        "base_lift": 0.05,
        "base_response": 0.10,
        "success_metric": "Offer activation and merchant spend",
        "guardrail_notes": "Low credit-risk impact, but still exclude very high risk customers.",
        "offers": [
            ("Retail Partner Cashback", "Earn cashback at selected retail partners.", "Retail"),
            ("Streaming Subscription Offer", "Earn statement credit on streaming subscriptions.", "Subscription"),
            ("Rideshare Partner Offer", "Earn cashback on rideshare purchases.", "Transportation"),
            ("Food Delivery Partner Offer", "Earn cashback with selected food delivery partners.", "Dining"),
            ("Fitness Partner Offer", "Earn rewards with selected fitness or wellness merchants.", "Wellness"),
            ("Home Improvement Offer", "Earn cashback at home improvement merchants.", "Home"),
            ("Electronics Partner Offer", "Earn cashback at selected electronics retailers.", "Retail"),
            ("Back-to-School Merchant Offer", "Earn rewards at school supply and retail partners.", "Retail"),
        ],
    },
    {
        "family": "Digital Engagement",
        "campaign_type": "Digital Engagement",
        "base_goal": "Increase mobile app usage, digital servicing, and offer activation.",
        "risk_level": "Low",
        "risk_sensitivity": "Low",
        "target_segments": [SEGMENTS["core"], SEGMENTS["underused"], SEGMENTS["dormant"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 3.00,
        "base_lift": 0.04,
        "base_response": 0.12,
        "success_metric": "Digital activation",
        "guardrail_notes": "Safe campaign type; avoid customers with major risk guardrail triggers.",
        "offers": [
            ("Mobile App Activation", "Earn a small reward after logging into the mobile app.", "Digital"),
            ("Card-Linked Offer Activation", "Earn rewards for activating a card-linked offer.", "Digital"),
            ("Digital Wallet Enrollment", "Earn a bonus for adding the card to a digital wallet.", "Digital"),
            ("Spend Alert Enrollment", "Earn a small reward for setting up spend alerts.", "Digital"),
            ("Credit Score Tool Engagement", "Encourage use of the credit score monitoring tool.", "Digital"),
            ("Mobile Payment Challenge", "Use mobile wallet purchases to earn bonus rewards.", "Digital"),
            ("Online Account Setup", "Encourage digital account setup and profile completion.", "Digital"),
            ("Security Feature Enrollment", "Encourage customers to activate card lock or security alerts.", "Digital"),
        ],
    },
    {
        "family": "Dormant Reactivation",
        "campaign_type": "Reactivation",
        "base_goal": "Reactivate customers with low recent card usage.",
        "risk_level": "Low-Medium",
        "risk_sensitivity": "Medium",
        "target_segments": [SEGMENTS["dormant"], SEGMENTS["underused"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 7.00,
        "base_lift": 0.07,
        "base_response": 0.06,
        "success_metric": "Reactivation rate",
        "guardrail_notes": "Do not push additional spend to dormant customers with elevated risk or negative risk-adjusted profit.",
        "offers": [
            ("Three Purchase Reactivation", "Use the card three times in 30 days to earn bonus rewards.", "General Spend"),
            ("First Purchase Back Offer", "Earn cashback after the first purchase in the campaign period.", "General Spend"),
            ("Dormant Grocery Restart", "Earn grocery rewards after returning to card usage.", "Grocery"),
            ("Dormant Digital Restart", "Reactivate through app login and card use.", "Digital"),
            ("Dormant Small Spend Challenge", "Make a small number of purchases to earn a reward.", "General Spend"),
            ("Dormant Merchant Comeback", "Earn rewards at selected merchants after inactivity.", "Merchant"),
            ("Dormant Bill Pay Restart", "Use the card for bill pay to earn bonus rewards.", "Utilities"),
            ("Dormant Everyday Spend Restart", "Earn bonus rewards after returning to everyday spend.", "General Spend"),
        ],
    },
    {
        "family": "Balance and Revolver",
        "campaign_type": "Balance Transfer / Revolver",
        "base_goal": "Grow balance-related revenue while controlling repayment risk.",
        "risk_level": "High",
        "risk_sensitivity": "High",
        "target_segments": [SEGMENTS["core"], SEGMENTS["loyal"], SEGMENTS["high_util"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 14.00,
        "base_lift": 0.12,
        "base_response": 0.04,
        "success_metric": "Risk-adjusted balance growth",
        "guardrail_notes": "Strict default probability, late payment, utilization, and profitability guardrails required.",
        "offers": [
            ("Balance Transfer Offer", "Promotional balance transfer offer for selected customers.", "Balance"),
            ("Low APR Purchase Promo", "Promotional APR on purchases for a defined period.", "Purchase APR"),
            ("Planned Purchase Financing", "Promotional financing for planned larger purchases.", "Purchase APR"),
            ("Revolver Value Offer", "Targeted offer for selected revolvers with acceptable risk.", "Balance"),
            ("APR Education and Planning", "Provide APR and repayment education before promotional use.", "Education"),
            ("Installment Plan Invitation", "Invite selected customers to use installment-style repayment options.", "Installment"),
            ("Balance Consolidation Review", "Review selected customers for balance consolidation offers.", "Balance"),
            ("Controlled Revolver Test", "Small controlled test for selected revolvers with strict guardrails.", "Balance"),
        ],
    },
    {
        "family": "Credit Line Strategy",
        "campaign_type": "Credit Line Review",
        "base_goal": "Increase card usage capacity for strong customers while managing risk exposure.",
        "risk_level": "High",
        "risk_sensitivity": "Very High",
        "target_segments": [SEGMENTS["loyal"], SEGMENTS["premium"], SEGMENTS["underused"]],
        "excluded_segments": [SEGMENTS["risk"], SEGMENTS["high_util"]],
        "base_cost": 9.00,
        "base_lift": 0.11,
        "base_response": 0.04,
        "success_metric": "Spend lift without risk deterioration",
        "guardrail_notes": "Require strong credit profile, stable utilization, low default probability, and no recent late payments.",
        "offers": [
            ("Credit Line Increase Review", "Invite eligible customers to be reviewed for a potential credit line increase.", "Credit Line"),
            ("Low Utilization Capacity Review", "Review low-utilization strong customers for capacity growth.", "Credit Line"),
            ("Premium Capacity Review", "Review premium customers for additional card capacity.", "Credit Line"),
            ("High-Score Line Review", "Review strong credit score customers for line growth.", "Credit Line"),
            ("Income-Supported Line Review", "Review customers with strong income and repayment profile.", "Credit Line"),
            ("Low-Risk Spend Capacity Offer", "Invite low-risk high-spend customers for capacity review.", "Credit Line"),
            ("Selective Line Growth Test", "Small test of line increase review among safest customers.", "Credit Line"),
            ("Relationship-Based Line Review", "Review long-tenure profitable customers for capacity growth.", "Credit Line"),
        ],
    },
    {
        "family": "Balance Health",
        "campaign_type": "Protective Engagement",
        "base_goal": "Reduce credit stress while maintaining customer engagement.",
        "risk_level": "Protective",
        "risk_sensitivity": "Protective",
        "target_segments": [SEGMENTS["high_util"], SEGMENTS["core"]],
        "excluded_segments": [],
        "base_cost": 2.50,
        "base_lift": 0.02,
        "base_response": 0.09,
        "success_metric": "Utilization reduction and repayment stability",
        "guardrail_notes": "This is not a spend-growth campaign. It is a protective engagement campaign.",
        "offers": [
            ("High-Utilization Balance Health", "Send balance health messaging and payment reminders.", "Risk Prevention"),
            ("Autopay Reminder Campaign", "Encourage autopay setup for repayment stability.", "Servicing"),
            ("Payment Due Reminder Test", "Test payment reminder messaging for higher utilization customers.", "Servicing"),
            ("Budgeting Support Message", "Send budgeting and balance management education.", "Education"),
            ("Utilization Awareness Message", "Explain utilization impact and balance management.", "Education"),
            ("Minimum Payment Education", "Educate customers on repayment behavior and interest impact.", "Education"),
            ("High Balance Digital Nudges", "Send digital nudges for balance and payment management.", "Digital"),
            ("Payment Flexibility Awareness", "Inform selected customers about payment planning tools.", "Servicing"),
        ],
    },
    {
        "family": "Retention and Loyalty",
        "campaign_type": "Retention",
        "base_goal": "Retain and deepen relationships with profitable high-value customers.",
        "risk_level": "Low-Medium",
        "risk_sensitivity": "Medium",
        "target_segments": [SEGMENTS["loyal"], SEGMENTS["premium"]],
        "excluded_segments": [SEGMENTS["risk"]],
        "base_cost": 15.00,
        "base_lift": 0.08,
        "base_response": 0.06,
        "success_metric": "Retention and spend stability",
        "guardrail_notes": "Prioritize positive risk-adjusted profit and strong engagement.",
        "offers": [
            ("Premium Retention Campaign", "Provide bonus rewards or statement credit for selected high-value customers.", "Retention"),
            ("Anniversary Loyalty Bonus", "Offer bonus rewards around relationship anniversary.", "Retention"),
            ("High-Value Spend Bonus", "Offer bonus rewards after reaching a spend threshold.", "Retention"),
            ("Premium Experience Benefit", "Offer experience benefits to premium customers.", "Retention"),
            ("Relationship Deepening Offer", "Encourage multi-category card engagement.", "Retention"),
            ("High-Value Merchant Perk", "Offer selected high-value merchant benefits.", "Retention"),
            ("Card Upgrade Interest Test", "Test interest in premium card upgrade pathways.", "Retention"),
            ("Loyalty Statement Credit", "Offer statement credit to selected profitable customers.", "Retention"),
        ],
    },
    {
        "family": "Servicing and Enrollment",
        "campaign_type": "Servicing Engagement",
        "base_goal": "Improve repayment stability, digital servicing, and lower friction.",
        "risk_level": "Low",
        "risk_sensitivity": "Low",
        "target_segments": [SEGMENTS["core"], SEGMENTS["high_util"], SEGMENTS["dormant"], SEGMENTS["underused"]],
        "excluded_segments": [],
        "base_cost": 2.00,
        "base_lift": 0.03,
        "base_response": 0.11,
        "success_metric": "Enrollment and servicing adoption",
        "guardrail_notes": "Useful for risk prevention and servicing. Not primarily a spend-growth campaign.",
        "offers": [
            ("Autopay and Paperless Enrollment", "Offer a small reward for enrolling in autopay or paperless statements.", "Servicing"),
            ("Paperless Statement Campaign", "Encourage paperless statement enrollment.", "Servicing"),
            ("Autopay Setup Campaign", "Encourage autopay enrollment.", "Servicing"),
            ("Payment Alert Setup", "Encourage payment reminders and alerts.", "Servicing"),
            ("Fraud Alert Enrollment", "Encourage fraud alert and account security enrollment.", "Servicing"),
            ("Profile Completion Campaign", "Encourage customers to complete account profile details.", "Servicing"),
            ("Communication Preference Update", "Encourage customers to update preferred communication channels.", "Servicing"),
            ("Account Health Check", "Encourage customers to review account health and settings.", "Servicing"),
        ],
    },
    {
        "family": "New-to-Credit and Student",
        "campaign_type": "Early Relationship",
        "base_goal": "Build healthy early card behavior and engagement.",
        "risk_level": "Medium",
        "risk_sensitivity": "Medium",
        "target_segments": [SEGMENTS["underused"], SEGMENTS["core"]],
        "excluded_segments": [SEGMENTS["risk"], SEGMENTS["high_util"]],
        "base_cost": 4.50,
        "base_lift": 0.04,
        "base_response": 0.08,
        "success_metric": "Healthy engagement and on-time payment behavior",
        "guardrail_notes": "Focus on education, low-risk engagement, and repayment health.",
        "offers": [
            ("First Year Engagement Offer", "Encourage healthy spend habits in the first year.", "Education"),
            ("Student Essentials Cashback", "Reward small everyday purchases for student-like profiles.", "Everyday Spend"),
            ("Credit Education Journey", "Encourage education around credit use and payments.", "Education"),
            ("Small Spend Starter Challenge", "Encourage low-risk card usage through small purchase goals.", "Everyday Spend"),
            ("On-Time Payment Reward", "Reward consistent on-time payment behavior.", "Servicing"),
            ("Digital Wallet Starter", "Encourage safe digital wallet usage.", "Digital"),
            ("First Autopay Setup", "Encourage early autopay enrollment.", "Servicing"),
            ("Responsible Credit Builder", "Encourage low utilization and on-time payments.", "Education"),
        ],
    },
]


def rollout_from_risk(risk_level: str) -> str:
    if risk_level in ["Low", "Protective"]:
        return "Scale for eligible customers; Test messaging variants."
    if risk_level in ["Low-Medium", "Medium"]:
        return "Scale for low-risk customers; Test moderate-risk or uncertain groups."
    if risk_level in ["Medium-High", "High"]:
        return "Test first with strict guardrails; Scale only after validation."
    return "Test only unless risk performance is validated."


def max_default_probability(risk_sensitivity: str) -> float:
    mapping = {
        "Low": 0.10,
        "Medium": 0.07,
        "High": 0.045,
        "Very High": 0.03,
        "Protective": 0.16,
    }
    return mapping.get(risk_sensitivity, 0.07)


def max_utilization(risk_sensitivity: str) -> float:
    mapping = {
        "Low": 0.90,
        "Medium": 0.78,
        "High": 0.65,
        "Very High": 0.45,
        "Protective": 1.00,
    }
    return mapping.get(risk_sensitivity, 0.78)


def min_credit_score(risk_sensitivity: str) -> int:
    mapping = {
        "Low": 600,
        "Medium": 630,
        "High": 670,
        "Very High": 700,
        "Protective": 560,
    }
    return mapping.get(risk_sensitivity, 630)


def main() -> None:
    campaigns = []
    campaign_number = 1

    for family in CAMPAIGN_FAMILIES:
        for offer_index, (campaign_name, offer_description, spend_category) in enumerate(family["offers"], start=1):
            cost_adjustment = 1 + ((offer_index - 4.5) * 0.025)
            lift_adjustment = 1 + ((offer_index % 4) * 0.04)
            response_adjustment = 1 + (((offer_index + 1) % 3) * 0.035)

            campaign = {
                "campaign_id": f"CMP{campaign_number:03d}",
                "campaign_name": campaign_name,
                "campaign_family": family["family"],
                "campaign_type": family["campaign_type"],
                "business_goal": family["base_goal"],
                "offer_description": offer_description,
                "spend_category_focus": spend_category,
                "target_segments": "; ".join(family["target_segments"]),
                "excluded_segments": "; ".join(family["excluded_segments"]),
                "risk_level": family["risk_level"],
                "risk_sensitivity": family["risk_sensitivity"],
                "cost_per_customer": round(family["base_cost"] * cost_adjustment, 2),
                "expected_lift_pct": round(family["base_lift"] * lift_adjustment, 4),
                "response_rate_assumption": round(family["base_response"] * response_adjustment, 4),
                "primary_success_metric": family["success_metric"],
                "guardrail_notes": family["guardrail_notes"],
                "recommended_rollout": rollout_from_risk(family["risk_level"]),
                "max_default_probability": max_default_probability(family["risk_sensitivity"]),
                "max_utilization": max_utilization(family["risk_sensitivity"]),
                "min_credit_score": min_credit_score(family["risk_sensitivity"]),
                "allow_high_utilization": "Yes" if family["risk_sensitivity"] == "Protective" else "No",
                "client_editable": "Yes",
                "active_flag": "Yes",
            }

            campaigns.append(campaign)
            campaign_number += 1

    campaigns_df = pd.DataFrame(campaigns)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    campaigns_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Created: {OUTPUT_PATH}")
    print("Shape:", campaigns_df.shape)
    print("\nCampaign families:")
    print(campaigns_df["campaign_family"].value_counts().sort_index().to_string())
    print("\nPreview:")
    print(
        campaigns_df[
            [
                "campaign_id",
                "campaign_name",
                "campaign_family",
                "risk_level",
                "cost_per_customer",
                "expected_lift_pct",
                "recommended_rollout",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
