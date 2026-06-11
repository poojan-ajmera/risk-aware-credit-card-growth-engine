from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42

INPUT_PATH = Path("data/synthetic_case_data/synthetic_credit_card_customers.csv")
OUTPUT_PATH = Path("data/synthetic_case_data/synthetic_customer_profiles.csv")


FIRST_NAMES = [
    "Aarav", "Maya", "Rohan", "Anika", "Ethan", "Sophia", "Noah", "Emma",
    "Liam", "Olivia", "Arjun", "Isha", "Kabir", "Meera", "Lucas", "Ava",
    "Mason", "Mia", "Nikhil", "Priya", "Daniel", "Grace", "Jay", "Nina",
    "Ryan", "Zoe", "Aditya", "Sana", "Leo", "Ella",
]

LAST_NAMES = [
    "Shah", "Patel", "Mehta", "Kapoor", "Singh", "Brown", "Johnson", "Williams",
    "Garcia", "Martinez", "Davis", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Lee", "Walker", "Young", "King", "Scott", "Green", "Baker",
    "Carter", "Morris", "Reddy", "Desai", "Jain", "Iyer", "Khan",
]

STATE_CITY_MAP = {
    "CA": [("Los Angeles", "90001"), ("San Diego", "92101"), ("San Jose", "95112"), ("Sacramento", "95814")],
    "NY": [("New York", "10001"), ("Buffalo", "14201"), ("Albany", "12207"), ("Rochester", "14604")],
    "TX": [("Austin", "78701"), ("Dallas", "75201"), ("Houston", "77002"), ("San Antonio", "78205")],
    "FL": [("Miami", "33101"), ("Orlando", "32801"), ("Tampa", "33602"), ("Jacksonville", "32202")],
    "IL": [("Chicago", "60601"), ("Aurora", "60505"), ("Naperville", "60540"), ("Springfield", "62701")],
    "MA": [("Boston", "02108"), ("Cambridge", "02139"), ("Somerville", "02143"), ("Worcester", "01608")],
    "WA": [("Seattle", "98101"), ("Bellevue", "98004"), ("Tacoma", "98402"), ("Spokane", "99201")],
    "GA": [("Atlanta", "30303"), ("Savannah", "31401"), ("Augusta", "30901"), ("Athens", "30601")],
    "NC": [("Charlotte", "28202"), ("Raleigh", "27601"), ("Durham", "27701"), ("Cary", "27511")],
    "NJ": [("Jersey City", "07302"), ("Newark", "07102"), ("Edison", "08817"), ("Hoboken", "07030")],
}

SIGNUP_CHANNELS = [
    "Mobile App",
    "Website",
    "Branch Referral",
    "Direct Mail",
    "Partner Offer",
    "Pre-Approved Campaign",
]

OCCUPATION_GROUPS = [
    "Technology",
    "Healthcare",
    "Education",
    "Finance",
    "Retail",
    "Hospitality",
    "Operations",
    "Public Sector",
    "Student",
    "Self-Employed",
]

PREFERRED_CHANNELS = [
    "Mobile App",
    "Email",
    "SMS",
    "Web Portal",
    "Branch / Phone",
]


def assign_relationship_tier(row) -> str:
    if row["customer_tenure_months"] >= 48 and row["monthly_spend"] >= 4500 and row["credit_score"] >= 700:
        return "Premium"
    if row["customer_tenure_months"] >= 24 and row["monthly_spend"] >= 3000 and row["credit_score"] >= 660:
        return "Preferred"
    if row["customer_tenure_months"] >= 12:
        return "Standard"
    return "Starter"


def assign_preferred_channel(row, rng: np.random.Generator) -> str:
    age = row["age"]

    if age <= 35:
        return rng.choice(["Mobile App", "Email", "SMS"], p=[0.60, 0.25, 0.15])

    if age <= 55:
        return rng.choice(["Email", "Mobile App", "Web Portal", "SMS"], p=[0.38, 0.32, 0.20, 0.10])

    return rng.choice(["Email", "Web Portal", "Branch / Phone", "Mobile App"], p=[0.38, 0.25, 0.22, 0.15])


def assign_occupation_group(row, rng: np.random.Generator) -> str:
    employment_status = row["employment_status"]
    age = row["age"]

    if employment_status == "Student" or age < 24:
        return "Student"

    if employment_status == "Self-employed":
        return "Self-Employed"

    if employment_status in ["Retired", "Unemployed"]:
        return employment_status

    return rng.choice(
        [
            "Technology",
            "Healthcare",
            "Education",
            "Finance",
            "Retail",
            "Hospitality",
            "Operations",
            "Public Sector",
        ],
        p=[0.18, 0.16, 0.12, 0.14, 0.12, 0.08, 0.12, 0.08],
    )


def assign_digital_score(row, preferred_channel: str, rng: np.random.Generator) -> int:
    base = 55

    if preferred_channel == "Mobile App":
        base += 25
    elif preferred_channel == "Email":
        base += 12
    elif preferred_channel == "Web Portal":
        base += 8
    elif preferred_channel == "Branch / Phone":
        base -= 8

    if row["transactions_count"] >= 80:
        base += 8

    if row["customer_tenure_months"] >= 24:
        base += 5

    score = base + rng.normal(0, 8)
    return int(np.clip(score, 5, 100))


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    customers = pd.read_csv(INPUT_PATH)

    profiles = customers[
        [
            "customer_id",
            "age",
            "employment_status",
            "state",
            "customer_tenure_months",
            "monthly_spend",
            "transactions_count",
            "credit_score",
            "card_type",
            "rewards_preference",
        ]
    ].copy()

    profiles["first_name"] = rng.choice(FIRST_NAMES, size=len(profiles))
    profiles["last_name"] = rng.choice(LAST_NAMES, size=len(profiles))
    profiles["customer_name"] = profiles["first_name"] + " " + profiles["last_name"]

    city_zip_pairs = profiles["state"].apply(
        lambda state: rng.choice(STATE_CITY_MAP.get(state, [("Metro Area", "00000")]))
    )
    profiles["city"] = [pair[0] for pair in city_zip_pairs]
    profiles["zip_code"] = [pair[1] for pair in city_zip_pairs]

    profiles["occupation_group"] = profiles.apply(
        lambda row: assign_occupation_group(row, rng), axis=1
    )

    profiles["preferred_channel"] = profiles.apply(
        lambda row: assign_preferred_channel(row, rng), axis=1
    )

    profiles["relationship_tier"] = profiles.apply(assign_relationship_tier, axis=1)

    profiles["signup_channel"] = rng.choice(
        SIGNUP_CHANNELS,
        size=len(profiles),
        p=[0.28, 0.25, 0.10, 0.12, 0.10, 0.15],
    )

    profiles["customer_email"] = profiles.apply(
        lambda row: (
            f"{str(row['first_name']).lower()}.{str(row['last_name']).lower()}."
            f"{int(row['customer_id'])}@syntheticmail.com"
        ),
        axis=1,
    )

    profiles["phone_number"] = profiles["customer_id"].apply(
        lambda x: f"(555) {str(int(x))[-3:]}-{str(int(x) * 7)[-4:]}"
    )

    profiles["account_open_date"] = pd.to_datetime("2025-12-31") - pd.to_timedelta(
        profiles["customer_tenure_months"] * 30,
        unit="D",
    )

    profiles["account_open_date"] = profiles["account_open_date"].dt.strftime("%Y-%m-%d")

    profiles["digital_engagement_score"] = profiles.apply(
        lambda row: assign_digital_score(row, row["preferred_channel"], rng),
        axis=1,
    )

    profiles["last_app_login_days"] = profiles["digital_engagement_score"].apply(
        lambda score: int(np.clip(rng.normal(45 - score * 0.35, 12), 0, 180))
    )

    profiles["autopay_enrolled"] = rng.choice(
        ["Yes", "No"],
        size=len(profiles),
        p=[0.62, 0.38],
    )

    profiles["paperless_enrolled"] = rng.choice(
        ["Yes", "No"],
        size=len(profiles),
        p=[0.74, 0.26],
    )

    profiles = profiles[
        [
            "customer_id",
            "customer_name",
            "customer_email",
            "phone_number",
            "city",
            "state",
            "zip_code",
            "employment_status",
            "occupation_group",
            "preferred_channel",
            "relationship_tier",
            "signup_channel",
            "account_open_date",
            "digital_engagement_score",
            "last_app_login_days",
            "autopay_enrolled",
            "paperless_enrolled",
            "card_type",
            "rewards_preference",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(OUTPUT_PATH, index=False)

    print(f"Created: {OUTPUT_PATH}")
    print("Shape:", profiles.shape)
    print("\nPreview:")
    print(profiles.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
