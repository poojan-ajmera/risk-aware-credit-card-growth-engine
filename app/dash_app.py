from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table


BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
SEGMENT_PATH = BASE_DIR / "data" / "processed" / "segment_summary.csv"
PORTFOLIO_KPI_PATH = BASE_DIR / "data" / "processed" / "portfolio_kpis.csv"
CAMPAIGN_LIBRARY_PATH = BASE_DIR / "data" / "campaigns" / "campaign_library.csv"
CAMPAIGN_RECOMMENDATIONS_PATH = BASE_DIR / "data" / "campaigns" / "campaign_recommendations.csv"


COLORS = {
    "background": "#f5f7fb",
    "card": "#ffffff",
    "text": "#111827",
    "muted": "#6b7280",
    "border": "#e5e7eb",
    "navy": "#111827",
    "blue": "#2563eb",
    "light_blue": "#eff6ff",
    "soft_blue": "#eff6ff",
    "orange_light": "#fff7ed",
    "orange_border": "#fed7aa",
}


DECISION_COLOR_MAP = {
    "Scale": "#16a34a",
    "Test": "#2563eb",
    "Do Not Launch": "#9ca3af",
    "Block": "#dc2626",
}


RISK_COLOR_MAP = {
    "Low Risk": "#16a34a",
    "Moderate Risk": "#f59e0b",
    "High Risk": "#f97316",
    "Very High Risk": "#dc2626",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customer_features = pd.read_csv(FEATURES_PATH)

    customer_profile_path = Path("data/synthetic_case_data/synthetic_customer_profiles.csv")

    if customer_profile_path.exists():
        customer_profiles = pd.read_csv(customer_profile_path)

        profile_columns = [
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

        overlapping_profile_columns = [
            col for col in profile_columns
            if col in customer_features.columns and col != "customer_id"
        ]

        customer_features = customer_features.drop(
            columns=overlapping_profile_columns,
            errors="ignore",
        )

        customer_features = customer_features.merge(
            customer_profiles[profile_columns],
            on="customer_id",
            how="left",
        )
    else:
        customer_features["customer_name"] = customer_features["customer_id"].astype(str)
        customer_features["customer_email"] = "unknown@syntheticmail.com"
        customer_features["phone_number"] = "Unknown"
        customer_features["city"] = "Unknown"
        customer_features["zip_code"] = "Unknown"
        customer_features["occupation_group"] = "Unknown"
        customer_features["preferred_channel"] = "Unknown"
        customer_features["relationship_tier"] = "Unknown"
        customer_features["signup_channel"] = "Unknown"
        customer_features["account_open_date"] = "Unknown"
        customer_features["digital_engagement_score"] = 0
        customer_features["last_app_login_days"] = 0
        customer_features["autopay_enrolled"] = "Unknown"
        customer_features["paperless_enrolled"] = "Unknown"

    segment_summary = pd.read_csv(SEGMENT_PATH)
    portfolio_kpis = pd.read_csv(PORTFOLIO_KPI_PATH)

    return customer_features, segment_summary, portfolio_kpis


def load_campaign_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load campaign-level files only.

    This keeps the Dash app lightweight. Large customer-level scoring should
    happen upstream in scripts or warehouse jobs, not inside the dashboard.
    """
    if CAMPAIGN_LIBRARY_PATH.exists():
        campaign_library = pd.read_csv(CAMPAIGN_LIBRARY_PATH)
    else:
        campaign_library = pd.DataFrame()

    if CAMPAIGN_RECOMMENDATIONS_PATH.exists():
        campaign_recommendations = pd.read_csv(CAMPAIGN_RECOMMENDATIONS_PATH)
    else:
        campaign_recommendations = pd.DataFrame()

    return campaign_library, campaign_recommendations


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def create_kpi_card(title: str, value: str, note: str, accent: str = "#2563eb") -> html.Div:
    return html.Div(
        children=[
            html.Div(
                style={
                    "height": "4px",
                    "width": "42px",
                    "borderRadius": "999px",
                    "backgroundColor": accent,
                    "marginBottom": "14px",
                }
            ),
            html.Div(title, style={"fontSize": "13px", "color": COLORS["muted"], "fontWeight": "700"}),
            html.Div(value, style={"fontSize": "30px", "fontWeight": "800", "marginTop": "8px", "color": COLORS["text"]}),
            html.Div(note, style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "8px", "lineHeight": "1.4"}),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "20px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "minHeight": "135px",
        },
    )


def create_chart_card(title: str, subtitle: str, figure, graph_id: str | None = None) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H3(title, style={"margin": "0", "fontSize": "20px", "fontWeight": "800", "color": COLORS["text"]}),
                    html.P(subtitle, style={"margin": "6px 0 0 0", "fontSize": "14px", "color": COLORS["muted"], "lineHeight": "1.4"}),
                ],
                style={"marginBottom": "10px"},
            ),
            dcc.Graph(id=graph_id, figure=figure, config={"displayModeBar": True}),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "20px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
        },
    )


def create_insight_card(title: str, text: str, variant: str = "default") -> html.Div:
    background = COLORS["card"]
    border = COLORS["border"]
    marker = COLORS["blue"]

    if variant == "warning":
        background = COLORS["orange_light"]
        border = COLORS["orange_border"]
        marker = "#f97316"

    return html.Div(
        children=[
            html.Div(
                style={
                    "height": "100%",
                    "width": "5px",
                    "borderRadius": "999px",
                    "backgroundColor": marker,
                    "marginRight": "16px",
                }
            ),
            html.Div(
                children=[
                    html.H3(title, style={"margin": "0 0 8px 0", "fontSize": "19px", "fontWeight": "800"}),
                    html.P(text, style={"margin": "0", "fontSize": "15px", "color": "#374151", "lineHeight": "1.6"}),
                ]
            ),
        ],
        style={
            "display": "flex",
            "backgroundColor": background,
            "border": f"1px solid {border}",
            "borderRadius": "18px",
            "padding": "22px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.05)",
            "marginTop": "22px",
        },
    )


def create_tab_intro(title: str, text: str) -> html.Div:
    return html.Div(
        children=[
            html.H2(
                title,
                style={
                    "margin": "0 0 8px 0",
                    "fontSize": "26px",
                    "fontWeight": "900",
                    "color": COLORS["text"],
                },
            ),
            html.P(
                text,
                style={
                    "margin": "0",
                    "fontSize": "15px",
                    "color": COLORS["muted"],
                    "lineHeight": "1.55",
                    "maxWidth": "980px",
                },
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "22px 24px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.05)",
            "marginTop": "22px",
            "marginBottom": "18px",
        },
    )


def create_placeholder_card(title: str, text: str) -> html.Div:
    return html.Div(
        children=[
            html.Div("Coming Next", style={"fontSize": "13px", "fontWeight": "800", "color": COLORS["blue"], "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H3(title, style={"fontSize": "28px", "margin": "10px 0 8px 0"}),
            html.P(text, style={"fontSize": "16px", "color": COLORS["muted"], "maxWidth": "720px", "lineHeight": "1.6"}),
        ],
        style={
            "minHeight": "320px",
            "backgroundColor": COLORS["card"],
            "border": f"1px dashed #bfdbfe",
            "borderRadius": "20px",
            "padding": "40px",
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "textAlign": "center",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.04)",
        },
    )


def create_table(rows) -> html.Table:
    """Render a simple table from either html.Tr rows or list-of-dict records."""
    if not rows:
        return html.Table(
            [
                html.Tr(
                    [
                        html.Td(
                            "No rows available.",
                            style={
                                "padding": "12px",
                                "color": COLORS["muted"],
                                "borderBottom": f"1px solid {COLORS['border']}",
                            },
                        )
                    ]
                )
            ],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "14px",
            },
        )

    # Newer sections pass rows as list[dict]. Convert them into html rows.
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        columns = list(rows[0].keys())

        header = html.Tr(
            [
                html.Th(
                    column,
                    style={
                        "textAlign": "left",
                        "padding": "10px 12px",
                        "fontSize": "12px",
                        "fontWeight": "900",
                        "color": COLORS["muted"],
                        "textTransform": "uppercase",
                        "borderBottom": f"1px solid {COLORS['border']}",
                        "whiteSpace": "nowrap",
                    },
                )
                for column in columns
            ]
        )

        body = [
            html.Tr(
                [
                    html.Td(
                        str(row.get(column, "")),
                        style={
                            "padding": "10px 12px",
                            "borderBottom": f"1px solid {COLORS['border']}",
                            "color": COLORS["text"],
                            "verticalAlign": "top",
                            "whiteSpace": "nowrap" if column not in ["Campaign"] else "normal",
                        },
                    )
                    for column in columns
                ]
            )
            for row in rows
        ]

        table_children = [header] + body
    else:
        # Older sections already pass html.Tr rows.
        table_children = rows

    return html.Table(
        table_children,
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "14px",
        },
    )




def create_campaign_table_rows(campaign_recommendations: pd.DataFrame, limit: int = 10) -> list[dict]:
    """Build a compact dashboard table from campaign-level recommendation output."""
    if campaign_recommendations.empty:
        return []

    required_columns = [
        "dashboard_recommendation_rank",
        "campaign_name",
        "campaign_family",
        "risk_level",
        "recommended_rollout_decision",
        "eligible_customers",
        "scale_customers",
        "blocked_customers",
        "expected_campaign_profit",
        "expected_campaign_roi",
        "campaign_score",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in campaign_recommendations.columns
    ]

    if missing_columns:
        raise ValueError(
            "Campaign recommendations missing required columns: "
            + ", ".join(missing_columns)
        )

    table = campaign_recommendations.head(limit).copy()

    table["Expected Profit"] = table["expected_campaign_profit"].apply(format_currency)
    table["Expected ROI"] = table["expected_campaign_roi"].apply(lambda value: f"{value:.2f}x")
    table["Eligible Customers"] = table["eligible_customers"].apply(lambda value: f"{int(value):,}")
    table["Scale Customers"] = table["scale_customers"].apply(lambda value: f"{int(value):,}")
    table["Blocked Customers"] = table["blocked_customers"].apply(lambda value: f"{int(value):,}")
    table["Score"] = table["campaign_score"].apply(lambda value: f"{value:.1f}")

    return table[
        [
            "dashboard_recommendation_rank",
            "campaign_name",
            "campaign_family",
            "risk_level",
            "recommended_rollout_decision",
            "Eligible Customers",
            "Scale Customers",
            "Blocked Customers",
            "Expected Profit",
            "Expected ROI",
            "Score",
        ]
    ].rename(
        columns={
            "dashboard_recommendation_rank": "Rank",
            "campaign_name": "Campaign",
            "campaign_family": "Family",
            "risk_level": "Risk",
            "recommended_rollout_decision": "Rollout",
        }
    ).to_dict("records")


customer_features, segment_summary, portfolio_kpis = load_data()
campaign_library, campaign_recommendations = load_campaign_data()
kpis = portfolio_kpis.iloc[0]


if campaign_recommendations.empty:
    campaign_top10 = pd.DataFrame()
    campaign_family_fig = create_empty_figure("Campaign recommendation data is not available.")
    campaign_rollout_fig = create_empty_figure("Campaign recommendation data is not available.")
    campaign_profit_fig = create_empty_figure("Campaign recommendation data is not available.")
    campaign_table_rows = []
    total_campaigns_available = 0
    top_campaign_profit = 0
    top_campaign_eligible = 0
    top_campaign_scale = 0
else:
    campaign_top10 = campaign_recommendations.head(10).copy()
    total_campaigns_available = len(campaign_recommendations)
    top_campaign_profit = campaign_top10["expected_campaign_profit"].sum()
    top_campaign_eligible = campaign_top10["eligible_customers"].sum()
    top_campaign_scale = campaign_top10["scale_customers"].sum()
    campaign_table_rows = create_campaign_table_rows(campaign_recommendations, limit=10)

    campaign_family_counts = (
        campaign_top10.groupby("campaign_family", as_index=False)
        .agg(campaign_count=("campaign_id", "count"))
        .sort_values("campaign_count", ascending=True)
    )

    campaign_family_fig = px.bar(
        campaign_family_counts,
        x="campaign_count",
        y="campaign_family",
        orientation="h",
        title="Top Campaign Families",
        labels={
            "campaign_count": "Campaigns",
            "campaign_family": "Campaign Family",
        },
    )
    campaign_family_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=340,
        margin=dict(l=20, r=20, t=55, b=30),
        font=dict(family="Arial", color=COLORS["text"]),
    )

    rollout_counts = (
        campaign_top10.groupby("recommended_rollout_decision", as_index=False)
        .agg(campaign_count=("campaign_id", "count"))
        .sort_values("campaign_count", ascending=False)
    )

    campaign_rollout_fig = px.pie(
        rollout_counts,
        names="recommended_rollout_decision",
        values="campaign_count",
        hole=0.55,
        title="Recommended Rollout Mix",
    )
    campaign_rollout_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=340,
        margin=dict(l=20, r=20, t=55, b=30),
        font=dict(family="Arial", color=COLORS["text"]),
    )

    profit_chart_data = campaign_top10.sort_values("expected_campaign_profit", ascending=True)

    campaign_profit_fig = px.bar(
        profit_chart_data,
        x="expected_campaign_profit",
        y="campaign_name",
        orientation="h",
        title="Expected Campaign Profit by Recommendation",
        labels={
            "expected_campaign_profit": "Expected Profit",
            "campaign_name": "Campaign",
        },
    )
    campaign_profit_fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=460,
        margin=dict(l=20, r=20, t=55, b=30),
        font=dict(family="Arial", color=COLORS["text"]),
    )


decision_counts = customer_features["decision_status"].value_counts().reset_index()
decision_counts.columns = ["decision_status", "customer_count"]

decision_fig = px.pie(
    decision_counts,
    names="decision_status",
    values="customer_count",
    hole=0.45,
    title="Decision Status Share",
    color="decision_status",
    color_discrete_map=DECISION_COLOR_MAP,
)
decision_fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=55, b=20),
    font=dict(family="Arial", size=13, color="#1f2937"),
    legend_title_text="Decision",
)

decision_bar_fig = px.bar(
    decision_counts,
    x="decision_status",
    y="customer_count",
    text="customer_count",
    title="Decision Status Count",
    color="decision_status",
    color_discrete_map=DECISION_COLOR_MAP,
)
decision_bar_fig.update_traces(texttemplate="%{text:,}", textposition="outside")
decision_bar_fig.update_layout(
    xaxis_title="Decision Status",
    yaxis_title="Customer Count",
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=40),
    font=dict(family="Arial", size=13, color="#1f2937"),
)

segment_count_fig = px.bar(
    segment_summary.sort_values("customer_count", ascending=True),
    x="customer_count",
    y="customer_segment",
    orientation="h",
    text="customer_count",
    title="Customer Count by Segment",
)
segment_count_fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#2563eb")
segment_count_fig.update_layout(
    xaxis_title="Customer Count",
    yaxis_title="Customer Segment",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=40),
    font=dict(family="Arial", size=13, color="#1f2937"),
)

eligible_fig = px.bar(
    segment_summary.sort_values("campaign_eligible_rate", ascending=True),
    x="campaign_eligible_rate",
    y="customer_segment",
    orientation="h",
    text="campaign_eligible_rate",
    title="Campaign Eligible Rate by Segment",
)
eligible_fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", marker_color="#0ea5e9")
eligible_fig.update_layout(
    xaxis_title="Eligible Rate",
    yaxis_title="Customer Segment",
    xaxis_tickformat=".0%",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=40),
    font=dict(family="Arial", size=13, color="#1f2937"),
)

segment_decision_mix = (
    customer_features.groupby(["customer_segment", "decision_status"], as_index=False)
    .agg(customer_count=("customer_id", "count"))
)

segment_stack_fig = px.bar(
    segment_decision_mix,
    x="customer_segment",
    y="customer_count",
    color="decision_status",
    title="Decision Mix by Segment",
    color_discrete_map=DECISION_COLOR_MAP,
)
segment_stack_fig.update_layout(
    xaxis_title="Customer Segment",
    yaxis_title="Customer Count",
    xaxis_tickangle=-25,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=110),
    font=dict(family="Arial", size=13, color="#1f2937"),
    legend_title_text="Decision",
)

action_counts = customer_features["recommended_action"].value_counts().reset_index()
action_counts.columns = ["recommended_action", "customer_count"]

action_fig = px.pie(
    action_counts,
    names="recommended_action",
    values="customer_count",
    hole=0.42,
    title="Recommended Action Mix",
)
action_fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=55, b=20),
    font=dict(family="Arial", size=13, color="#1f2937"),
    legend_title_text="Action",
)

offer_counts = customer_features["offer_type"].value_counts().reset_index()
offer_counts.columns = ["offer_type", "customer_count"]

offer_fig = px.bar(
    offer_counts.sort_values("customer_count", ascending=True),
    x="customer_count",
    y="offer_type",
    orientation="h",
    text="customer_count",
    title="Offer Type Distribution",
)
offer_fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#7c3aed")
offer_fig.update_layout(
    xaxis_title="Customer Count",
    yaxis_title="Offer Type",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=40),
    font=dict(family="Arial", size=13, color="#1f2937"),
)

risk_counts = customer_features["risk_band"].value_counts().reset_index()
risk_counts.columns = ["risk_band", "customer_count"]

risk_fig = px.pie(
    risk_counts,
    names="risk_band",
    values="customer_count",
    hole=0.45,
    title="Risk Band Distribution",
    color="risk_band",
    color_discrete_map=RISK_COLOR_MAP,
)
risk_fig.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=20, r=20, t=55, b=20),
    font=dict(family="Arial", size=13, color="#1f2937"),
    legend_title_text="Risk Band",
)

high_utilization_mix = (
    customer_features[customer_features["customer_segment"] == "High-Utilization Revolver"]
    ["decision_status"]
    .value_counts()
    .reset_index()
)
high_utilization_mix.columns = ["decision_status", "customer_count"]

high_utilization_fig = px.bar(
    high_utilization_mix,
    x="decision_status",
    y="customer_count",
    text="customer_count",
    title="High-Utilization Revolver Decision Mix",
    color="decision_status",
    color_discrete_map=DECISION_COLOR_MAP,
)
high_utilization_fig.update_traces(texttemplate="%{text:,}", textposition="outside")
high_utilization_fig.update_layout(
    xaxis_title="Decision Status",
    yaxis_title="Customer Count",
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=40),
    font=dict(family="Arial", size=13, color="#1f2937"),
)

block_by_segment = (
    customer_features[customer_features["decision_status"] == "Block"]
    .groupby("customer_segment", as_index=False)
    .agg(blocked_customers=("customer_id", "count"))
    .sort_values("blocked_customers", ascending=True)
)

block_segment_fig = px.bar(
    block_by_segment,
    x="blocked_customers",
    y="customer_segment",
    orientation="h",
    text="blocked_customers",
    title="Blocked Customers by Segment",
)
block_segment_fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#dc2626")
block_segment_fig.update_layout(
    xaxis_title="Blocked Customers",
    yaxis_title="Customer Segment",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=30, t=55, b=40),
    font=dict(family="Arial", size=13, color="#1f2937"),
)

risk_watch_count = len(customer_features[customer_features["customer_segment"] == "Risk Watch"])
risk_watch_blocked = len(
    customer_features[
        (customer_features["customer_segment"] == "Risk Watch")
        & (customer_features["decision_status"] == "Block")
    ]
)
high_utilization_scaled = len(
    customer_features[
        (customer_features["customer_segment"] == "High-Utilization Revolver")
        & (customer_features["decision_status"] == "Scale")
    ]
)
total_blocked = len(customer_features[customer_features["decision_status"] == "Block"])

priority_table = segment_summary[
    [
        "customer_segment",
        "customer_count",
        "avg_risk_adjusted_profit",
        "avg_expected_roi",
        "scale_count",
        "test_count",
        "campaign_eligible_rate",
    ]
].sort_values("customer_count", ascending=False)

priority_table_display = priority_table.copy()
priority_table_display["avg_risk_adjusted_profit"] = priority_table_display["avg_risk_adjusted_profit"].map(lambda x: f"${x:,.0f}")
priority_table_display["avg_expected_roi"] = priority_table_display["avg_expected_roi"].map(lambda x: f"{x:.1f}")
priority_table_display["campaign_eligible_rate"] = priority_table_display["campaign_eligible_rate"].map(lambda x: f"{x * 100:.1f}%")

header_style = {
    "textAlign": "left",
    "backgroundColor": "#f3f4f6",
    "padding": "12px",
    "borderBottom": "1px solid #e5e7eb",
    "fontWeight": "800",
    "color": "#374151",
}

cell_style = {
    "padding": "12px",
    "borderBottom": "1px solid #f1f5f9",
    "color": "#111827",
}

priority_rows = [
    html.Tr([html.Th(col.replace("_", " ").title(), style=header_style) for col in priority_table_display.columns])
]
for _, row in priority_table_display.iterrows():
    priority_rows.append(html.Tr([html.Td(row[col], style=cell_style) for col in priority_table_display.columns]))





def format_large_number(value: float) -> str:
    if pd.isna(value):
        return "0"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def create_metric_chip(label: str, value: str) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                label,
                style={
                    "fontSize": "11px",
                    "fontWeight": "800",
                    "color": COLORS["muted"],
                    "textTransform": "uppercase",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "15px",
                    "fontWeight": "900",
                    "color": COLORS["text"],
                    "marginTop": "2px",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["soft_blue"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "14px",
            "padding": "10px",
        },
    )


def create_campaign_recommendation_card(row: pd.Series) -> html.Div:
    rollout = row.get("recommended_rollout_decision", "Unknown")

    rollout_colors = {
        "Scale": "#16a34a",
        "Test": "#7c3aed",
        "Constrain": "#f97316",
        "Do Not Launch": "#64748b",
        "Block": "#dc2626",
    }

    accent = rollout_colors.get(rollout, "#2563eb")

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        f"#{int(row.get('dashboard_recommendation_rank', 0))}",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "900",
                            "color": "white",
                            "backgroundColor": accent,
                            "borderRadius": "999px",
                            "padding": "6px 10px",
                            "display": "inline-block",
                        },
                    ),
                    html.Div(
                        rollout,
                        style={
                            "fontSize": "12px",
                            "fontWeight": "800",
                            "color": accent,
                            "backgroundColor": f"{accent}18",
                            "border": f"1px solid {accent}33",
                            "borderRadius": "999px",
                            "padding": "6px 10px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "12px",
                },
            ),
            html.H3(
                row.get("campaign_name", "Campaign"),
                style={
                    "fontSize": "18px",
                    "fontWeight": "900",
                    "margin": "0 0 6px 0",
                    "color": COLORS["text"],
                },
            ),
            html.Div(
                row.get("campaign_family", "Campaign Family"),
                style={
                    "fontSize": "13px",
                    "fontWeight": "700",
                    "color": COLORS["muted"],
                    "marginBottom": "12px",
                },
            ),
            html.P(
                row.get("business_goal", ""),
                style={
                    "fontSize": "13px",
                    "lineHeight": "1.5",
                    "color": COLORS["muted"],
                    "margin": "0 0 14px 0",
                },
            ),
            html.Div(
                children=[
                    create_metric_chip("Eligible", format_large_number(row.get("eligible_customers", 0))),
                    create_metric_chip("Scale", format_large_number(row.get("scale_customers", 0))),
                    create_metric_chip("Profit", format_currency(row.get("expected_campaign_profit", 0))),
                    create_metric_chip("ROI", f"{row.get('expected_campaign_roi', 0):.2f}x"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "10px",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderTop": f"5px solid {accent}",
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
        },
    )


def create_campaign_table_rows(campaign_recommendations: pd.DataFrame, limit: int = 10) -> list[dict]:
    if campaign_recommendations.empty:
        return []

    table = campaign_recommendations.head(limit).copy()

    table["Expected Profit"] = table["expected_campaign_profit"].apply(format_currency)
    table["Expected ROI"] = table["expected_campaign_roi"].apply(lambda value: f"{value:.2f}x")
    table["Eligible Customers"] = table["eligible_customers"].apply(lambda value: f"{int(value):,}")
    table["Scale Customers"] = table["scale_customers"].apply(lambda value: f"{int(value):,}")
    table["Blocked Customers"] = table["blocked_customers"].apply(lambda value: f"{int(value):,}")
    table["Score"] = table["campaign_score"].apply(lambda value: f"{value:.1f}")

    return table[
        [
            "dashboard_recommendation_rank",
            "campaign_name",
            "campaign_family",
            "risk_level",
            "recommended_rollout_decision",
            "Eligible Customers",
            "Scale Customers",
            "Blocked Customers",
            "Expected Profit",
            "Expected ROI",
            "Score",
        ]
    ].rename(
        columns={
            "dashboard_recommendation_rank": "Rank",
            "campaign_name": "Campaign",
            "campaign_family": "Family",
            "risk_level": "Risk",
            "recommended_rollout_decision": "Rollout",
        }
    ).to_dict("records")


def create_campaign_detail_panel(campaign_recommendations: pd.DataFrame) -> html.Div:
    if campaign_recommendations.empty:
        return create_insight_card(
            "Campaign Detail",
            "Campaign recommendation data is not available yet. Run src/generate_campaign_library.py and src/score_campaign_recommendations.py first.",
        )

    top_campaign = campaign_recommendations.iloc[0]

    return html.Div(
        children=[
            html.H3(
                "Top Campaign Detail",
                style={
                    "fontSize": "20px",
                    "fontWeight": "900",
                    "margin": "0 0 12px 0",
                    "color": COLORS["text"],
                },
            ),
            html.Div(
                top_campaign["campaign_name"],
                style={
                    "fontSize": "24px",
                    "fontWeight": "900",
                    "color": COLORS["text"],
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                f"{top_campaign['campaign_family']} • {top_campaign['campaign_type']} • {top_campaign['risk_level']} risk",
                style={
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "color": COLORS["muted"],
                    "marginBottom": "14px",
                },
            ),
            html.P(
                top_campaign["offer_description"],
                style={
                    "fontSize": "14px",
                    "lineHeight": "1.6",
                    "color": COLORS["muted"],
                    "margin": "0 0 14px 0",
                },
            ),
            html.Div(
                children=[
                    create_metric_chip("Target Segments", top_campaign["target_segments"]),
                    create_metric_chip("Success Metric", top_campaign["primary_success_metric"]),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px",
                    "marginBottom": "12px",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        "Guardrail Notes",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "900",
                            "color": COLORS["text"],
                            "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        top_campaign["guardrail_notes"],
                        style={
                            "fontSize": "13px",
                            "lineHeight": "1.5",
                            "color": COLORS["muted"],
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#fff7ed",
                    "border": "1px solid #fed7aa",
                    "borderRadius": "14px",
                    "padding": "14px",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "20px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
        },
    )


def create_filter_panel() -> html.Div:
    segment_options = [
        {"label": segment, "value": segment}
        for segment in sorted(customer_features["customer_segment"].unique())
    ]

    decision_options = [
        {"label": decision, "value": decision}
        for decision in sorted(customer_features["decision_status"].unique())
    ]

    risk_options = [
        {"label": risk, "value": risk}
        for risk in sorted(customer_features["risk_band"].unique())
    ]

    action_options = [
        {"label": action, "value": action}
        for action in sorted(customer_features["recommended_action"].unique())
    ]

    dropdown_style = {
        "fontSize": "14px",
    }

    label_style = {
        "fontSize": "13px",
        "fontWeight": "800",
        "color": COLORS["text"],
        "marginBottom": "6px",
        "display": "block",
    }

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                "Interactive Filters",
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "color": COLORS["blue"],
                                    "marginBottom": "6px",
                                },
                            ),
                            html.H3(
                                "Explore the portfolio by segment, risk, action, or decision",
                                style={
                                    "margin": "0",
                                    "fontSize": "22px",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                },
                            ),
                            html.P(
                                "Use these filters to narrow the portfolio before reviewing strategy, guardrails, or campaign actions. Leave filters blank to view the full portfolio.",
                                style={
                                    "margin": "8px 0 0 0",
                                    "fontSize": "14px",
                                    "color": COLORS["muted"],
                                    "lineHeight": "1.5",
                                },
                            ),
                        ],
                        style={"marginBottom": "18px"},
                    ),

                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Customer Segment", style=label_style),
                                    dcc.Dropdown(
                                        id="filter-segment",
                                        options=segment_options,
                                        multi=True,
                                        placeholder="All segments",
                                        style=dropdown_style,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Decision Status", style=label_style),
                                    dcc.Dropdown(
                                        id="filter-decision",
                                        options=decision_options,
                                        multi=True,
                                        placeholder="All decisions",
                                        style=dropdown_style,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Risk Band", style=label_style),
                                    dcc.Dropdown(
                                        id="filter-risk",
                                        options=risk_options,
                                        multi=True,
                                        placeholder="All risk bands",
                                        style=dropdown_style,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Recommended Action", style=label_style),
                                    dcc.Dropdown(
                                        id="filter-action",
                                        options=action_options,
                                        multi=True,
                                        placeholder="All actions",
                                        style=dropdown_style,
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(4, 1fr)",
                            "gap": "14px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "20px",
                    "padding": "22px",
                    "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                },
            ),

            html.Div(
                id="filtered-summary-output",
                style={"marginTop": "16px"},
            ),
        ],
        style={"marginBottom": "24px"},
    )



def create_small_metric_card(title: str, value: str, note: str, accent: str = "#2563eb") -> html.Div:
    return html.Div(
        children=[
            html.Div(title, style={"fontSize": "13px", "fontWeight": "800", "color": COLORS["muted"]}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": "900", "marginTop": "8px", "color": COLORS["text"]}),
            html.Div(note, style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "6px", "lineHeight": "1.35"}),
        ],
        style={
            "backgroundColor": "#ffffff",
            "borderTop": f"4px solid {accent}",
            "borderLeft": f"1px solid {COLORS['border']}",
            "borderRight": f"1px solid {COLORS['border']}",
            "borderBottom": f"1px solid {COLORS['border']}",
            "borderRadius": "16px",
            "padding": "18px",
            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
        },
    )



def build_customer_lookup_layout() -> html.Div:
    customer_directory = customer_features[
        [
            "customer_id",
            "customer_name",
            "customer_email",
            "city",
            "state",
            "customer_segment",
            "risk_band",
            "decision_status",
        ]
    ].copy()

    customer_directory["location"] = customer_directory["city"] + ", " + customer_directory["state"]

    customer_directory = customer_directory[
        [
            "customer_id",
            "customer_name",
            "customer_email",
            "location",
            "customer_segment",
            "risk_band",
            "decision_status",
        ]
    ]

    segment_options = [
        {"label": segment, "value": segment}
        for segment in sorted(customer_features["customer_segment"].unique())
    ]

    decision_options = [
        {"label": decision, "value": decision}
        for decision in sorted(customer_features["decision_status"].unique())
    ]

    return html.Div(
        children=[
            create_tab_intro(
                "Customer Lookup",
                "Search the customer directory, select a customer, and review the profile, decision, risk band, recommended action, and explanation behind the engine output.",
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                "Customer Directory",
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "color": COLORS["blue"],
                                    "marginBottom": "8px",
                                },
                            ),
                            html.H3(
                                "Find a customer",
                                style={
                                    "margin": "0 0 8px 0",
                                    "fontSize": "20px",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                },
                            ),
                            html.P(
                                "Search by ID, name, email, segment, risk, or decision. Select a row to inspect the customer decision profile.",
                                style={
                                    "margin": "0 0 14px 0",
                                    "fontSize": "14px",
                                    "color": COLORS["muted"],
                                    "lineHeight": "1.45",
                                },
                            ),
                            html.Div(
                                children=[
                                    dcc.Input(
                                        id="customer-directory-search",
                                        type="text",
                                        placeholder="Search customer ID, name, email, segment, risk, decision...",
                                        debounce=True,
                                        style={
                                            "width": "100%",
                                            "height": "40px",
                                            "border": f"1px solid {COLORS['border']}",
                                            "borderRadius": "10px",
                                            "padding": "0 12px",
                                            "fontSize": "14px",
                                            "boxSizing": "border-box",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="customer-directory-segment-filter",
                                        options=segment_options,
                                        placeholder="All segments",
                                        clearable=True,
                                        style={"fontSize": "13px"},
                                    ),
                                    dcc.Dropdown(
                                        id="customer-directory-decision-filter",
                                        options=decision_options,
                                        placeholder="All decisions",
                                        clearable=True,
                                        style={"fontSize": "13px"},
                                    ),
                                ],
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "1.4fr 0.8fr 0.8fr",
                                    "gap": "10px",
                                    "marginBottom": "12px",
                                },
                            ),
                            dash_table.DataTable(
                                id="customer-directory-table",
                                data=customer_directory.head(500).to_dict("records"),
                                columns=[
                                    {"name": "ID", "id": "customer_id"},
                                    {"name": "Name", "id": "customer_name"},
                                    {"name": "Location", "id": "location"},
                                    {"name": "Segment", "id": "customer_segment"},
                                    {"name": "Risk", "id": "risk_band"},
                                    {"name": "Decision", "id": "decision_status"},
                                ],
                                page_size=8,
                                sort_action="native",
                                filter_action="none",
                                row_selectable="single",
                                selected_rows=[0],
                                style_table={
                                    "overflowX": "auto",
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "14px",
                                },
                                style_header={
                                    "backgroundColor": "#f8fafc",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                    "borderBottom": f"1px solid {COLORS['border']}",
                                    "fontSize": "12px",
                                    "textTransform": "uppercase",
                                },
                                style_cell={
                                    "fontFamily": "Arial, sans-serif",
                                    "fontSize": "13px",
                                    "padding": "9px 10px",
                                    "textAlign": "left",
                                    "borderBottom": f"1px solid {COLORS['border']}",
                                    "whiteSpace": "normal",
                                    "height": "auto",
                                    "lineHeight": "1.25",
                                    "minWidth": "70px",
                                    "maxWidth": "180px",
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "customer_id"}, "width": "82px"},
                                    {"if": {"column_id": "customer_name"}, "width": "135px"},
                                    {"if": {"column_id": "location"}, "width": "130px"},
                                    {"if": {"column_id": "customer_segment"}, "width": "165px"},
                                    {"if": {"column_id": "risk_band"}, "width": "110px"},
                                    {"if": {"column_id": "decision_status"}, "width": "115px"},
                                ],
                                style_data_conditional=[
                                    {
                                        "if": {"state": "selected"},
                                        "backgroundColor": "#eff6ff",
                                        "border": "1px solid #2563eb",
                                    },
                                    {
                                        "if": {"filter_query": "{decision_status} = Scale"},
                                        "color": "#166534",
                                        "fontWeight": "800",
                                    },
                                    {
                                        "if": {"filter_query": "{decision_status} = Block"},
                                        "color": "#991b1b",
                                        "fontWeight": "800",
                                    },
                                    {
                                        "if": {"filter_query": "{decision_status} = Test"},
                                        "color": "#1d4ed8",
                                        "fontWeight": "800",
                                    },
                                ],
                            ),
                            html.Div(
                                children=[
                                    html.Div(
                                        "Synthetic data note",
                                        style={
                                            "fontSize": "13px",
                                            "fontWeight": "900",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "1px",
                                            "color": COLORS["blue"],
                                            "marginBottom": "6px",
                                        },
                                    ),
                                    html.P(
                                        "Names, emails, phone numbers, and profile attributes are synthetic and used only to demonstrate how the decision engine could be inspected.",
                                        style={
                                            "margin": "0",
                                            "fontSize": "13px",
                                            "color": COLORS["muted"],
                                            "lineHeight": "1.45",
                                        },
                                    ),
                                ],
                                style={
                                    "backgroundColor": "#eff6ff",
                                    "border": "1px solid #bfdbfe",
                                    "borderRadius": "14px",
                                    "padding": "13px",
                                    "marginTop": "14px",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "20px",
                            "padding": "18px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                            "minWidth": "0",
                        },
                    ),
                    html.Div(
                        id="customer-lookup-output",
                        style={"minWidth": "0"},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "0.9fr 1.1fr",
                    "gap": "16px",
                    "alignItems": "start",
                },
            ),
        ]
    )


def explain_customer_decision(customer: pd.Series) -> list[str]:
    explanations = []

    decision = customer["decision_status"]
    segment = customer["customer_segment"]
    risk_band = customer["risk_band"]
    action = customer["recommended_action"]

    roi = customer["expected_roi"]
    default_probability = customer["default_probability"]
    utilization = customer["utilization_rate"]
    late_payments = customer["late_payments_12m"]
    risk_adjusted_profit = customer["risk_adjusted_profit"]

    explanations.append(
        f"The customer is classified as {segment} based on spend behavior, tenure, utilization, and repayment signals."
    )

    explanations.append(
        f"The customer is in the {risk_band} risk band with an estimated default probability of {default_probability * 100:.2f}%."
    )

    if utilization >= 0.75:
        explanations.append(
            "Utilization is high, so the engine applies extra caution before recommending any broad growth campaign."
        )
    elif utilization <= 0.25:
        explanations.append(
            "Utilization is relatively low, which may indicate unused capacity or lower immediate credit pressure."
        )
    else:
        explanations.append(
            "Utilization is in a moderate range, so the engine balances growth opportunity with risk exposure."
        )

    if late_payments > 0:
        explanations.append(
            f"The customer has {int(late_payments)} late payment(s) in the last 12 months, which increases risk sensitivity."
        )
    else:
        explanations.append(
            "No late payments are observed in the last 12 months, which supports a cleaner repayment profile."
        )

    if roi >= 5:
        explanations.append(
            f"Expected ROI is strong at {roi:.1f}x, which supports a growth or testing recommendation if risk guardrails are satisfied."
        )
    elif roi >= 0:
        explanations.append(
            f"Expected ROI is positive but not strong at {roi:.1f}x, so controlled testing may be more appropriate than broad scaling."
        )
    else:
        explanations.append(
            f"Expected ROI is negative at {roi:.1f}x, so the engine avoids growth investment for this customer."
        )

    if risk_adjusted_profit < 0:
        explanations.append(
            "Risk-adjusted profit is negative after expected credit loss, which is a major reason to avoid or block growth action."
        )

    if decision == "Scale":
        explanations.append(
            f"Final decision: Scale. The customer has enough economic upside and passes the core risk guardrails. Recommended action: {action}."
        )
    elif decision == "Test":
        explanations.append(
            f"Final decision: Test. The customer shows some opportunity, but uncertainty or risk is high enough that controlled testing is safer than broad rollout. Recommended action: {action}."
        )
    elif decision == "Do Not Launch":
        explanations.append(
            f"Final decision: Do Not Launch. The customer does not currently show enough risk-adjusted upside for a campaign. Recommended action: {action}."
        )
    else:
        explanations.append(
            f"Final decision: Block. Risk guardrails override campaign growth logic for this customer. Recommended action: {action}."
        )

    return explanations


def create_explanation_list(explanations: list[str]) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                explanation,
                style={
                    "backgroundColor": "#f8fafc",
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "14px",
                    "padding": "12px 14px",
                    "fontSize": "14px",
                    "color": COLORS["text"],
                    "lineHeight": "1.45",
                    "minWidth": "0",
                },
            )
            for explanation in explanations
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
            "gap": "10px",
        },
    )


def create_profile_field(label: str, value: str) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                label,
                style={
                    "fontSize": "12px",
                    "fontWeight": "900",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.4px",
                    "color": COLORS["muted"],
                    "marginBottom": "4px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "14px",
                    "fontWeight": "900",
                    "color": COLORS["text"],
                    "lineHeight": "1.25",
                    "overflowWrap": "break-word",
                    "wordBreak": "break-word",
                },
            ),
        ],
        style={
            "backgroundColor": "#f8fafc",
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "12px",
            "padding": "10px 12px",
            "minWidth": "0",
        },
    )


def create_profile_section(title: str, rows: list[tuple[str, str]], columns: int = 2) -> html.Div:
    return html.Div(
        children=[
            html.H4(
                title,
                style={
                    "margin": "0 0 10px 0",
                    "fontSize": "16px",
                    "fontWeight": "900",
                    "color": COLORS["text"],
                },
            ),
            html.Div(
                children=[
                    create_profile_field(label, value)
                    for label, value in rows
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": f"repeat({columns}, minmax(0, 1fr))",
                    "gap": "10px",
                },
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "16px",
            "padding": "14px",
            "minWidth": "0",
        },
    )


def create_customer_profile_table(customer: pd.Series) -> html.Div:
    contact_rows = [
        ("Customer ID", str(customer["customer_id"])),
        ("Name", customer["customer_name"]),
        ("Email", customer["customer_email"]),
        ("Phone", customer["phone_number"]),
        ("Location", f"{customer['city']}, {customer['state']} {customer['zip_code']}"),
        ("Employment", customer["employment_status"]),
        ("Occupation", customer["occupation_group"]),
        ("Preferred Channel", customer["preferred_channel"]),
    ]

    account_rows = [
        ("Relationship Tier", customer["relationship_tier"]),
        ("Signup Channel", customer["signup_channel"]),
        ("Open Date", customer["account_open_date"]),
        ("Card Type", customer["card_type"]),
        ("Rewards Preference", customer["rewards_preference"]),
        ("Digital Score", f"{int(customer['digital_engagement_score'])}/100"),
        ("Last App Login", f"{int(customer['last_app_login_days'])} days ago"),
        ("Autopay / Paperless", f"{customer['autopay_enrolled']} / {customer['paperless_enrolled']}"),
    ]

    credit_rows = [
        ("Age", f"{int(customer['age'])}"),
        ("Income", f"${customer['income']:,.0f}"),
        ("Credit Score", f"{int(customer['credit_score'])}"),
        ("Credit Limit", f"${customer['credit_limit']:,.0f}"),
        ("Current Balance", f"${customer['current_balance']:,.0f}"),
        ("Utilization", f"{customer['utilization_rate'] * 100:.1f}%"),
        ("Monthly Spend", f"${customer['monthly_spend']:,.0f}"),
        ("Late Payments", f"{int(customer['late_payments_12m'])}"),
    ]

    return html.Div(
        children=[
            create_profile_section("Contact & Identity", contact_rows, columns=2),
            create_profile_section("Account Relationship", account_rows, columns=2),
            create_profile_section("Credit Behavior", credit_rows, columns=2),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr",
            "gap": "12px",
        },
    )


def build_decision_workbench_layout() -> html.Div:
    return html.Div(
        children=[
            create_tab_intro(
                "Decision Workbench",
                "Use this section for deeper decision tools. The main dashboard shows portfolio strategy, while this workbench helps users inspect individual customers, test campaign scenarios, and plan experiments before launch.",
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                "Workbench Tools",
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "color": COLORS["blue"],
                                    "marginBottom": "8px",
                                },
                            ),
                            html.H3(
                                "Inspect, simulate, test, then decide",
                                style={
                                    "margin": "0 0 6px 0",
                                    "fontSize": "22px",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                },
                            ),
                            html.P(
                                "Customer Lookup explains individual decisions, Scenario Simulator tests campaign assumptions, and A/B Test Planner designs controlled experiments before rollout.",
                                style={
                                    "margin": "0",
                                    "fontSize": "14px",
                                    "color": COLORS["muted"],
                                    "lineHeight": "1.5",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "20px",
                            "padding": "20px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                            "marginBottom": "16px",
                        },
                    ),
                    dcc.Tabs(
                        id="workbench-tabs",
                        value="customer-lookup",
                        vertical=False,
                        children=[
                            dcc.Tab(
                                label="Customer Lookup",
                                value="customer-lookup",
                                children=[build_customer_lookup_layout()],
                                selected_style={
                                    "backgroundColor": COLORS["blue"],
                                    "color": "white",
                                    "fontWeight": "900",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "padding": "12px 16px",
                                    "fontSize": "14px",
                                },
                                style={
                                    "backgroundColor": "#f8fafc",
                                    "color": COLORS["text"],
                                    "fontWeight": "800",
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "12px",
                                    "padding": "12px 16px",
                                    "fontSize": "14px",
                                },
                            ),
                            dcc.Tab(
                                label="Scenario Simulator",
                                value="scenario",
                                children=[build_scenario_simulator_layout()],
                                selected_style={
                                    "backgroundColor": COLORS["blue"],
                                    "color": "white",
                                    "fontWeight": "900",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "padding": "12px 16px",
                                    "fontSize": "14px",
                                },
                                style={
                                    "backgroundColor": "#f8fafc",
                                    "color": COLORS["text"],
                                    "fontWeight": "800",
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "12px",
                                    "padding": "12px 16px",
                                    "fontSize": "14px",
                                },
                            ),
                            dcc.Tab(
                                label="A/B Test Planner",
                                value="ab-test",
                                children=[build_ab_test_planner_layout()],
                                selected_style={
                                    "backgroundColor": COLORS["blue"],
                                    "color": "white",
                                    "fontWeight": "900",
                                    "border": "none",
                                    "borderRadius": "12px",
                                    "padding": "12px 16px",
                                    "fontSize": "14px",
                                },
                                style={
                                    "backgroundColor": "#f8fafc",
                                    "color": COLORS["text"],
                                    "fontWeight": "800",
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "12px",
                                    "padding": "12px 16px",
                                    "fontSize": "14px",
                                },
                            ),
                        ],
                        style={
                            "backgroundColor": "#eef2ff",
                            "border": "1px solid #dbeafe",
                            "borderRadius": "16px",
                            "padding": "8px",
                            "marginBottom": "16px",
                        },
                        colors={
                            "border": "transparent",
                            "primary": "#2563eb",
                            "background": "#eef2ff",
                        },
                    ),
                ]
            ),
        ]
    )


def build_scenario_simulator_layout() -> html.Div:
    segment_options = [{"label": "All Segments", "value": "All Segments"}] + [
        {"label": segment, "value": segment}
        for segment in sorted(customer_features["customer_segment"].unique())
    ]

    return html.Div(
        children=[
            create_tab_intro(
                "Scenario Simulator",
                "Use this section to test how campaign economics change when marketing cost, expected spend lift, and risk tolerance change. This helps decision-makers compare aggressive versus conservative rollout assumptions before launching a campaign.",
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H3("Scenario Inputs", style={"marginTop": "0"}),
                            html.Label("Customer Segment", style={"fontWeight": "800"}),
                            dcc.Dropdown(
                                id="scenario-segment",
                                options=segment_options,
                                value="All Segments",
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Label("Marketing Cost per Customer", style={"fontWeight": "800"}),
                            dcc.Slider(
                                id="scenario-marketing-cost",
                                min=1,
                                max=25,
                                step=1,
                                value=5,
                                marks={1: "$1", 5: "$5", 10: "$10", 15: "$15", 20: "$20", 25: "$25"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            html.Div(style={"height": "22px"}),

                            html.Label("Expected Spend Lift", style={"fontWeight": "800"}),
                            dcc.Slider(
                                id="scenario-spend-lift",
                                min=-2,
                                max=20,
                                step=1,
                                value=8,
                                marks={-2: "-2%", 0: "0%", 5: "5%", 10: "10%", 15: "15%", 20: "20%"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            html.Div(style={"height": "22px"}),

                            html.Label("Max Default Probability Allowed", style={"fontWeight": "800"}),
                            dcc.Slider(
                                id="scenario-risk-threshold",
                                min=2,
                                max=20,
                                step=1,
                                value=8,
                                marks={2: "2%", 5: "5%", 8: "8%", 12: "12%", 16: "16%", 20: "20%"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={
                            "backgroundColor": COLORS["card"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "18px",
                            "padding": "22px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                        },
                    ),

                    html.Div(
                        children=[
                            html.H3("Scenario Output", style={"marginTop": "0"}),
                            html.Div(id="scenario-output"),
                        ],
                        style={
                            "backgroundColor": COLORS["card"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "18px",
                            "padding": "22px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "0.9fr 1.1fr",
                    "gap": "18px",
                },
            ),
        ]
    )


def build_ab_test_planner_layout() -> html.Div:
    segment_options = [
        {"label": segment, "value": segment}
        for segment in sorted(customer_features["customer_segment"].unique())
        if segment != "Risk Watch"
    ]

    return html.Div(
        children=[
            create_tab_intro(
                "A/B Test Planner",
                "Use this section to plan a controlled experiment before scaling a campaign. It helps estimate control/treatment group sizes, expected incremental responders, and whether the test is large enough to make a confident decision.",
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H3("Test Inputs", style={"marginTop": "0"}),

                            html.Label("Target Segment", style={"fontWeight": "800"}),
                            dcc.Dropdown(
                                id="ab-segment",
                                options=segment_options,
                                value="High-Utilization Revolver",
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Label("Baseline Response Rate", style={"fontWeight": "800"}),
                            dcc.Slider(
                                id="ab-baseline-rate",
                                min=1,
                                max=20,
                                step=1,
                                value=6,
                                marks={1: "1%", 5: "5%", 10: "10%", 15: "15%", 20: "20%"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            html.Div(style={"height": "22px"}),

                            html.Label("Expected Lift from Treatment", style={"fontWeight": "800"}),
                            dcc.Slider(
                                id="ab-lift",
                                min=1,
                                max=15,
                                step=1,
                                value=4,
                                marks={1: "+1pp", 3: "+3pp", 5: "+5pp", 10: "+10pp", 15: "+15pp"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            html.Div(style={"height": "22px"}),

                            html.Label("Test Population Available", style={"fontWeight": "800"}),
                            dcc.Slider(
                                id="ab-test-population",
                                min=100,
                                max=3000,
                                step=100,
                                value=1000,
                                marks={100: "100", 500: "500", 1000: "1k", 2000: "2k", 3000: "3k"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style={
                            "backgroundColor": COLORS["card"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "18px",
                            "padding": "22px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                        },
                    ),

                    html.Div(
                        children=[
                            html.H3("Test Recommendation", style={"marginTop": "0"}),
                            html.Div(id="ab-output"),
                        ],
                        style={
                            "backgroundColor": COLORS["card"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "18px",
                            "padding": "22px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "0.9fr 1.1fr",
                    "gap": "18px",
                },
            ),
        ]
    )



app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Risk-Aware Credit Card Growth Decision Engine"

tab_style = {
    "padding": "14px 18px",
    "fontWeight": "700",
    "fontSize": "14px",
    "border": "none",
    "backgroundColor": "#eef2ff",
    "color": "#374151",
}

selected_tab_style = {
    "padding": "14px 18px",
    "fontWeight": "800",
    "fontSize": "14px",
    "border": "none",
    "backgroundColor": "#2563eb",
    "color": "white",
    "borderRadius": "12px",
}

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            "Credit Card Portfolio Analytics",
                            style={
                                "textTransform": "uppercase",
                                "letterSpacing": "1.5px",
                                "fontSize": "12px",
                                "fontWeight": "800",
                                "color": "#bfdbfe",
                                "marginBottom": "10px",
                            },
                        ),
                        html.H1(
                            "Risk-Aware Credit Card Growth Decision Engine",
                            style={
                                "fontSize": "42px",
                                "lineHeight": "1.1",
                                "margin": "0",
                                "fontWeight": "900",
                            },
                        ),
                        html.P(
                            "Identify which existing credit card customers to scale, test, protect, or block using segmentation, profitability, risk, and responsible-lending guardrails.",
                            style={
                                "fontSize": "17px",
                                "lineHeight": "1.55",
                                "color": "#e5e7eb",
                                "maxWidth": "980px",
                                "marginTop": "14px",
                            },
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    children=[
                        html.Div("Decision Framework", style={"fontSize": "13px", "fontWeight": "800", "color": "#dbeafe", "marginBottom": "12px"}),
                        html.Div("Scale", style={"padding": "10px 14px", "backgroundColor": "rgba(22,163,74,0.25)", "borderRadius": "999px", "marginBottom": "10px", "fontWeight": "800"}),
                        html.Div("Test", style={"padding": "10px 14px", "backgroundColor": "rgba(37,99,235,0.25)", "borderRadius": "999px", "marginBottom": "10px", "fontWeight": "800"}),
                        html.Div("Do Not Launch", style={"padding": "10px 14px", "backgroundColor": "rgba(156,163,175,0.25)", "borderRadius": "999px", "marginBottom": "10px", "fontWeight": "800"}),
                        html.Div("Block", style={"padding": "10px 14px", "backgroundColor": "rgba(220,38,38,0.28)", "borderRadius": "999px", "fontWeight": "800"}),
                    ],
                    style={
                        "width": "260px",
                        "backgroundColor": "rgba(255,255,255,0.12)",
                        "border": "1px solid rgba(255,255,255,0.18)",
                        "borderRadius": "18px",
                        "padding": "20px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "gap": "24px",
                "background": "linear-gradient(135deg, #111827, #243b6b)",
                "color": "white",
                "borderRadius": "24px",
                "padding": "34px",
                "boxShadow": "0 16px 36px rgba(15,23,42,0.18)",
                "marginBottom": "24px",
            },
        ),

        html.Div(
            children=[
                create_kpi_card("Total Customers", f"{int(kpis['total_customers']):,}", "Existing customers analyzed", "#2563eb"),
                create_kpi_card("Monthly Spend", format_currency(kpis["total_monthly_spend"]), "Total card portfolio spend", "#0ea5e9"),
                create_kpi_card("Risk-Adjusted Profit", format_currency(kpis["total_monthly_risk_adjusted_profit"]), "Monthly profit after risk cost", "#16a34a"),
                create_kpi_card("Campaign Eligible", format_percent(kpis["campaign_eligible_rate"]), "Scale or Test customers", "#7c3aed"),
                create_kpi_card("Blocked by Guardrails", format_percent(kpis["block_rate"]), "Protected from growth offers", "#dc2626"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(5, 1fr)",
                "gap": "16px",
                "marginBottom": "24px",
            },
        ),

        create_filter_panel(),

        dcc.Tabs(
            children=[
                dcc.Tab(
                    label="Overview",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Portfolio Overview",
                            "Start here for the executive view of the portfolio. This page shows how many customers are ready to scale, how many should be tested, and how many should be held back or blocked before any campaign rollout.",
                        ),
                        html.Div(
                            children=[
                                create_chart_card("Decision Mix", "Share of customers assigned to Scale, Test, Do Not Launch, or Block.", decision_fig, "overview-decision-share"),
                                create_chart_card("Decision Counts", "Volume behind each portfolio decision.", decision_bar_fig, "overview-decision-counts"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginTop": "22px"},
                        ),
                        create_insight_card(
                            "Executive Takeaway",
                            "The engine avoids a simple campaign-blast approach. It separates customers into launch-ready, test-worthy, not-ready, and blocked groups so growth decisions are tied to risk-adjusted economics.",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Segment Strategy",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Segment Strategy",
                            "Use this tab to understand which customer groups drive the largest opportunity. It compares segment size, campaign eligibility, and decision mix so business teams can prioritize where to scale, test, or apply guardrails.",
                        ),
                        html.Div(
                            children=[
                                create_chart_card("Segment Size", "Customer concentration across the portfolio.", segment_count_fig, "segment-size-chart"),
                                create_chart_card("Eligibility Rate", "Which segments have the highest share of Scale/Test decisions.", eligible_fig, "segment-eligibility-chart"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginTop": "22px"},
                        ),
                        html.Div(style={"height": "18px"}),
                        create_chart_card("Segment Decision Mix", "How Scale, Test, Do Not Launch, and Block decisions differ across customer segments.", segment_stack_fig, "segment-decision-mix-chart"),
                        html.Div(
                            children=[
                                html.H3("Priority Segment Table", style={"margin": "0 0 14px 0", "fontSize": "20px", "fontWeight": "800"}),
                                create_table(priority_rows),
                            ],
                            style={
                                "backgroundColor": COLORS["card"],
                                "border": f"1px solid {COLORS['border']}",
                                "borderRadius": "18px",
                                "padding": "20px",
                                "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                                "marginTop": "18px",
                                "overflowX": "auto",
                            },
                        ),
                        create_insight_card(
                            "Segment-Level Takeaway",
                            "Core Customer and Loyal High-Value Customer are the strongest broad-scale opportunities. High-Utilization Revolvers may show financial value, but should remain a controlled test and guardrail segment.",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Campaigns & Offers",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Campaign Recommendation Engine",
                            "This page ranks campaign opportunities from a reusable campaign library. It shows which campaigns are viable for the current portfolio, where to scale, where to test, and where risk should constrain rollout.",
                        ),
                        html.Div(
                            children=[
                                create_kpi_card(
                                    "Campaign Templates",
                                    f"{total_campaigns_available:,}",
                                    "Available campaign options scored",
                                    "#2563eb",
                                ),
                                create_kpi_card(
                                    "Top-10 Expected Profit",
                                    format_currency(top_campaign_profit),
                                    "Projected profit from recommended campaigns",
                                    "#16a34a",
                                ),
                                create_kpi_card(
                                    "Top-10 Eligible Customers",
                                    f"{int(top_campaign_eligible):,}",
                                    "Customer-campaign matches passing guardrails",
                                    "#7c3aed",
                                ),
                                create_kpi_card(
                                    "Top-10 Scale Customers",
                                    f"{int(top_campaign_scale):,}",
                                    "Customers recommended for broad rollout",
                                    "#0ea5e9",
                                ),
                            ],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(4, 1fr)",
                                "gap": "16px",
                                "marginTop": "22px",
                                "marginBottom": "18px",
                            },
                        ),
                        html.Div(
                            children=[
                                create_chart_card(
                                    "Campaign Family Mix",
                                    "Family distribution across the top recommended campaigns.",
                                    campaign_family_fig,
                                    "campaign-family-mix-chart",
                                ),
                                create_chart_card(
                                    "Rollout Recommendation Mix",
                                    "How top campaigns split across Scale, Test, and Constrain decisions.",
                                    campaign_rollout_fig,
                                    "campaign-rollout-mix-chart",
                                ),
                            ],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1fr 1fr",
                                "gap": "18px",
                                "marginTop": "22px",
                            },
                        ),
                        html.Div(style={"height": "18px"}),
                        create_chart_card(
                            "Profit Potential",
                            "Expected campaign profit across the current top recommendations.",
                            campaign_profit_fig,
                            "campaign-profit-chart",
                        ),
                        html.Div(
                            children=[
                                html.H3(
                                    "Top Recommended Campaigns",
                                    style={
                                        "margin": "0 0 14px 0",
                                        "fontSize": "20px",
                                        "fontWeight": "900",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        create_campaign_recommendation_card(row)
                                        for _, row in campaign_top10.iterrows()
                                    ],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(2, 1fr)",
                                        "gap": "16px",
                                    },
                                ),
                            ],
                            style={"marginTop": "18px"},
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.H3(
                                            "Campaign Recommendation Table",
                                            style={
                                                "margin": "0 0 14px 0",
                                                "fontSize": "20px",
                                                "fontWeight": "900",
                                            },
                                        ),
                                        create_table(campaign_table_rows),
                                    ],
                                    style={
                                        "backgroundColor": COLORS["card"],
                                        "border": f"1px solid {COLORS['border']}",
                                        "borderRadius": "18px",
                                        "padding": "20px",
                                        "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                                        "overflowX": "auto",
                                    },
                                ),
                                create_campaign_detail_panel(campaign_recommendations),
                            ],
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "1.35fr 0.85fr",
                                "gap": "18px",
                                "marginTop": "18px",
                            },
                        ),
                        create_insight_card(
                            "Why this page matters",
                            "The campaign layer connects customer segmentation to business action. It helps a user choose which campaigns to run, where to scale, where to test, and where risk guardrails should constrain rollout.",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Offer Engine",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Offer Decision Engine",
                            "This tab explains the next-best-action logic. Instead of giving every customer the same campaign, the engine assigns different offers or treatments based on segment, risk, profitability, and expected ROI.",
                        ),
                        html.Div(
                            children=[
                                create_chart_card("Recommended Action Mix", "How the engine assigns next-best-actions across the portfolio.", action_fig, "offer-action-mix-chart"),
                                create_chart_card("Offer Type Distribution", "The actual offer or treatment associated with each recommendation.", offer_fig, "offer-type-chart"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "18px", "marginTop": "22px"},
                        ),
                        create_insight_card(
                            "Next-Best-Action Logic",
                            "The engine does not recommend one generic offer to every customer. It assigns different treatments for growth, retention, reactivation, payment health, and risk protection.",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Guardrails",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Responsible Lending Guardrails",
                            "This tab checks whether the engine is protecting customers who may carry higher credit risk. The goal is to separate profitable growth from risky growth and prevent aggressive offers from going to the wrong groups.",
                        ),
                        html.Div(
                            children=[
                                create_kpi_card("Total Blocked Customers", f"{total_blocked:,}", "Blocked from growth campaigns", "#dc2626"),
                                create_kpi_card("Risk Watch Block Check", f"{risk_watch_blocked:,} / {risk_watch_count:,}", "Risk Watch customers blocked", "#f97316"),
                                create_kpi_card("High-Utilization Scaled", f"{high_utilization_scaled:,}", "Should remain zero", "#16a34a"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "16px", "marginTop": "22px", "marginBottom": "18px"},
                        ),
                        html.Div(
                            children=[
                                create_chart_card("Risk Band Distribution", "Portfolio split by estimated risk band.", risk_fig, "guardrail-risk-chart"),
                                create_chart_card("High-Utilization Revolver Mix", "Decision mix for customers who may need extra guardrails.", high_utilization_fig, "guardrail-high-util-chart"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "0.9fr 1.1fr", "gap": "18px"},
                        ),
                        html.Div(style={"height": "18px"}),
                        create_chart_card("Blocked Customers by Segment", "Where risk guardrails are triggered.", block_segment_fig, "guardrail-blocked-segment-chart"),
                        create_insight_card(
                            "Guardrail Interpretation",
                            "This section separates revenue potential from responsible growth. Some customers may generate interest income, but that does not mean they should receive aggressive spend or upgrade offers.",
                            variant="warning",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Decision Workbench",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        build_decision_workbench_layout()
                    ],
                ),
            ],
            style={
                "backgroundColor": "#eef2ff",
                "borderRadius": "16px",
                "padding": "8px",
                "border": "1px solid #dbeafe",
                "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.04)",
            },
            colors={
                "border": "transparent",
                "primary": "#2563eb",
                "background": "#eef2ff",
            },
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "padding": "28px 36px",
        "width": "100%",
        "maxWidth": "none",
        "boxSizing": "border-box",
        "margin": "0",
        "color": COLORS["text"],
    },
)





def get_active_view_label(selected_segments, selected_decisions, selected_risks, selected_actions) -> str:
    has_filters = any([selected_segments, selected_decisions, selected_risks, selected_actions])
    return "Filtered View" if has_filters else "Full Portfolio View"


def apply_view_label_to_figure(fig, view_label: str):
    current_title = fig.layout.title.text if fig.layout.title.text else "Chart"

    # Avoid stacking labels if callback fires multiple times
    current_title = current_title.replace(" — Filtered View", "").replace(" — Full Portfolio View", "")

    fig.update_layout(
        title=f"{current_title} — {view_label}"
    )
    return fig



@app.callback(
    Output("overview-decision-share", "figure"),
    Output("overview-decision-counts", "figure"),
    Output("segment-size-chart", "figure"),
    Output("segment-eligibility-chart", "figure"),
    Output("segment-decision-mix-chart", "figure"),
    Output("offer-action-mix-chart", "figure"),
    Output("offer-type-chart", "figure"),
    Output("guardrail-risk-chart", "figure"),
    Output("guardrail-high-util-chart", "figure"),
    Output("guardrail-blocked-segment-chart", "figure"),
    Input("filter-segment", "value"),
    Input("filter-decision", "value"),
    Input("filter-risk", "value"),
    Input("filter-action", "value"),
)
def update_filtered_charts(selected_segments, selected_decisions, selected_risks, selected_actions):
    filtered = apply_global_filters(
        selected_segments,
        selected_decisions,
        selected_risks,
        selected_actions,
    )

    view_label = get_active_view_label(
        selected_segments,
        selected_decisions,
        selected_risks,
        selected_actions,
    )

    figures = (
        build_decision_share_fig(filtered),
        build_decision_count_fig(filtered),
        build_segment_size_fig(filtered),
        build_segment_eligibility_fig(filtered),
        build_segment_decision_mix_fig(filtered),
        build_action_mix_fig(filtered),
        build_offer_type_fig(filtered),
        build_risk_band_fig(filtered),
        build_high_utilization_fig(filtered),
        build_blocked_segment_fig(filtered),
    )

    return tuple(apply_view_label_to_figure(fig, view_label) for fig in figures)





@app.callback(
    Output("customer-directory-table", "data"),
    Output("customer-directory-table", "selected_rows"),
    Input("customer-directory-search", "value"),
    Input("customer-directory-segment-filter", "value"),
    Input("customer-directory-decision-filter", "value"),
)
def update_customer_directory(search_value, selected_segment, selected_decision):
    directory = customer_features[
        [
            "customer_id",
            "customer_name",
            "customer_email",
            "city",
            "state",
            "customer_segment",
            "risk_band",
            "decision_status",
        ]
    ].copy()

    directory["location"] = directory["city"] + ", " + directory["state"]

    if selected_segment:
        directory = directory[directory["customer_segment"] == selected_segment]

    if selected_decision:
        directory = directory[directory["decision_status"] == selected_decision]

    if search_value:
        search_text = str(search_value).strip().lower()

        searchable = (
            directory["customer_id"].astype(str)
            + " "
            + directory["customer_name"].astype(str)
            + " "
            + directory["customer_email"].astype(str)
            + " "
            + directory["location"].astype(str)
            + " "
            + directory["customer_segment"].astype(str)
            + " "
            + directory["risk_band"].astype(str)
            + " "
            + directory["decision_status"].astype(str)
        ).str.lower()

        directory = directory[searchable.str.contains(search_text, na=False)]

    directory = directory[
        [
            "customer_id",
            "customer_name",
            "location",
            "customer_segment",
            "risk_band",
            "decision_status",
        ]
    ].head(500)

    return directory.to_dict("records"), [0] if len(directory) > 0 else []



@app.callback(
    Output("customer-lookup-output", "children"),
    Input("customer-directory-table", "selected_rows"),
    State("customer-directory-table", "derived_virtual_data"),
)
def update_customer_lookup(selected_rows, table_data):
    if not table_data:
        return html.Div(
            "No customers are available in the current directory view.",
            style={
                "backgroundColor": "#ffffff",
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "20px",
                "padding": "22px",
            },
        )

    if not selected_rows:
        selected_rows = [0]

    selected_index = selected_rows[0]

    if selected_index >= len(table_data):
        selected_index = 0

    customer_id = table_data[selected_index]["customer_id"]

    selected = customer_features[customer_features["customer_id"] == customer_id]

    if selected.empty:
        return html.Div(
            "Customer not found.",
            style={
                "backgroundColor": "#fff7ed",
                "border": "1px solid #fed7aa",
                "borderRadius": "20px",
                "padding": "22px",
            },
        )

    customer = selected.iloc[0]
    explanations = explain_customer_decision(customer)

    decision_color = DECISION_COLOR_MAP.get(customer["decision_status"], COLORS["blue"])
    risk_color = RISK_COLOR_MAP.get(customer["risk_band"], COLORS["muted"])

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                "Customer Decision Profile",
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "color": COLORS["blue"],
                                    "marginBottom": "8px",
                                },
                            ),
                            html.H2(
                                f"{customer['customer_name']}",
                                style={
                                    "margin": "0",
                                    "fontSize": "28px",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                },
                            ),
                            html.P(
                                f"Customer {customer['customer_id']} | {customer['city']}, {customer['state']} | {customer['customer_segment']}",
                                style={
                                    "margin": "8px 0 0 0",
                                    "fontSize": "14px",
                                    "color": COLORS["muted"],
                                },
                            ),
                            html.P(
                                f"{customer['relationship_tier']} relationship | {customer['card_type']} card | {customer['preferred_channel']}",
                                style={
                                    "margin": "6px 0 0 0",
                                    "fontSize": "14px",
                                    "color": COLORS["muted"],
                                },
                            ),
                            html.P(
                                f"{customer['customer_email']} | {customer['phone_number']}",
                                style={
                                    "margin": "6px 0 0 0",
                                    "fontSize": "13px",
                                    "color": COLORS["muted"],
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                customer["decision_status"],
                                style={
                                    "backgroundColor": decision_color,
                                    "color": "white",
                                    "borderRadius": "999px",
                                    "padding": "8px 12px",
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "marginBottom": "8px",
                                    "textAlign": "center",
                                },
                            ),
                            html.Div(
                                customer["risk_band"],
                                style={
                                    "backgroundColor": risk_color,
                                    "color": "white",
                                    "borderRadius": "999px",
                                    "padding": "8px 12px",
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "textAlign": "center",
                                },
                            ),
                        ],
                        style={"minWidth": "150px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "start",
                    "gap": "18px",
                    "marginBottom": "18px",
                },
            ),

            html.Div(
                children=[
                    create_small_metric_card(
                        "Expected ROI",
                        f"{customer['expected_roi']:.1f}x",
                        "Estimated return per marketing dollar",
                        "#2563eb",
                    ),
                    create_small_metric_card(
                        "Risk-Adjusted Profit",
                        f"${customer['risk_adjusted_profit']:,.0f}",
                        "Monthly profit after expected loss",
                        "#16a34a",
                    ),
                    create_small_metric_card(
                        "Default Probability",
                        f"{customer['default_probability'] * 100:.2f}%",
                        "Estimated customer risk",
                        "#f97316",
                    ),
                    create_small_metric_card(
                        "Recommended Action",
                        customer["recommended_action"],
                        "Next-best-action from engine",
                        "#7c3aed",
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                    "gap": "12px",
                    "marginBottom": "18px",
                },
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H3(
                                "Customer Profile",
                                style={
                                    "margin": "0 0 12px 0",
                                    "fontSize": "18px",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                },
                            ),
                            create_customer_profile_table(customer),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "18px",
                            "padding": "18px",
                            "minWidth": "0",
                        },
                    ),
                    html.Div(
                        children=[
                            html.H3(
                                "Why This Decision?",
                                style={
                                    "margin": "0 0 12px 0",
                                    "fontSize": "18px",
                                    "fontWeight": "900",
                                    "color": COLORS["text"],
                                },
                            ),
                            create_explanation_list(explanations),
                        ],
                        style={
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "18px",
                            "padding": "18px",
                            "minWidth": "0",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "16px",
                },
            ),
        ],
        style={
            "backgroundColor": "#ffffff",
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "20px",
            "padding": "22px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
        },
    )



@app.callback(
    Output("scenario-output", "children"),
    Input("scenario-segment", "value"),
    Input("scenario-marketing-cost", "value"),
    Input("scenario-spend-lift", "value"),
    Input("scenario-risk-threshold", "value"),
)
def update_scenario_simulator(segment: str, marketing_cost: float, spend_lift_percent: float, risk_threshold_percent: float):
    scenario_df = customer_features.copy()

    if segment != "All Segments":
        scenario_df = scenario_df[scenario_df["customer_segment"] == segment].copy()

    spend_lift = spend_lift_percent / 100
    risk_threshold = risk_threshold_percent / 100

    interchange_rate = 0.018
    monthly_interest_rate = 0.245 / 12
    rewards_rate = 0.015
    loss_given_default = 0.72
    campaign_horizon_months = 12

    incremental_revenue = scenario_df["monthly_spend"] * spend_lift * interchange_rate
    incremental_interest = scenario_df["revolving_balance"] * spend_lift * monthly_interest_rate
    incremental_rewards_cost = scenario_df["monthly_spend"] * spend_lift * rewards_rate
    incremental_expected_loss = (
        scenario_df["default_probability"]
        * scenario_df["monthly_spend"]
        * spend_lift
        * loss_given_default
    )

    scenario_df["scenario_incremental_profit"] = (
        campaign_horizon_months
        * (
            incremental_revenue
            + incremental_interest
            - incremental_rewards_cost
            - incremental_expected_loss
        )
        - marketing_cost
    )

    scenario_df["scenario_roi"] = scenario_df["scenario_incremental_profit"] / marketing_cost

    scenario_df["scenario_decision"] = "Do Not Launch"

    block_mask = (
        (scenario_df["customer_segment"] == "Risk Watch")
        | (scenario_df["risk_band"] == "Very High Risk")
        | (scenario_df["default_probability"] > risk_threshold)
    )

    scale_mask = (
        (scenario_df["scenario_roi"] >= 5)
        & (~block_mask)
        & (scenario_df["customer_segment"] != "High-Utilization Revolver")
    )

    test_mask = (
        (scenario_df["scenario_roi"] >= 0)
        & (~block_mask)
        & (~scale_mask)
    )

    scenario_df.loc[block_mask, "scenario_decision"] = "Block"
    scenario_df.loc[test_mask, "scenario_decision"] = "Test"
    scenario_df.loc[scale_mask, "scenario_decision"] = "Scale"

    total_customers = len(scenario_df)
    scale_count = int((scenario_df["scenario_decision"] == "Scale").sum())
    test_count = int((scenario_df["scenario_decision"] == "Test").sum())
    block_count = int((scenario_df["scenario_decision"] == "Block").sum())
    eligible_count = scale_count + test_count
    total_profit = scenario_df["scenario_incremental_profit"].sum()
    avg_roi = scenario_df["scenario_roi"].mean()

    if total_customers == 0:
        return html.Div("No customers found for this scenario.")

    if block_count / total_customers > 0.25:
        recommendation = "Conservative scenario. A large share of customers would be blocked, so this setup is better for risk protection than growth."
        rec_color = "#dc2626"
    elif scale_count / total_customers > 0.35 and total_profit > 0:
        recommendation = "Strong growth scenario. The assumptions create a meaningful scale pool while keeping guardrails active."
        rec_color = "#16a34a"
    elif eligible_count / total_customers > 0.35:
        recommendation = "Test-first scenario. There is opportunity, but enough uncertainty remains that controlled testing is better than broad rollout."
        rec_color = "#2563eb"
    else:
        recommendation = "Limited launch scenario. Expected lift or economics may be too weak, so the business should revisit offer design before launch."
        rec_color = "#f97316"

    decision_mix = scenario_df["scenario_decision"].value_counts().reset_index()
    decision_mix.columns = ["scenario_decision", "customer_count"]

    scenario_fig = px.bar(
        decision_mix,
        x="scenario_decision",
        y="customer_count",
        text="customer_count",
        color="scenario_decision",
        color_discrete_map=DECISION_COLOR_MAP,
        title="Scenario Decision Mix",
    )
    scenario_fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    scenario_fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
        xaxis_title="Scenario Decision",
        yaxis_title="Customer Count",
    )

    return html.Div(
        children=[
            html.Div(
                children=[
                    create_small_metric_card("Customers in Scenario", f"{total_customers:,}", "Filtered portfolio size", "#2563eb"),
                    create_small_metric_card("Scale Customers", f"{scale_count:,}", "Customers eligible for broad rollout", "#16a34a"),
                    create_small_metric_card("Test Customers", f"{test_count:,}", "Customers recommended for controlled testing", "#2563eb"),
                    create_small_metric_card("Blocked Customers", f"{block_count:,}", "Customers blocked by risk guardrails", "#dc2626"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "18px"},
            ),
            html.Div(
                children=[
                    create_small_metric_card("Estimated Incremental Profit", f"${total_profit:,.0f}", "Scenario-level campaign profit", "#7c3aed"),
                    create_small_metric_card("Average Scenario ROI", f"{avg_roi:.1f}x", "Average return per marketing dollar", "#0ea5e9"),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "18px"},
            ),
            dcc.Graph(figure=scenario_fig),
            html.Div(
                children=[
                    html.H4("Recommendation", style={"margin": "0 0 8px 0"}),
                    html.P(recommendation, style={"margin": "0", "lineHeight": "1.55"}),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "borderLeft": f"5px solid {rec_color}",
                    "borderRadius": "14px",
                    "padding": "18px",
                    "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
                },
            ),
        ]
    )


@app.callback(
    Output("ab-output", "children"),
    Input("ab-segment", "value"),
    Input("ab-baseline-rate", "value"),
    Input("ab-lift", "value"),
    Input("ab-test-population", "value"),
)
def update_ab_test_planner(segment: str, baseline_rate_percent: float, lift_pp: float, test_population: int):
    segment_df = customer_features[customer_features["customer_segment"] == segment].copy()
    segment_size = len(segment_df)

    available_population = min(segment_size, test_population)
    control_size = available_population // 2
    treatment_size = available_population - control_size

    baseline_rate = baseline_rate_percent / 100
    treatment_rate = baseline_rate + (lift_pp / 100)

    expected_control_responders = control_size * baseline_rate
    expected_treatment_responders = treatment_size * treatment_rate
    incremental_responders = expected_treatment_responders - (treatment_size * baseline_rate)

    # Approximate sample size per group for detecting difference in two proportions
    z_alpha = 1.96
    z_beta = 0.84
    p1 = baseline_rate
    p2 = treatment_rate
    effect = max(abs(p2 - p1), 0.001)
    pooled_p = (p1 + p2) / 2
    required_per_group = int(
        ((z_alpha + z_beta) ** 2)
        * (pooled_p * (1 - pooled_p) * 2)
        / (effect ** 2)
    )

    total_required = required_per_group * 2

    if available_population >= total_required:
        readiness = "Ready to test"
        rec_color = "#16a34a"
        recommendation = "The available population is large enough for this test assumption. Proceed with a controlled A/B test before scaling."
    elif available_population >= total_required * 0.6:
        readiness = "Directional test"
        rec_color = "#f97316"
        recommendation = "The test may provide directional learning, but it may not be strong enough for a high-confidence decision. Consider extending the test window or combining similar segments."
    else:
        readiness = "Underpowered"
        rec_color = "#dc2626"
        recommendation = "The available population is likely too small for this expected lift. Increase the test population, target a larger segment, or test a stronger offer."

    ab_mix_df = pd.DataFrame(
        {
            "Group": ["Control", "Treatment"],
            "Customers": [control_size, treatment_size],
            "Expected Responders": [expected_control_responders, expected_treatment_responders],
        }
    )

    ab_fig = px.bar(
        ab_mix_df,
        x="Group",
        y="Expected Responders",
        text="Expected Responders",
        title="Expected Responders by Test Group",
    )
    ab_fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", marker_color=["#9ca3af", "#2563eb"])
    ab_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
        xaxis_title="Test Group",
        yaxis_title="Expected Responders",
    )

    return html.Div(
        children=[
            html.Div(
                children=[
                    create_small_metric_card("Segment Size", f"{segment_size:,}", "Customers in selected segment", "#2563eb"),
                    create_small_metric_card("Test Population Used", f"{available_population:,}", "Customers assigned to test", "#0ea5e9"),
                    create_small_metric_card("Required Sample", f"{total_required:,}", "Approximate total sample needed", "#7c3aed"),
                    create_small_metric_card("Readiness", readiness, "Power check result", rec_color),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "18px"},
            ),
            html.Div(
                children=[
                    create_small_metric_card("Control Group", f"{control_size:,}", f"Expected responders: {expected_control_responders:.0f}", "#9ca3af"),
                    create_small_metric_card("Treatment Group", f"{treatment_size:,}", f"Expected responders: {expected_treatment_responders:.0f}", "#2563eb"),
                    create_small_metric_card("Incremental Responders", f"{incremental_responders:.0f}", "Additional responders from treatment", "#16a34a"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px", "marginBottom": "18px"},
            ),
            dcc.Graph(figure=ab_fig),
            html.Div(
                children=[
                    html.H4("Test Recommendation", style={"margin": "0 0 8px 0"}),
                    html.P(recommendation, style={"margin": "0", "lineHeight": "1.55"}),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "borderLeft": f"5px solid {rec_color}",
                    "borderRadius": "14px",
                    "padding": "18px",
                    "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.05)",
                },
            ),
        ]
    )





def apply_global_filters(selected_segments, selected_decisions, selected_risks, selected_actions) -> pd.DataFrame:
    filtered = customer_features.copy()

    if selected_segments:
        filtered = filtered[filtered["customer_segment"].isin(selected_segments)]

    if selected_decisions:
        filtered = filtered[filtered["decision_status"].isin(selected_decisions)]

    if selected_risks:
        filtered = filtered[filtered["risk_band"].isin(selected_risks)]

    if selected_actions:
        filtered = filtered[filtered["recommended_action"].isin(selected_actions)]

    return filtered


def empty_figure(title: str, message: str = "No customers match the selected filters"):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#6b7280"),
    )
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig


def build_decision_share_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Decision Status Share")

    counts = df["decision_status"].value_counts().reset_index()
    counts.columns = ["decision_status", "customer_count"]

    fig = px.pie(
        counts,
        names="decision_status",
        values="customer_count",
        hole=0.45,
        title="Decision Status Share",
        color="decision_status",
        color_discrete_map=DECISION_COLOR_MAP,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Arial", size=13, color="#1f2937"),
        legend_title_text="Decision",
    )
    return fig


def build_decision_count_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Decision Status Count")

    counts = df["decision_status"].value_counts().reset_index()
    counts.columns = ["decision_status", "customer_count"]

    fig = px.bar(
        counts,
        x="decision_status",
        y="customer_count",
        text="customer_count",
        title="Decision Status Count",
        color="decision_status",
        color_discrete_map=DECISION_COLOR_MAP,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        xaxis_title="Decision Status",
        yaxis_title="Customer Count",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig


def build_segment_size_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Customer Count by Segment")

    grouped = (
        df.groupby("customer_segment", as_index=False)
        .agg(customer_count=("customer_id", "count"))
        .sort_values("customer_count", ascending=True)
    )

    fig = px.bar(
        grouped,
        x="customer_count",
        y="customer_segment",
        orientation="h",
        text="customer_count",
        title="Customer Count by Segment",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#2563eb")
    fig.update_layout(
        xaxis_title="Customer Count",
        yaxis_title="Customer Segment",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig


def build_segment_eligibility_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Campaign Eligible Rate by Segment")

    grouped = (
        df.groupby("customer_segment", as_index=False)
        .agg(campaign_eligible_rate=("campaign_eligible_flag", "mean"))
        .sort_values("campaign_eligible_rate", ascending=True)
    )

    fig = px.bar(
        grouped,
        x="campaign_eligible_rate",
        y="customer_segment",
        orientation="h",
        text="campaign_eligible_rate",
        title="Campaign Eligible Rate by Segment",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", marker_color="#0ea5e9")
    fig.update_layout(
        xaxis_title="Eligible Rate",
        yaxis_title="Customer Segment",
        xaxis_tickformat=".0%",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig


def build_segment_decision_mix_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Decision Mix by Segment")

    grouped = (
        df.groupby(["customer_segment", "decision_status"], as_index=False)
        .agg(customer_count=("customer_id", "count"))
    )

    fig = px.bar(
        grouped,
        x="customer_segment",
        y="customer_count",
        color="decision_status",
        title="Decision Mix by Segment",
        color_discrete_map=DECISION_COLOR_MAP,
    )
    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Customer Count",
        xaxis_tickangle=-25,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=110),
        font=dict(family="Arial", size=13, color="#1f2937"),
        legend_title_text="Decision",
    )
    return fig


def build_action_mix_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Recommended Action Mix")

    counts = df["recommended_action"].value_counts().reset_index()
    counts.columns = ["recommended_action", "customer_count"]

    fig = px.pie(
        counts,
        names="recommended_action",
        values="customer_count",
        hole=0.42,
        title="Recommended Action Mix",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Arial", size=13, color="#1f2937"),
        legend_title_text="Action",
    )
    return fig


def build_offer_type_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Offer Type Distribution")

    counts = df["offer_type"].value_counts().reset_index()
    counts.columns = ["offer_type", "customer_count"]

    fig = px.bar(
        counts.sort_values("customer_count", ascending=True),
        x="customer_count",
        y="offer_type",
        orientation="h",
        text="customer_count",
        title="Offer Type Distribution",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#7c3aed")
    fig.update_layout(
        xaxis_title="Customer Count",
        yaxis_title="Offer Type",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig


def build_risk_band_fig(df: pd.DataFrame):
    if df.empty:
        return empty_figure("Risk Band Distribution")

    counts = df["risk_band"].value_counts().reset_index()
    counts.columns = ["risk_band", "customer_count"]

    fig = px.pie(
        counts,
        names="risk_band",
        values="customer_count",
        hole=0.45,
        title="Risk Band Distribution",
        color="risk_band",
        color_discrete_map=RISK_COLOR_MAP,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(family="Arial", size=13, color="#1f2937"),
        legend_title_text="Risk Band",
    )
    return fig


def build_high_utilization_fig(df: pd.DataFrame):
    filtered = df[df["customer_segment"] == "High-Utilization Revolver"]

    if filtered.empty:
        return empty_figure(
            "High-Utilization Revolver Decision Mix",
            "No High-Utilization Revolver customers match the selected filters",
        )

    counts = filtered["decision_status"].value_counts().reset_index()
    counts.columns = ["decision_status", "customer_count"]

    fig = px.bar(
        counts,
        x="decision_status",
        y="customer_count",
        text="customer_count",
        title="High-Utilization Revolver Decision Mix",
        color="decision_status",
        color_discrete_map=DECISION_COLOR_MAP,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        xaxis_title="Decision Status",
        yaxis_title="Customer Count",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig


def build_blocked_segment_fig(df: pd.DataFrame):
    filtered = df[df["decision_status"] == "Block"]

    if filtered.empty:
        return empty_figure(
            "Blocked Customers by Segment",
            "No blocked customers match the selected filters",
        )

    grouped = (
        filtered.groupby("customer_segment", as_index=False)
        .agg(blocked_customers=("customer_id", "count"))
        .sort_values("blocked_customers", ascending=True)
    )

    fig = px.bar(
        grouped,
        x="blocked_customers",
        y="customer_segment",
        orientation="h",
        text="blocked_customers",
        title="Blocked Customers by Segment",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#dc2626")
    fig.update_layout(
        xaxis_title="Blocked Customers",
        yaxis_title="Customer Segment",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=55, b=40),
        font=dict(family="Arial", size=13, color="#1f2937"),
    )
    return fig



@app.callback(
    Output("filtered-summary-output", "children"),
    Input("filter-segment", "value"),
    Input("filter-decision", "value"),
    Input("filter-risk", "value"),
    Input("filter-action", "value"),
)
def update_filtered_summary(selected_segments, selected_decisions, selected_risks, selected_actions):
    filtered = customer_features.copy()

    active_filters = []

    if selected_segments:
        filtered = filtered[filtered["customer_segment"].isin(selected_segments)]
        active_filters.append("Segment: " + ", ".join(selected_segments))

    if selected_decisions:
        filtered = filtered[filtered["decision_status"].isin(selected_decisions)]
        active_filters.append("Decision: " + ", ".join(selected_decisions))

    if selected_risks:
        filtered = filtered[filtered["risk_band"].isin(selected_risks)]
        active_filters.append("Risk: " + ", ".join(selected_risks))

    if selected_actions:
        filtered = filtered[filtered["recommended_action"].isin(selected_actions)]
        active_filters.append("Action: " + ", ".join(selected_actions))

    customer_count = len(filtered)

    if active_filters:
        summary_title = "Filtered Portfolio Summary"
        view_note = "Current view: " + " | ".join(active_filters)
        view_badge = "Filtered View"
        badge_color = "#2563eb"
    else:
        summary_title = "Full Portfolio Summary"
        view_note = "Current view: All customers in the portfolio"
        view_badge = "Full Portfolio View"
        badge_color = "#16a34a"

    if customer_count == 0:
        return html.Div(
            children=[
                html.H4("No customers match the selected filters", style={"margin": "0 0 6px 0"}),
                html.P(
                    "Try removing one or more filters to expand the portfolio view.",
                    style={"margin": "0", "color": COLORS["muted"]},
                ),
            ],
            style={
                "backgroundColor": "#fff7ed",
                "border": "1px solid #fed7aa",
                "borderRadius": "16px",
                "padding": "18px",
            },
        )

    total_spend = filtered["monthly_spend"].sum()
    total_profit = filtered["risk_adjusted_profit"].sum()
    avg_default_probability = filtered["default_probability"].mean()
    eligible_rate = filtered["campaign_eligible_flag"].mean()
    block_rate = (filtered["decision_status"] == "Block").mean()

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                summary_title,
                                style={
                                    "fontSize": "13px",
                                    "fontWeight": "900",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px",
                                    "color": COLORS["blue"],
                                },
                            ),
                            html.Div(
                                view_badge,
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "900",
                                    "color": "white",
                                    "backgroundColor": badge_color,
                                    "borderRadius": "999px",
                                    "padding": "6px 10px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "space-between",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        view_note,
                        style={
                            "fontSize": "13px",
                            "color": COLORS["muted"],
                            "marginBottom": "14px",
                            "lineHeight": "1.45",
                        },
                    ),
                    html.Div(
                        children=[
                            create_small_metric_card(
                                "Filtered Customers",
                                f"{customer_count:,}",
                                "Customers matching current filters",
                                "#2563eb",
                            ),
                            create_small_metric_card(
                                "Monthly Spend",
                                f"${total_spend:,.0f}",
                                "Spend from filtered customers",
                                "#0ea5e9",
                            ),
                            create_small_metric_card(
                                "Risk-Adjusted Profit",
                                f"${total_profit:,.0f}",
                                "Profit after expected credit loss",
                                "#16a34a",
                            ),
                            create_small_metric_card(
                                "Avg Default Probability",
                                f"{avg_default_probability * 100:.2f}%",
                                "Average risk level in filtered view",
                                "#f97316",
                            ),
                            create_small_metric_card(
                                "Eligible Rate",
                                f"{eligible_rate * 100:.1f}%",
                                "Share marked Scale or Test",
                                "#7c3aed",
                            ),
                            create_small_metric_card(
                                "Block Rate",
                                f"{block_rate * 100:.1f}%",
                                "Share blocked by guardrails",
                                "#dc2626",
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(6, 1fr)",
                            "gap": "12px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "20px",
                    "padding": "20px",
                    "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                },
            )
        ]
    )



if __name__ == "__main__":
    app.run(debug=True)
