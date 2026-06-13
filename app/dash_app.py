from __future__ import annotations

import sys

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_engine import score_customer_portfolio
import base64
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, dash_table, no_update
from dash.exceptions import PreventUpdate


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


def help_label(text: str, help_text: str) -> html.Span:
    return html.Span(
        [
            text,
            html.Span(
                " ⓘ",
                style={
                    "fontSize": "12px",
                    "color": COLORS["muted"],
                    "fontWeight": "800",
                },
            ),
        ],
        title=help_text,
        style={
            "fontWeight": "800",
            "cursor": "help",
            "display": "inline-block",
        },
    )


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
                            "whiteSpace": "normal" if column in ["Campaign", "Family", "Rollout"] else "nowrap",
                            "maxWidth": "240px" if column in ["Campaign", "Family", "Rollout"] else "none",
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
    """Build a compact campaign audit table from campaign-level recommendation output."""
    if campaign_recommendations.empty:
        return []

    table = campaign_recommendations.copy()

    fallback_columns = {
        "dashboard_recommendation_rank": 0,
        "campaign_name": "Unknown campaign",
        "campaign_family": "Unknown family",
        "risk_level": "Unknown",
        "recommended_rollout_decision": "Review",
        "eligible_customers": 0,
        "scale_customers": 0,
        "blocked_customers": 0,
        "expected_campaign_profit": 0,
        "expected_campaign_roi": 0,
        "campaign_score": 0,
    }

    for column, default in fallback_columns.items():
        if column not in table.columns:
            table[column] = default

    table = table.head(limit).copy()

    table["Rank"] = table["dashboard_recommendation_rank"].apply(lambda value: str(int(float(value))) if str(value).replace(".", "", 1).isdigit() else str(value))
    table["Campaign"] = table["campaign_name"].astype(str)
    table["Family"] = table["campaign_family"].astype(str)
    table["Rollout"] = table["recommended_rollout_decision"].astype(str)
    table["Risk"] = table["risk_level"].astype(str)
    table["Profit"] = pd.to_numeric(table["expected_campaign_profit"], errors="coerce").fillna(0).apply(format_currency)
    table["ROI"] = pd.to_numeric(table["expected_campaign_roi"], errors="coerce").fillna(0).apply(lambda value: f"{value:.2f}x")
    table["Matches"] = pd.to_numeric(table["eligible_customers"], errors="coerce").fillna(0).astype(int).astype(str)
    table["Scale"] = pd.to_numeric(table["scale_customers"], errors="coerce").fillna(0).astype(int).astype(str)
    table["Blocked"] = pd.to_numeric(table["blocked_customers"], errors="coerce").fillna(0).astype(int).astype(str)
    table["Score"] = pd.to_numeric(table["campaign_score"], errors="coerce").fillna(0).apply(lambda value: f"{value:.1f}")

    return table[
        [
            "Rank",
            "Campaign",
            "Family",
            "Rollout",
            "Risk",
            "Profit",
            "ROI",
            "Matches",
            "Scale",
            "Blocked",
            "Score",
        ]
    ].to_dict("records")

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
    title="Treatment Type Distribution",
)
offer_fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_color="#7c3aed")
offer_fig.update_layout(
    xaxis_title="Customer Count",
    yaxis_title="Treatment Type",
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





def split_semicolon_values(value: str) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    return [item.strip() for item in str(value).split(";") if item.strip()]


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
        "Controlled Servicing": "#0ea5e9",
        "Do Not Launch": "#64748b",
        "Block": "#dc2626",
    }

    accent = rollout_colors.get(rollout, "#2563eb")

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        f"#{int(row.get('rank', row.get('dashboard_recommendation_rank', 0)))}",
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
                    "margin": "0 0 10px 0",
                },
            ),
            html.Div(
                children=[
                    html.Strong("Active rollout logic: "),
                    row.get("active_rollout_reason", "Rollout is inferred from the active master dataset."),
                ],
                style={
                    "fontSize": "12px",
                    "lineHeight": "1.45",
                    "color": "#1e3a8a",
                    "backgroundColor": "#eff6ff",
                    "border": "1px solid #bfdbfe",
                    "borderRadius": "10px",
                    "padding": "9px 10px",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                children=[
                    create_metric_chip("Matches", format_large_number(row.get("eligible_customers", 0))),
                    create_metric_chip("Scale Matches", format_large_number(row.get("scale_customers", 0))),
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
                    create_metric_chip("Active Matches", format_large_number(top_campaign.get("eligible_customers", 0))),
                    create_metric_chip("Scale / Blocked", f"{int(top_campaign.get('scale_customers', 0))} / {int(top_campaign.get('blocked_customers', 0))}"),
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
                        "Active Rollout Logic",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "900",
                            "color": "#1e3a8a",
                            "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        top_campaign.get("active_rollout_reason", "Rollout is inferred from active customer matches and guardrails."),
                        style={
                            "fontSize": "13px",
                            "lineHeight": "1.5",
                            "color": "#1e3a8a",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#eff6ff",
                    "border": "1px solid #bfdbfe",
                    "borderRadius": "14px",
                    "padding": "14px",
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



def create_color_legend() -> html.Details:
    legend_items = [
        ("Scale", "#16a34a", "Ready for broader rollout when economics and risk are acceptable."),
        ("Test", "#2563eb", "Run a controlled experiment before scaling."),
        ("Constrain", "#f97316", "Potential opportunity, but rollout should be limited by risk or uncertainty."),
        ("Controlled Servicing", "#0ea5e9", "Servicing or protective campaign; avoid aggressive growth framing."),
        ("Block", "#dc2626", "Do not target because guardrails or risk rules are triggered."),
        ("Do Not Launch", "#9ca3af", "Not enough upside or not a strong campaign fit."),
    ]

    return html.Details(
        children=[
            html.Summary(
                "Decision legend: what the colors mean",
                style={
                    "cursor": "pointer",
                    "fontWeight": "900",
                    "fontSize": "15px",
                    "color": COLORS["text"],
                    "padding": "12px 14px",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                style={
                                    "width": "10px",
                                    "height": "10px",
                                    "borderRadius": "999px",
                                    "backgroundColor": color,
                                    "flexShrink": "0",
                                    "marginTop": "4px",
                                },
                            ),
                            html.Div(
                                children=[
                                    html.Div(label, style={"fontWeight": "900", "fontSize": "13px"}),
                                    html.Div(description, style={"fontSize": "12px", "color": COLORS["muted"], "lineHeight": "1.35"}),
                                ],
                            ),
                        ],
                        title=description,
                        style={
                            "display": "flex",
                            "gap": "9px",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "12px",
                            "padding": "10px",
                            "backgroundColor": "#ffffff",
                        },
                    )
                    for label, color, description in legend_items
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(5, 1fr)",
                    "gap": "10px",
                    "padding": "0 14px 14px 14px",
                },
            ),
        ],
        open=False,
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "16px",
            "marginBottom": "12px",
            "boxShadow": "0 6px 16px rgba(15, 23, 42, 0.04)",
        },
    )


def create_workflow_guide() -> html.Details:
    steps = [
        ("1", "Review portfolio", "Start with Overview and Segment Strategy.", "Open Overview", "guide-open-overview"),
        ("2", "Choose campaign", "Use Campaigns & Offers to pick a recommended campaign.", "Open Campaigns", "guide-open-campaigns"),
        ("3", "Simulate impact", "Use Scenario Simulator to test cost, lift, risk, and profit.", "Open Scenario", "guide-open-scenario"),
        ("4", "Design experiment", "Use A/B Planner to size control and treatment groups.", "Open A/B Planner", "guide-open-ab"),
        ("5", "Export customers", "Download eligible, test, scale, or blocked audiences.", "Open Export", "guide-open-export"),
        ("6", "Check guardrails", "Review risk controls before rollout.", "Open Guardrails", "guide-open-guardrails"),
    ]

    return html.Details(
        children=[
            html.Summary(
                "How to use this dashboard",
                title="Expand this section for a step-by-step workflow across the dashboard.",
                style={
                    "cursor": "pointer",
                    "fontWeight": "900",
                    "fontSize": "15px",
                    "color": COLORS["text"],
                    "padding": "12px 14px",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                number,
                                style={
                                    "width": "24px",
                                    "height": "24px",
                                    "borderRadius": "999px",
                                    "backgroundColor": COLORS["blue"],
                                    "color": "white",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "fontWeight": "900",
                                    "fontSize": "12px",
                                    "marginBottom": "8px",
                                },
                            ),
                            html.Div(title, style={"fontWeight": "900", "fontSize": "13px", "marginBottom": "4px"}),
                            html.Div(detail, style={"fontSize": "12px", "color": COLORS["muted"], "lineHeight": "1.35", "marginBottom": "10px"}),
                            html.Button(
                                button_text,
                                id=button_id,
                                n_clicks=0,
                                title=f"Go to: {title}",
                                style={
                                    "border": "none",
                                    "borderRadius": "10px",
                                    "backgroundColor": COLORS["blue"],
                                    "color": "white",
                                    "fontWeight": "900",
                                    "fontSize": "12px",
                                    "padding": "8px 10px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        title=detail,
                        style={
                            "backgroundColor": "#f8fafc",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "12px",
                            "padding": "10px",
                        },
                    )
                    for number, title, detail, button_text, button_id in steps
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(6, 1fr)",
                    "gap": "10px",
                    "padding": "0 14px 14px 14px",
                },
            ),
        ],
        open=False,
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "16px",
            "marginBottom": "12px",
            "boxShadow": "0 6px 16px rgba(15, 23, 42, 0.04)",
        },
    )


def create_action_prompt_panel() -> html.Details:
    actions = [
        ("Campaign decision", "Use Campaigns & Offers to decide which campaign deserves scale, test, or constraint.", "Open Campaigns", "action-open-campaigns"),
        ("Customer action", "Use Customer 360 and Export Center to inspect who is included before campaign execution.", "Open Customer Tools", "action-open-customer-tools"),
        ("Risk action", "Use Guardrails before launch to avoid risky growth and protect sensitive audiences.", "Open Guardrails", "action-open-guardrails"),
    ]

    return html.Details(
        children=[
            html.Summary(
                "What should I do next?",
                title="Expand this section for quick actions based on the current workflow.",
                style={
                    "cursor": "pointer",
                    "fontWeight": "900",
                    "fontSize": "15px",
                    "color": COLORS["text"],
                    "padding": "12px 14px",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(title, style={"fontWeight": "900", "fontSize": "13px", "marginBottom": "4px"}),
                            html.Div(body, style={"fontSize": "12px", "color": COLORS["muted"], "lineHeight": "1.35", "marginBottom": "10px"}),
                            html.Button(
                                button_text,
                                id=button_id,
                                n_clicks=0,
                                title=body,
                                style={
                                    "border": "none",
                                    "borderRadius": "10px",
                                    "backgroundColor": COLORS["blue"],
                                    "color": "white",
                                    "fontWeight": "900",
                                    "fontSize": "12px",
                                    "padding": "8px 10px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        title=body,
                        style={
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "12px",
                            "padding": "10px",
                        },
                    )
                    for title, body, button_text, button_id in actions
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "10px",
                    "padding": "0 14px 14px 14px",
                },
            ),
        ],
        open=False,
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "16px",
            "marginBottom": "12px",
            "boxShadow": "0 6px 16px rgba(15, 23, 42, 0.04)",
        },
    )




def create_strategy_flow_diagram() -> html.Div:
    steps = [
        {
            "title": "Customer Portfolio",
            "body": "Active cardholders from either the synthetic demo portfolio or the uploaded customer file.",
            "accent": "#2563eb",
        },
        {
            "title": "Segment Strategy",
            "body": "Group customers into active portfolio segments using spend, credit risk, engagement, and profitability signals.",
            "accent": "#0ea5e9",
        },
        {
            "title": "Campaign Library",
            "body": "Match segments to growth, servicing, retention, merchant, or protective campaigns.",
            "accent": "#7c3aed",
        },
        {
            "title": "Scenario Simulation",
            "body": "Test cost, lift, risk tolerance, ROI, profit, and campaign economics.",
            "accent": "#f97316",
        },
        {
            "title": "A/B Test Design",
            "body": "Split control/treatment groups and check whether the experiment is powered.",
            "accent": "#16a34a",
        },
        {
            "title": "Customer Export",
            "body": "Download eligible, test, scale, or blocked customer lists for action.",
            "accent": "#059669",
        },
        {
            "title": "Guardrail Review",
            "body": "Final check before launch to avoid risky growth and protect customers.",
            "accent": "#dc2626",
        },
    ]

    diagram_items = []

    for idx, step in enumerate(steps):
        diagram_items.append(
            html.Div(
                children=[
                    html.Div(
                        step["title"],
                        style={
                            "fontSize": "14px",
                            "fontWeight": "900",
                            "color": COLORS["text"],
                            "marginBottom": "6px",
                        },
                    ),
                    html.Div(
                        step["body"],
                        style={
                            "fontSize": "12px",
                            "lineHeight": "1.35",
                            "color": COLORS["muted"],
                        },
                    ),
                ],
                title=step["body"],
                style={
                    "backgroundColor": "#ffffff",
                    "border": f"1px solid {COLORS['border']}",
                    "borderTop": f"5px solid {step['accent']}",
                    "borderRadius": "14px",
                    "padding": "12px",
                    "minHeight": "118px",
                },
            )
        )

        if idx < len(steps) - 1:
            diagram_items.append(
                html.Div(
                    "→",
                    style={
                        "fontSize": "26px",
                        "fontWeight": "900",
                        "color": COLORS["muted"],
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                )
            )

    return html.Div(
        children=[
            html.H3(
                "Decision Flow Sketch",
                style={"fontSize": "22px", "fontWeight": "900", "margin": "0 0 8px 0"},
            ),
            html.P(
                "This sketch shows how the dashboard moves from raw portfolio insight to campaign execution and final risk review.",
                style={"color": COLORS["muted"], "lineHeight": "1.5", "margin": "0 0 16px 0"},
            ),
            html.Div(
                children=diagram_items,
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.2fr 32px 1.2fr 32px 1.2fr 32px 1.2fr 32px 1.2fr 32px 1.2fr 32px 1.2fr",
                    "gap": "8px",
                    "alignItems": "stretch",
                },
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "22px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
            "overflowX": "auto",
        },
    )


def build_strategy_risk_return_figure():
    plot_df = segment_summary.copy()

    decision_map = {
        "Core Customer": "Scale",
        "Loyal High-Value Customer": "Scale",
        "Premium Growth Candidate": "Scale / Test",
        "High-Utilization Revolver": "Test / Constrain",
        "Underused Low-Risk Customer": "Test",
        "Dormant but Recoverable": "Test",
        "Risk Watch": "Block",
    }

    plot_df["strategy_decision"] = plot_df["customer_segment"].map(decision_map).fillna("Review")
    plot_df["avg_default_probability_pct"] = plot_df["avg_default_probability"] * 100

    fig = px.scatter(
        plot_df,
        x="avg_default_probability_pct",
        y="avg_risk_adjusted_profit",
        size="customer_count",
        color="strategy_decision",
        hover_name="customer_segment",
        hover_data={
            "customer_count": ":,",
            "avg_default_probability_pct": ":.2f",
            "avg_risk_adjusted_profit": ":.2f",
            "avg_expected_roi": ":.2f",
            "campaign_eligible_rate": ":.2%",
            "strategy_decision": True,
        },
        title="Segment Risk-Return Matrix",
        labels={
            "avg_default_probability_pct": "Average default probability (%)",
            "avg_risk_adjusted_profit": "Average risk-adjusted profit",
            "strategy_decision": "Strategy decision",
            "customer_count": "Customer count",
        },
        size_max=58,
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text="Profit break-even",
        annotation_position="bottom right",
    )

    fig.update_layout(
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=30),
        legend_title_text="Strategy decision",
        font=dict(family="Arial", size=12, color=COLORS["text"]),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickprefix="$")

    return fig



def strategy_placeholder_figure(message: str):
    """Small placeholder figure for Strategy Playbook before callbacks load."""
    fig = px.scatter(title=message)
    fig.update_layout(
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=30),
        font=dict(family="Arial", size=12, color=COLORS["text"]),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=15, color=COLORS["muted"]),
            )
        ],
    )
    return fig


def create_strategy_risk_return_section() -> html.Div:
    return html.Div(
        children=[
            html.H3(
                "Active Segment Risk-Return Matrix",
                style={"fontSize": "22px", "fontWeight": "900", "margin": "0 0 8px 0"},
            ),
            html.P(
                "Each bubble is a customer segment in the active master dataset. Higher profit is better, but higher default probability requires more control, testing, or guardrails.",
                style={"color": COLORS["muted"], "lineHeight": "1.5", "margin": "0 0 14px 0"},
            ),
            dcc.Graph(
                id="strategy-risk-return-chart",
                figure=strategy_placeholder_figure("Strategy risk-return chart will load from the active master dataset."),
                config={"displayModeBar": True},
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "22px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
        },
    )


def create_strategy_playbook_table() -> html.Div:
    return html.Div(
        children=[
            html.H3(
                "Active Segment Strategy Playbook",
                style={"fontSize": "22px", "fontWeight": "900", "margin": "0 0 8px 0"},
            ),
            html.P(
                "These strategy cards are rebuilt from the active master dataset. They show which segments exist, how valuable/risky they are, what strategy should be used, and what action should happen next.",
                style={"color": COLORS["muted"], "lineHeight": "1.5", "margin": "0 0 16px 0"},
            ),
            html.Div(
                id="strategy-playbook-table-container",
                children=html.Div(
                    "Strategy table will load from the active master dataset.",
                    style={"color": COLORS["muted"]},
                ),
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "22px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
            "overflowX": "auto",
        },
    )


def create_strategy_cta_panel() -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        "ACTIVE PORTFOLIO STRATEGY",
                        style={
                            "fontSize": "12px",
                            "fontWeight": "900",
                            "letterSpacing": "0.08em",
                            "color": COLORS["blue"],
                            "textTransform": "uppercase",
                            "marginBottom": "8px",
                        },
                    ),
                    html.H3(
                        "What should the business do next?",
                        style={"fontSize": "24px", "fontWeight": "900", "margin": "0 0 8px 0"},
                    ),
                    html.P(
                        "This section reads the active master dataset and turns the portfolio into a practical next-step strategy. When a customer file is uploaded, the playbook recalculates from that uploaded portfolio.",
                        style={"color": COLORS["muted"], "lineHeight": "1.5", "margin": "0"},
                    ),
                ]
            ),
            html.Div(
                id="strategy-executive-recommendation",
                children=html.Div(
                    "Strategy recommendations will load from the active master dataset.",
                    style={"color": COLORS["muted"], "marginTop": "14px"},
                ),
                style={"marginTop": "16px"},
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "22px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
        },
    )


def create_governance_layout() -> html.Div:
    governance_cards = [
        {
            "title": "Synthetic data only",
            "body": "The dashboard uses synthetic customer records for portfolio analytics demonstration. No real customer data, account data, or bank-confidential data is used.",
            "accent": "#2563eb",
        },
        {
            "title": "Not a credit approval model",
            "body": "This engine is not used to approve, deny, price, or underwrite credit. It is a campaign decisioning prototype for existing cardholder engagement.",
            "accent": "#dc2626",
        },
        {
            "title": "Campaign decisioning only",
            "body": "The model recommends campaign posture such as Scale, Test, Constrain, Do Not Launch, or Block for marketing and portfolio strategy use cases.",
            "accent": "#7c3aed",
        },
        {
            "title": "Human review required",
            "body": "Any real deployment would require review by risk, compliance, marketing, product, legal, and model governance teams before launch.",
            "accent": "#f97316",
        },
        {
            "title": "Fairness testing required",
            "body": "Before production, the bank would need fairness and adverse-impact testing across protected and sensitive groups using approved governance methods.",
            "accent": "#16a34a",
        },
        {
            "title": "Monitoring and drift checks",
            "body": "Campaign performance, risk distribution, default behavior, response rate, ROI, and segment drift should be monitored after deployment.",
            "accent": "#0ea5e9",
        },
    ]

    control_rows = [
        {
            "control": "Input validation",
            "purpose": "Check required fields, missing values, outliers, and valid ranges before scoring.",
            "status": "Prototype-ready",
            "owner": "Data / Analytics",
        },
        {
            "control": "Risk guardrails",
            "purpose": "Prevent aggressive growth campaigns from reaching high-risk or high-stress customers.",
            "status": "Implemented",
            "owner": "Credit Risk",
        },
        {
            "control": "Campaign eligibility rules",
            "purpose": "Match campaigns to the right segments while excluding risky or unsuitable audiences.",
            "status": "Implemented",
            "owner": "Portfolio Marketing",
        },
        {
            "control": "A/B test validation",
            "purpose": "Require controlled experiments before scaling uncertain campaigns.",
            "status": "Implemented",
            "owner": "Experimentation / Product",
        },
        {
            "control": "Fairness and bias review",
            "purpose": "Evaluate whether campaign eligibility or treatment decisions create unfair outcomes.",
            "status": "Required before production",
            "owner": "Model Risk / Compliance",
        },
        {
            "control": "Model documentation",
            "purpose": "Document inputs, assumptions, rules, thresholds, limitations, and approval workflow.",
            "status": "Needed for final project docs",
            "owner": "Analytics / Governance",
        },
        {
            "control": "Performance monitoring",
            "purpose": "Track response rate, lift, profit, losses, customer complaints, and risk drift after launch.",
            "status": "Recommended next",
            "owner": "Business + Risk",
        },
        {
            "control": "Audit trail",
            "purpose": "Preserve campaign version, scoring date, selected audience, decision rules, and approval record.",
            "status": "Recommended next",
            "owner": "Governance / Operations",
        },
    ]

    header_style = {
        "padding": "12px",
        "fontSize": "12px",
        "fontWeight": "900",
        "color": COLORS["muted"],
        "textTransform": "uppercase",
        "borderBottom": f"1px solid {COLORS['border']}",
        "backgroundColor": "#f8fafc",
        "textAlign": "left",
    }

    cell_style = {
        "padding": "12px",
        "fontSize": "13px",
        "lineHeight": "1.4",
        "borderBottom": f"1px solid {COLORS['border']}",
        "verticalAlign": "top",
    }

    return html.Div(
        children=[
            create_tab_intro(
                "Governance & Responsible Use",
                "This page explains how the decision engine should be controlled in a banking environment. It documents what the prototype is, what it is not, and which checks are required before any real-world deployment.",
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                style={
                                    "height": "5px",
                                    "width": "44px",
                                    "backgroundColor": card["accent"],
                                    "borderRadius": "999px",
                                    "marginBottom": "12px",
                                },
                            ),
                            html.H4(
                                card["title"],
                                style={"margin": "0 0 8px 0", "fontSize": "16px", "fontWeight": "900"},
                            ),
                            html.P(
                                card["body"],
                                style={"margin": "0", "fontSize": "13px", "lineHeight": "1.5", "color": COLORS["muted"]},
                            ),
                        ],
                        title=card["body"],
                        style={
                            "backgroundColor": COLORS["card"],
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "16px",
                            "padding": "16px",
                            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                        },
                    )
                    for card in governance_cards
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(3, 1fr)",
                    "gap": "16px",
                    "marginTop": "18px",
                },
            ),

            html.Div(
                children=[
                    html.H3(
                        "Model Controls Checklist",
                        style={"fontSize": "22px", "fontWeight": "900", "margin": "0 0 8px 0"},
                    ),
                    html.P(
                        "These controls show how the prototype would need to be governed before being used in a real financial institution.",
                        style={"color": COLORS["muted"], "lineHeight": "1.5", "margin": "0 0 16px 0"},
                    ),
                    html.Table(
                        children=[
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Control", style=header_style),
                                        html.Th("Purpose", style=header_style),
                                        html.Th("Status", style=header_style),
                                        html.Th("Owner", style=header_style),
                                    ]
                                )
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        children=[
                                            html.Td(row["control"], style={**cell_style, "fontWeight": "900"}),
                                            html.Td(row["purpose"], style=cell_style),
                                            html.Td(row["status"], style=cell_style),
                                            html.Td(row["owner"], style=cell_style),
                                        ],
                                        title=f"{row['control']}: {row['purpose']}",
                                    )
                                    for row in control_rows
                                ]
                            ),
                        ],
                        style={
                            "width": "100%",
                            "borderCollapse": "collapse",
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "14px",
                            "overflow": "hidden",
                        },
                    ),
                ],
                style={
                    "backgroundColor": COLORS["card"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "18px",
                    "padding": "22px",
                    "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                    "marginTop": "18px",
                    "overflowX": "auto",
                },
            ),

            create_insight_card(
                "Production-readiness note",
                "For millions of customers, this prototype should move scoring and filtering out of the Dash front end and into a scalable data layer such as SQL, Spark, Snowflake, Databricks, or a batch scoring pipeline. The dashboard should query summarized and paginated outputs instead of loading every row into memory.",
                variant="warning",
            ),
        ]
    )




def create_compact_governance_panel() -> html.Details:
    governance_items = [
        ("Synthetic data only", "No real customer, account, or bank-confidential data is used."),
        ("Not credit approval", "This is campaign decisioning for existing cardholders, not underwriting, pricing, approval, or denial."),
        ("Human review required", "Risk, compliance, product, marketing, legal, and model governance teams would review before production."),
        ("Fairness review required", "Real deployment would require fairness and adverse-impact testing using approved governance methods."),
        ("Monitoring required", "Track response rate, lift, losses, complaints, ROI, risk drift, and segment drift after launch."),
        ("Scalable production design", "For millions of customers, scoring and filtering should move to SQL/Spark/Snowflake/Databricks or a batch pipeline."),
    ]

    controls = [
        ("Input validation", "Required fields, missing values, ranges, and outliers."),
        ("Risk guardrails", "Prevent aggressive growth offers for high-risk or high-stress customers."),
        ("Campaign eligibility rules", "Match campaigns to suitable audiences and exclude unsuitable ones."),
        ("Experiment validation", "Use A/B testing before scaling uncertain campaigns."),
        ("Audit trail", "Preserve campaign version, scoring date, audience, rules, and approval record."),
    ]

    return html.Details(
        children=[
            html.Summary(
                "Governance & responsible-use checklist",
                title="Expand this section to review responsible-use, compliance, and production-readiness controls.",
                style={
                    "fontSize": "18px",
                    "fontWeight": "900",
                    "cursor": "pointer",
                    "color": COLORS["text"],
                    "padding": "4px 0",
                },
            ),
            html.P(
                "Use this as the final control checklist before any campaign is launched. It keeps the dashboard workflow clean while still documenting responsible use.",
                style={"color": COLORS["muted"], "lineHeight": "1.5", "margin": "12px 0 16px 0"},
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(title, style={"fontWeight": "900", "fontSize": "14px", "marginBottom": "6px"}),
                            html.Div(body, style={"color": COLORS["muted"], "fontSize": "13px", "lineHeight": "1.45"}),
                        ],
                        title=body,
                        style={
                            "backgroundColor": "#ffffff",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "14px",
                            "padding": "14px",
                        },
                    )
                    for title, body in governance_items
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "12px"},
            ),
            html.Div(
                children=[
                    html.H4(
                        "Production controls",
                        style={"fontSize": "16px", "fontWeight": "900", "margin": "18px 0 10px 0"},
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.Span(control, style={"fontWeight": "900"}),
                                    html.Span(f" — {purpose}", style={"color": COLORS["muted"]}),
                                ],
                                title=purpose,
                                style={
                                    "padding": "10px 12px",
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "12px",
                                    "backgroundColor": "#f8fafc",
                                    "fontSize": "13px",
                                },
                            )
                            for control, purpose in controls
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"},
                    ),
                ]
            ),
        ],
        open=False,
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
            "marginBottom": "18px",
        },
    )




def create_workflow_cta_panel(title: str, description: str, buttons: list[dict]) -> html.Div:
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.H3(
                        title,
                        style={"margin": "0 0 6px 0", "fontSize": "20px", "fontWeight": "900"},
                    ),
                    html.P(
                        description,
                        style={"margin": "0", "color": COLORS["muted"], "lineHeight": "1.5"},
                    ),
                ],
                style={"marginBottom": "14px"},
            ),
            html.Div(
                children=[
                    html.Button(
                        button["label"],
                        id=button["id"],
                        n_clicks=0,
                        title=button.get("title", button["label"]),
                        style={
                            "backgroundColor": button.get("color", COLORS["blue"]),
                            "color": "white",
                            "border": "none",
                            "borderRadius": "12px",
                            "padding": "10px 14px",
                            "fontWeight": "900",
                            "cursor": "pointer",
                            "minWidth": "150px",
                            "boxShadow": "0 6px 14px rgba(37, 99, 235, 0.18)",
                        },
                    )
                    for button in buttons
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
            "marginBottom": "18px",
        },
    )


def create_campaigns_action_panel() -> html.Div:
    return create_workflow_cta_panel(
        "Recommended next actions",
        "After selecting a campaign, move into simulation, experiment design, customer export, or final guardrail review.",
        [
            {
                "label": "Open Scenario",
                "id": "cta-open-scenario",
                "title": "Go to the Scenario Simulator to test cost, lift, risk threshold, ROI, and profit impact.",
                "color": "#2563eb",
            },
            {
                "label": "Open A/B Planner",
                "id": "cta-open-ab",
                "title": "Go to the A/B Test Planner to size control and treatment groups before rollout.",
                "color": "#7c3aed",
            },
            {
                "label": "Open Audience Export",
                "id": "cta-open-audience",
                "title": "Go to Audience Explorer to filter and download the customer list.",
                "color": "#16a34a",
            },
            {
                "label": "Review Guardrails",
                "id": "cta-open-guardrails",
                "title": "Go to Guardrails for risk and responsible-use review before launch.",
                "color": "#dc2626",
            },
        ],
    )


def create_playbook_action_panel() -> html.Div:
    return create_workflow_cta_panel(
        "Use the playbook to take action",
        "Use these shortcuts to move from active strategy into execution.",
        [
            {
                "label": "Choose Campaign",
                "id": "cta-playbook-campaigns",
                "title": "Open Campaigns & Offers to choose a recommended campaign from the library.",
                "color": "#2563eb",
            },
            {
                "label": "Simulate Impact",
                "id": "cta-playbook-scenario",
                "title": "Open Scenario Simulator to test campaign assumptions.",
                "color": "#f97316",
            },
            {
                "label": "Export Audience",
                "id": "cta-playbook-audience",
                "title": "Open Audience Explorer to export eligible, scale, test, or blocked customers.",
                "color": "#16a34a",
            },
            {
                "label": "Final Risk Review",
                "id": "cta-playbook-guardrails",
                "title": "Open Guardrails as the final review before launch.",
                "color": "#dc2626",
            },
        ],
    )


def create_guardrails_action_panel() -> html.Div:
    return create_workflow_cta_panel(
        "Guardrail follow-up actions",
        "Use these shortcuts when a campaign needs review, export, or additional testing before rollout.",
        [
            {
                "label": "View Risk Audience",
                "id": "cta-guardrails-audience",
                "title": "Open Audience Explorer to inspect blocked, high-risk, or constrained customer groups.",
                "color": "#16a34a",
            },
            {
                "label": "Back to Playbook",
                "id": "cta-guardrails-playbook",
                "title": "Return to Strategy Playbook to review segment-level recommendations.",
                "color": "#2563eb",
            },
            {
                "label": "Test Campaign",
                "id": "cta-guardrails-ab",
                "title": "Open A/B Test Planner to design a controlled experiment before rollout.",
                "color": "#7c3aed",
            },
        ],
    )





REQUIRED_UPLOAD_COLUMNS = [
    "customer_id",
    "age",
    "income",
    "credit_score",
    "credit_limit",
    "current_balance",
    "monthly_spend",
    "transactions_count",
    "customer_tenure_months",
    "late_payments_12m",
    "revolving_balance",
]


OPTIONAL_UPLOAD_COLUMNS = [
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
    "grocery_spend",
    "dining_spend",
    "travel_spend",
    "gas_spend",
    "online_spend",
]


DERIVED_ENGINE_COLUMNS = [
    "utilization_rate",
    "default_probability",
    "risk_band",
    "customer_segment",
    "risk_adjusted_profit",
    "expected_roi",
    "recommended_action",
    "offer_type",
    "decision_status",
    "campaign_eligible_flag",
    "risk_guardrail_flag",
]


def build_schema_template_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customer_id": "100001",
                "age": 34,
                "income": 85000,
                "credit_score": 710,
                "credit_limit": 12000,
                "current_balance": 2800,
                "monthly_spend": 3200,
                "transactions_count": 42,
                "customer_tenure_months": 36,
                "late_payments_12m": 0,
                "revolving_balance": 1400,
                "customer_name": "Demo Customer One",
                "customer_email": "demo.customer.one@example.com",
                "phone_number": "555-000-1001",
                "city": "Charlotte",
                "state": "NC",
                "zip_code": "28202",
                "employment_status": "Full-Time",
                "occupation_group": "Technology",
                "preferred_channel": "Email",
                "relationship_tier": "Gold",
                "signup_channel": "Mobile App",
                "account_open_date": "2022-04-15",
                "digital_engagement_score": 82,
                "last_app_login_days": 4,
                "autopay_enrolled": "Yes",
                "paperless_enrolled": "Yes",
                "card_type": "Platinum",
                "rewards_preference": "Cashback",
                "grocery_spend": 620,
                "dining_spend": 410,
                "travel_spend": 300,
                "gas_spend": 180,
                "online_spend": 720,
            },
            {
                "customer_id": "100002",
                "age": 46,
                "income": 112000,
                "credit_score": 675,
                "credit_limit": 18000,
                "current_balance": 9200,
                "monthly_spend": 4100,
                "transactions_count": 55,
                "customer_tenure_months": 72,
                "late_payments_12m": 1,
                "revolving_balance": 6800,
                "customer_name": "Demo Customer Two",
                "customer_email": "demo.customer.two@example.com",
                "phone_number": "555-000-1002",
                "city": "Boston",
                "state": "MA",
                "zip_code": "02115",
                "employment_status": "Full-Time",
                "occupation_group": "Healthcare",
                "preferred_channel": "Mobile App",
                "relationship_tier": "Standard",
                "signup_channel": "Branch",
                "account_open_date": "2020-08-20",
                "digital_engagement_score": 64,
                "last_app_login_days": 18,
                "autopay_enrolled": "No",
                "paperless_enrolled": "Yes",
                "card_type": "Quicksilver",
                "rewards_preference": "Dining",
                "grocery_spend": 510,
                "dining_spend": 780,
                "travel_spend": 120,
                "gas_spend": 260,
                "online_spend": 550,
            },
        ]
    )


def create_data_source_center() -> html.Details:
    def schema_card(column: str, group: str, description: str, accent: str) -> html.Div:
        return html.Div(
            children=[
                html.Div(column, style={"fontWeight": "900", "fontSize": "13px"}),
                html.Div(
                    description,
                    style={"fontSize": "12px", "color": COLORS["muted"], "lineHeight": "1.35", "marginTop": "3px"},
                ),
            ],
            title=f"{group}: {column}",
            style={
                "border": f"1px solid {COLORS['border']}",
                "borderTop": f"4px solid {accent}",
                "borderRadius": "12px",
                "padding": "10px 12px",
                "backgroundColor": "#ffffff",
            },
        )

    required_rows = [
        schema_card(
            column,
            "Required",
            "Minimum field needed to score the customer.",
            COLORS["blue"],
        )
        for column in REQUIRED_UPLOAD_COLUMNS
    ]

    optional_rows = [
        schema_card(
            column,
            "Optional",
            "Improves profile, targeting, segmentation, or campaign context.",
            "#16a34a",
        )
        for column in OPTIONAL_UPLOAD_COLUMNS
    ]

    derived_rows = [
        schema_card(
            column,
            "Derived",
            "Calculated by the decision engine after upload.",
            "#7c3aed",
        )
        for column in DERIVED_ENGINE_COLUMNS
    ]

    return html.Details(
        children=[
            html.Summary(
                "Data Source & Schema Center",
                title="Expand to review upload requirements, download templates, and prepare external customer data.",
                style={
                    "fontSize": "18px",
                    "fontWeight": "900",
                    "cursor": "pointer",
                    "color": COLORS["text"],
                    "padding": "4px 0",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                "Current data mode",
                                style={"fontSize": "12px", "fontWeight": "900", "color": COLORS["muted"], "textTransform": "uppercase"},
                            ),
                            html.H3(
                                id="data-source-mode-title",
                                children="Synthetic demo portfolio",
                                style={"margin": "6px 0 8px 0", "fontSize": "22px", "fontWeight": "900"},
                            ),
                            html.P(
                                id="data-source-mode-description",
                                children="The dashboard is currently powered by the built-in synthetic customer portfolio. Upload a valid CSV or Excel file to replace the active master dataset.",
                                style={"margin": "0", "color": COLORS["muted"], "lineHeight": "1.5"},
                            ),
                        ],
                        title="Current mode shows whether the dashboard is using the synthetic demo portfolio or an uploaded customer file as the active master dataset.",
                        style={
                            "backgroundColor": "#eff6ff",
                            "border": "1px solid #bfdbfe",
                            "borderRadius": "16px",
                            "padding": "16px",
                        },
                    ),
                    html.Div(
                        children=[
                            html.Div(
                                "Schema logic",
                                style={"fontSize": "12px", "fontWeight": "900", "color": COLORS["muted"], "textTransform": "uppercase"},
                            ),
                            html.H3(
                                "Minimum required + optional enrichment",
                                style={"margin": "6px 0 8px 0", "fontSize": "22px", "fontWeight": "900"},
                            ),
                            html.P(
                                "The required columns are the minimum fields needed for scoring. Optional fields make the dashboard richer. Derived fields are calculated by the engine.",
                                style={"margin": "0", "color": COLORS["muted"], "lineHeight": "1.5"},
                            ),
                        ],
                        title="Required fields score the customer. Optional fields enrich the product workflow. Derived fields are generated by the engine.",
                        style={
                            "backgroundColor": "#f0fdf4",
                            "border": "1px solid #bbf7d0",
                            "borderRadius": "16px",
                            "padding": "16px",
                        },
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "16px"},
            ),
            html.Div(
                children=[
                    dcc.Upload(
                        id="customer-data-upload",
                        accept=".csv,.xlsx,.xls",
                        multiple=False,
                        children=html.Div(
                            children=[
                                html.Div(
                                    "Click here to upload customer data",
                                    style={"fontSize": "18px", "fontWeight": "900", "marginBottom": "8px"},
                                ),
                                html.Div(
                                    "CSV, XLSX, or XLS accepted",
                                    style={"fontSize": "14px", "color": COLORS["muted"], "marginBottom": "12px"},
                                ),
                                html.Div(
                                    "Choose File",
                                    style={
                                        "display": "inline-block",
                                        "backgroundColor": COLORS["blue"],
                                        "color": "white",
                                        "borderRadius": "12px",
                                        "padding": "10px 18px",
                                        "fontWeight": "900",
                                    },
                                ),
                            ],
                            style={"textAlign": "center"},
                        ),
                        style={
                            "width": "100%",
                            "border": "2px dashed #60a5fa",
                            "borderRadius": "18px",
                            "padding": "34px",
                            "backgroundColor": "#f8fafc",
                            "cursor": "pointer",
                            "boxSizing": "border-box",
                        },
                    ),
                    html.Div(id="customer-data-upload-message"),
                    html.Div(id="upload-preview-modal-container"),
                ],
                title="Upload a CSV or Excel file to validate columns and preview the data.",
                style={"marginTop": "16px"},
            ),
            html.Div(
                children=[
                    html.Button(
                        "Download CSV Template",
                        id="download-schema-csv-button",
                        n_clicks=0,
                        title="Download a CSV file with required and optional upload columns.",
                        style={
                            "backgroundColor": COLORS["blue"],
                            "color": "white",
                            "border": "none",
                            "borderRadius": "12px",
                            "padding": "10px 14px",
                            "fontWeight": "900",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "Download Excel Template",
                        id="download-schema-excel-button",
                        n_clicks=0,
                        title="Download an Excel file with required and optional upload columns.",
                        style={
                            "backgroundColor": "#16a34a",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "12px",
                            "padding": "10px 14px",
                            "fontWeight": "900",
                            "cursor": "pointer",
                        },
                    ),
                    dcc.Download(id="download-schema-csv"),
                    dcc.Download(id="download-schema-excel"),
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginTop": "16px"},
            ),
            html.Div(
                children=[
                    html.H4("Minimum required upload columns", style={"fontSize": "16px", "fontWeight": "900", "marginBottom": "10px"}),
                    html.Div(
                        children=required_rows,
                        style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"},
                    ),
                ],
                style={"marginTop": "16px"},
            ),
            html.Details(
                children=[
                    html.Summary(
                        "Recommended optional fields",
                        style={"cursor": "pointer", "fontWeight": "900", "fontSize": "16px", "marginBottom": "12px"},
                    ),
                    html.Div(
                        children=optional_rows,
                        style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"},
                    ),
                ],
                open=False,
                style={"marginTop": "16px"},
            ),
            html.Details(
                children=[
                    html.Summary(
                        "Fields created by the engine",
                        style={"cursor": "pointer", "fontWeight": "900", "fontSize": "16px", "marginBottom": "12px"},
                    ),
                    html.Div(
                        children=derived_rows,
                        style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"},
                    ),
                ],
                open=False,
                style={"marginTop": "16px"},
            ),
            html.Div(
                children=[
                    html.Strong("Active master data: "),
                    "uploaded files are validated, scored through the decision engine, and stored as the active master dataset. Dashboard views, campaign recommendations, workbench tools, and strategy playbook now refresh from this active dataset.",
                ],
                title="This explains how uploaded files are used as the active dashboard dataset.",
                style={
                    "marginTop": "16px",
                    "backgroundColor": "#fff7ed",
                    "border": "1px solid #fed7aa",
                    "borderRadius": "14px",
                    "padding": "14px",
                    "color": "#7c2d12",
                    "lineHeight": "1.5",
                },
            ),
        ],
        open=False,
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "marginTop": "18px",
            "marginBottom": "18px",
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
                "Customer 360",
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
                                        "if": {"filter_query": '{decision_status} = "Scale"'},
                                        "color": "#166534",
                                        "fontWeight": "800",
                                    },
                                    {
                                        "if": {"filter_query": '{decision_status} = "Block"'},
                                        "color": "#991b1b",
                                        "fontWeight": "800",
                                    },
                                    {
                                        "if": {"filter_query": '{decision_status} = "Test"'},
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



def create_zero_state_card(title: str, message: str, action: str = "") -> html.Div:
    return html.Div(
        children=[
            html.H4(title, style={"margin": "0 0 8px 0", "color": "#991b1b", "fontSize": "16px"}),
            html.P(message, style={"margin": "0", "color": "#7f1d1d", "lineHeight": "1.5"}),
            html.P(action, style={"margin": "8px 0 0 0", "color": "#7f1d1d", "lineHeight": "1.5"}) if action else html.Div(),
        ],
        style={
            "backgroundColor": "#fef2f2",
            "border": "1px solid #fecaca",
            "borderLeft": "5px solid #dc2626",
            "borderRadius": "14px",
            "padding": "14px",
            "margin": "12px 0",
        },
    )


def build_customer_explorer_dataframe(
    search_text: str | None,
    segment: str | None,
    decision: str | None,
    risk_band: str | None,
    state: str | None,
    card_type: str | None,
    campaign_id: str | None,
    audience_type: str | None,
) -> pd.DataFrame:
    if campaign_id and campaign_id != "None":
        df = get_campaign_audience_df(campaign_id, segment or "All Segments", audience_type or "Eligible").copy()
    else:
        df = customer_features.copy()

        if segment and segment != "All Segments":
            df = df[df["customer_segment"] == segment].copy()

    if decision and decision != "All Decisions":
        df = df[df["decision_status"] == decision].copy()

    if risk_band and risk_band != "All Risk Bands":
        df = df[df["risk_band"] == risk_band].copy()

    if state and state != "All States" and "state" in df.columns:
        df = df[df["state"] == state].copy()

    if card_type and card_type != "All Card Types" and "card_type" in df.columns:
        df = df[df["card_type"] == card_type].copy()

    if search_text:
        search_value = str(search_text).strip().lower()

        searchable_columns = [
            column for column in [
                "customer_id",
                "customer_name",
                "customer_email",
                "city",
                "state",
                "customer_segment",
                "decision_status",
                "risk_band",
                "recommended_action",
                "card_type",
                "rewards_preference",
            ]
            if column in df.columns
        ]

        if searchable_columns:
            search_mask = pd.Series(False, index=df.index)

            for column in searchable_columns:
                search_mask = search_mask | df[column].astype(str).str.lower().str.contains(search_value, na=False)

            df = df[search_mask].copy()

    return df


def format_customer_explorer_preview(df: pd.DataFrame, limit: int = 500) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    preview = df.copy()

    rename_map = {
        "customer_id": "Customer ID",
        "customer_name": "Name",
        "customer_email": "Email",
        "city": "City",
        "state": "State",
        "customer_segment": "Segment",
        "risk_band": "Risk Band",
        "decision_status": "Decision",
        "recommended_action": "Recommended Action",
        "offer_type": "Treatment Type",
        "card_type": "Card Type",
        "rewards_preference": "Rewards",
        "credit_score": "Credit Score",
        "utilization_rate": "Utilization",
        "default_probability": "Default Probability",
        "monthly_spend": "Monthly Spend",
        "risk_adjusted_profit": "Risk-Adjusted Profit",
        "expected_roi": "Expected ROI",
        "preferred_channel": "Preferred Channel",
    }

    preview = preview.rename(columns=rename_map)

    for column in ["Utilization", "Default Probability"]:
        if column in preview.columns:
            preview[column] = preview[column].apply(lambda value: f"{value * 100:.1f}%")

    for column in ["Monthly Spend", "Risk-Adjusted Profit"]:
        if column in preview.columns:
            preview[column] = preview[column].apply(lambda value: f"${value:,.0f}")

    if "Expected ROI" in preview.columns:
        preview["Expected ROI"] = preview["Expected ROI"].apply(lambda value: f"{value:.2f}x")

    preview_columns = [
        "Customer ID",
        "Name",
        "Email",
        "City",
        "State",
        "Segment",
        "Risk Band",
        "Decision",
        "Recommended Action",
        "Treatment Type",
        "Card Type",
        "Rewards",
        "Credit Score",
        "Utilization",
        "Default Probability",
        "Monthly Spend",
        "Risk-Adjusted Profit",
        "Expected ROI",
        "Preferred Channel",
    ]

    preview_columns = [column for column in preview_columns if column in preview.columns]

    return preview[preview_columns].head(limit)


def create_customer_explorer_layout() -> html.Div:
    segment_options = [{"label": "All Segments", "value": "All Segments"}] + [
        {"label": segment, "value": segment}
        for segment in sorted(customer_features["customer_segment"].dropna().unique())
    ]

    decision_options = [{"label": "All Decisions", "value": "All Decisions"}] + [
        {"label": decision, "value": decision}
        for decision in sorted(customer_features["decision_status"].dropna().unique())
    ]

    risk_options = [{"label": "All Risk Bands", "value": "All Risk Bands"}] + [
        {"label": risk, "value": risk}
        for risk in sorted(customer_features["risk_band"].dropna().unique())
    ]

    state_options = [{"label": "All States", "value": "All States"}] + [
        {"label": state, "value": state}
        for state in sorted(customer_features["state"].dropna().unique())
    ]

    card_options = [{"label": "All Card Types", "value": "All Card Types"}] + [
        {"label": card_type, "value": card_type}
        for card_type in sorted(customer_features["card_type"].dropna().unique())
    ]

    campaign_options = [{"label": "No campaign filter", "value": "None"}]

    if not campaign_recommendations.empty:
        campaign_options += [
            {
                "label": f"{int(row['dashboard_recommendation_rank'])}. {row['campaign_name']} ({row['recommended_rollout_decision']})",
                "value": row["campaign_id"],
            }
            for _, row in campaign_recommendations.head(25).iterrows()
        ]

    return html.Div(
        children=[
            create_tab_intro(
                "Audience Explorer",
                "Use this section as the operational audience database. Filter customers by segment, risk, decision, state, card type, or campaign audience, then export the selected list for review or campaign setup.",
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Label(help_label("Search", "Search by customer ID, name, email, city, state, segment, decision, or treatment type.")),
                            dcc.Input(
                                id="customer-explorer-search",
                                type="text",
                                placeholder="Search customer ID, name, email, city...",
                                debounce=True,
                                style={
                                    "width": "100%",
                                    "padding": "10px",
                                    "border": f"1px solid {COLORS['border']}",
                                    "borderRadius": "10px",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label(help_label("Campaign Filter", "Optional. Select a campaign to show only customers eligible for that campaign audience.")),
                            dcc.Dropdown(
                                id="customer-explorer-campaign",
                                options=campaign_options,
                                value="None",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label(help_label("Campaign Audience", "When a campaign is selected, choose eligible, scale, test, or blocked customers.")),
                            dcc.Dropdown(
                                id="customer-explorer-audience-type",
                                options=[
                                    {"label": "Eligible customers", "value": "Eligible"},
                                    {"label": "Scale customers", "value": "Scale"},
                                    {"label": "Test customers", "value": "Test"},
                                    {"label": "Blocked customers", "value": "Blocked"},
                                ],
                                value="Eligible",
                                clearable=False,
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Segment", style={"fontWeight": "800"}),
                            dcc.Dropdown(id="customer-explorer-segment", options=segment_options, value="All Segments", clearable=False),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Decision", style={"fontWeight": "800"}),
                            dcc.Dropdown(id="customer-explorer-decision", options=decision_options, value="All Decisions", clearable=False),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Risk Band", style={"fontWeight": "800"}),
                            dcc.Dropdown(id="customer-explorer-risk", options=risk_options, value="All Risk Bands", clearable=False),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("State", style={"fontWeight": "800"}),
                            dcc.Dropdown(id="customer-explorer-state", options=state_options, value="All States", clearable=False),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Card Type", style={"fontWeight": "800"}),
                            dcc.Dropdown(id="customer-explorer-card-type", options=card_options, value="All Card Types", clearable=False),
                        ],
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "14px",
                    "backgroundColor": COLORS["card"],
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "18px",
                    "padding": "18px",
                    "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                    "marginBottom": "18px",
                },
            ),
            html.Div(id="customer-explorer-summary"),
            html.Div(
                children=[
                    html.Button(
                        "Download CSV",
                        id="download-customer-explorer-csv-button",
                        n_clicks=0,
                        style={
                            "backgroundColor": COLORS["blue"],
                            "color": "white",
                            "border": "none",
                            "borderRadius": "10px",
                            "padding": "10px 14px",
                            "fontWeight": "900",
                            "cursor": "pointer",
                            "width": "150px",
                        },
                    ),
                    html.Button(
                        "Download Excel",
                        id="download-customer-explorer-excel-button",
                        n_clicks=0,
                        style={
                            "backgroundColor": "#16a34a",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "10px",
                            "padding": "10px 14px",
                            "fontWeight": "900",
                            "cursor": "pointer",
                            "width": "160px",
                        },
                    ),
                    dcc.Download(id="download-customer-explorer-csv"),
                    dcc.Download(id="download-customer-explorer-excel"),
                ],
                style={"display": "flex", "gap": "10px", "marginBottom": "14px", "flexWrap": "wrap"},
            ),
            dash_table.DataTable(
                id="customer-explorer-table",
                columns=[],
                data=[],
                page_size=15,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#f8fafc",
                    "fontWeight": "900",
                    "color": COLORS["muted"],
                    "border": f"1px solid {COLORS['border']}",
                },
                style_cell={
                    "padding": "10px",
                    "fontSize": "13px",
                    "fontFamily": "Arial",
                    "border": f"1px solid {COLORS['border']}",
                    "whiteSpace": "normal",
                    "height": "auto",
                    "textAlign": "left",
                    "minWidth": "90px",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{Decision} = "Scale"'}, "backgroundColor": "#f0fdf4"},
                    {"if": {"filter_query": '{Decision} = "Test"'}, "backgroundColor": "#eff6ff"},
                    {"if": {"filter_query": '{Decision} = "Block"'}, "backgroundColor": "#fef2f2"},
                    {"if": {"filter_query": '{Risk Band} = "Very High Risk"'}, "backgroundColor": "#fff1f2"},
                ],
            ),
        ],
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
                                "Customer 360 explains individual decisions, Scenario Simulator tests campaign assumptions, A/B Test Planner designs experiments, and Audience Explorer exports customer lists before rollout.",
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
                                label="Customer 360",
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
                                value="scenario-simulator",
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
                                value="ab-test-planner",
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
                        
                                dcc.Tab(
                                    label="Audience Explorer",
                                    value="customer-explorer",
                                    style=tab_style,
                                    selected_style=selected_tab_style,
                                    children=[
                                        create_customer_explorer_layout()
                                    ],
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

    if campaign_recommendations.empty:
        campaign_options = [{"label": "No campaign recommendations available", "value": "None"}]
        default_campaign_id = "None"
    else:
        scenario_campaigns = campaign_recommendations.head(10).copy()
        campaign_options = [
            {
                "label": f"{row['dashboard_recommendation_rank']}. {row['campaign_name']} ({row['recommended_rollout_decision']})",
                "value": row["campaign_id"],
            }
            for _, row in scenario_campaigns.iterrows()
        ]
        default_campaign_id = scenario_campaigns.iloc[0]["campaign_id"]

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

                            html.Label("Campaign", style={"fontWeight": "800"}),
                            dcc.Dropdown(
                                id="scenario-campaign",
                                options=campaign_options,
                                value=default_campaign_id,
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Div(
                                "Campaign assumptions auto-fill the sliders below. You can still override them manually.",
                                style={
                                    "fontSize": "13px",
                                    "color": COLORS["muted"],
                                    "lineHeight": "1.45",
                                    "marginBottom": "18px",
                                },
                            ),

                            html.Label("Customer Segment", style={"fontWeight": "800"}),
                            dcc.Dropdown(
                                id="scenario-segment",
                                options=segment_options,
                                value="All Segments",
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Label(help_label("Marketing Cost per Customer", "Estimated cost to target one customer with this campaign, including marketing, servicing, rewards, or partner offer cost.")),
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

                            html.Label(help_label("Expected Spend Lift", "Estimated percent increase in customer spend if the campaign is launched. Higher lift usually improves revenue but can also increase risk exposure.")),
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

                            html.Label(help_label("Max Default Probability Allowed", "Risk threshold used to block customers whose estimated default probability is above the selected limit.")),
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
    segment_options = [{"label": "All Segments", "value": "All Segments"}] + [
        {"label": segment, "value": segment}
        for segment in sorted(customer_features["customer_segment"].unique())
        if segment != "Risk Watch"
    ]

    if campaign_recommendations.empty:
        campaign_options = [{"label": "No campaign recommendations available", "value": "None"}]
        default_campaign_id = "None"
    else:
        ab_campaigns = campaign_recommendations.head(10).copy()
        campaign_options = [
            {
                "label": f"{row['dashboard_recommendation_rank']}. {row['campaign_name']} ({row['recommended_rollout_decision']})",
                "value": row["campaign_id"],
            }
            for _, row in ab_campaigns.iterrows()
        ]
        default_campaign_id = ab_campaigns.iloc[0]["campaign_id"]

    return html.Div(
        children=[
            create_tab_intro(
                "A/B Test Planner",
                "Use this section to design a controlled experiment before scaling a campaign. It focuses on test population, control/treatment split, test duration, expected responders, and whether the test is powered enough to support a rollout decision.",
            ),

            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H3("Experiment Setup", style={"marginTop": "0"}),

                            html.Label("Campaign", style={"fontWeight": "800"}),
                            dcc.Dropdown(
                                id="ab-campaign",
                                options=campaign_options,
                                value=default_campaign_id,
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Div(
                                "Campaign assumptions auto-fill response rate, lift, and test population. You can still override them manually.",
                                style={
                                    "fontSize": "13px",
                                    "color": COLORS["muted"],
                                    "lineHeight": "1.45",
                                    "marginBottom": "18px",
                                },
                            ),

                            html.Label("Target Segment", style={"fontWeight": "800"}),
                            dcc.Dropdown(
                                id="ab-segment",
                                options=segment_options,
                                value="All Segments",
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Label(help_label("Baseline Response Rate", "Expected response rate without the new campaign treatment. Used as the control group benchmark in the experiment.")),
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

                            html.Label(help_label("Expected Lift from Treatment", "Expected percentage-point improvement from the treatment group compared with the control group.")),
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

                            html.Label(help_label("Test Population Available", "Maximum number of eligible customers available for the A/B test after campaign targeting and guardrails.")),
                            dcc.Slider(
                                id="ab-test-population",
                                min=100,
                                max=3000,
                                step=100,
                                value=1000,
                                marks={100: "100", 500: "500", 1000: "1k", 2000: "2k", 3000: "3k"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),

                            html.Div(style={"height": "22px"}),

                            html.Label(help_label("Control / Treatment Split", "Share of customers assigned to control versus treatment. 50/50 is best for power, while 70/30 or 80/20 limits exposure.")),
                            dcc.Dropdown(
                                id="ab-test-split",
                                options=[
                                    {"label": "50 / 50 balanced test", "value": "50/50"},
                                    {"label": "70 / 30 conservative rollout", "value": "70/30"},
                                    {"label": "80 / 20 small treatment pilot", "value": "80/20"},
                                ],
                                value="50/50",
                                clearable=False,
                                style={"marginBottom": "18px"},
                            ),

                            html.Label(help_label("Test Duration", "Planned experiment length in weeks. Longer tests can improve confidence but delay rollout decisions.")),
                            dcc.Slider(
                                id="ab-test-duration",
                                min=2,
                                max=12,
                                step=1,
                                value=6,
                                marks={2: "2w", 4: "4w", 6: "6w", 8: "8w", 12: "12w"},
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
                            html.H3("Experiment Design Output", style={"marginTop": "0"}),
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
                    "gridTemplateColumns": "1fr",
                    "gap": "18px",
                },
            ),
        ]
    )





def get_campaign_audience_df(
    campaign_id,
    segment="All Segments",
    audience_type="Eligible",
    source_df: pd.DataFrame | None = None,
):
    """Build campaign audience from the current master dataset, not only synthetic data."""
    audience_df = source_df.copy() if source_df is not None else customer_features.copy()

    if campaign_id and campaign_id != "None" and not campaign_recommendations.empty:
        campaign_match = campaign_recommendations[
            campaign_recommendations["campaign_id"] == campaign_id
        ]

        if not campaign_match.empty:
            selected_campaign = campaign_match.iloc[0]
            target_segments = split_semicolon_values(selected_campaign.get("target_segments", ""))
            excluded_segments = split_semicolon_values(selected_campaign.get("excluded_segments", ""))

            if target_segments and "customer_segment" in audience_df.columns:
                audience_df = audience_df[
                    audience_df["customer_segment"].isin(target_segments)
                ].copy()

            if excluded_segments and "customer_segment" in audience_df.columns:
                audience_df = audience_df[
                    ~audience_df["customer_segment"].isin(excluded_segments)
                ].copy()

    if segment and segment != "All Segments" and "customer_segment" in audience_df.columns:
        audience_df = audience_df[audience_df["customer_segment"] == segment].copy()

    if audience_type == "Eligible" and "decision_status" in audience_df.columns:
        audience_df = audience_df[audience_df["decision_status"].isin(["Scale", "Test"])].copy()
    elif audience_type == "Scale" and "decision_status" in audience_df.columns:
        audience_df = audience_df[audience_df["decision_status"] == "Scale"].copy()
    elif audience_type == "Test" and "decision_status" in audience_df.columns:
        audience_df = audience_df[audience_df["decision_status"] == "Test"].copy()
    elif audience_type == "Blocked" and "decision_status" in audience_df.columns:
        audience_df = audience_df[audience_df["decision_status"] == "Block"].copy()

    return audience_df.copy()



def build_customer_export_preview_rows(df: pd.DataFrame, limit: int = 25) -> list[dict]:
    if df.empty:
        return []

    preview = df.head(limit).copy()

    rename_map = {
        "customer_id": "Customer ID",
        "customer_name": "Name",
        "customer_email": "Email",
        "city": "City",
        "state": "State",
        "customer_segment": "Segment",
        "risk_band": "Risk Band",
        "decision_status": "Portfolio Decision",
        "campaign_customer_action": "Campaign Action",
        "campaign_test_group": "Test Group",
        "credit_score": "Credit Score",
        "utilization_rate": "Utilization",
        "default_probability": "Default Probability",
        "monthly_spend": "Monthly Spend",
        "risk_adjusted_profit": "Risk-Adjusted Profit",
        "expected_roi": "Expected ROI",
        "preferred_channel": "Preferred Channel",
        "card_type": "Card Type",
        "rewards_preference": "Rewards Preference",
    }

    preview = preview.rename(columns=rename_map)

    for column in ["Utilization", "Default Probability"]:
        if column in preview.columns:
            preview[column] = preview[column].apply(lambda value: f"{value * 100:.1f}%")

    for column in ["Monthly Spend", "Risk-Adjusted Profit"]:
        if column in preview.columns:
            preview[column] = preview[column].apply(format_currency)

    if "Expected ROI" in preview.columns:
        preview["Expected ROI"] = preview["Expected ROI"].apply(lambda value: f"{value:.2f}x")

    preview_columns = [
        "Customer ID",
        "Name",
        "Segment",
        "Risk Band",
        "Portfolio Decision",
        "Campaign Action",
        "Test Group",
        "Credit Score",
        "Utilization",
        "Default Probability",
        "Monthly Spend",
        "Risk-Adjusted Profit",
        "Preferred Channel",
    ]

    preview_columns = [column for column in preview_columns if column in preview.columns]

    return preview[preview_columns].to_dict("records")

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
        dcc.Store(id="active-customer-data-store", storage_type="memory"),
        dcc.Store(id="active-data-mode-store", storage_type="memory", data={"mode": "synthetic", "rows": 0, "filename": "synthetic_demo_portfolio"}),
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
                        id="top-kpi-container",
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

        create_data_source_center(),

        create_filter_panel(),

        create_workflow_guide(),

        create_color_legend(),

        create_action_prompt_panel(),

        html.Div(
            children=[
                dcc.Graph(id="offer-action-mix-chart"),
                dcc.Graph(id="offer-type-chart"),
            ],
            id="hidden-callback-placeholders",
            style={"display": "none"},
        ),

        dcc.Tabs(
            id="main-tabs",
            value="overview",
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
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
                    value="segment-strategy",
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
                                html.Div(id="priority-segment-table-container", children=create_table(priority_rows)),
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
                            "Use the priority table above to decide which active segments should scale, test, or remain under guardrail review. Uploaded files may change this takeaway based on the active portfolio.",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Campaigns & Offers",
                    value="campaigns-offers",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Campaign Recommendation Engine",
                            "This page ranks campaign opportunities from a reusable campaign library. It shows which campaigns are viable for the current portfolio, where to scale, where to test, and where risk should constrain rollout.",
                        ),
                        create_campaigns_action_panel(),
                        html.Div(
                            id="campaign-kpi-container",
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
                                    "How top campaigns split across active rollout recommendations.",
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
                                    "Top Active Campaign Opportunities",
                                    style={
                                        "margin": "0 0 14px 0",
                                        "fontSize": "20px",
                                        "fontWeight": "900",
                                    },
                                ),
                                html.Div(
                                    id="campaign-top-cards-container",
                                    children=[
                                        create_campaign_recommendation_card(row)
                                        for _, row in campaign_top10.iterrows()
                                    ],
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(auto-fit, minmax(440px, 1fr))",
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
                                            "Campaign Audit Table",
                                            style={
                                                "margin": "0 0 6px 0",
                                                "fontSize": "20px",
                                                "fontWeight": "900",
                                            },
                                        ),
                                        html.P(
                                            "Compact audit view of the top campaign opportunities. Matches can include servicing/protective audiences; Scale and Blocked columns show the active customer decision mix behind each campaign.",
                                            style={
                                                "margin": "0 0 14px 0",
                                                "fontSize": "13px",
                                                "color": COLORS["muted"],
                                                "lineHeight": "1.45",
                                            },
                                        ),
                                        html.Div(
                                            id="campaign-table-container",
                                            children=create_table(campaign_table_rows),
                                            style={"overflowX": "auto"},
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": COLORS["card"],
                                        "border": f"1px solid {COLORS['border']}",
                                        "borderRadius": "18px",
                                        "padding": "20px",
                                        "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
                                    },
                                ),
                                html.Div(
                                    id="campaign-detail-container",
                                    children=create_campaign_detail_panel(campaign_recommendations),
                                    style={"marginTop": "18px"},
                                ),
                            ],
                            style={"marginTop": "18px"},
                        ),
                        create_insight_card(
                            "Why this page matters",
                            "The campaign layer connects customer segmentation to business action. It helps a user choose which campaigns to run, where to scale, where to test, and where risk guardrails should constrain rollout.",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Strategy Playbook",
                    value="strategy-playbook",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Strategy Playbook",
                            "This page turns the active portfolio into an operating plan. It shows which segments to scale, test, constrain, or protect before campaign launch.",
                        ),
                        create_strategy_cta_panel(),
                        create_playbook_action_panel(),
                        create_strategy_flow_diagram(),
                        create_strategy_risk_return_section(),
                        create_strategy_playbook_table(),
                        create_insight_card(
                            "How to read this playbook",
                            "Use this page as the business decision layer. The flow sketch explains the end-to-end process, the risk-return matrix shows why each segment needs a different rollout posture, and the strategy cards turn the analysis into segment-level actions.",
                        ),
                    ],
                ),
                                dcc.Tab(
                    label="Decision Workbench",
                    value="decision-workbench",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        build_decision_workbench_layout()
                    ],
                ),

                dcc.Tab(
                    label="Guardrails",
                    value="guardrails",
                    style=tab_style,
                    selected_style=selected_tab_style,
                    children=[
                        create_tab_intro(
                            "Responsible Lending Guardrails",
                            "This tab checks whether the engine is protecting customers who may carry higher credit risk. The goal is to separate profitable growth from risky growth and prevent aggressive offers from going to the wrong groups.",
                        ),
                        create_compact_governance_panel(),
                        create_guardrails_action_panel(),
                        html.Div(
                            id="guardrails-kpi-container",
                            children=[],
                            style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "16px", "marginTop": "22px", "marginBottom": "18px"},
                        ),
                        html.Div(
                            id="guardrails-interpretation-container",
                            children=create_insight_card(
                                "Guardrail Interpretation",
                                "This section will refresh from the active master dataset and explain whether any customers fail risk rules.",
                                variant="warning",
                            ),
                            style={"marginBottom": "18px"},
                        ),
                        html.Div(
                            id="guardrails-rule-review-container",
                            children=[],
                            style={"marginBottom": "18px"},
                        ),
                        html.Div(
                            id="guardrails-customer-review-container",
                            children=[],
                            style={"marginBottom": "18px"},
                        ),
                        html.Div(
                            children=[
                                create_chart_card("Risk Band Distribution", "Portfolio split by estimated risk band.", risk_fig, "guardrail-risk-chart"),
                                create_chart_card("High Utilization Watchlist", "Decision mix for customers whose utilization rate suggests extra review.", high_utilization_fig, "guardrail-high-util-chart"),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "0.9fr 1.1fr", "gap": "18px"},
                        ),
                        html.Div(style={"height": "18px"}),
                        create_chart_card("Blocked Customers by Segment", "Where risk guardrails are triggered.", block_segment_fig, "guardrail-blocked-segment-chart"),
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
    return "Filtered View" if has_filters else "Active Portfolio View"


def apply_view_label_to_figure(fig, view_label: str):
    current_title = fig.layout.title.text if fig.layout.title.text else "Chart"

    # Avoid stacking labels if callback fires multiple times
    current_title = current_title.replace(" — Filtered View", "").replace(" — Full Portfolio View", "").replace(" — Active Portfolio View", "")

    fig.update_layout(
        title=f"{current_title} — {view_label}"
    )
    return fig





def apply_global_filters(
    selected_segments,
    selected_decisions,
    selected_risks,
    selected_actions,
    source_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply global dashboard filters to either uploaded active data or synthetic data."""
    filtered = source_df.copy() if source_df is not None else customer_features.copy()

    if selected_segments and "customer_segment" in filtered.columns:
        filtered = filtered[filtered["customer_segment"].isin(selected_segments)]

    if selected_decisions and "decision_status" in filtered.columns:
        filtered = filtered[filtered["decision_status"].isin(selected_decisions)]

    if selected_risks and "risk_band" in filtered.columns:
        filtered = filtered[filtered["risk_band"].isin(selected_risks)]

    if selected_actions and "recommended_action" in filtered.columns:
        filtered = filtered[filtered["recommended_action"].isin(selected_actions)]

    return filtered


def empty_figure(title: str = "No data available", message: str = "No matching records for the selected filters."):
    try:
        return create_empty_figure(message)
    except NameError:
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
            xref="paper",
            yref="paper",
        )
        fig.update_layout(
            title=title,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig


def build_decision_bar_fig(df: pd.DataFrame):
    return build_decision_count_fig(df)


def build_decision_counts_fig(df: pd.DataFrame):
    return build_decision_count_fig(df)


def build_segment_stack_fig(df: pd.DataFrame):
    return build_segment_decision_mix_fig(df)


def build_segment_mix_fig(df: pd.DataFrame):
    return build_segment_decision_mix_fig(df)


def build_eligible_fig(df: pd.DataFrame):
    return build_segment_eligibility_fig(df)


def build_eligibility_fig(df: pd.DataFrame):
    return build_segment_eligibility_fig(df)


def build_guardrail_risk_fig(df: pd.DataFrame):
    return build_risk_band_fig(df)


def build_guardrail_high_util_fig(df: pd.DataFrame):
    return build_high_util_fig(df)


def build_high_utilization_fig(df: pd.DataFrame):
    return build_high_util_fig(df)


def build_high_utilization_revolver_fig(df: pd.DataFrame):
    return build_high_util_fig(df)


def build_block_segment_fig(df: pd.DataFrame):
    return build_blocked_segment_fig(df)


def build_blocked_customers_fig(df: pd.DataFrame):
    return build_blocked_segment_fig(df)


def build_guardrail_blocked_segment_fig(df: pd.DataFrame):
    return build_blocked_segment_fig(df)









def parse_test_split_value(value) -> float:
    """Convert A/B split dropdown values like '50/50' or numeric 50 into treatment percentage."""
    if value is None:
        return 50.0

    if isinstance(value, (int, float)):
        return float(value)

    text_value = str(value).strip()

    if "/" in text_value:
        first_part = text_value.split("/", 1)[0].strip()
        try:
            return float(first_part)
        except ValueError:
            return 50.0

    digits = "".join(ch for ch in text_value if ch.isdigit() or ch == ".")
    if digits:
        try:
            return float(digits)
        except ValueError:
            return 50.0

    return 50.0



def get_active_customer_features(active_records):
    """Return uploaded active dataset when available; otherwise return synthetic demo data."""
    if active_records:
        try:
            active_df = pd.DataFrame(active_records)
            if not active_df.empty and "customer_id" in active_df.columns:
                return active_df.copy()
        except Exception:
            pass
    return customer_features.copy()


def score_uploaded_customer_file(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Validate, score, and enrich an uploaded customer file for dashboard use."""
    missing_required = [col for col in REQUIRED_UPLOAD_COLUMNS if col not in uploaded_df.columns]
    if missing_required:
        raise ValueError("Missing required fields: " + ", ".join(missing_required))

    base_input = uploaded_df[REQUIRED_UPLOAD_COLUMNS].copy()
    scored_df = score_customer_portfolio(base_input)

    # Preserve optional and extra uploaded fields without overwriting scored engine fields.
    preserve_cols = [col for col in uploaded_df.columns if col not in scored_df.columns or col == "customer_id"]
    if preserve_cols:
        scored_df = scored_df.merge(
            uploaded_df[preserve_cols],
            on="customer_id",
            how="left",
        )

    # Fill profile/enrichment fields that the dashboard expects.
    fallback_values = {
        "customer_name": "Uploaded Customer",
        "customer_email": "not_provided@example.com",
        "phone_number": "Not provided",
        "city": "Not provided",
        "state": "NA",
        "zip_code": "NA",
        "employment_status": "Not provided",
        "occupation_group": "Not provided",
        "preferred_channel": "Not provided",
        "relationship_tier": "Standard",
        "signup_channel": "Not provided",
        "account_open_date": "Not provided",
        "digital_engagement_score": 0,
        "last_app_login_days": 0,
        "autopay_enrolled": "Unknown",
        "paperless_enrolled": "Unknown",
        "card_type": "Unknown",
        "rewards_preference": "Unknown",
    }

    for col, value in fallback_values.items():
        if col not in scored_df.columns:
            scored_df[col] = value
        else:
            scored_df[col] = scored_df[col].fillna(value)

    # Friendly defaults for fields used in tables/charts.
    if "recommended_action" not in scored_df.columns and "decision_status" in scored_df.columns:
        scored_df["recommended_action"] = scored_df["decision_status"]

    if "offer_type" not in scored_df.columns:
        scored_df["offer_type"] = scored_df.get("recommended_action", "Not assigned")

    return scored_df


@app.callback(
    Output("customer-data-upload-message", "children"),
    Output("active-customer-data-store", "data"),
    Output("active-data-mode-store", "data"),
    Input("customer-data-upload", "contents"),
    State("customer-data-upload", "filename"),
    prevent_initial_call=True,
)
def validate_customer_data_upload(contents, filename):
    if not contents:
        raise PreventUpdate

    try:
        content_type, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string)

        lower_filename = (filename or "").lower()

        if lower_filename.endswith(".csv"):
            uploaded_df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif lower_filename.endswith((".xlsx", ".xls")):
            uploaded_df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(
                children=[
                    html.Strong("Upload fired, but file type is unsupported."),
                    html.Div(f"File: {filename}", style={"marginTop": "4px"}),
                    html.Div("Please upload CSV, XLSX, or XLS.", style={"marginTop": "4px"}),
                ],
                style={
                    "marginTop": "12px",
                    "backgroundColor": "#fef2f2",
                    "border": "1px solid #fecaca",
                    "borderRadius": "12px",
                    "padding": "12px",
                    "color": "#7f1d1d",
                },
            ), no_update, no_update

        uploaded_columns = list(uploaded_df.columns)
        missing_required = [col for col in REQUIRED_UPLOAD_COLUMNS if col not in uploaded_columns]
        detected_required = [col for col in REQUIRED_UPLOAD_COLUMNS if col in uploaded_columns]
        detected_optional = [col for col in OPTIONAL_UPLOAD_COLUMNS if col in uploaded_columns]

        ready = len(missing_required) == 0
        preview_df = uploaded_df.head(5).copy()

        scored_active_df = None
        active_records = no_update
        active_mode = no_update

        if ready:
            scored_active_df = score_uploaded_customer_file(uploaded_df)
            active_records = scored_active_df.to_dict("records")
            active_mode = {
                "mode": "uploaded",
                "rows": int(len(scored_active_df)),
                "filename": filename or "uploaded_file",
            }

        return html.Div(
            children=[
                html.Div(
                    children=[
                        html.Strong("Upload fired successfully: "),
                        filename,
                        html.Div(
                            f"Rows: {len(uploaded_df):,} | Columns: {len(uploaded_columns):,} | Required fields: {len(detected_required)} / {len(REQUIRED_UPLOAD_COLUMNS)} | Optional fields: {len(detected_optional)} / {len(OPTIONAL_UPLOAD_COLUMNS)}",
                            style={"marginTop": "6px"},
                        ),
                        html.Div(
                            "Uploaded file is now active in the dashboard." if ready else "Missing required fields: " + ", ".join(missing_required),
                            style={"marginTop": "6px", "fontWeight": "800"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#f0fdf4" if ready else "#fff7ed",
                        "border": "1px solid #bbf7d0" if ready else "1px solid #fed7aa",
                        "borderRadius": "12px",
                        "padding": "14px",
                        "color": "#14532d" if ready else "#7c2d12",
                        "marginTop": "12px",
                    },
                ),
                html.Div(
                    children=[
                        html.H4("Upload preview", style={"margin": "0 0 10px 0"}),
                        dash_table.DataTable(
                            columns=[{"name": col, "id": col} for col in preview_df.columns],
                            data=preview_df.to_dict("records"),
                            page_size=5,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#f8fafc",
                                "fontWeight": "900",
                                "border": f"1px solid {COLORS['border']}",
                            },
                            style_cell={
                                "padding": "8px",
                                "fontSize": "12px",
                                "fontFamily": "Arial",
                                "border": f"1px solid {COLORS['border']}",
                                "whiteSpace": "normal",
                                "height": "auto",
                                "textAlign": "left",
                            },
                        ),
                    ],
                    style={
                        "marginTop": "12px",
                        "backgroundColor": "#ffffff",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "14px",
                        "padding": "14px",
                    },
                ),
            ]
        ), active_records, active_mode

    except Exception as error:
        return html.Div(
            children=[
                html.Strong("Upload fired, but file could not be parsed."),
                html.Div(str(error), style={"marginTop": "6px"}),
            ],
            style={
                "marginTop": "12px",
                "backgroundColor": "#fef2f2",
                "border": "1px solid #fecaca",
                "borderRadius": "12px",
                "padding": "12px",
                "color": "#7f1d1d",
            },
        ), no_update, no_update



@app.callback(
    Output("download-schema-csv", "data"),
    Input("download-schema-csv-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_schema_csv(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    template_df = build_schema_template_df()
    return dcc.send_data_frame(
        template_df.to_csv,
        "credit_card_customer_upload_template.csv",
        index=False,
    )


@app.callback(
    Output("download-schema-excel", "data"),
    Input("download-schema-excel-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_schema_excel(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    template_df = build_schema_template_df()
    return dcc.send_data_frame(
        template_df.to_excel,
        "credit_card_customer_upload_template.xlsx",
        index=False,
        sheet_name="required_schema",
    )






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
    Input("active-customer-data-store", "data"),
)
def update_filtered_charts(selected_segments, selected_decisions, selected_risks, selected_actions, active_data):
    master_df = get_active_customer_features(active_data)

    filtered = apply_global_filters(
        selected_segments,
        selected_decisions,
        selected_risks,
        selected_actions,
        master_df,
    )

    view_label = get_active_view_label(
        selected_segments,
        selected_decisions,
        selected_risks,
        selected_actions,
    )

    def master_empty_figure(title, message="No records available for this view."):
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": COLORS["muted"]},
        )
        fig.update_layout(
            title=title,
            height=360,
            margin={"l": 40, "r": 30, "t": 70, "b": 40},
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        return fig

    def polish(fig, title):
        fig.update_layout(
            title=title,
            height=390,
            margin={"l": 50, "r": 30, "t": 70, "b": 50},
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"family": "Arial", "size": 12, "color": COLORS["text"]},
        )
        return fig

    def value_counts_frame(df, column, label_name, value_name):
        if df.empty or column not in df.columns:
            return pd.DataFrame(columns=[label_name, value_name])

        counts = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )
        counts.columns = [label_name, value_name]
        return counts

    def decision_share_figure(df):
        counts = value_counts_frame(df, "decision_status", "decision_status", "customers")
        if counts.empty:
            return master_empty_figure("Decision Status Share")

        fig = px.pie(
            counts,
            names="decision_status",
            values="customers",
            hole=0.55,
            title="Decision Status Share",
            color="decision_status",
            color_discrete_map=DECISION_COLOR_MAP,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return polish(fig, "Decision Status Share")

    def decision_count_figure(df):
        counts = value_counts_frame(df, "decision_status", "decision_status", "customers")
        if counts.empty:
            return master_empty_figure("Decision Status Count")

        fig = px.bar(
            counts,
            x="decision_status",
            y="customers",
            title="Decision Status Count",
            color="decision_status",
            color_discrete_map=DECISION_COLOR_MAP,
            text="customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Customer Count", xaxis_title="")
        return polish(fig, "Decision Status Count")

    def segment_size_figure(df):
        counts = value_counts_frame(df, "customer_segment", "customer_segment", "customers")
        if counts.empty:
            return master_empty_figure("Customer Count by Segment")

        fig = px.bar(
            counts.sort_values("customers", ascending=True),
            x="customers",
            y="customer_segment",
            orientation="h",
            title="Customer Count by Segment",
            text="customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Customer Count", yaxis_title="")
        return polish(fig, "Customer Count by Segment")

    def segment_eligibility_figure(df):
        if df.empty or "customer_segment" not in df.columns or "decision_status" not in df.columns:
            return master_empty_figure("Scale/Test Eligibility by Segment")

        segment_frame = df.copy()
        segment_frame["eligible_flag"] = segment_frame["decision_status"].isin(["Scale", "Test"]).astype(int)

        summary = (
            segment_frame
            .groupby("customer_segment", as_index=False)
            .agg(customers=("customer_id", "count"), eligible=("eligible_flag", "sum"))
        )
        summary["eligible_rate"] = summary["eligible"] / summary["customers"] * 100

        fig = px.bar(
            summary.sort_values("eligible_rate", ascending=True),
            x="eligible_rate",
            y="customer_segment",
            orientation="h",
            title="Scale/Test Eligibility by Segment",
            text=summary["eligible_rate"].round(1),
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Eligible Rate", yaxis_title="")
        return polish(fig, "Scale/Test Eligibility by Segment")

    def segment_decision_mix_figure(df):
        if df.empty or "customer_segment" not in df.columns or "decision_status" not in df.columns:
            return master_empty_figure("Decision Mix by Segment")

        mix = (
            df.groupby(["customer_segment", "decision_status"], as_index=False)
            .size()
            .rename(columns={"size": "customers"})
        )

        fig = px.bar(
            mix,
            x="customer_segment",
            y="customers",
            color="decision_status",
            title="Decision Mix by Segment",
            color_discrete_map=DECISION_COLOR_MAP,
        )
        fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Customer Count")
        return polish(fig, "Decision Mix by Segment")

    def action_mix_figure(df):
        counts = value_counts_frame(df, "recommended_action", "recommended_action", "customers")
        if counts.empty:
            return master_empty_figure("Recommended Action Mix")

        fig = px.bar(
            counts.sort_values("customers", ascending=True),
            x="customers",
            y="recommended_action",
            orientation="h",
            title="Recommended Action Mix",
            text="customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Customer Count", yaxis_title="")
        return polish(fig, "Recommended Action Mix")

    def offer_type_figure(df):
        counts = value_counts_frame(df, "offer_type", "offer_type", "customers")
        if counts.empty:
            return master_empty_figure("Offer Type Distribution")

        fig = px.bar(
            counts.sort_values("customers", ascending=True),
            x="customers",
            y="offer_type",
            orientation="h",
            title="Offer Type Distribution",
            text="customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Customer Count", yaxis_title="")
        return polish(fig, "Offer Type Distribution")

    def risk_band_figure(df):
        counts = value_counts_frame(df, "risk_band", "risk_band", "customers")
        if counts.empty:
            return master_empty_figure("Risk Band Distribution")

        fig = px.bar(
            counts,
            x="risk_band",
            y="customers",
            title="Risk Band Distribution",
            color="risk_band",
            color_discrete_map=RISK_COLOR_MAP,
            text="customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customer Count")
        return polish(fig, "Risk Band Distribution")

    def high_utilization_figure(df):
        if df.empty or "utilization_rate" not in df.columns:
            return master_empty_figure(
                "High Utilization Watchlist",
                "No utilization values available in this active view.",
            )

        util = pd.to_numeric(df["utilization_rate"], errors="coerce").fillna(0)
        high_util_df = df[util >= 0.70].copy()

        if high_util_df.empty:
            return master_empty_figure(
                "High Utilization Watchlist",
                "No customers in this active view have utilization at or above 70%.",
            )

        return decision_count_figure(high_util_df).update_layout(title="High Utilization Watchlist Decisions")

    def blocked_segment_figure(df):
        if df.empty or "decision_status" not in df.columns:
            return master_empty_figure("Blocked Customers by Segment")

        blocked_df = df[df["decision_status"] == "Block"].copy()

        if blocked_df.empty:
            return master_empty_figure(
                "Blocked Customers by Segment",
                "No blocked customers in this active view.",
            )

        counts = value_counts_frame(blocked_df, "customer_segment", "customer_segment", "customers")
        fig = px.bar(
            counts.sort_values("customers", ascending=True),
            x="customers",
            y="customer_segment",
            orientation="h",
            title="Blocked Customers by Segment",
            text="customers",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="Blocked Customer Count", yaxis_title="")
        return polish(fig, "Blocked Customers by Segment")

    figures = (
        decision_share_figure(filtered),
        decision_count_figure(filtered),
        segment_size_figure(filtered),
        segment_eligibility_figure(filtered),
        segment_decision_mix_figure(filtered),
        action_mix_figure(filtered),
        offer_type_figure(filtered),
        risk_band_figure(filtered),
        high_utilization_figure(filtered),
        blocked_segment_figure(filtered),
    )

    return tuple(apply_view_label_to_figure(fig, view_label) for fig in figures)



@app.callback(
    Output("customer-directory-table", "data"),
    Output("customer-directory-table", "selected_rows"),
    Input("customer-directory-search", "value"),
    Input("customer-directory-segment-filter", "value"),
    Input("customer-directory-decision-filter", "value"),
    Input("active-customer-data-store", "data"),
)
def update_customer_directory(search_value, selected_segment, selected_decision, active_data):
    active_features = get_active_customer_features(active_data).copy()

    directory_columns = [
        "customer_id",
        "customer_name",
        "customer_email",
        "city",
        "state",
        "customer_segment",
        "risk_band",
        "decision_status",
    ]

    for column in directory_columns:
        if column not in active_features.columns:
            active_features[column] = "Unknown"

    directory = active_features[directory_columns].copy()
    directory["location"] = directory["city"].astype(str) + ", " + directory["state"].astype(str)

    if selected_segment and "customer_segment" in directory.columns:
        directory = directory[directory["customer_segment"] == selected_segment]

    if selected_decision and "decision_status" in directory.columns:
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

    directory = directory.sort_values("customer_id").head(500)

    if directory.empty:
        return [], []

    return directory.to_dict("records"), [0]



@app.callback(
    Output("customer-lookup-output", "children"),
    Input("customer-directory-table", "selected_rows"),
    Input("customer-directory-table", "derived_virtual_data"),
    Input("active-customer-data-store", "data"),
)
def update_customer_lookup(selected_rows, table_data, active_data):
    active_features = get_active_customer_features(active_data)
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

    selected = active_features[active_features["customer_id"].astype(str) == str(customer_id)]

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
    Output("scenario-marketing-cost", "value"),
    Output("scenario-spend-lift", "value"),
    Output("scenario-risk-threshold", "value"),
    Input("scenario-campaign", "value"),
)
def sync_scenario_inputs_with_campaign(campaign_id: str):
    if campaign_recommendations.empty or campaign_id in [None, "None"]:
        return 5, 8, 8

    selected = campaign_recommendations[
        campaign_recommendations["campaign_id"] == campaign_id
    ]

    if selected.empty:
        return 5, 8, 8

    campaign = selected.iloc[0]

    marketing_cost = int(round(float(campaign.get("cost_per_customer", 5))))
    spend_lift = int(round(float(campaign.get("expected_lift_pct", 0.08)) * 100))

    # Conservative default by campaign risk sensitivity.
    risk_sensitivity = campaign.get("risk_sensitivity", "Medium")
    risk_threshold_map = {
        "Low": 10,
        "Medium": 7,
        "High": 5,
        "Very High": 3,
        "Protective": 16,
    }
    risk_threshold = risk_threshold_map.get(risk_sensitivity, 8)

    marketing_cost = max(1, min(25, marketing_cost))
    spend_lift = max(-2, min(20, spend_lift))
    risk_threshold = max(2, min(20, risk_threshold))

    return marketing_cost, spend_lift, risk_threshold


@app.callback(
    Output("scenario-output", "children"),
    Input("scenario-campaign", "value"),
    Input("scenario-segment", "value"),
    Input("scenario-marketing-cost", "value"),
    Input("scenario-spend-lift", "value"),
    Input("scenario-risk-threshold", "value"),
    Input("active-customer-data-store", "data"),
)
def update_scenario_simulator(
    campaign_id,
    segment,
    marketing_cost,
    spend_lift_percent,
    risk_threshold_percent,
    active_data,
):
    master_df = get_active_customer_features(active_data)

    scenario_df = get_campaign_audience_df(
        campaign_id,
        segment or "All Segments",
        "Eligible",
        source_df=master_df,
    )

    if scenario_df.empty:
        return create_zero_state_card(
            "No scenario audience available",
            "The selected campaign and segment do not return any customers in the active master dataset.",
            "Try All Segments, a different campaign, or upload a larger customer file.",
        )

    marketing_cost = float(marketing_cost or 0)
    spend_lift_percent = float(spend_lift_percent or 0)
    risk_threshold_percent = float(risk_threshold_percent or 0)

    monthly_spend = pd.to_numeric(scenario_df.get("monthly_spend", 0), errors="coerce").fillna(0)
    risk_adjusted_profit = pd.to_numeric(scenario_df.get("risk_adjusted_profit", 0), errors="coerce").fillna(0)
    default_probability = pd.to_numeric(scenario_df.get("default_probability", 0), errors="coerce").fillna(0)

    total_customers = int(len(scenario_df))
    avg_default_probability = float(default_probability.mean()) if total_customers else 0
    total_monthly_spend = float(monthly_spend.sum())
    baseline_profit = float(risk_adjusted_profit.sum())

    incremental_spend = total_monthly_spend * (spend_lift_percent / 100)
    assumed_margin_rate = 0.018
    incremental_margin = incremental_spend * assumed_margin_rate
    total_campaign_cost = total_customers * marketing_cost
    net_incremental_value = incremental_margin - total_campaign_cost
    projected_profit = baseline_profit + net_incremental_value

    risk_pass = avg_default_probability <= (risk_threshold_percent / 100 if risk_threshold_percent > 1 else risk_threshold_percent)
    economics_pass = net_incremental_value > 0

    if risk_pass and economics_pass:
        recommendation = "Scale"
        recommendation_detail = "Economics are positive and the average default probability is within the selected risk threshold."
        accent = "#16a34a"
    elif risk_pass and not economics_pass:
        recommendation = "Test"
        recommendation_detail = "Risk is acceptable, but economics are not strong enough for full rollout without an experiment."
        accent = "#2563eb"
    else:
        recommendation = "Do Not Launch"
        recommendation_detail = "Risk threshold is breached for this active audience."
        accent = "#dc2626"

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div("Scenario Recommendation", style={"fontSize": "13px", "fontWeight": "900", "color": COLORS["muted"], "textTransform": "uppercase"}),
                    html.H2(recommendation, style={"margin": "6px 0 4px 0", "fontSize": "28px", "fontWeight": "900", "color": accent}),
                    html.P(recommendation_detail, style={"margin": "0", "color": COLORS["muted"], "lineHeight": "1.45"}),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "border": f"1px solid {COLORS['border']}",
                    "borderRadius": "18px",
                    "padding": "18px",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                children=[
                    create_kpi_card("Audience Size", f"{total_customers:,}", "Active master dataset customers", "#2563eb"),
                    create_kpi_card("Avg Default Probability", f"{avg_default_probability:.2%}", "Compared with risk threshold", "#dc2626" if not risk_pass else "#16a34a"),
                    create_kpi_card("Incremental Spend", f"${incremental_spend:,.0f}", f"{spend_lift_percent:.1f}% lift assumption", "#0ea5e9"),
                    create_kpi_card("Net Incremental Value", f"${net_incremental_value:,.0f}", "Margin minus campaign cost", "#16a34a" if economics_pass else "#f97316"),
                    create_kpi_card("Projected Profit", f"${projected_profit:,.0f}", "Baseline risk-adjusted profit plus scenario value", "#7c3aed"),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "14px"},
            ),
        ]
    )


@app.callback(
    Output("ab-baseline-rate", "value"),
    Output("ab-lift", "value"),
    Output("ab-test-population", "value"),
    Input("ab-campaign", "value"),
)
def sync_ab_inputs_with_campaign(campaign_id: str):
    if campaign_recommendations.empty or campaign_id in [None, "None"]:
        return 6, 4, 1000

    selected = campaign_recommendations[
        campaign_recommendations["campaign_id"] == campaign_id
    ]

    if selected.empty:
        return 6, 4, 1000

    campaign = selected.iloc[0]

    baseline_rate = int(round(float(campaign.get("avg_predicted_response_rate", 0.06)) * 100))
    expected_lift = int(round(float(campaign.get("expected_lift_pct", 0.04)) * 100))
    test_population = int(min(3000, max(100, round(float(campaign.get("eligible_customers", 1000)) / 2, -2))))

    baseline_rate = max(1, min(20, baseline_rate))
    expected_lift = max(1, min(15, expected_lift))
    test_population = max(100, min(3000, test_population))

    return baseline_rate, expected_lift, test_population


@app.callback(
    Output("ab-output", "children"),
    Input("ab-campaign", "value"),
    Input("ab-segment", "value"),
    Input("ab-baseline-rate", "value"),
    Input("ab-lift", "value"),
    Input("ab-test-population", "value"),
    Input("ab-test-split", "value"),
    Input("ab-test-duration", "value"),
    Input("active-customer-data-store", "data"),
)
def update_ab_test_planner(
    campaign_id,
    segment,
    baseline_rate_percent,
    lift_pp,
    test_population,
    test_split,
    test_duration,
    active_data,
):
    try:
        master_df = get_active_customer_features(active_data)

        safe_segment = segment or "All Segments"

        # If uploaded data is active and the selected segment no longer exists, fall back to All Segments.
        if (
            safe_segment != "All Segments"
            and "customer_segment" in master_df.columns
            and safe_segment not in set(master_df["customer_segment"].dropna().astype(str).unique())
        ):
            safe_segment = "All Segments"

        audience_df = get_campaign_audience_df(
            campaign_id,
            safe_segment,
            "Eligible",
            source_df=master_df,
        )

        if audience_df.empty:
            available_segments = []
            if "customer_segment" in master_df.columns:
                available_segments = sorted(master_df["customer_segment"].dropna().astype(str).unique())

            return html.Div(
                children=[
                    create_zero_state_card(
                        "No A/B audience available",
                        "The selected campaign and segment do not return any eligible customers in the active master dataset.",
                        "Use All Segments, select a segment that exists in the uploaded file, choose a different campaign, or upload a larger customer file.",
                    ),
                    html.Div(
                        children=[
                            html.Strong("Active uploaded segments: "),
                            ", ".join(available_segments) if available_segments else "No segment values found",
                        ],
                        style={
                            "marginTop": "12px",
                            "backgroundColor": "#eff6ff",
                            "border": "1px solid #bfdbfe",
                            "borderRadius": "12px",
                            "padding": "12px",
                            "color": "#1e3a8a",
                        },
                    ),
                ]
            )

        baseline_rate_percent = float(baseline_rate_percent or 0)
        lift_pp = float(lift_pp or 0)
        test_population = int(test_population or 0)
        test_split = parse_test_split_value(test_split)
        test_duration = int(test_duration or 14)

        available_customers = int(len(audience_df))
        planned_population = min(test_population, available_customers) if test_population > 0 else available_customers

        treatment_customers = int(round(planned_population * test_split / 100))
        control_customers = planned_population - treatment_customers

        expected_control_responses = control_customers * (baseline_rate_percent / 100)
        expected_treatment_responses = treatment_customers * ((baseline_rate_percent + lift_pp) / 100)
        incremental_responses = expected_treatment_responses - (treatment_customers * baseline_rate_percent / 100)

        ready = planned_population >= 100 and treatment_customers > 0 and control_customers > 0
        readiness = "Ready to Test" if ready else "Small Sample / Directional Only"
        accent = "#16a34a" if ready else "#f97316"

        return html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            "A/B Test Readiness",
                            style={
                                "fontSize": "13px",
                                "fontWeight": "900",
                                "color": COLORS["muted"],
                                "textTransform": "uppercase",
                            },
                        ),
                        html.H2(
                            readiness,
                            style={
                                "margin": "6px 0 4px 0",
                                "fontSize": "28px",
                                "fontWeight": "900",
                                "color": accent,
                            },
                        ),
                        html.P(
                            f"The planner is using {safe_segment}. Uploaded files automatically become the active master dataset.",
                            style={"margin": "0", "color": COLORS["muted"], "lineHeight": "1.45"},
                        ),
                    ],
                    style={
                        "backgroundColor": "#ffffff",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "18px",
                        "padding": "18px",
                        "marginBottom": "14px",
                    },
                ),
                html.Div(
                    children=[
                        create_kpi_card("Available Audience", f"{available_customers:,}", "From active master dataset", "#2563eb"),
                        create_kpi_card("Planned Test Size", f"{planned_population:,}", f"{test_duration} week test window", "#0ea5e9"),
                        create_kpi_card("Control / Treatment", f"{control_customers:,} / {treatment_customers:,}", f"{100 - test_split:.0f}% / {test_split:.0f}% split", "#7c3aed"),
                        create_kpi_card("Expected Incremental Responses", f"{incremental_responses:,.1f}", f"+{lift_pp:.1f} pp lift assumption", "#16a34a"),
                        create_kpi_card("Decision Gate", "Scale if lift wins", "Scale only if risk guardrails remain clean", "#111827"),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "14px"},
                ),
            ]
        )

    except Exception as exc:
        return html.Div(
            children=[
                html.H4("A/B planner could not run", style={"marginTop": "0", "color": "#991b1b"}),
                html.Div(
                    f"{type(exc).__name__}: {exc}",
                    style={"fontFamily": "monospace", "fontSize": "13px", "whiteSpace": "pre-wrap"},
                ),
                html.Div(
                    "This is now handled inside the dashboard instead of breaking the server.",
                    style={"marginTop": "8px", "color": COLORS["muted"]},
                ),
            ],
            style={
                "backgroundColor": "#fef2f2",
                "border": "1px solid #fecaca",
                "borderRadius": "14px",
                "padding": "16px",
                "color": "#7f1d1d",
            },
        )


@app.callback(
    Output("ab-customer-list-table", "data"),
    Output("ab-customer-list-table", "columns"),
    Output("ab-customer-export-summary", "children"),
    Input("ab-campaign", "value"),
    Input("ab-segment", "value"),
    Input("ab-audience-type", "value"),
    Input("active-customer-data-store", "data"),
)
def update_ab_customer_export_preview(campaign_id, segment, audience_type, active_data):
    master_df = get_active_customer_features(active_data)

    audience_df = get_campaign_audience_df(
        campaign_id,
        segment or "All Segments",
        audience_type or "Eligible",
        source_df=master_df,
    )

    if audience_df.empty:
        summary = create_zero_state_card(
            "No customers available for export",
            "The selected A/B audience returned zero customers in the active master dataset.",
            "Try All Segments, Eligible audience, or upload a larger file.",
        )
        return [], [], summary

    preview_df = format_customer_explorer_preview(audience_df, limit=500)
    columns = [{"name": column, "id": column} for column in preview_df.columns]

    summary = html.Div(
        children=[
            create_kpi_card("Export Audience", f"{len(audience_df):,}", "Customers matching A/B audience", "#2563eb"),
            create_kpi_card("Preview Rows", f"{len(preview_df):,}", "Shown in table preview", "#0ea5e9"),
            create_kpi_card("Data Source", "Uploaded" if active_data else "Synthetic", "Active master dataset", "#7c3aed"),
        ],
        style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px", "marginBottom": "14px"},
    )

    return preview_df.to_dict("records"), columns, summary


@app.callback(
    Output("download-ab-customer-list", "data"),
    Input("download-ab-customer-list-button", "n_clicks"),
    State("ab-campaign", "value"),
    State("ab-segment", "value"),
    State("ab-audience-type", "value"),
    State("active-customer-data-store", "data"),
    prevent_initial_call=True,
)
def download_ab_customer_list(n_clicks, campaign_id, segment, audience_type, active_data):
    if not n_clicks:
        raise PreventUpdate

    master_df = get_active_customer_features(active_data)

    export_df = get_campaign_audience_df(
        campaign_id,
        segment or "All Segments",
        audience_type or "Eligible",
        source_df=master_df,
    )

    if export_df.empty:
        raise PreventUpdate

    export_df = format_customer_explorer_preview(export_df, limit=len(export_df))
    return dcc.send_data_frame(export_df.to_csv, "ab_customer_list.csv", index=False)


@app.callback(
    Output("customer-explorer-summary", "children"),
    Output("customer-explorer-table", "data"),
    Output("customer-explorer-table", "columns"),
    Input("customer-explorer-search", "value"),
    Input("customer-explorer-segment", "value"),
    Input("customer-explorer-decision", "value"),
    Input("customer-explorer-risk", "value"),
    Input("customer-explorer-state", "value"),
    Input("customer-explorer-card-type", "value"),
    Input("customer-explorer-campaign", "value"),
    Input("customer-explorer-audience-type", "value"),
    Input("active-customer-data-store", "data"),
)
def update_customer_explorer(
    search_text,
    segment,
    decision,
    risk_band,
    state,
    card_type,
    campaign_id,
    audience_type,
    active_data,
):
    active_features = get_active_customer_features(active_data)
    explorer_df = active_features.copy()

    # Campaign targeting layer, applied to the active dataset.
    if campaign_id and campaign_id != "None" and not campaign_recommendations.empty:
        campaign_match = campaign_recommendations[
            campaign_recommendations["campaign_id"] == campaign_id
        ]

        if not campaign_match.empty:
            selected_campaign = campaign_match.iloc[0]
            target_segments = split_semicolon_values(selected_campaign.get("target_segments", ""))
            excluded_segments = split_semicolon_values(selected_campaign.get("excluded_segments", ""))

            if target_segments and "customer_segment" in explorer_df.columns:
                explorer_df = explorer_df[
                    explorer_df["customer_segment"].isin(target_segments)
                ].copy()

            if excluded_segments and "customer_segment" in explorer_df.columns:
                explorer_df = explorer_df[
                    ~explorer_df["customer_segment"].isin(excluded_segments)
                ].copy()

            if audience_type == "Eligible" and "decision_status" in explorer_df.columns:
                explorer_df = explorer_df[
                    explorer_df["decision_status"].isin(["Scale", "Test"])
                ].copy()
            elif audience_type == "Scale" and "decision_status" in explorer_df.columns:
                explorer_df = explorer_df[explorer_df["decision_status"] == "Scale"].copy()
            elif audience_type == "Test" and "decision_status" in explorer_df.columns:
                explorer_df = explorer_df[explorer_df["decision_status"] == "Test"].copy()
            elif audience_type == "Blocked" and "decision_status" in explorer_df.columns:
                explorer_df = explorer_df[explorer_df["decision_status"] == "Block"].copy()

    # Manual filters.
    if segment and segment != "All Segments" and "customer_segment" in explorer_df.columns:
        explorer_df = explorer_df[explorer_df["customer_segment"] == segment].copy()

    if decision and decision != "All Decisions" and "decision_status" in explorer_df.columns:
        explorer_df = explorer_df[explorer_df["decision_status"] == decision].copy()

    if risk_band and risk_band != "All Risk Bands" and "risk_band" in explorer_df.columns:
        explorer_df = explorer_df[explorer_df["risk_band"] == risk_band].copy()

    if state and state != "All States" and "state" in explorer_df.columns:
        explorer_df = explorer_df[explorer_df["state"] == state].copy()

    if card_type and card_type != "All Card Types" and "card_type" in explorer_df.columns:
        explorer_df = explorer_df[explorer_df["card_type"] == card_type].copy()

    if search_text:
        search_value = str(search_text).strip().lower()
        searchable_columns = [
            column for column in [
                "customer_id",
                "customer_name",
                "customer_email",
                "city",
                "state",
                "customer_segment",
                "decision_status",
                "risk_band",
                "recommended_action",
                "card_type",
                "rewards_preference",
            ]
            if column in explorer_df.columns
        ]

        if searchable_columns:
            search_mask = pd.Series(False, index=explorer_df.index)

            for column in searchable_columns:
                search_mask = search_mask | explorer_df[column].astype(str).str.lower().str.contains(search_value, na=False)

            explorer_df = explorer_df[search_mask].copy()

    if explorer_df.empty:
        summary = create_zero_state_card(
            "No customers match these filters",
            "The selected filters, campaign, or audience type returned zero customers.",
            "Try widening the segment, selecting All Decisions, clearing search, or choosing a different campaign audience.",
        )
        return summary, [], []

    preview_df = format_customer_explorer_preview(explorer_df, limit=500)

    decision_counts = explorer_df["decision_status"].value_counts().to_dict() if "decision_status" in explorer_df.columns else {}
    scale_count = int(decision_counts.get("Scale", 0))
    test_count = int(decision_counts.get("Test", 0))
    block_count = int(decision_counts.get("Block", 0))
    do_not_launch_count = int(decision_counts.get("Do Not Launch", 0))

    summary = html.Div(
        children=[
            create_kpi_card("Matched Customers", f"{len(explorer_df):,}", "Customers matching current filters", "#2563eb"),
            create_kpi_card("Preview Rows", f"{len(preview_df):,}", "Rows shown in table preview", "#0ea5e9"),
            create_kpi_card("Scale / Test / Block", f"{scale_count:,} / {test_count:,} / {block_count:,}", "Decision split", "#7c3aed"),
            create_kpi_card("Do Not Launch", f"{do_not_launch_count:,}", "Customers not recommended for launch", "#9ca3af"),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "14px",
            "marginBottom": "14px",
        },
    )

    columns = [{"name": column, "id": column} for column in preview_df.columns]

    return summary, preview_df.to_dict("records"), columns



@app.callback(
    Output("download-customer-explorer-csv", "data"),
    Input("download-customer-explorer-csv-button", "n_clicks"),
    State("customer-explorer-search", "value"),
    State("customer-explorer-segment", "value"),
    State("customer-explorer-decision", "value"),
    State("customer-explorer-risk", "value"),
    State("customer-explorer-state", "value"),
    State("customer-explorer-card-type", "value"),
    State("customer-explorer-campaign", "value"),
    State("customer-explorer-audience-type", "value"),
    State("active-customer-data-store", "data"),
    prevent_initial_call=True,
)
def download_customer_explorer_csv(
    n_clicks,
    search_text,
    segment,
    decision,
    risk_band,
    state,
    card_type,
    campaign_id,
    audience_type,
    active_data,
):
    if not n_clicks:
        raise PreventUpdate

    master_df = get_active_customer_features(active_data)

    explorer_df = build_customer_explorer_dataframe(
        search_text,
        segment,
        decision,
        risk_band,
        state,
        card_type,
        campaign_id,
        audience_type,
        source_df=master_df,
    )

    if explorer_df.empty:
        raise PreventUpdate

    filename = "customer_explorer_export.csv"

    return dcc.send_data_frame(explorer_df.to_csv, filename, index=False)


@app.callback(
    Output("download-customer-explorer-excel", "data"),
    Input("download-customer-explorer-excel-button", "n_clicks"),
    State("customer-explorer-search", "value"),
    State("customer-explorer-segment", "value"),
    State("customer-explorer-decision", "value"),
    State("customer-explorer-risk", "value"),
    State("customer-explorer-state", "value"),
    State("customer-explorer-card-type", "value"),
    State("customer-explorer-campaign", "value"),
    State("customer-explorer-audience-type", "value"),
    State("active-customer-data-store", "data"),
    prevent_initial_call=True,
)
def download_customer_explorer_excel(
    n_clicks,
    search_text,
    segment,
    decision,
    risk_band,
    state,
    card_type,
    campaign_id,
    audience_type,
    active_data,
):
    if not n_clicks:
        raise PreventUpdate

    master_df = get_active_customer_features(active_data)

    explorer_df = build_customer_explorer_dataframe(
        search_text,
        segment,
        decision,
        risk_band,
        state,
        card_type,
        campaign_id,
        audience_type,
        source_df=master_df,
    )

    if explorer_df.empty:
        raise PreventUpdate

    filename = "customer_explorer_export.xlsx"

    return dcc.send_data_frame(explorer_df.to_excel, filename, index=False)



@app.callback(
    Output("download-ab-customer-list-excel", "data"),
    Input("download-ab-customer-list-excel-button", "n_clicks"),
    State("ab-campaign", "value"),
    State("ab-segment", "value"),
    State("ab-audience-type", "value"),
    State("active-customer-data-store", "data"),
    prevent_initial_call=True,
)
def download_ab_customer_list_excel(n_clicks, campaign_id, segment, audience_type, active_data):
    if not n_clicks:
        raise PreventUpdate

    master_df = get_active_customer_features(active_data)

    export_df = get_campaign_audience_df(
        campaign_id,
        segment or "All Segments",
        audience_type or "Eligible",
        source_df=master_df,
    )

    if export_df.empty:
        raise PreventUpdate

    export_df = format_customer_explorer_preview(export_df, limit=len(export_df))
    return dcc.send_data_frame(export_df.to_excel, "ab_customer_list.xlsx", index=False, engine="openpyxl")


@app.callback(
    Output("main-tabs", "value"),
    Output("workbench-tabs", "value"),
    Input("guide-open-overview", "n_clicks"),
    Input("guide-open-campaigns", "n_clicks"),
    Input("guide-open-scenario", "n_clicks"),
    Input("guide-open-ab", "n_clicks"),
    Input("guide-open-export", "n_clicks"),
    Input("guide-open-guardrails", "n_clicks"),
    Input("action-open-campaigns", "n_clicks"),
    Input("action-open-customer-tools", "n_clicks"),
    Input("action-open-guardrails", "n_clicks"),
    Input("cta-open-scenario", "n_clicks"),
    Input("cta-open-ab", "n_clicks"),
    Input("cta-open-audience", "n_clicks"),
    Input("cta-open-guardrails", "n_clicks"),
    Input("cta-playbook-campaigns", "n_clicks"),
    Input("cta-playbook-scenario", "n_clicks"),
    Input("cta-playbook-audience", "n_clicks"),
    Input("cta-playbook-guardrails", "n_clicks"),
    Input("cta-guardrails-audience", "n_clicks"),
    Input("cta-guardrails-playbook", "n_clicks"),
    Input("cta-guardrails-ab", "n_clicks"),
    prevent_initial_call=True,
)
def navigate_from_dashboard_guides(
    guide_open_overview,
    guide_open_campaigns,
    guide_open_scenario,
    guide_open_ab,
    guide_open_export,
    guide_open_guardrails,
    action_open_campaigns,
    action_open_customer_tools,
    action_open_guardrails,
    cta_open_scenario,
    cta_open_ab,
    cta_open_audience,
    cta_open_guardrails,
    cta_playbook_campaigns,
    cta_playbook_scenario,
    cta_playbook_audience,
    cta_playbook_guardrails,
    cta_guardrails_audience,
    cta_guardrails_playbook,
    cta_guardrails_ab,
):
    ctx = callback_context

    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    route_map = {
        "guide-open-overview": ("overview", "customer-lookup"),
        "guide-open-campaigns": ("campaigns-offers", "customer-lookup"),
        "guide-open-scenario": ("decision-workbench", "scenario"),
        "guide-open-ab": ("decision-workbench", "ab-test-planner"),
        "guide-open-export": ("decision-workbench", "customer-explorer"),
        "guide-open-guardrails": ("guardrails", "customer-lookup"),

        "action-open-campaigns": ("campaigns-offers", "customer-lookup"),
        "action-open-customer-tools": ("decision-workbench", "customer-explorer"),
        "action-open-guardrails": ("guardrails", "customer-lookup"),

        "cta-open-scenario": ("decision-workbench", "scenario"),
        "cta-open-ab": ("decision-workbench", "ab-test-planner"),
        "cta-open-audience": ("decision-workbench", "customer-explorer"),
        "cta-open-guardrails": ("guardrails", "customer-lookup"),

        "cta-playbook-campaigns": ("campaigns-offers", "customer-lookup"),
        "cta-playbook-scenario": ("decision-workbench", "scenario"),
        "cta-playbook-audience": ("decision-workbench", "customer-explorer"),
        "cta-playbook-guardrails": ("guardrails", "customer-lookup"),

        "cta-guardrails-audience": ("decision-workbench", "customer-explorer"),
        "cta-guardrails-playbook": ("strategy-playbook", "customer-lookup"),
        "cta-guardrails-ab": ("decision-workbench", "ab-test-planner"),
    }

    if trigger not in route_map:
        raise PreventUpdate

    return route_map[trigger]




@app.callback(
    Output("filtered-summary-output", "children"),
    Input("filter-segment", "value"),
    Input("filter-decision", "value"),
    Input("filter-risk", "value"),
    Input("filter-action", "value"),
    Input("active-customer-data-store", "data"),
)
def update_filtered_summary(selected_segments, selected_decisions, selected_risks, selected_actions, active_data):
    active_features = get_active_customer_features(active_data)
    filtered = apply_global_filters(
        selected_segments,
        selected_decisions,
        selected_risks,
        selected_actions,
        active_features,
    )

    def safe_sum(column_name):
        if column_name in filtered.columns:
            return pd.to_numeric(filtered[column_name], errors="coerce").fillna(0).sum()
        return 0

    def safe_count_decision(values):
        if "decision_status" not in filtered.columns:
            return 0
        return int(filtered["decision_status"].isin(values).sum())

    total_customers = int(len(filtered))
    total_spend = safe_sum("monthly_spend")
    total_profit = safe_sum("risk_adjusted_profit")
    eligible_count = safe_count_decision(["Scale", "Test"])
    blocked_count = safe_count_decision(["Block"])

    eligible_rate = (eligible_count / total_customers * 100) if total_customers else 0
    block_rate = (blocked_count / total_customers * 100) if total_customers else 0

    mode_label = "Uploaded active file" if active_data else "Synthetic demo portfolio"

    def live_card(title, value, subtitle, accent):
        return html.Div(
            children=[
                html.Div(
                    style={
                        "width": "42px",
                        "height": "4px",
                        "borderRadius": "999px",
                        "backgroundColor": accent,
                        "marginBottom": "10px",
                    }
                ),
                html.Div(title, style={"fontSize": "12px", "fontWeight": "900", "color": COLORS["muted"], "textTransform": "uppercase"}),
                html.Div(value, style={"fontSize": "26px", "fontWeight": "900", "marginTop": "6px", "color": COLORS["text"]}),
                html.Div(subtitle, style={"fontSize": "13px", "color": COLORS["muted"], "marginTop": "5px"}),
            ],
            style={
                "backgroundColor": "#ffffff",
                "border": f"1px solid {COLORS['border']}",
                "borderRadius": "16px",
                "padding": "16px",
                "boxShadow": "0 8px 20px rgba(15, 23, 42, 0.05)",
            },
        )

    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Strong(f"Active view: {mode_label}"),
                    html.Span(
                        f" | Current filtered population: {total_customers:,} customers",
                        style={"color": COLORS["muted"]},
                    ),
                ],
                style={
                    "gridColumn": "1 / -1",
                    "backgroundColor": "#eff6ff" if active_data else "#f8fafc",
                    "border": "1px solid #bfdbfe" if active_data else f"1px solid {COLORS['border']}",
                    "borderRadius": "14px",
                    "padding": "12px 14px",
                    "fontSize": "14px",
                    "marginBottom": "2px",
                },
            ),
            live_card("Filtered Customers", f"{total_customers:,}", "After active filters", "#2563eb"),
            live_card("Monthly Spend", f"${total_spend:,.0f}", "Filtered portfolio spend", "#0ea5e9"),
            live_card("Risk-Adjusted Profit", f"${total_profit:,.0f}", "Profit after risk cost", "#16a34a"),
            live_card("Scale/Test Rate", f"{eligible_rate:.1f}%", f"{eligible_count:,} eligible customers", "#7c3aed"),
            live_card("Block Rate", f"{block_rate:.1f}%", f"{blocked_count:,} protected customers", "#dc2626"),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(5, 1fr)",
            "gap": "14px",
            "marginTop": "14px",
            "marginBottom": "14px",
        },
    )


def build_priority_segment_rows_from_active_data(df: pd.DataFrame) -> list[dict]:
    """Build the Segment Strategy priority table from the active master dataset."""
    if df is None or df.empty or "customer_segment" not in df.columns:
        return []

    working = df.copy()

    for column in ["risk_adjusted_profit", "expected_roi"]:
        if column not in working.columns:
            working[column] = 0
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)

    if "decision_status" not in working.columns:
        working["decision_status"] = "Unknown"

    working["scale_flag"] = (working["decision_status"] == "Scale").astype(int)
    working["test_flag"] = (working["decision_status"] == "Test").astype(int)
    working["eligible_flag"] = working["decision_status"].isin(["Scale", "Test"]).astype(int)

    grouped = (
        working.groupby("customer_segment", as_index=False)
        .agg(
            customer_count=("customer_segment", "size"),
            avg_risk_adjusted_profit=("risk_adjusted_profit", "mean"),
            avg_expected_roi=("expected_roi", "mean"),
            scale_count=("scale_flag", "sum"),
            test_count=("test_flag", "sum"),
            eligible_count=("eligible_flag", "sum"),
        )
    )

    grouped["campaign_eligible_rate"] = grouped["eligible_count"] / grouped["customer_count"] * 100
    grouped = grouped.sort_values(["campaign_eligible_rate", "customer_count"], ascending=[False, False])

    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "Customer Segment": row["customer_segment"],
                "Customer Count": f"{int(row['customer_count']):,}",
                "Avg Risk Adjusted Profit": format_currency(row["avg_risk_adjusted_profit"]),
                "Avg Expected Roi": f"{row['avg_expected_roi']:.1f}",
                "Scale Count": f"{int(row['scale_count']):,}",
                "Test Count": f"{int(row['test_count']):,}",
                "Campaign Eligible Rate": f"{row['campaign_eligible_rate']:.1f}%",
            }
        )

    return rows


@app.callback(
    Output("priority-segment-table-container", "children"),
    Input("active-customer-data-store", "data"),
)
def update_priority_segment_table(active_data):
    master_df = get_active_customer_features(active_data)
    rows = build_priority_segment_rows_from_active_data(master_df)

    if not rows:
        return create_zero_state_card(
            "No segment table available",
            "The active master dataset does not contain enough segment information.",
            "Upload a valid customer file or return to the synthetic demo portfolio.",
        )

    return create_table(rows)


@app.callback(
    Output("data-source-mode-title", "children"),
    Output("data-source-mode-description", "children"),
    Input("active-data-mode-store", "data"),
)
def update_data_source_mode_text(mode_data):
    if mode_data and mode_data.get("mode") == "uploaded":
        filename = mode_data.get("filename", "uploaded file")
        rows = mode_data.get("rows", 0)
        return (
            "Uploaded customer file active",
            f"The dashboard is currently powered by {filename}. {rows:,} uploaded customers were scored through the decision engine and stored as the active master dataset.",
        )

    return (
        "Synthetic demo portfolio",
        "The dashboard is currently powered by the built-in synthetic customer portfolio. Upload a valid CSV or Excel file to replace the active master dataset.",
    )





CAMPAIGN_PROFIT_COLUMNS = [
    "expected_profit",
    "projected_profit",
    "campaign_profit",
    "profit",
    "expected_campaign_profit",
    "incremental_profit",
]

CAMPAIGN_ROI_COLUMNS = [
    "expected_roi",
    "roi",
    "campaign_roi",
    "projected_roi",
]


def get_first_numeric_campaign_value(row_dict: dict, candidate_columns: list[str], default: float = 0) -> float:
    """Return first available numeric campaign metric from possible column names."""
    for column in candidate_columns:
        if column in row_dict:
            value = pd.to_numeric(pd.Series([row_dict.get(column)]), errors="coerce").fillna(default).iloc[0]
            return float(value)
    return float(default)


def set_campaign_metric_aliases(row_dict: dict, profit_value: float, roi_value: float) -> dict:
    """Keep campaign KPI, chart, card, and table views reading the same numbers."""
    for column in CAMPAIGN_PROFIT_COLUMNS:
        row_dict[column] = float(profit_value)

    for column in CAMPAIGN_ROI_COLUMNS:
        row_dict[column] = float(roi_value)

    return row_dict


def parse_customer_id_set(value) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    }




def get_campaign_economics_profile(campaign_dict: dict) -> dict:
    """Return simple economic assumptions by campaign family/risk type."""
    family = str(campaign_dict.get("campaign_family", "")).lower()
    risk_level = str(campaign_dict.get("risk_level", "")).lower()

    profile = {
        "spend_lift_rate": 0.010,
        "profit_capture_rate": 0.08,
        "cost_per_customer": 2.00,
        "risk_penalty_multiplier": 0.015,
    }

    if "dining" in family or "merchant" in family:
        profile.update(
            {
                "spend_lift_rate": 0.025,
                "profit_capture_rate": 0.10,
                "cost_per_customer": 3.00,
                "risk_penalty_multiplier": 0.010,
            }
        )
    elif "retention" in family or "loyalty" in family:
        profile.update(
            {
                "spend_lift_rate": 0.010,
                "profit_capture_rate": 0.22,
                "cost_per_customer": 5.00,
                "risk_penalty_multiplier": 0.008,
            }
        )
    elif "balance" in family or "revolver" in family:
        profile.update(
            {
                "spend_lift_rate": 0.018,
                "profit_capture_rate": 0.18,
                "cost_per_customer": 4.00,
                "risk_penalty_multiplier": 0.030,
            }
        )
    elif "credit line" in family:
        profile.update(
            {
                "spend_lift_rate": 0.015,
                "profit_capture_rate": 0.16,
                "cost_per_customer": 4.00,
                "risk_penalty_multiplier": 0.040,
            }
        )
    elif "servicing" in family or "enrollment" in family:
        profile.update(
            {
                "spend_lift_rate": 0.004,
                "profit_capture_rate": 0.12,
                "cost_per_customer": 1.50,
                "risk_penalty_multiplier": 0.006,
            }
        )

    if "high" in risk_level:
        profile["risk_penalty_multiplier"] *= 1.35
    elif "low" in risk_level:
        profile["risk_penalty_multiplier"] *= 0.75

    return profile


def calculate_campaign_economics_from_customers(campaign_dict: dict, eligible_df: pd.DataFrame) -> tuple[float, float, float]:
    """Calculate expected campaign profit, ROI, and score from active customer-level economics."""
    if eligible_df is None or eligible_df.empty:
        return 0.0, 0.0, -999.0

    profile = get_campaign_economics_profile(campaign_dict)

    monthly_spend = pd.to_numeric(
        eligible_df["monthly_spend"] if "monthly_spend" in eligible_df.columns else 0,
        errors="coerce",
    ).fillna(0)

    risk_adjusted_profit = pd.to_numeric(
        eligible_df["risk_adjusted_profit"] if "risk_adjusted_profit" in eligible_df.columns else 0,
        errors="coerce",
    ).fillna(0)

    default_probability = pd.to_numeric(
        eligible_df["default_probability"] if "default_probability" in eligible_df.columns else 0,
        errors="coerce",
    ).fillna(0)

    customers = int(len(eligible_df))
    total_spend = float(monthly_spend.sum())
    total_profit = float(risk_adjusted_profit.sum())
    avg_default_probability = float(default_probability.mean()) if customers else 0.0

    interchange_margin_rate = 0.018
    incremental_margin = total_spend * profile["spend_lift_rate"] * interchange_margin_rate
    retained_profit_value = total_profit * profile["profit_capture_rate"]
    campaign_cost = customers * profile["cost_per_customer"]
    risk_penalty = total_spend * avg_default_probability * profile["risk_penalty_multiplier"]

    expected_profit = incremental_margin + retained_profit_value - campaign_cost - risk_penalty
    expected_roi = expected_profit / campaign_cost if campaign_cost > 0 else 0.0

    # Keep score interpretable: economics + safer risk + larger matched opportunity.
    score = (
        expected_profit
        + (expected_roi * 8)
        + (customers * 2)
        - (avg_default_probability * 100)
    )

    return float(expected_profit), float(expected_roi), float(score)




def infer_active_campaign_rollout(campaign_dict: dict, audience_df: pd.DataFrame, eligible_df: pd.DataFrame, scale_df: pd.DataFrame, blocked_df: pd.DataFrame) -> tuple[str, str]:
    """Infer campaign rollout from active customer decisions, not static campaign-library labels."""
    eligible_count = int(len(eligible_df)) if eligible_df is not None else 0
    scale_count = int(len(scale_df)) if scale_df is not None else 0
    blocked_count = int(len(blocked_df)) if blocked_df is not None else 0
    audience_count = int(len(audience_df)) if audience_df is not None else 0

    family = str(campaign_dict.get("campaign_family", "")).lower()
    risk_level = str(campaign_dict.get("risk_level", "")).lower()
    campaign_name = str(campaign_dict.get("campaign_name", campaign_dict.get("campaign", ""))).lower()

    servicing_family = any(term in family for term in ["servicing", "enrollment", "balance health", "account health"])
    protective_campaign = any(term in campaign_name for term in ["payment", "autopay", "paperless", "fraud", "account health", "balance health", "alert"])

    if eligible_count <= 0:
        return "Do Not Launch", "No active customers match the campaign audience after guardrails."

    if scale_count > 0 and blocked_count == 0 and scale_count == eligible_count:
        return "Scale", "All matched active customers are Scale-ready and no blocked customers are in the campaign audience."

    if servicing_family or protective_campaign:
        if blocked_count > 0:
            return "Controlled Servicing", "This is a servicing/protective campaign, but the audience includes blocked or high-risk customers, so use controlled servicing language and avoid growth framing."
        if scale_count == 0:
            return "Test", "Matched customers are eligible but not Scale-ready; use a controlled test or servicing pilot."
        return "Controlled Servicing", "Campaign is servicing/protective; rollout should focus on risk prevention and customer support rather than aggressive growth."

    if blocked_count > 0 and audience_count > 0:
        blocked_share = blocked_count / max(audience_count, 1)
        if blocked_share >= 0.25:
            return "Constrain", "Audience includes meaningful blocked/risky customers; limit rollout and review guardrails before launch."

    if scale_count == 0:
        return "Test", "Matched active customers are eligible but none are Scale-ready, so broad rollout is not justified."

    if scale_count < eligible_count:
        return "Test", "Audience mixes Scale and Test customers; validate with an experiment before broader rollout."

    if "high" in risk_level or "protective" in risk_level:
        return "Constrain", "Campaign has elevated risk/protective posture, so rollout should be limited."

    return "Test", "Defaulting to controlled test because active data does not support a full Scale recommendation."



def build_active_campaign_recommendations(master_df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate campaign recommendation counts and economics from the active master dataset."""
    if campaign_recommendations.empty or master_df is None or master_df.empty:
        return pd.DataFrame()

    active_rows = []

    for _, campaign in campaign_recommendations.copy().iterrows():
        campaign_dict = campaign.to_dict()

        audience_df = master_df.copy()

        target_segments = split_semicolon_values(campaign_dict.get("target_segments", ""))
        excluded_segments = split_semicolon_values(campaign_dict.get("excluded_segments", ""))

        if target_segments and "customer_segment" in audience_df.columns:
            audience_df = audience_df[audience_df["customer_segment"].isin(target_segments)].copy()

        if excluded_segments and "customer_segment" in audience_df.columns:
            audience_df = audience_df[~audience_df["customer_segment"].isin(excluded_segments)].copy()

        if "decision_status" in audience_df.columns:
            eligible_df = audience_df[audience_df["decision_status"].isin(["Scale", "Test"])].copy()
            scale_df = audience_df[audience_df["decision_status"] == "Scale"].copy()
            blocked_df = audience_df[audience_df["decision_status"] == "Block"].copy()
        else:
            eligible_df = audience_df.copy()
            scale_df = audience_df.iloc[0:0].copy()
            blocked_df = audience_df.iloc[0:0].copy()

        eligible_customers = int(len(eligible_df))
        scale_customers = int(len(scale_df))
        blocked_customers = int(len(blocked_df))

        expected_profit, expected_roi, active_campaign_score = calculate_campaign_economics_from_customers(
            campaign_dict,
            eligible_df,
        )

        active_rollout, active_rollout_reason = infer_active_campaign_rollout(
            campaign_dict,
            audience_df,
            eligible_df,
            scale_df,
            blocked_df,
        )

        campaign_dict["recommended_rollout_decision"] = active_rollout
        campaign_dict["rollout_recommendation"] = active_rollout
        campaign_dict["recommended_rollout"] = active_rollout
        campaign_dict["active_rollout_reason"] = active_rollout_reason

        campaign_dict["eligible_customers"] = eligible_customers
        campaign_dict["scale_customers"] = scale_customers
        campaign_dict["blocked_customers"] = blocked_customers
        campaign_dict["eligible_customer_ids"] = ";".join(sorted(eligible_df["customer_id"].astype(str).unique())) if "customer_id" in eligible_df.columns else ""
        campaign_dict["scale_customer_ids"] = ";".join(sorted(scale_df["customer_id"].astype(str).unique())) if "customer_id" in scale_df.columns else ""
        campaign_dict["blocked_customer_ids"] = ";".join(sorted(blocked_df["customer_id"].astype(str).unique())) if "customer_id" in blocked_df.columns else ""
        campaign_dict["active_unique_customers"] = int(master_df["customer_id"].nunique()) if "customer_id" in master_df.columns else int(len(master_df))
        campaign_dict = set_campaign_metric_aliases(campaign_dict, expected_profit, expected_roi)
        campaign_dict["campaign_score"] = active_campaign_score
        campaign_dict["active_campaign_score"] = active_campaign_score

        # Do not recommend campaigns with no active audience.
        if eligible_customers > 0:
            active_rows.append(campaign_dict)

    if not active_rows:
        return pd.DataFrame()

    active_df = pd.DataFrame(active_rows)

    if "campaign_score" in active_df.columns:
        active_df["campaign_score"] = pd.to_numeric(active_df["campaign_score"], errors="coerce").fillna(-999)
        active_df["expected_profit"] = pd.to_numeric(active_df["expected_profit"], errors="coerce").fillna(0)
        active_df = active_df.sort_values(["campaign_score", "expected_profit"], ascending=[False, False])
    elif "expected_profit" in active_df.columns:
        active_df["expected_profit"] = pd.to_numeric(active_df["expected_profit"], errors="coerce").fillna(0)
        active_df = active_df.sort_values("expected_profit", ascending=False)

    active_df = active_df.reset_index(drop=True)
    active_df["rank"] = active_df.index + 1
    active_df["dashboard_recommendation_rank"] = active_df["rank"]

    return active_df


def empty_campaign_figure(title: str, message: str = "No campaign recommendations available for the active dataset."):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": COLORS["muted"]},
    )
    fig.update_layout(
        title=title,
        height=380,
        margin={"l": 50, "r": 30, "t": 70, "b": 50},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


@app.callback(
    Output("campaign-kpi-container", "children"),
    Output("campaign-family-mix-chart", "figure"),
    Output("campaign-rollout-mix-chart", "figure"),
    Output("campaign-profit-chart", "figure"),
    Output("campaign-top-cards-container", "children"),
    Output("campaign-table-container", "children"),
    Output("campaign-detail-container", "children"),
    Input("active-customer-data-store", "data"),
)
def update_campaigns_from_active_master(active_data):
    master_df = get_active_customer_features(active_data)
    active_campaigns = build_active_campaign_recommendations(master_df)

    total_templates = len(campaign_library) if not campaign_library.empty else len(campaign_recommendations)

    if active_campaigns.empty:
        kpis = [
            create_kpi_card("Campaign Templates", f"{total_templates:,}", "Available campaign options scored", "#2563eb"),
            create_kpi_card("Active Recommendations", "0", "No matching active customer audience", "#dc2626"),
            create_kpi_card("Eligible Customers", "0", "Customer-campaign matches passing guardrails", "#7c3aed"),
            create_kpi_card("Scale Customers", "0", "Customers recommended for broad rollout", "#0ea5e9"),
        ]

        empty_fig = empty_campaign_figure("No active campaign recommendations")
        zero_state = create_zero_state_card(
            "No campaign recommendations available",
            "The active master dataset does not match any campaign audience rules.",
            "Try All Segments, upload a larger file, or review campaign target-segment rules.",
        )

        return kpis, empty_fig, empty_fig, empty_fig, zero_state, zero_state, zero_state

    top10 = active_campaigns.head(10).copy()

    top_campaign_profit = float(pd.to_numeric(top10["expected_profit"], errors="coerce").fillna(0).sum())
    top_campaign_eligible = int(pd.to_numeric(top10["eligible_customers"], errors="coerce").fillna(0).sum())
    top_campaign_scale = int(pd.to_numeric(top10["scale_customers"], errors="coerce").fillna(0).sum())

    unique_covered_customer_ids = set()
    if "eligible_customer_ids" in top10.columns:
        for ids in top10["eligible_customer_ids"]:
            unique_covered_customer_ids.update(parse_customer_id_set(ids))

    unique_customers_covered = len(unique_covered_customer_ids)

    kpis = [
        create_kpi_card("Campaign Templates", f"{total_templates:,}", "Available campaign options scored", "#2563eb"),
        create_kpi_card("Unique Customers Covered", f"{unique_customers_covered:,}", "Unique people appearing in top campaign opportunities", "#0ea5e9"),
        create_kpi_card("Top-10 Customer-Campaign Matches", f"{top_campaign_eligible:,}", "Customers can appear in more than one campaign", "#7c3aed"),
        create_kpi_card("Top-10 Expected Profit", format_currency(top_campaign_profit), "Projected profit from active recommendations", "#16a34a"),
    ]

    if "campaign_family" in top10.columns:
        family_counts = top10["campaign_family"].fillna("Unknown").value_counts().reset_index()
        family_counts.columns = ["campaign_family", "campaigns"]
        family_fig = px.bar(
            family_counts.sort_values("campaigns", ascending=True),
            x="campaigns",
            y="campaign_family",
            orientation="h",
            title="Top Campaign Families",
            text="campaigns",
        )
        family_fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        family_fig.update_layout(showlegend=False, height=380, margin={"l": 50, "r": 30, "t": 70, "b": 50})
    else:
        family_fig = empty_campaign_figure("Top Campaign Families")

    rollout_col = None
    for candidate_col in [
        "recommended_rollout_decision",
        "rollout_recommendation",
        "recommended_rollout",
        "rollout",
    ]:
        if candidate_col in top10.columns:
            rollout_col = candidate_col
            break

    if rollout_col:
        rollout_counts = top10[rollout_col].fillna("Unknown").astype(str).value_counts().reset_index()
        rollout_counts.columns = ["rollout", "campaigns"]
        rollout_fig = px.pie(
            rollout_counts,
            names="rollout",
            values="campaigns",
            hole=0.55,
            title="Recommended Rollout Mix",
        )
        rollout_fig.update_traces(textposition="inside", textinfo="percent+label")
        rollout_fig.update_layout(height=380, margin={"l": 50, "r": 30, "t": 70, "b": 50})
    else:
        rollout_fig = empty_campaign_figure(
            "Recommended Rollout Mix",
            "No rollout decision column was found for the active campaign recommendations.",
        )

    campaign_name_col = "campaign_name" if "campaign_name" in top10.columns else "campaign"
    if campaign_name_col in top10.columns:
        profit_fig = px.bar(
            top10.sort_values("expected_profit", ascending=True),
            x="expected_profit",
            y=campaign_name_col,
            orientation="h",
            title="Expected Campaign Profit by Recommendation",
            text="expected_profit",
        )
        profit_fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        profit_fig.update_layout(showlegend=False, height=520, margin={"l": 80, "r": 40, "t": 70, "b": 50}, xaxis_title="Expected Profit", yaxis_title="Campaign")
    else:
        profit_fig = empty_campaign_figure("Expected Campaign Profit by Recommendation")

    top_cards = [
        html.Div(
            children=[
                html.Strong("How to read this section: "),
                "this page ranks campaign opportunities. One customer can qualify for multiple campaigns, so campaign-level counts are customer-campaign matches, not unique customer counts.",
            ],
            style={
                "gridColumn": "1 / -1",
                "backgroundColor": "#eff6ff",
                "border": "1px solid #bfdbfe",
                "borderRadius": "14px",
                "padding": "12px 14px",
                "color": "#1e3a8a",
                "lineHeight": "1.45",
            },
        )
    ] + [
        create_campaign_recommendation_card(row)
        for _, row in top10.iterrows()
    ]

    try:
        table_rows = create_campaign_table_rows(top10)
    except Exception:
        table_rows = top10.to_dict("records")

    cleaned_table_rows = []
    for row in table_rows:
        row = dict(row)
        if "Eligible Customers" in row:
            row["Customer-Campaign Matches"] = row.pop("Eligible Customers")
        if "Scale Customers" in row:
            row["Scale Matches"] = row.pop("Scale Customers")
        if "Blocked Customers" in row:
            row["Blocked Matches"] = row.pop("Blocked Customers")
        cleaned_table_rows.append(row)

    table = create_table(cleaned_table_rows)
    detail = create_campaign_detail_panel(top10)

    return kpis, family_fig, rollout_fig, profit_fig, top_cards, table, detail




@app.callback(
    Output("top-kpi-container", "children"),
    Input("active-customer-data-store", "data"),
)
def update_top_kpi_row(active_data):
    master_df = get_active_customer_features(active_data)

    total_customers = int(len(master_df))

    def safe_sum(column_name):
        if column_name in master_df.columns:
            return float(pd.to_numeric(master_df[column_name], errors="coerce").fillna(0).sum())
        return 0.0

    total_spend = safe_sum("monthly_spend")
    total_profit = safe_sum("risk_adjusted_profit")

    if total_customers and "decision_status" in master_df.columns:
        eligible_count = int(master_df["decision_status"].isin(["Scale", "Test"]).sum())
        blocked_count = int((master_df["decision_status"] == "Block").sum())
        eligible_rate = eligible_count / total_customers
        block_rate = blocked_count / total_customers
    else:
        eligible_rate = 0
        block_rate = 0

    return [
        create_kpi_card("Total Customers", f"{total_customers:,}", "Customers in active master dataset", "#2563eb"),
        create_kpi_card("Monthly Spend", format_currency(total_spend), "Total card portfolio spend", "#0ea5e9"),
        create_kpi_card("Risk-Adjusted Profit", format_currency(total_profit), "Monthly profit after risk cost", "#16a34a"),
        create_kpi_card("Campaign Eligible", format_percent(eligible_rate), "Scale or Test customers", "#7c3aed"),
        create_kpi_card("Blocked by Guardrails", format_percent(block_rate), "Protected from growth offers", "#dc2626"),
    ]




def classify_segment_strategy(row: pd.Series) -> tuple[str, str, str, str]:
    """Classify segment strategy from active portfolio economics and risk."""
    segment = str(row.get("customer_segment", "Unknown"))
    customer_count = int(row.get("customer_count", 0))
    avg_default_probability = float(row.get("avg_default_probability", 0))
    avg_profit = float(row.get("avg_risk_adjusted_profit", 0))
    eligible_rate = float(row.get("campaign_eligible_rate", 0))
    scale_rate = float(row.get("scale_rate", 0))
    block_rate = float(row.get("block_rate", 0))

    if customer_count == 0:
        return (
            "Review",
            "No customers available in this segment.",
            "Data Quality / Analytics",
            "Upload a larger file or check segmentation inputs.",
        )

    if block_rate >= 0.40 or avg_default_probability >= 0.08 or segment == "Risk Watch":
        return (
            "Block / Protect",
            "Risk is too high for growth offers. Use servicing, monitoring, or protective engagement.",
            "Credit Risk / Compliance",
            "Review Guardrails before any campaign launch.",
        )

    if scale_rate >= 0.50 and avg_profit > 0 and avg_default_probability <= 0.04:
        return (
            "Scale",
            "Segment has positive risk-adjusted value and enough Scale decisions for broader rollout.",
            "Growth / Portfolio Marketing",
            "Move to Campaigns & Offers, then validate economics in Scenario Simulator.",
        )

    if eligible_rate >= 0.50 and avg_profit > 0 and avg_default_probability <= 0.07:
        return (
            "Test",
            "Segment has opportunity, but the best next step is a controlled test before broad rollout.",
            "Lifecycle / Experimentation",
            "Use A/B Planner and export a test audience.",
        )

    if avg_profit > 0 and avg_default_probability > 0.04:
        return (
            "Constrain",
            "Segment is profitable but risk is elevated, so growth should be limited and monitored.",
            "Risk + Customer Management",
            "Run a smaller audience test with strict guardrails.",
        )

    return (
        "Do Not Launch",
        "Current economics or eligibility do not justify a campaign launch.",
        "Portfolio Strategy",
        "Monitor the segment or improve targeting before launch.",
    )


def build_active_strategy_summary(master_df: pd.DataFrame) -> pd.DataFrame:
    """Create active segment-level summary for the strategy playbook."""
    if master_df is None or master_df.empty or "customer_segment" not in master_df.columns:
        return pd.DataFrame()

    df = master_df.copy()

    for column in [
        "default_probability",
        "risk_adjusted_profit",
        "expected_roi",
        "monthly_spend",
    ]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if "decision_status" not in df.columns:
        df["decision_status"] = "Unknown"

    df["scale_flag"] = (df["decision_status"] == "Scale").astype(int)
    df["test_flag"] = (df["decision_status"] == "Test").astype(int)
    df["block_flag"] = (df["decision_status"] == "Block").astype(int)
    df["eligible_flag"] = df["decision_status"].isin(["Scale", "Test"]).astype(int)

    summary = (
        df.groupby("customer_segment", as_index=False)
        .agg(
            customer_count=("customer_segment", "size"),
            total_monthly_spend=("monthly_spend", "sum"),
            total_risk_adjusted_profit=("risk_adjusted_profit", "sum"),
            avg_risk_adjusted_profit=("risk_adjusted_profit", "mean"),
            avg_default_probability=("default_probability", "mean"),
            avg_expected_roi=("expected_roi", "mean"),
            scale_count=("scale_flag", "sum"),
            test_count=("test_flag", "sum"),
            block_count=("block_flag", "sum"),
            eligible_count=("eligible_flag", "sum"),
        )
    )

    summary["scale_rate"] = summary["scale_count"] / summary["customer_count"]
    summary["test_rate"] = summary["test_count"] / summary["customer_count"]
    summary["block_rate"] = summary["block_count"] / summary["customer_count"]
    summary["campaign_eligible_rate"] = summary["eligible_count"] / summary["customer_count"]

    strategy_results = summary.apply(classify_segment_strategy, axis=1)
    summary["strategy_decision"] = [result[0] for result in strategy_results]
    summary["strategy_reason"] = [result[1] for result in strategy_results]
    summary["strategy_owner"] = [result[2] for result in strategy_results]
    summary["next_step"] = [result[3] for result in strategy_results]

    summary["opportunity_score"] = (
        summary["total_risk_adjusted_profit"]
        + summary["eligible_count"] * 10
        - summary["avg_default_probability"] * 100
        - summary["block_count"] * 25
    )

    strategy_sort_order = {
        "Scale": 1,
        "Test": 2,
        "Constrain": 3,
        "Block / Protect": 4,
        "Do Not Launch": 5,
        "Review": 6,
    }
    summary["strategy_sort_order"] = summary["strategy_decision"].map(strategy_sort_order).fillna(99)

    summary = summary.sort_values(
        ["strategy_sort_order", "opportunity_score", "customer_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    summary["priority"] = summary.index + 1

    return summary


def build_active_strategy_risk_return_figure(strategy_df: pd.DataFrame):
    if strategy_df is None or strategy_df.empty:
        return strategy_placeholder_figure("No strategy data available for the active master dataset.")

    plot_df = strategy_df.copy()
    plot_df["avg_default_probability_pct"] = plot_df["avg_default_probability"] * 100

    fig = px.scatter(
        plot_df,
        x="avg_default_probability_pct",
        y="avg_risk_adjusted_profit",
        size="customer_count",
        color="strategy_decision",
        hover_name="customer_segment",
        hover_data={
            "customer_count": ":,",
            "avg_default_probability_pct": ":.2f",
            "avg_risk_adjusted_profit": ":.2f",
            "campaign_eligible_rate": ":.1%",
            "strategy_decision": True,
        },
        title="Active Segment Risk-Return Matrix",
        labels={
            "avg_default_probability_pct": "Average default probability (%)",
            "avg_risk_adjusted_profit": "Average risk-adjusted profit",
            "strategy_decision": "Strategy decision",
            "customer_count": "Customer count",
        },
        size_max=58,
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#9ca3af",
        annotation_text="Profit break-even",
        annotation_position="bottom right",
    )

    fig.update_layout(
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=70, b=30),
        legend_title_text="Strategy decision",
        font=dict(family="Arial", size=12, color=COLORS["text"]),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False, tickprefix="$")

    return fig


def render_strategy_executive_panel(master_df: pd.DataFrame, strategy_df: pd.DataFrame):
    if strategy_df is None or strategy_df.empty:
        return create_zero_state_card(
            "No active strategy available",
            "The active master dataset does not contain enough segment information to produce a strategy.",
            "Upload a valid customer file or use the synthetic demo portfolio.",
        )

    total_customers = int(len(master_df))
    top_row = strategy_df.iloc[0]
    top_segment = top_row["customer_segment"]
    top_decision = top_row["strategy_decision"]

    scale_segments = strategy_df[strategy_df["strategy_decision"] == "Scale"]["customer_segment"].tolist()
    test_segments = strategy_df[strategy_df["strategy_decision"] == "Test"]["customer_segment"].tolist()
    protect_segments = strategy_df[strategy_df["strategy_decision"].isin(["Block / Protect", "Constrain"])]["customer_segment"].tolist()

    if scale_segments:
        next_move = f"Prioritize {scale_segments[0]} for a controlled scale-ready campaign."
    elif test_segments:
        next_move = f"Run a controlled test for {test_segments[0]} before scaling."
    elif protect_segments:
        next_move = f"Focus on risk control for {protect_segments[0]} before any growth campaign."
    else:
        next_move = f"Monitor {top_segment}; no broad launch is recommended yet."

    active_mode = "Active"

    return html.Div(
        children=[
            html.Div(
                children=[
                    create_kpi_card("Active Customers", f"{total_customers:,}", "Current master dataset", "#2563eb"),
                    create_kpi_card("Segments Found", f"{len(strategy_df):,}", "Segments in active data", "#0ea5e9"),
                    create_kpi_card("Top Strategy", top_decision, f"Primary segment: {top_segment}", "#7c3aed"),
                    create_kpi_card("Next Move", "Action Required", next_move, "#16a34a"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "14px",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                children=[
                    html.Strong("Strategy explanation: "),
                    f"The playbook is using the active master dataset and recommends '{top_decision}' for {top_segment}. "
                    f"{top_row['strategy_reason']}",
                ],
                style={
                    "backgroundColor": "#eff6ff",
                    "border": "1px solid #bfdbfe",
                    "borderRadius": "14px",
                    "padding": "14px",
                    "lineHeight": "1.5",
                    "color": "#1e3a8a",
                    "marginBottom": "14px",
                },
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div("1", style={"fontWeight": "900", "fontSize": "18px", "color": "#2563eb"}),
                            html.Strong("Select priority segment"),
                            html.Div(top_segment, style={"color": COLORS["muted"], "marginTop": "4px"}),
                        ],
                        style={"backgroundColor": "#ffffff", "border": f"1px solid {COLORS['border']}", "borderRadius": "14px", "padding": "14px"},
                    ),
                    html.Div(
                        children=[
                            html.Div("2", style={"fontWeight": "900", "fontSize": "18px", "color": "#7c3aed"}),
                            html.Strong("Choose campaign"),
                            html.Div("Use Campaigns & Offers to pick matching campaign opportunities.", style={"color": COLORS["muted"], "marginTop": "4px"}),
                        ],
                        style={"backgroundColor": "#ffffff", "border": f"1px solid {COLORS['border']}", "borderRadius": "14px", "padding": "14px"},
                    ),
                    html.Div(
                        children=[
                            html.Div("3", style={"fontWeight": "900", "fontSize": "18px", "color": "#f97316"}),
                            html.Strong("Simulate and test"),
                            html.Div("Use Scenario Simulator and A/B Planner before broad rollout.", style={"color": COLORS["muted"], "marginTop": "4px"}),
                        ],
                        style={"backgroundColor": "#ffffff", "border": f"1px solid {COLORS['border']}", "borderRadius": "14px", "padding": "14px"},
                    ),
                    html.Div(
                        children=[
                            html.Div("4", style={"fontWeight": "900", "fontSize": "18px", "color": "#dc2626"}),
                            html.Strong("Guardrail review"),
                            html.Div("Check blocked/risky groups before launch or export.", style={"color": COLORS["muted"], "marginTop": "4px"}),
                        ],
                        style={"backgroundColor": "#ffffff", "border": f"1px solid {COLORS['border']}", "borderRadius": "14px", "padding": "14px"},
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px"},
            ),
        ]
    )


def render_strategy_playbook_table(strategy_df: pd.DataFrame):
    if strategy_df is None or strategy_df.empty:
        return create_zero_state_card(
            "No strategy table available",
            "The active master dataset does not contain segment-level records.",
            "Upload a valid customer file or use the synthetic demo portfolio.",
        )

    strategy_colors = {
        "Scale": "#16a34a",
        "Test": "#2563eb",
        "Constrain": "#f97316",
        "Block / Protect": "#dc2626",
        "Do Not Launch": "#64748b",
        "Review": "#7c3aed",
    }

    cards = []

    for _, row in strategy_df.iterrows():
        strategy = row["strategy_decision"]
        accent = strategy_colors.get(strategy, "#2563eb")

        cards.append(
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                f"Priority {int(row['priority'])}",
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "900",
                                    "color": COLORS["muted"],
                                    "textTransform": "uppercase",
                                },
                            ),
                            html.Div(
                                strategy,
                                style={
                                    "fontSize": "12px",
                                    "fontWeight": "900",
                                    "color": "white",
                                    "backgroundColor": accent,
                                    "borderRadius": "999px",
                                    "padding": "6px 10px",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "10px",
                        },
                    ),
                    html.H4(
                        row["customer_segment"],
                        style={
                            "fontSize": "18px",
                            "fontWeight": "900",
                            "margin": "0 0 12px 0",
                            "color": COLORS["text"],
                        },
                    ),
                    html.Div(
                        children=[
                            create_metric_chip("Customers", f"{int(row['customer_count']):,}"),
                            create_metric_chip("Default Risk", f"{row['avg_default_probability']:.2%}"),
                            create_metric_chip("Avg Profit", format_currency(row["avg_risk_adjusted_profit"])),
                            create_metric_chip("Eligible", f"{row['campaign_eligible_rate']:.1%}"),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(4, 1fr)",
                            "gap": "8px",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(
                        children=[
                            html.Strong("Why: "),
                            row["strategy_reason"],
                        ],
                        style={
                            "fontSize": "13px",
                            "lineHeight": "1.45",
                            "color": COLORS["text"],
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div(
                        children=[
                            html.Strong("Next step: "),
                            row["next_step"],
                        ],
                        style={
                            "fontSize": "13px",
                            "lineHeight": "1.45",
                            "color": COLORS["muted"],
                            "backgroundColor": "#f8fafc",
                            "border": f"1px solid {COLORS['border']}",
                            "borderRadius": "12px",
                            "padding": "10px",
                        },
                    ),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "border": f"1px solid {COLORS['border']}",
                    "borderTop": f"5px solid {accent}",
                    "borderRadius": "16px",
                    "padding": "16px",
                    "boxShadow": "0 6px 16px rgba(15, 23, 42, 0.05)",
                },
            )
        )

    return html.Div(
        children=cards,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(460px, 1fr))",
            "gap": "14px",
        },
    )


@app.callback(
    Output("strategy-executive-recommendation", "children"),
    Output("strategy-risk-return-chart", "figure"),
    Output("strategy-playbook-table-container", "children"),
    Input("active-customer-data-store", "data"),
)
def update_strategy_playbook_from_active_master(active_data):
    master_df = get_active_customer_features(active_data)
    strategy_df = build_active_strategy_summary(master_df)

    executive_panel = render_strategy_executive_panel(master_df, strategy_df)
    risk_return_fig = build_active_strategy_risk_return_figure(strategy_df)
    playbook_table = render_strategy_playbook_table(strategy_df)

    return executive_panel, risk_return_fig, playbook_table




def build_active_guardrail_review(master_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create customer-level and rule-level guardrail review from active master data."""
    if master_df is None or master_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = master_df.copy()

    numeric_defaults = {
        "default_probability": 0,
        "utilization_rate": 0,
        "credit_score": 999,
        "late_payments_12m": 0,
        "risk_adjusted_profit": 0,
    }

    for column, default in numeric_defaults.items():
        if column not in df.columns:
            df[column] = default
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)

    if "decision_status" not in df.columns:
        df["decision_status"] = "Unknown"

    if "risk_guardrail_flag" in df.columns:
        risk_guardrail_series = df["risk_guardrail_flag"]
        if risk_guardrail_series.dtype == bool:
            df["rule_engine_guardrail_flag"] = risk_guardrail_series
        else:
            df["rule_engine_guardrail_flag"] = risk_guardrail_series.astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        df["rule_engine_guardrail_flag"] = False

    df["rule_block_decision"] = df["decision_status"].astype(str).eq("Block")
    df["rule_high_default_probability"] = df["default_probability"] >= 0.08
    df["rule_high_utilization"] = df["utilization_rate"] >= 0.70
    df["rule_low_credit_score"] = df["credit_score"] < 600
    df["rule_late_payment_risk"] = df["late_payments_12m"] >= 2
    df["rule_negative_profit"] = df["risk_adjusted_profit"] <= 0

    df["hard_guardrail_fail"] = (
        df["rule_block_decision"]
        | df["rule_engine_guardrail_flag"]
        | df["rule_high_default_probability"]
        | (df["rule_low_credit_score"] & df["rule_high_utilization"])
        | (df["rule_late_payment_risk"] & df["rule_high_utilization"])
    )

    df["review_guardrail_flag"] = (
        df["rule_high_utilization"]
        | df["rule_low_credit_score"]
        | df["rule_late_payment_risk"]
        | df["rule_negative_profit"]
        | df["rule_high_default_probability"]
    )

    rule_specs = [
        {
            "Rule": "Block decision",
            "Threshold": "decision_status = Block",
            "Column": "rule_block_decision",
            "Severity": "Hard stop",
        },
        {
            "Rule": "Engine guardrail flag",
            "Threshold": "risk_guardrail_flag = true",
            "Column": "rule_engine_guardrail_flag",
            "Severity": "Hard stop",
        },
        {
            "Rule": "High default probability",
            "Threshold": "default_probability >= 8%",
            "Column": "rule_high_default_probability",
            "Severity": "Hard stop",
        },
        {
            "Rule": "High utilization watchlist",
            "Threshold": "utilization_rate >= 70%",
            "Column": "rule_high_utilization",
            "Severity": "Review",
        },
        {
            "Rule": "Low credit score",
            "Threshold": "credit_score < 600",
            "Column": "rule_low_credit_score",
            "Severity": "Review",
        },
        {
            "Rule": "Late payment risk",
            "Threshold": "late_payments_12m >= 2",
            "Column": "rule_late_payment_risk",
            "Severity": "Review",
        },
        {
            "Rule": "Negative risk-adjusted profit",
            "Threshold": "risk_adjusted_profit <= 0",
            "Column": "rule_negative_profit",
            "Severity": "Review",
        },
    ]

    total_customers = max(len(df), 1)
    rule_rows = []

    for rule in rule_specs:
        flagged = int(df[rule["Column"]].sum()) if rule["Column"] in df.columns else 0
        share = flagged / total_customers
        status = "Clear" if flagged == 0 else ("Stop" if rule["Severity"] == "Hard stop" else "Review")

        rule_rows.append(
            {
                "Rule": rule["Rule"],
                "Threshold": rule["Threshold"],
                "Severity": rule["Severity"],
                "Customers Flagged": f"{flagged:,}",
                "Share": format_percent(share),
                "Status": status,
            }
        )

    rule_df = pd.DataFrame(rule_rows)

    return df, rule_df


def render_guardrail_rule_cards(rule_df: pd.DataFrame):
    if rule_df is None or rule_df.empty:
        return create_zero_state_card(
            "No guardrail rules available",
            "The active dataset could not be evaluated against risk rules.",
            "Check whether the uploaded file contains the required risk and decision columns.",
        )

    status_colors = {
        "Clear": "#16a34a",
        "Review": "#f97316",
        "Stop": "#dc2626",
    }

    cards = []

    for _, row in rule_df.iterrows():
        accent = status_colors.get(row["Status"], "#64748b")
        cards.append(
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(row["Rule"], style={"fontWeight": "900", "fontSize": "14px", "color": COLORS["text"]}),
                            html.Div(
                                row["Status"],
                                style={
                                    "fontSize": "11px",
                                    "fontWeight": "900",
                                    "color": "white",
                                    "backgroundColor": accent,
                                    "borderRadius": "999px",
                                    "padding": "5px 9px",
                                },
                            ),
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "10px"},
                    ),
                    html.Div(row["Threshold"], style={"fontSize": "12px", "color": COLORS["muted"], "marginTop": "8px", "lineHeight": "1.35"}),
                    html.Div(
                        children=[
                            create_metric_chip("Flagged", row["Customers Flagged"]),
                            create_metric_chip("Share", row["Share"]),
                            create_metric_chip("Severity", row["Severity"]),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "8px", "marginTop": "12px"},
                    ),
                ],
                style={
                    "backgroundColor": "#ffffff",
                    "border": f"1px solid {COLORS['border']}",
                    "borderTop": f"4px solid {accent}",
                    "borderRadius": "14px",
                    "padding": "14px",
                    "boxShadow": "0 6px 16px rgba(15, 23, 42, 0.05)",
                },
            )
        )

    return html.Div(
        children=[
            html.H3("Active Guardrail Rule Checklist", style={"margin": "0 0 12px 0", "fontSize": "20px", "fontWeight": "900"}),
            html.Div(
                cards,
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))", "gap": "12px"},
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


def render_guardrail_customer_review(review_df: pd.DataFrame):
    if review_df is None or review_df.empty:
        return html.Div(
            children=[
                html.H3(
                    "No active customers failed hard guardrails",
                    style={"margin": "0 0 8px 0", "fontSize": "20px", "fontWeight": "900", "color": "#166534"},
                ),
                html.P(
                    "No customer in the active master dataset is currently blocked or failing hard risk rules.",
                    style={"margin": "0 0 8px 0", "color": "#166534", "lineHeight": "1.45"},
                ),
                html.P(
                    "Continue to monitor high utilization, late payment risk, and low credit scores before campaign launch.",
                    style={"margin": "0", "color": "#166534", "lineHeight": "1.45"},
                ),
            ],
            style={
                "backgroundColor": "#f0fdf4",
                "border": "1px solid #bbf7d0",
                "borderLeft": "5px solid #16a34a",
                "borderRadius": "14px",
                "padding": "16px",
            },
        )

    display_df = review_df.copy()

    rename_map = {
        "customer_id": "Customer ID",
        "customer_name": "Name",
        "customer_segment": "Segment",
        "decision_status": "Decision",
        "risk_band": "Risk Band",
        "credit_score": "Credit Score",
        "utilization_rate": "Utilization",
        "default_probability": "Default Probability",
        "late_payments_12m": "Late Payments 12M",
        "risk_adjusted_profit": "Risk-Adjusted Profit",
    }

    display_df = display_df.rename(columns=rename_map)

    for column in ["Utilization", "Default Probability"]:
        if column in display_df.columns:
            display_df[column] = pd.to_numeric(display_df[column], errors="coerce").fillna(0).apply(lambda value: f"{value:.1%}")

    if "Risk-Adjusted Profit" in display_df.columns:
        display_df["Risk-Adjusted Profit"] = pd.to_numeric(display_df["Risk-Adjusted Profit"], errors="coerce").fillna(0).apply(format_currency)

    columns = [
        "Customer ID",
        "Name",
        "Segment",
        "Decision",
        "Risk Band",
        "Credit Score",
        "Utilization",
        "Default Probability",
        "Late Payments 12M",
        "Risk-Adjusted Profit",
    ]

    columns = [column for column in columns if column in display_df.columns]

    rows = display_df[columns].head(25).to_dict("records")

    return html.Div(
        children=[
            html.H3("Customers Requiring Guardrail Review", style={"margin": "0 0 10px 0", "fontSize": "20px", "fontWeight": "900"}),
            html.P(
                "This table shows customers who are blocked or flagged by active risk rules. Use Audience Explorer for export and deeper review.",
                style={"color": COLORS["muted"], "margin": "0 0 14px 0", "lineHeight": "1.45"},
            ),
            create_table(rows),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "18px",
            "padding": "20px",
            "boxShadow": "0 8px 22px rgba(15, 23, 42, 0.06)",
            "overflowX": "auto",
        },
    )


@app.callback(
    Output("guardrails-kpi-container", "children"),
    Output("guardrails-rule-review-container", "children"),
    Output("guardrails-customer-review-container", "children"),
    Output("guardrails-interpretation-container", "children"),
    Input("active-customer-data-store", "data"),
)
def update_guardrails_from_active_master(active_data):
    master_df = get_active_customer_features(active_data)
    review_df, rule_df = build_active_guardrail_review(master_df)

    total_customers = int(len(review_df))
    blocked_count = int(review_df["rule_block_decision"].sum()) if "rule_block_decision" in review_df.columns else 0
    hard_fail_count = int(review_df["hard_guardrail_fail"].sum()) if "hard_guardrail_fail" in review_df.columns else 0
    review_count = int(review_df["review_guardrail_flag"].sum()) if "review_guardrail_flag" in review_df.columns else 0
    high_util_count = int(review_df["rule_high_utilization"].sum()) if "rule_high_utilization" in review_df.columns else 0

    block_rate = blocked_count / total_customers if total_customers else 0
    hard_fail_rate = hard_fail_count / total_customers if total_customers else 0

    kpis = [
        create_kpi_card("Active Customers", f"{total_customers:,}", "Customers in active master dataset", "#2563eb"),
        create_kpi_card("Blocked Customers", f"{blocked_count:,}", f"{format_percent(block_rate)} of active customers", "#dc2626"),
        create_kpi_card("Hard Rule Fails", f"{hard_fail_count:,}", f"{format_percent(hard_fail_rate)} fail stop rules", "#f97316"),
        create_kpi_card("Review Watchlist", f"{review_count:,}", "Customers flagged by review rules", "#7c3aed"),
        create_kpi_card("High Utilization", f"{high_util_count:,}", "Utilization at or above 70%", "#0ea5e9"),
    ]

    rule_cards = render_guardrail_rule_cards(rule_df)

    customer_review_df = review_df[
        (review_df["hard_guardrail_fail"]) | (review_df["review_guardrail_flag"])
    ].copy() if not review_df.empty else pd.DataFrame()

    customer_review = render_guardrail_customer_review(customer_review_df)

    if total_customers == 0:
        interpretation = create_insight_card(
            "Guardrail Interpretation",
            "No active customers are available for guardrail review.",
            variant="warning",
        )
    elif hard_fail_count == 0 and review_count == 0:
        interpretation = create_insight_card(
            "Guardrail Interpretation",
            "No active customers failed hard stop rules or review watchlist rules. The current active portfolio is clean from a guardrail perspective, but campaign launches should still be reviewed before execution.",
            variant="success",
        )
    elif hard_fail_count == 0:
        interpretation = create_insight_card(
            "Guardrail Interpretation",
            f"No active customers failed hard stop rules, but {review_count:,} customer(s) are on the review watchlist. Use controlled testing or manual review before scaling offers.",
            variant="warning",
        )
    else:
        interpretation = create_insight_card(
            "Guardrail Interpretation",
            f"{hard_fail_count:,} active customer(s) failed hard stop rules. These customers should not receive aggressive growth offers until risk conditions improve.",
            variant="warning",
        )

    return kpis, rule_cards, customer_review, interpretation



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
