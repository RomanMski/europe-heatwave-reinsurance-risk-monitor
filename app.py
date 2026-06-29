from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from heatwave_risk.climate import (  # noqa: E402
    fallback_projection_from_baseline,
    fetch_climate_projection,
    projection_decade_summary,
    projection_period,
    summarize_projection,
)
from heatwave_risk.data import build_weather_dataset  # noqa: E402
from heatwave_risk.risk import DEFAULT_LINE_WEIGHTS, compute_risk_scores, selected_day_view  # noqa: E402


st.set_page_config(page_title="Heat Stress Reinsurance Workbench", layout="wide")

CSS = """
<style>
    html, body, [class*="css"] { color: #172033; }
    .main .block-container { padding-top: 1.25rem; max-width: 1360px; }
    .hero {
        padding: 1.0rem 1.15rem 1.05rem 1.15rem;
        border: 1px solid #dbe4ea;
        border-radius: 8px;
        background: #ffffff;
    }
    .hero h1 {
        font-size: 1.84rem;
        line-height: 1.1;
        margin-bottom: 0.25rem;
        color: #102a43;
    }
    .hero p {
        color: #425466;
        font-size: 0.98rem;
        margin-bottom: 0;
    }
    .status-row {
        display: flex;
        gap: .5rem;
        flex-wrap: wrap;
        margin-top: .65rem;
    }
    .status-pill {
        border: 1px solid #dbe4ea;
        border-radius: 999px;
        padding: .28rem .65rem;
        color: #425466;
        background: #f8fafc;
        font-size: .82rem;
    }
    .insight-panel {
        border: 1px solid #dbe4ea;
        border-radius: 8px;
        background: #fbfefd;
        padding: 0.95rem 1.05rem;
        margin-top: 0.75rem;
        margin-bottom: 0.95rem;
    }
    .insight-panel strong { color: #123238; }
    .insight-panel span { color: #425466; }
    .brief-card {
        border: 1px solid #dbe4ea;
        border-radius: 8px;
        background: #ffffff;
        padding: .95rem 1.05rem;
        min-height: 8.5rem;
    }
    .brief-card h4 {
        margin: 0 0 .35rem 0;
        color: #102a43;
        font-size: 1.02rem;
    }
    .brief-card p {
        margin: 0;
        color: #425466;
        font-size: .93rem;
        line-height: 1.45;
    }
    .formula-card {
        border: 1px solid #dbe4ea;
        border-radius: 8px;
        background: #f8fafc;
        padding: .75rem .85rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        color: #1f2937;
        font-size: .88rem;
        white-space: normal;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #dbe4ea;
        border-radius: 8px;
        padding: 0.78rem 0.9rem;
        background: #ffffff;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

MODULE_LABELS = {
    "life_health_stress": "life and health",
    "agriculture_stress": "agriculture",
    "energy_stress": "energy",
    "property_infra_stress": "property and infrastructure",
    "business_interruption_stress": "business interruption",
}

ACTION_RULES = {
    "life_health_stress": "Check mortality, health and worker safety exposure around the affected region.",
    "agriculture_stress": "Review crop, livestock and drought sensitive agriculture exposure.",
    "energy_stress": "Watch cooling demand, grid stress and energy linked business interruption.",
    "property_infra_stress": "Check wildfire, transport and infrastructure vulnerability.",
    "business_interruption_stress": "Review dependent business interruption and operational continuity assumptions.",
}

COMPACT_MODULE_LABELS = {
    "life_health_stress": "Life and health",
    "agriculture_stress": "Agriculture",
    "energy_stress": "Energy",
    "property_infra_stress": "Infra",
    "business_interruption_stress": "BI",
}


def fmt_chf(value: float) -> str:
    return f"CHF {value:,.1f}m"


def fmt_chf_compact(value: float) -> str:
    return f"CHF {value:,.0f}m" if value >= 10 else f"CHF {value:,.1f}m"


def fmt_pct(value: float) -> str:
    return f"{value:.0%}"


def metric_grid(items: list[tuple[str, object] | tuple[str, object, object]], columns: int = 2) -> None:
    for start in range(0, len(items), columns):
        row = st.columns(columns)
        for col, item in zip(row, items[start : start + columns]):
            label, value, *rest = item
            delta = rest[0] if rest else None
            col.metric(label, value, delta)


def cache_freshness(cache_dir: Path) -> str:
    files = list((cache_dir / "open_meteo_recent").glob("*.json"))
    if not files:
        return "Weather cache not available yet"
    latest = max(file.stat().st_mtime for file in files)
    return f"Weather cache refreshed {datetime.fromtimestamp(latest):%Y-%m-%d %H:%M}"


def chart_theme(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial, sans-serif", color="#172033", size=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=58, b=72),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def load_scored_data(refresh: bool, weights_tuple: tuple[tuple[str, float], ...]):
    daily, regions = build_weather_dataset(
        ROOT / "data" / "regions.csv",
        ROOT / "data" / "cache",
        refresh=refresh,
    )
    return compute_risk_scores(daily, dict(weights_tuple)), regions


@st.cache_data(show_spinner=False)
def load_projection(region_id: str, latitude: float, longitude: float, end_year: int, refresh: bool):
    try:
        projection = fetch_climate_projection(
            region_id=region_id,
            latitude=latitude,
            longitude=longitude,
            cache_dir=ROOT / "data" / "cache",
            end_year=end_year,
            refresh=refresh,
        )
        return projection, "Open Meteo CMIP6 climate API"
    except Exception:
        projection = fallback_projection_from_baseline(
            region_id=region_id,
            cache_dir=ROOT / "data" / "cache",
            end_year=end_year,
        )
        return projection, "Fallback local trend scenario from historical archive cache"


def baseline_status(scored: pd.DataFrame) -> str:
    if "baseline_source" not in scored.columns:
        return "Historical baseline status unavailable"
    sources = scored[["region_id", "city", "baseline_source"]].drop_duplicates()
    archive_count = int((sources["baseline_source"] == "open_meteo_archive").sum())
    total = len(sources)
    fallback = sources[sources["baseline_source"] != "open_meteo_archive"]["city"].tolist()
    if not fallback:
        return f"Historical baseline: Open Meteo archive for all {total} regions"
    return f"Historical baseline: Open Meteo archive for {archive_count} of {total} regions. Fallback used for {', '.join(fallback)}."


def main_driver(row: pd.Series) -> str:
    scores = {column: float(row[column]) for column in MODULE_LABELS}
    return max(scores, key=scores.get)


def top_regions(day_view: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("day_view", day_view)
    return con.execute(
        """
        select
            city,
            country,
            round(temperature_2m_max, 1) as temp_max_c,
            round(trigger_threshold_c, 1) as trigger_threshold_c,
            round(temp_anomaly, 1) as anomaly_c,
            p95_streak as heatwave_days,
            round(composite_risk_score, 1) as risk_score,
            trigger_active,
            round(modeled_payout_chf_m, 2) as scenario_payout_chf_m
        from day_view
        order by risk_score desc, scenario_payout_chf_m desc
        limit ?
        """,
        [limit],
    ).fetchdf()


def action_queue(day_view: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    rows = []
    for _, row in day_view.sort_values("composite_risk_score", ascending=False).head(limit).iterrows():
        lead = main_driver(row)
        rows.append(
            {
                "Region": f"{row['city']}, {row['country']}",
                "Score": round(float(row["composite_risk_score"]), 1),
                "Driver": COMPACT_MODULE_LABELS[lead],
                "Trigger": "Active" if bool(row["trigger_active"]) else "Not active",
                "Payout": fmt_chf_compact(float(row["modeled_payout_chf_m"])),
            }
        )
    return pd.DataFrame(rows)


def basis_risk_table(day_view: pd.DataFrame) -> pd.DataFrame:
    rows = []
    median_score = float(day_view["composite_risk_score"].median())
    for _, row in day_view.iterrows():
        high_no_pay = row["composite_risk_score"] >= median_score and not bool(row["trigger_active"])
        low_with_pay = row["composite_risk_score"] < median_score and bool(row["trigger_active"])
        if not high_no_pay and not low_with_pay:
            continue
        rows.append(
            {
                "Region": f"{row['city']}, {row['country']}",
                "Flag": "High stress, no trigger" if high_no_pay else "Trigger active, lower stress",
                "Risk score": round(float(row["composite_risk_score"]), 1),
                "Threshold": f"{float(row['trigger_threshold_c']):.1f} deg C",
                "Heat degree days": round(float(row["event_degree_days"]), 1),
                "Scenario payout": fmt_chf(float(row["modeled_payout_chf_m"])),
            }
        )
    return pd.DataFrame(rows)


def trigger_sensitivity(scored: pd.DataFrame, selected_date: pd.Timestamp, notional: float, payout_rate: float, cap_pct: float) -> pd.DataFrame:
    rows = []
    for percentile in ["95", "98", "99"]:
        for streak in range(2, 7):
            view = selected_day_view(
                scored,
                selected_date,
                threshold_percentile=percentile,
                trigger_days=streak,
                notional_chf_m=notional,
                payout_per_degree_day=payout_rate,
                cap_pct=cap_pct,
            )
            rows.append(
                {
                    "percentile": f"p{percentile}",
                    "streak_days": streak,
                    "active_regions": int(view["trigger_active"].sum()),
                    "total_payout_chf_m": float(view["modeled_payout_chf_m"].sum()),
                }
            )
    return pd.DataFrame(rows)


def underwriter_brief(day_view: pd.DataFrame, selected_date: pd.Timestamp) -> dict[str, str]:
    leader = day_view.iloc[0]
    lead = main_driver(leader)
    basis = basis_risk_table(day_view)
    high_no_pay = basis[basis["Flag"].eq("High stress, no trigger")] if not basis.empty else pd.DataFrame()
    return {
        "event": (
            f"{leader['city']} leads the portfolio on {selected_date.date()} with a "
            f"{float(leader['composite_risk_score']):.1f} risk score. The main driver is {MODULE_LABELS[lead]}."
        ),
        "contract": (
            f"{int(day_view['trigger_active'].sum())} regions activate the selected heat trigger. "
            f"The current scenario payout is {fmt_chf(float(day_view['modeled_payout_chf_m'].sum()))}."
        ),
        "basis": (
            "No major high stress no trigger flags in the current top view."
            if high_no_pay.empty
            else f"{len(high_no_pay)} regions show high stress without trigger activation. That is the first basis risk review."
        ),
        "next": ACTION_RULES[lead],
    }


def add_projection_payout(annual: pd.DataFrame, trigger_days: int, notional: float, payout_rate: float, cap_pct: float) -> pd.DataFrame:
    projected = annual.copy()
    cap = cap_pct * notional
    projected["projected_trigger_active"] = projected["max_local_extreme_streak"] >= trigger_days
    raw = projected["local_heat_degree_days"] * payout_rate * notional
    projected["projected_payout_chf_m"] = np.where(projected["projected_trigger_active"], raw.clip(upper=cap), 0).round(2)
    projected["period"] = projected["year"].map(projection_period)
    return projected


def projected_payout_summary(projected: pd.DataFrame) -> pd.DataFrame:
    summary = (
        projected.groupby("period", observed=True)
        .agg(
            avg_payout_chf_m=("projected_payout_chf_m", "mean"),
            p75_payout_chf_m=("projected_payout_chf_m", lambda values: values.quantile(0.75)),
            p95_payout_chf_m=("projected_payout_chf_m", lambda values: values.quantile(0.95)),
            trigger_activation_rate=("projected_trigger_active", "mean"),
            avg_max_trigger_streak=("max_local_extreme_streak", "mean"),
            model_years=("year", "count"),
        )
        .reset_index()
    )
    numeric = ["avg_payout_chf_m", "p75_payout_chf_m", "p95_payout_chf_m", "trigger_activation_rate", "avg_max_trigger_streak"]
    summary[numeric] = summary[numeric].round(2)
    summary["trigger_activation_rate"] = (summary["trigger_activation_rate"] * 100).round(0).astype(int).astype(str) + "%"
    return summary


def plot_map(day_view: pd.DataFrame):
    fig = px.scatter_geo(
        day_view,
        lat="latitude",
        lon="longitude",
        color="composite_risk_score",
        size="composite_risk_score",
        hover_name="city",
        hover_data={
            "country": True,
            "temperature_2m_max": ":.1f",
            "trigger_threshold_c": ":.1f",
            "temp_anomaly": ":.1f",
            "modeled_payout_chf_m": ":.2f",
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale=["#d9f99d", "#fde68a", "#fb923c", "#b91c1c"],
        range_color=(0, 100),
        projection="natural earth",
        title="Current heat stress across monitored regions",
    )
    fig.update_geos(scope="europe", showland=True, landcolor="#f8fafc", showcountries=True, countrycolor="#cbd5e1", showocean=True, oceancolor="#edf6fb")
    fig.update_layout(coloraxis_colorbar_title="Risk score")
    return chart_theme(fig, 540)


def plot_risk_decomposition(day_view: pd.DataFrame):
    labels = {
        "life_health_stress": "Life and health",
        "agriculture_stress": "Agriculture",
        "energy_stress": "Energy",
        "property_infra_stress": "Property and infrastructure",
        "business_interruption_stress": "Business interruption",
    }
    top = day_view.sort_values("composite_risk_score", ascending=False).head(9)
    tidy = top.melt(id_vars=["city"], value_vars=list(labels), var_name="module", value_name="stress")
    tidy["module"] = tidy["module"].map(labels)
    fig = px.bar(
        tidy,
        x="stress",
        y="city",
        color="module",
        orientation="h",
        title="Line of business drivers behind the top regions",
        labels={"stress": "Stress score", "city": "", "module": ""},
        color_discrete_map={
            "Life and health": "#0ea5e9",
            "Agriculture": "#65a30d",
            "Energy": "#f59e0b",
            "Property and infrastructure": "#64748b",
            "Business interruption": "#8b5cf6",
        },
    )
    fig.update_layout(barmode="stack")
    return chart_theme(fig, 430)


def plot_module_heatmap(day_view: pd.DataFrame):
    cols = {
        "life_health_stress": "Life and health",
        "agriculture_stress": "Agriculture",
        "energy_stress": "Energy",
        "property_infra_stress": "Property and infrastructure",
        "business_interruption_stress": "Business interruption",
    }
    top = day_view.sort_values("composite_risk_score", ascending=False).head(10)
    matrix = top[["city", *cols.keys()]].set_index("city").rename(columns=cols)
    fig = px.imshow(
        matrix,
        color_continuous_scale=["#f8fafc", "#fed7aa", "#ef4444", "#7f1d1d"],
        zmin=0,
        zmax=100,
        aspect="auto",
        labels=dict(color="Stress"),
        title="Multi-line stress matrix",
    )
    return chart_theme(fig, 390)


def plot_trigger_heatmap(sensitivity: pd.DataFrame):
    matrix = sensitivity.pivot(index="percentile", columns="streak_days", values="total_payout_chf_m")
    fig = px.imshow(
        matrix,
        text_auto=".1f",
        color_continuous_scale=["#f8fafc", "#c7f9e8", "#facc15", "#f97316", "#b91c1c"],
        labels=dict(x="Required consecutive days", y="Local trigger percentile", color="CHF m"),
        title="Trigger design sensitivity: total scenario payout",
        aspect="auto",
    )
    fig.update_traces(hovertemplate="Percentile %{y}<br>Streak %{x} days<br>Payout CHF %{z:.1f}m<extra></extra>")
    return chart_theme(fig, 330)


def plot_basis_scatter(day_view: pd.DataFrame):
    fig = px.scatter(
        day_view,
        x="composite_risk_score",
        y="modeled_payout_chf_m",
        color="trigger_active",
        size="event_degree_days",
        hover_name="city",
        hover_data={"country": True, "trigger_threshold_c": ":.1f", "event_degree_days": ":.1f"},
        title="Basis risk check: stress score versus contract response",
        labels={
            "composite_risk_score": "Composite risk score",
            "modeled_payout_chf_m": "Scenario payout in CHF millions",
            "trigger_active": "Trigger active",
            "event_degree_days": "Heat degree days",
        },
        color_discrete_map={True: "#ef4444", False: "#64748b"},
    )
    return chart_theme(fig, 420)


def plot_city_timeline(scored: pd.DataFrame, city: str, percentile: str):
    city_df = scored[scored["city"] == city].sort_values("date")
    threshold_col = f"clim_p{percentile}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=city_df["date"], y=city_df["temperature_2m_max"], mode="lines+markers", name="Daily max", line=dict(color="#ef4444", width=3)))
    fig.add_trace(go.Scatter(x=city_df["date"], y=city_df["clim_mean"], mode="lines", name="Local normal", line=dict(color="#64748b", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=city_df["date"], y=city_df[threshold_col], mode="lines", name=f"Local p{percentile} trigger", line=dict(color="#111827", width=2, dash="dash")))
    fig.update_layout(title=f"{city}: current event against local threshold", yaxis_title="Temperature in deg C", hovermode="x unified")
    return chart_theme(fig, 430)


def plot_city_radar(city_row: pd.Series):
    labels = ["Life and health", "Agriculture", "Energy", "Property and infrastructure", "Business interruption"]
    values = [
        float(city_row["life_health_stress"]),
        float(city_row["agriculture_stress"]),
        float(city_row["energy_stress"]),
        float(city_row["property_infra_stress"]),
        float(city_row["business_interruption_stress"]),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=labels + [labels[0]], fill="toself", line=dict(color="#0f766e", width=3), fillcolor="rgba(15,118,110,0.20)", name=str(city_row["city"])))
    fig.update_layout(title=f"{city_row['city']}: line stress profile", polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#e5e7eb")), showlegend=False)
    return chart_theme(fig, 430)


def plot_projection_fan(annual: pd.DataFrame, city: str):
    model_mean = annual.groupby("year", as_index=False).agg(local_extreme_days=("local_extreme_days", "mean"))
    bands = annual.groupby("year", as_index=False).agg(
        p10=("local_extreme_days", lambda values: values.quantile(0.10)),
        p90=("local_extreme_days", lambda values: values.quantile(0.90)),
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(bands["year"]) + list(bands["year"])[::-1], y=list(bands["p90"]) + list(bands["p10"])[::-1], fill="toself", name="Model range p10 to p90", line=dict(color="rgba(14,165,233,0)"), fillcolor="rgba(14,165,233,0.18)", hoverinfo="skip"))
    for model, frame in annual.groupby("model"):
        fig.add_trace(go.Scatter(x=frame["year"], y=frame["local_extreme_days"], mode="lines", name=model, line=dict(width=1.4), opacity=0.5))
    z = np.polyfit(model_mean["year"], model_mean["local_extreme_days"], 1)
    fig.add_trace(go.Scatter(x=model_mean["year"], y=model_mean["local_extreme_days"], mode="lines+markers", name="Average", line=dict(color="#0f766e", width=3)))
    fig.add_trace(go.Scatter(x=model_mean["year"], y=np.poly1d(z)(model_mean["year"]), mode="lines", name="Trend", line=dict(color="#111827", width=2, dash="dash")))
    fig.update_layout(title=f"{city}: projected days above today's trigger threshold", xaxis_title="Summer year", yaxis_title="Days above local threshold", hovermode="x unified")
    return chart_theme(fig, 440)


def plot_projection_payout(projected: pd.DataFrame, city: str):
    fig = px.box(
        projected,
        x="period",
        y="projected_payout_chf_m",
        color="model",
        points="all",
        title=f"{city}: projected parametric payout distribution",
        labels={"period": "", "projected_payout_chf_m": "Scenario payout in CHF millions", "model": "Projection"},
        color_discrete_sequence=["#0f766e", "#f97316", "#2563eb"],
    )
    fig.update_layout(boxmode="group")
    return chart_theme(fig, 420)


def plot_exceedance(projected: pd.DataFrame, city: str):
    data = projected.sort_values("projected_payout_chf_m", ascending=False).copy()
    data["rank"] = data.groupby("period").cumcount() + 1
    data["period_count"] = data.groupby("period")["projected_payout_chf_m"].transform("count")
    data["exceedance_probability"] = data["rank"] / (data["period_count"] + 1)
    fig = px.line(
        data,
        x="projected_payout_chf_m",
        y="exceedance_probability",
        color="period",
        markers=True,
        title=f"{city}: exceedance curve for selected heat trigger",
        labels={"projected_payout_chf_m": "Scenario payout in CHF millions", "exceedance_probability": "Probability of exceedance", "period": ""},
        color_discrete_sequence=["#0f766e", "#f97316", "#2563eb", "#8b5cf6", "#be123c"],
    )
    fig.update_yaxes(tickformat=".0%")
    return chart_theme(fig, 390)


with st.sidebar:
    st.header("Workbench controls")
    refresh = st.toggle("Refresh live weather", value=False)
    refresh_projection = st.toggle("Refresh climate projection", value=False)
    projection_horizon = st.selectbox("Climate horizon", [2035, 2040, 2050], index=0)
    threshold_percentile = st.selectbox("Trigger percentile", ["95", "98", "99"], index=1)
    trigger_days = st.slider("Trigger streak in days", 2, 6, 3)
    notional = st.slider("Notional per region in CHF millions", 5, 100, 25, step=5)
    payout_rate = st.slider("Payout rate per heat degree day", 0.02, 0.20, 0.08, step=0.01)
    cap_pct = st.slider("Payout cap as percent of notional", 0.05, 0.50, 0.25, step=0.05)
    with st.expander("Portfolio line weights"):
        weights = {
            "life_health": st.slider("Life and health", 0.0, 1.0, DEFAULT_LINE_WEIGHTS["life_health"], step=0.01),
            "agriculture": st.slider("Agriculture", 0.0, 1.0, DEFAULT_LINE_WEIGHTS["agriculture"], step=0.01),
            "energy": st.slider("Energy", 0.0, 1.0, DEFAULT_LINE_WEIGHTS["energy"], step=0.01),
            "property_infra": st.slider("Property and infrastructure", 0.0, 1.0, DEFAULT_LINE_WEIGHTS["property_infra"], step=0.01),
            "business_interruption": st.slider("Business interruption", 0.0, 1.0, DEFAULT_LINE_WEIGHTS["business_interruption"], step=0.01),
        }

with st.spinner("Loading live weather, local thresholds and portfolio stress..."):
    scored, regions = load_scored_data(refresh, tuple(sorted(weights.items())))

available_dates = sorted(scored["date"].dt.date.unique())
default_date = date.today() if date.today() in available_dates else available_dates[-1]
with st.sidebar:
    selected_date_value = st.selectbox("Analysis date", available_dates, index=available_dates.index(default_date))
selected_date = pd.Timestamp(selected_date_value)

day_view = selected_day_view(
    scored,
    selected_date,
    threshold_percentile=threshold_percentile,
    trigger_days=trigger_days,
    notional_chf_m=notional,
    payout_per_degree_day=payout_rate,
    cap_pct=cap_pct,
)

leader = day_view.iloc[0]
brief = underwriter_brief(day_view, selected_date)
basis_flags = basis_risk_table(day_view)
high_stress_no_pay = int((basis_flags["Flag"] == "High stress, no trigger").sum()) if not basis_flags.empty else 0
sensitivity = trigger_sensitivity(scored, selected_date, notional, payout_rate, cap_pct)

st.markdown(
    f"""
    <div class="hero">
        <h1>Heat Stress Reinsurance Workbench</h1>
        <p>Live European heat signal, contract response and forward trigger stress for accumulation risk.</p>
        <div class="status-row">
            <span class="status-pill">Open Meteo weather</span>
            <span class="status-pill">1991 to 2020 local thresholds</span>
            <span class="status-pill">CMIP6 forward stress</span>
            <span class="status-pill">{cache_freshness(ROOT / "data" / "cache")}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_grid(
    [
        ("Highest region", str(leader["city"]), f"{float(leader['composite_risk_score']):.1f} score"),
        ("Active triggers", int(day_view["trigger_active"].sum())),
        ("Scenario payout", fmt_chf_compact(float(day_view["modeled_payout_chf_m"].sum()))),
        ("High stress no trigger", high_stress_no_pay),
    ]
)

st.markdown(
    f"""
    <div class="insight-panel">
        <strong>Current signal.</strong>
        <span>{brief["event"]} {brief["contract"]} {brief["basis"]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_today, tab_contract, tab_forward, tab_city, tab_brief = st.tabs(
    ["Live portfolio", "Trigger lab", "Climate forward", "City drilldown", "Reviewer brief"]
)

with tab_today:
    map_col, queue_col = st.columns([1.25, 1.05])
    with map_col:
        st.plotly_chart(plot_map(day_view), width="stretch")
    with queue_col:
        st.subheader("Underwriting queue")
        st.dataframe(action_queue(day_view), width="stretch", hide_index=True)
    st.plotly_chart(plot_risk_decomposition(day_view), width="stretch")
    st.plotly_chart(plot_module_heatmap(day_view), width="stretch")
    with st.expander("Full attention list"):
        st.dataframe(top_regions(day_view), width="stretch", hide_index=True)

with tab_contract:
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(plot_trigger_heatmap(sensitivity), width="stretch")
        st.markdown(
            f"""
            <div class="formula-card">
            payout = min({fmt_pct(cap_pct)} x notional, heat degree days x {payout_rate:.2f} x notional)<br>
            trigger = active only after {trigger_days} consecutive days above local p{threshold_percentile}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.plotly_chart(plot_basis_scatter(day_view), width="stretch")

    trigger_df = day_view.sort_values("modeled_payout_chf_m", ascending=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.bar(
            trigger_df.head(12),
            x="city",
            y="modeled_payout_chf_m",
            color="trigger_active",
            labels={"modeled_payout_chf_m": "Scenario payout in CHF millions", "city": ""},
            title="Contract response by region",
            color_discrete_map={True: "#ef4444", False: "#94a3b8"},
        )
        st.plotly_chart(chart_theme(fig, 390), width="stretch")
    with c2:
        st.subheader("Basis risk flags")
        if basis_flags.empty:
            st.info("No major basis risk flags for the selected trigger.")
        else:
            st.dataframe(basis_flags, width="stretch", hide_index=True)

with tab_forward:
    projection_city = st.selectbox(
        "Region for forward trigger stress",
        options=day_view.sort_values("composite_risk_score", ascending=False)["city"].tolist(),
    )
    region_row = day_view[day_view["city"] == projection_city].iloc[0]
    local_threshold = float(region_row["trigger_threshold_c"])

    try:
        with st.spinner(f"Loading {projection_city} climate projection to {projection_horizon}..."):
            projection, projection_source = load_projection(
                str(region_row["region_id"]),
                float(region_row["latitude"]),
                float(region_row["longitude"]),
                int(projection_horizon),
                refresh_projection,
            )
            annual = summarize_projection(projection, local_threshold)
            period_summary = projection_decade_summary(annual)
            projected = add_projection_payout(annual, trigger_days, notional, payout_rate, cap_pct)
            payout_summary = projected_payout_summary(projected)

        model_mean = annual.groupby("year", as_index=False)["local_extreme_days"].mean()
        trend_slope = float(np.polyfit(model_mean["year"], model_mean["local_extreme_days"], 1)[0])
        metric_grid(
            [
                ("Local trigger", f"{local_threshold:.1f} deg C"),
                ("Trigger activation", fmt_pct(float(projected["projected_trigger_active"].mean()))),
                ("Trend", f"{trend_slope:+.2f}", "days/year"),
                ("Projection source", "CMIP6" if "CMIP6" in projection_source else "Fallback"),
            ]
        )
        st.caption(
            f"Projection source: {projection_source}. The selected trigger design is applied to each projected summer. "
            "This is a forward stress test, not a market loss forecast."
        )

        p1, p2 = st.columns([1, 1])
        with p1:
            st.plotly_chart(plot_projection_fan(annual, projection_city), width="stretch")
        with p2:
            st.plotly_chart(plot_projection_payout(projected, projection_city), width="stretch")
        e1, e2 = st.columns([1, 1])
        with e1:
            st.plotly_chart(plot_exceedance(projected, projection_city), width="stretch")
        with e2:
            st.subheader("Forward stress table")
            st.dataframe(payout_summary, width="stretch", hide_index=True)
            with st.expander("Climate threshold summary"):
                st.dataframe(period_summary, width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Forward projection could not be loaded for {projection_city}: {exc}")

with tab_city:
    city = st.selectbox(
        "Region for detailed review",
        options=day_view.sort_values("composite_risk_score", ascending=False)["city"].tolist(),
    )
    city_row = day_view[day_view["city"] == city].iloc[0]
    metric_grid(
        [
            ("Composite risk", f"{city_row['composite_risk_score']:.1f}"),
            ("Max temp", f"{city_row['temperature_2m_max']:.1f} deg C"),
            ("Anomaly", f"{city_row['temp_anomaly']:.1f} deg C"),
            ("Scenario payout", fmt_chf(float(city_row["modeled_payout_chf_m"]))),
        ]
    )
    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(plot_city_timeline(scored, city, threshold_percentile), width="stretch")
    with right:
        st.plotly_chart(plot_city_radar(city_row), width="stretch")

with tab_brief:
    cards = [("Event", brief["event"]), ("Contract response", brief["contract"]), ("Basis risk", brief["basis"]), ("Next review", brief["next"])]
    for start in range(0, len(cards), 2):
        card_cols = st.columns(2)
        for col, (title, body) in zip(card_cols, cards[start : start + 2]):
            with col:
                st.markdown(f'<div class="brief-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

    st.subheader("Model logic in plain English")
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown(
            """
            <div class="formula-card">
            heat severity = 30% anomaly + 25% local threshold excess + 25% heat streak + 20% heat degree days<br><br>
            portfolio score = weighted stress across life and health, agriculture, energy, infrastructure and business interruption
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="formula-card">
            contract response = active trigger x capped payout<br><br>
            current terms: p{threshold_percentile}, {trigger_days} day streak, {fmt_chf(notional)} notional, {fmt_pct(cap_pct)} cap
            </div>
            """,
            unsafe_allow_html=True,
        )
    with st.expander("Data status and limits"):
        st.write(cache_freshness(ROOT / "data" / "cache"))
        st.write(baseline_status(scored))
        st.write(
            "The dashboard uses live and forecast weather, local historical thresholds, exposure proxies and scenario terms. "
            "It is not a catastrophe model and does not estimate insured market loss. Production use would require real insured values, policy terms, claims, attachment points and audited exposure data."
        )
