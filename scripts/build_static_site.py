from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_risk.climate import (  # noqa: E402
    fallback_projection_from_baseline,
    fetch_climate_projection,
    summarize_projection,
)
from heatwave_risk.data import build_weather_dataset  # noqa: E402
from heatwave_risk.risk import DEFAULT_LINE_WEIGHTS, compute_risk_scores, selected_day_view  # noqa: E402


MODULE_LABELS = {
    "life_health_stress": "Life and health",
    "agriculture_stress": "Agriculture",
    "energy_stress": "Energy",
    "property_infra_stress": "Infrastructure",
    "business_interruption_stress": "Business interruption",
}


def fmt_chf(value: float) -> str:
    return f"CHF {value:,.1f}m"


def chart_theme(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=12),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
        margin=dict(l=16, r=16, t=62, b=72),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf2f7", zeroline=False)
    return fig


def main_driver(row: pd.Series) -> str:
    scores = {column: float(row[column]) for column in MODULE_LABELS}
    return MODULE_LABELS[max(scores, key=scores.get)]


def basis_flags(day_view: pd.DataFrame) -> pd.DataFrame:
    median_score = float(day_view["composite_risk_score"].median())
    rows = []
    for _, row in day_view.iterrows():
        high_no_pay = row["composite_risk_score"] >= median_score and not bool(row["trigger_active"])
        low_with_pay = row["composite_risk_score"] < median_score and bool(row["trigger_active"])
        if high_no_pay or low_with_pay:
            rows.append(
                {
                    "region": f"{row['city']}, {row['country']}",
                    "flag": "High stress without trigger" if high_no_pay else "Trigger active at lower stress",
                    "score": round(float(row["composite_risk_score"]), 1),
                    "payout": fmt_chf(float(row["modeled_payout_chf_m"])),
                }
            )
    return pd.DataFrame(rows)


def trigger_sensitivity(scored: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for percentile in ["95", "98", "99"]:
        for streak in range(2, 7):
            view = selected_day_view(
                scored,
                selected_date,
                threshold_percentile=percentile,
                trigger_days=streak,
                notional_chf_m=25,
                payout_per_degree_day=0.08,
                cap_pct=0.25,
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


def add_projection_payout(annual: pd.DataFrame) -> pd.DataFrame:
    projected = annual.copy()
    cap = 0.25 * 25
    projected["projected_trigger_active"] = projected["max_local_extreme_streak"] >= 3
    raw = projected["local_heat_degree_days"] * 0.08 * 25
    projected["projected_payout_chf_m"] = np.where(projected["projected_trigger_active"], raw.clip(upper=cap), 0).round(2)
    return projected


def build_figures(day_view: pd.DataFrame, scored: pd.DataFrame, selected_date: pd.Timestamp) -> dict[str, str]:
    map_fig = px.scatter_geo(
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
            "modeled_payout_chf_m": ":.2f",
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale=["#d9f99d", "#fde68a", "#fb923c", "#b91c1c"],
        range_color=(0, 100),
        projection="natural earth",
        title="Current heat stress across monitored European regions",
    )
    map_fig.update_geos(scope="europe", showland=True, landcolor="#f8fafc", showcountries=True, countrycolor="#cbd5e1", showocean=True, oceancolor="#edf6fb")
    map_fig.update_layout(coloraxis_colorbar_title="Risk score")

    top = day_view.sort_values("composite_risk_score", ascending=False).head(9)
    tidy = top.melt(id_vars=["city"], value_vars=list(MODULE_LABELS), var_name="module", value_name="stress")
    tidy["module"] = tidy["module"].map(MODULE_LABELS)
    drivers_fig = px.bar(
        tidy,
        x="stress",
        y="city",
        color="module",
        orientation="h",
        title="Why the highest regions are flagged",
        labels={"stress": "Stress contribution", "city": "", "module": ""},
        color_discrete_map={
            "Life and health": "#0ea5e9",
            "Agriculture": "#65a30d",
            "Energy": "#f59e0b",
            "Infrastructure": "#64748b",
            "Business interruption": "#8b5cf6",
        },
    )
    drivers_fig.update_layout(barmode="stack")

    sensitivity = trigger_sensitivity(scored, selected_date)
    sensitivity_fig = px.imshow(
        sensitivity.pivot(index="percentile", columns="streak_days", values="total_payout_chf_m"),
        text_auto=".1f",
        color_continuous_scale=["#f8fafc", "#c7f9e8", "#facc15", "#f97316", "#b91c1c"],
        labels=dict(x="Required consecutive days", y="Local trigger percentile", color="CHF m"),
        title="Trigger wording changes the scenario payout",
        aspect="auto",
    )

    basis_fig = px.scatter(
        day_view,
        x="composite_risk_score",
        y="modeled_payout_chf_m",
        color="trigger_active",
        size="event_degree_days",
        hover_name="city",
        title="Basis risk check: stress score versus contract response",
        labels={
            "composite_risk_score": "Composite risk score",
            "modeled_payout_chf_m": "Scenario payout in CHF millions",
            "trigger_active": "Trigger active",
            "event_degree_days": "Heat degree days",
        },
        color_discrete_map={True: "#ef4444", False: "#64748b"},
    )

    leader = day_view.iloc[0]
    timeline_data = scored[scored["city"] == leader["city"]].sort_values("date")
    timeline_fig = go.Figure()
    timeline_fig.add_trace(go.Scatter(x=timeline_data["date"], y=timeline_data["temperature_2m_max"], mode="lines+markers", name="Daily max", line=dict(color="#ef4444", width=3)))
    timeline_fig.add_trace(go.Scatter(x=timeline_data["date"], y=timeline_data["clim_mean"], mode="lines", name="Local normal", line=dict(color="#64748b", width=2, dash="dot")))
    timeline_fig.add_trace(go.Scatter(x=timeline_data["date"], y=timeline_data["clim_p98"], mode="lines", name="Local p98 trigger", line=dict(color="#111827", width=2, dash="dash")))
    timeline_fig.update_layout(title=f"{leader['city']}: current event against local climatology", yaxis_title="Daily max temperature in deg C", hovermode="x unified")

    figures = {
        "map": to_html(chart_theme(map_fig, 500), include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False, "responsive": True}),
        "drivers": to_html(chart_theme(drivers_fig, 440), include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True}),
        "sensitivity": to_html(chart_theme(sensitivity_fig, 360), include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True}),
        "basis": to_html(chart_theme(basis_fig, 410), include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True}),
        "timeline": to_html(chart_theme(timeline_fig, 390), include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True}),
    }

    try:
        projection = fetch_climate_projection(
            region_id=str(leader["region_id"]),
            latitude=float(leader["latitude"]),
            longitude=float(leader["longitude"]),
            cache_dir=ROOT / "data" / "cache",
            end_year=2035,
            refresh=False,
        )
        projection_source = "Open Meteo CMIP6"
    except Exception:
        projection = fallback_projection_from_baseline(str(leader["region_id"]), ROOT / "data" / "cache", end_year=2035)
        projection_source = "Local fallback"

    annual = summarize_projection(projection, float(leader["trigger_threshold_c"]))
    projected = add_projection_payout(annual)
    mean = annual.groupby("year", as_index=False)["local_extreme_days"].mean()
    bands = annual.groupby("year", as_index=False).agg(
        p10=("local_extreme_days", lambda values: values.quantile(0.10)),
        p90=("local_extreme_days", lambda values: values.quantile(0.90)),
    )
    forward_fig = go.Figure()
    forward_fig.add_trace(go.Scatter(x=list(bands["year"]) + list(bands["year"])[::-1], y=list(bands["p90"]) + list(bands["p10"])[::-1], fill="toself", name="Model range p10 to p90", line=dict(color="rgba(14,165,233,0)"), fillcolor="rgba(14,165,233,0.18)", hoverinfo="skip"))
    forward_fig.add_trace(go.Scatter(x=mean["year"], y=mean["local_extreme_days"], mode="lines+markers", name="Model average", line=dict(color="#0f766e", width=3)))
    z = np.polyfit(mean["year"], mean["local_extreme_days"], 1)
    forward_fig.add_trace(go.Scatter(x=mean["year"], y=np.poly1d(z)(mean["year"]), mode="lines", name="Trend", line=dict(color="#111827", width=2, dash="dash")))
    forward_fig.update_layout(
        title=f"{leader['city']}: projected summers above today's trigger threshold",
        xaxis_title="Summer year",
        yaxis_title="Days above local threshold",
        hovermode="x unified",
    )
    figures["forward"] = to_html(chart_theme(forward_fig, 410), include_plotlyjs=False, full_html=False, config={"displayModeBar": False, "responsive": True})
    figures["projection_source"] = projection_source
    figures["activation_rate"] = f"{projected['projected_trigger_active'].mean():.0%}"
    figures["p95_forward_payout"] = fmt_chf(float(projected["projected_payout_chf_m"].quantile(0.95)))
    return figures


def write_preview_svg(day_view: pd.DataFrame, selected_date: pd.Timestamp) -> None:
    leader = day_view.iloc[0]
    rows = day_view.sort_values("composite_risk_score", ascending=False).head(5)
    bars = []
    y = 270
    for _, row in rows.iterrows():
        width = int(float(row["composite_risk_score"]) * 4.5)
        bars.append(
            f'<text x="70" y="{y}" class="label">{row["city"]}</text>'
            f'<rect x="230" y="{y - 18}" width="{width}" height="18" rx="5" fill="#0f766e"/>'
            f'<text x="{245 + width}" y="{y}" class="small">{float(row["composite_risk_score"]):.1f}</text>'
        )
        y += 42
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">
<style>
.bg {{ fill: #f7fafc; }}
.card {{ fill: #ffffff; stroke: #dbe4ea; stroke-width: 1; }}
.title {{ font: 700 42px Arial, sans-serif; fill: #102a43; }}
.subtitle {{ font: 400 22px Arial, sans-serif; fill: #425466; }}
.metric {{ font: 700 44px Arial, sans-serif; fill: #172033; }}
.small {{ font: 400 18px Arial, sans-serif; fill: #425466; }}
.label {{ font: 700 20px Arial, sans-serif; fill: #172033; }}
</style>
<rect class="bg" width="1200" height="675"/>
<rect class="card" x="44" y="42" width="1112" height="590" rx="10"/>
<text x="70" y="105" class="title">Heat Stress Reinsurance Workbench</text>
<text x="70" y="145" class="subtitle">Live heat signal, trigger response and forward stress test for Europe.</text>
<rect class="card" x="70" y="185" width="245" height="110" rx="8"/>
<text x="92" y="226" class="small">Highest region</text>
<text x="92" y="274" class="metric">{leader["city"]}</text>
<rect class="card" x="340" y="185" width="245" height="110" rx="8"/>
<text x="362" y="226" class="small">Active triggers</text>
<text x="362" y="274" class="metric">{int(day_view["trigger_active"].sum())}</text>
<rect class="card" x="610" y="185" width="245" height="110" rx="8"/>
<text x="632" y="226" class="small">Scenario payout</text>
<text x="632" y="274" class="metric">{fmt_chf(float(day_view["modeled_payout_chf_m"].sum()))}</text>
<rect class="card" x="880" y="185" width="245" height="110" rx="8"/>
<text x="902" y="226" class="small">Analysis date</text>
<text x="902" y="274" class="metric">{selected_date.date()}</text>
<text x="70" y="350" class="subtitle">Current portfolio queue</text>
{''.join(bars)}
</svg>'''
    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "preview.svg").write_text(svg, encoding="utf-8")


def render_html(day_view: pd.DataFrame, selected_date: pd.Timestamp, figures: dict[str, str]) -> str:
    leader = day_view.iloc[0]
    flags = basis_flags(day_view)
    flag_text = "No major basis risk flags in the top view." if flags.empty else f"{len(flags)} regions show stress that the trigger does not fully mirror."
    queue_rows = "\n".join(
        f"<tr><td>{row['city']}, {row['country']}</td><td>{float(row['composite_risk_score']):.1f}</td><td>{main_driver(row)}</td><td>{'Active' if bool(row['trigger_active']) else 'Not active'}</td><td>{fmt_chf(float(row['modeled_payout_chf_m']))}</td></tr>"
        for _, row in day_view.sort_values("composite_risk_score", ascending=False).head(8).iterrows()
    )
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heat Stress Reinsurance Workbench</title>
  <meta name="description" content="A browser preview of a European heat stress reinsurance analytics workbench.">
  <style>
    :root {{ color-scheme: light; --ink: #172033; --muted: #425466; --line: #dbe4ea; --paper: #ffffff; --soft: #f7fafc; --accent: #0f766e; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: var(--ink); background: var(--soft); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 56px; }}
    header {{ background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 28px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(2rem, 5vw, 4rem); line-height: 1; letter-spacing: 0; color: #102a43; }}
    h2 {{ margin: 0 0 14px; font-size: 1.45rem; color: #102a43; }}
    p {{ font-size: 1.02rem; line-height: 1.65; color: var(--muted); }}
    a {{ color: #0f766e; font-weight: 700; }}
    .lede {{ max-width: 840px; margin: 0; }}
    .pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 7px 12px; color: var(--muted); background: #fbfdff; font-size: .9rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .metric {{ background: var(--paper); border: 1px solid var(--line); border-radius: 8px; padding: 18px; min-height: 116px; }}
    .metric span {{ display: block; color: var(--muted); font-size: .92rem; margin-bottom: 12px; }}
    .metric strong {{ display: block; font-size: clamp(1.8rem, 4vw, 2.8rem); line-height: 1; color: var(--ink); }}
    section {{ background: var(--paper); border: 1px solid var(--line); border-radius: 10px; padding: 22px; margin: 18px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .note {{ background: #fbfefd; border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; margin: 18px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .95rem; }}
    th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px 8px; text-align: left; }}
    th {{ color: #102a43; }}
    .footer {{ color: var(--muted); font-size: .95rem; margin-top: 18px; }}
    @media (max-width: 860px) {{
      .metrics, .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 14px; }}
      header, section {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Heat Stress Reinsurance Workbench</h1>
      <p class="lede">I built this as a small reinsurance analytics case study around a current European heat event. The idea is simple: take live weather data, compare it with local climatology, translate it into line-of-business stress, and test whether a parametric heat trigger would respond in a sensible way.</p>
      <div class="pills">
        <span class="pill">Open Meteo weather</span>
        <span class="pill">1991 to 2020 local thresholds</span>
        <span class="pill">Parametric trigger simulator</span>
        <span class="pill">CMIP6 forward stress</span>
      </div>
    </header>

    <div class="metrics">
      <div class="metric"><span>Highest region</span><strong>{leader["city"]}</strong></div>
      <div class="metric"><span>Active triggers</span><strong>{int(day_view["trigger_active"].sum())}</strong></div>
      <div class="metric"><span>Scenario payout</span><strong>{fmt_chf(float(day_view["modeled_payout_chf_m"].sum()))}</strong></div>
      <div class="metric"><span>Analysis date</span><strong>{selected_date.date()}</strong></div>
    </div>

    <div class="note">
      <p><strong>Current read.</strong> {leader["city"]} leads the monitored portfolio with a {float(leader["composite_risk_score"]):.1f} risk score. The main driver is {main_driver(leader).lower()}. {flag_text}</p>
    </div>

    <section>{figures["map"]}</section>

    <section>
      <h2>Underwriting queue</h2>
      <table>
        <thead><tr><th>Region</th><th>Score</th><th>Main driver</th><th>Trigger</th><th>Payout</th></tr></thead>
        <tbody>{queue_rows}</tbody>
      </table>
    </section>

    <div class="grid">
      <section>{figures["drivers"]}</section>
      <section>{figures["timeline"]}</section>
      <section>{figures["sensitivity"]}</section>
      <section>{figures["basis"]}</section>
    </div>

    <section>
      <h2>Forward stress</h2>
      <p>The forward view applies the same trigger logic to projected summers for the current leading region. In this static preview the projection source is {figures["projection_source"]}; the projected trigger activation rate is {figures["activation_rate"]}, with a p95 scenario payout of {figures["p95_forward_payout"]}. This is a stress test, not a loss forecast.</p>
      {figures["forward"]}
    </section>

    <p class="footer">Full source code, Streamlit app and methodology notes are in the GitHub repository. This project is a portfolio case study, not underwriting advice.</p>
  </main>
</body>
</html>'''


def main() -> None:
    daily, _regions = build_weather_dataset(ROOT / "data" / "regions.csv", ROOT / "data" / "cache", refresh=False)
    scored = compute_risk_scores(daily, DEFAULT_LINE_WEIGHTS)
    available_dates = sorted(scored["date"].dt.date.unique())
    selected_date_value = date.today() if date.today() in available_dates else available_dates[-1]
    selected_date = pd.Timestamp(selected_date_value)
    day_view = selected_day_view(
        scored,
        selected_date,
        threshold_percentile="98",
        trigger_days=3,
        notional_chf_m=25,
        payout_per_degree_day=0.08,
        cap_pct=0.25,
    )
    figures = build_figures(day_view, scored, selected_date)
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    write_preview_svg(day_view, selected_date)
    (docs / "index.html").write_text(render_html(day_view, selected_date, figures), encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built docs/index.html at {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
