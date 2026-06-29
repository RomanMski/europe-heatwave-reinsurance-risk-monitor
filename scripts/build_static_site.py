from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_risk.climate import (  # noqa: E402
    fallback_projection_from_baseline,
    fetch_climate_projection,
    summarize_projection,
)
from heatwave_risk.data import build_weather_dataset  # noqa: E402
from heatwave_risk.risk import DEFAULT_LINE_WEIGHTS, compute_risk_scores  # noqa: E402


MODULES = {
    "life_health_stress": "Life and health",
    "agriculture_stress": "Agriculture",
    "energy_stress": "Energy",
    "property_infra_stress": "Infrastructure",
    "business_interruption_stress": "Business interruption",
}


def clean_float(value: object, digits: int = 2) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), digits)


def max_streak(values: pd.Series) -> int:
    current = 0
    best = 0
    for value in values.fillna(False):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def scenario_rows(scored: pd.DataFrame, selected_date: pd.Timestamp) -> list[dict]:
    start_date = selected_date - pd.Timedelta(days=9)
    window = scored[(scored["date"] >= start_date) & (scored["date"] <= selected_date)].copy()
    current = scored[scored["date"] == selected_date].copy()

    event_degree_days: dict[str, pd.Series] = {}
    for percentile in ("95", "98", "99"):
        threshold = f"clim_p{percentile}"
        excess = (window["temperature_2m_max"] - window[threshold]).clip(lower=0)
        tmp = window[["region_id"]].copy()
        tmp["excess"] = excess
        event_degree_days[percentile] = tmp.groupby("region_id")["excess"].sum()

    rows = []
    for _, row in current.iterrows():
        region_id = str(row["region_id"])
        rows.append(
            {
                "region_id": region_id,
                "city": str(row["city"]),
                "country": str(row["country"]),
                "lat": clean_float(row["latitude"], 4),
                "lon": clean_float(row["longitude"], 4),
                "temp": clean_float(row["temperature_2m_max"], 1),
                "mean_temp": clean_float(row["temperature_2m_mean"], 1),
                "anomaly": clean_float(row["temp_anomaly"], 1),
                "score": clean_float(row["composite_risk_score"], 1),
                "hdd30": clean_float(row["hdd_30"], 1),
                "stress": {label: clean_float(row[column], 1) for column, label in MODULES.items()},
                "thresholds": {
                    "95": clean_float(row["clim_p95"], 1),
                    "98": clean_float(row["clim_p98"], 1),
                    "99": clean_float(row["clim_p99"], 1),
                },
                "streaks": {
                    "95": int(row["p95_streak"]),
                    "98": int(row["p98_streak"]),
                    "99": int(row["p99_streak"]),
                },
                "event_degree_days": {
                    pct: clean_float(event_degree_days[pct].get(region_id, 0), 2)
                    for pct in ("95", "98", "99")
                },
            }
        )
    return sorted(rows, key=lambda item: item["score"], reverse=True)


def timeline_rows(scored: pd.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for city, frame in scored.sort_values("date").groupby("city"):
        out[str(city)] = [
            {
                "date": str(row["date"].date()),
                "temp": clean_float(row["temperature_2m_max"], 1),
                "normal": clean_float(row["clim_mean"], 1),
                "p95": clean_float(row["clim_p95"], 1),
                "p98": clean_float(row["clim_p98"], 1),
                "p99": clean_float(row["clim_p99"], 1),
            }
            for _, row in frame.iterrows()
        ]
    return out


def projection_rows(leader: dict) -> tuple[list[dict], str]:
    region_id = leader["region_id"]
    try:
        projection = fetch_climate_projection(
            region_id=region_id,
            latitude=float(leader["lat"]),
            longitude=float(leader["lon"]),
            cache_dir=ROOT / "data" / "cache",
            end_year=2035,
            refresh=False,
        )
        source = "Open Meteo CMIP6 climate API"
    except Exception:
        projection = fallback_projection_from_baseline(region_id, ROOT / "data" / "cache", end_year=2035)
        source = "historical-cache fallback"

    annual = summarize_projection(projection, float(leader["thresholds"]["98"]))
    cap = 0.25 * 25
    raw_payout = annual["local_heat_degree_days"] * 0.08 * 25
    annual["projected_trigger_active"] = annual["max_local_extreme_streak"] >= 3
    annual["projected_payout_chf_m"] = np.where(annual["projected_trigger_active"], raw_payout.clip(upper=cap), 0)
    return (
        [
            {
                "model": str(row["model"]),
                "year": int(row["year"]),
                "extreme_days": clean_float(row["local_extreme_days"], 1),
                "max_streak": int(row["max_local_extreme_streak"]),
                "payout": clean_float(row["projected_payout_chf_m"], 2),
            }
            for _, row in annual.iterrows()
        ],
        source,
    )


def write_preview_svg(rows: list[dict], selected_date: pd.Timestamp) -> None:
    leader = rows[0]
    bars = []
    y = 320
    for row in rows[:5]:
        width = int(row["score"] * 4.6)
        bars.append(
            f'<text x="84" y="{y}" class="city">{row["city"]}</text>'
            f'<rect x="255" y="{y - 18}" width="{width}" height="18" rx="5" fill="#0f766e"/>'
            f'<text x="{270 + width}" y="{y}" class="note">{row["score"]:.1f}</text>'
        )
        y += 43

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">
<style>
.bg {{ fill: #f4f7f9; }}
.panel {{ fill: #fff; stroke: #dbe4ea; stroke-width: 1; }}
.title {{ font: 700 52px Arial, sans-serif; fill: #102a43; }}
.copy {{ font: 400 23px Arial, sans-serif; fill: #425466; }}
.metric {{ font: 700 52px Arial, sans-serif; fill: #172033; }}
.note {{ font: 400 18px Arial, sans-serif; fill: #425466; }}
.city {{ font: 700 20px Arial, sans-serif; fill: #172033; }}
</style>
<rect class="bg" width="1200" height="675"/>
<rect class="panel" x="44" y="38" width="1112" height="598" rx="10"/>
<text x="80" y="112" class="title">Heat Stress Reinsurance Workbench</text>
<text x="80" y="154" class="copy">A current heat event turned into a reinsurance trigger and basis-risk view.</text>
<rect class="panel" x="80" y="195" width="245" height="98" rx="8"/>
<text x="102" y="232" class="note">Highest region</text>
<text x="102" y="275" class="metric">{leader["city"]}</text>
<rect class="panel" x="350" y="195" width="245" height="98" rx="8"/>
<text x="372" y="232" class="note">Risk score</text>
<text x="372" y="275" class="metric">{leader["score"]:.1f}</text>
<rect class="panel" x="620" y="195" width="245" height="98" rx="8"/>
<text x="642" y="232" class="note">Analysis date</text>
<text x="642" y="275" class="metric">{selected_date.date()}</text>
<text x="80" y="354" class="copy">Current underwriting queue</text>
{''.join(bars)}
</svg>'''
    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "preview.svg").write_text(svg, encoding="utf-8")


def build_payload() -> dict:
    daily, _regions = build_weather_dataset(ROOT / "data" / "regions.csv", ROOT / "data" / "cache", refresh=False)
    scored = compute_risk_scores(daily, DEFAULT_LINE_WEIGHTS)
    available_dates = sorted(scored["date"].dt.date.unique())
    recent_cache = list((ROOT / "data" / "cache" / "open_meteo_recent").glob("*.json"))
    cache_date = None
    if recent_cache:
        latest_cache = max(file.stat().st_mtime for file in recent_cache)
        cache_date = datetime.fromtimestamp(latest_cache).date()
    selected_date_value = cache_date if cache_date in available_dates else date.today()
    if selected_date_value not in available_dates:
        selected_date_value = available_dates[-1]
    selected_date = pd.Timestamp(selected_date_value)
    rows = scenario_rows(scored, selected_date)
    projection, projection_source = projection_rows(rows[0])
    write_preview_svg(rows, selected_date)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "analysis_date": str(selected_date.date()),
        "regions": rows,
        "timeline": timeline_rows(scored),
        "projection": projection,
        "projection_source": projection_source,
        "repo_url": "https://github.com/RomanMski/europe-heatwave-reinsurance-risk-monitor",
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heat Stress Reinsurance Workbench</title>
  <meta name="description" content="Interactive European heat stress reinsurance case study by Roman Mirosenski.">
  <style>
    :root {
      --ink: #172033;
      --muted: #425466;
      --line: #d8e1e8;
      --paper: #ffffff;
      --soft: #f4f7f9;
      --teal: #0f766e;
      --blue: #2563eb;
      --amber: #f59e0b;
      --red: #dc2626;
      --violet: #7c3aed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: var(--ink);
      background: var(--soft);
    }
    a { color: var(--teal); font-weight: 700; text-decoration-thickness: 1px; }
    main { max-width: 1280px; margin: 0 auto; padding: 22px 18px 56px; }
    .topbar {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .topbar nav { display: flex; gap: 14px; flex-wrap: wrap; }
    .hero {
      display: grid;
      grid-template-columns: minmax(360px, 0.78fr) minmax(560px, 1.22fr);
      gap: 18px;
      align-items: start;
    }
    .hero > *, .visual, .intro, .panel, .chart, .table-panel, .note { min-width: 0; }
    .intro, .panel, .metric, .control, .chart, .note, .table-panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .intro { padding: 26px 28px; display: flex; flex-direction: column; gap: 20px; }
    h1 {
      margin: 0 0 16px;
      font-size: clamp(2.35rem, 4vw, 4.25rem);
      line-height: 1.01;
      letter-spacing: 0;
      color: #102a43;
    }
    h2 { margin: 0 0 12px; color: #102a43; font-size: 1.55rem; }
    h3 { margin: 0 0 8px; color: #102a43; font-size: 1.1rem; }
    p { color: var(--muted); line-height: 1.58; font-size: 1.01rem; }
    .intro p { font-size: 1.02rem; }
    .signature { margin: 0; color: var(--ink); font-weight: 700; }
    .live-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }
    .metric { padding: 14px 16px; min-height: 96px; }
    .metric span { display: block; color: var(--muted); font-size: 0.88rem; margin-bottom: 10px; }
    .metric strong { display: block; font-size: clamp(1.8rem, 3vw, 2.65rem); line-height: 1; }
    .metric small { display: block; color: var(--teal); margin-top: 8px; font-weight: 700; font-size: 0.88rem; }
    .visual { display: grid; grid-template-rows: auto 1fr; gap: 12px; }
    .controls {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }
    .control { padding: 12px; }
    .control label { display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 8px; }
    select, input[type="range"] { width: 100%; }
    select {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
    }
    input[type="range"] { accent-color: var(--teal); }
    .control b { font-size: 1.1rem; }
    .map-panel { min-height: 510px; padding: 10px; }
    .chart { min-height: 390px; padding: 12px; }
    .chart svg, .map-panel svg { width: 100%; max-width: 100%; height: 100%; min-height: 360px; display: block; }
    .map-panel svg { min-height: 490px; }
    .axis { stroke: #cbd5e1; stroke-width: 1; }
    .gridline { stroke: #edf2f7; stroke-width: 1; }
    .chart-title { font: 700 20px Inter, Arial, sans-serif; fill: #102a43; }
    .chart-label { font: 400 13px Inter, Arial, sans-serif; fill: #425466; }
    .chart-small { font: 700 12px Inter, Arial, sans-serif; fill: #172033; }
    .section { margin-top: 18px; }
    .two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .three { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 18px; align-items: start; }
    .note { padding: 22px; }
    .queue { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
    .queue th, .queue td { padding: 12px 10px; border-bottom: 1px solid #edf2f7; text-align: left; }
    .queue th { color: #102a43; }
    .badge {
      display: inline-flex;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 0.82rem;
      font-weight: 700;
      background: #eef7f5;
      color: #0f766e;
    }
    .badge.off { background: #f1f5f9; color: #475569; }
    .region-row { cursor: pointer; }
    .region-row:hover td { background: #f8fafc; }
    .footer { color: var(--muted); font-size: 0.95rem; margin-top: 20px; }
    @media (max-width: 1020px) {
      .hero, .two, .three { grid-template-columns: minmax(0, 1fr); width: 100%; }
      .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .map-panel { min-height: 420px; }
      .visual { margin-top: 18px; }
    }
    @media (max-width: 640px) {
      main { width: 100%; padding: 14px; overflow-x: hidden; }
      .hero, .two, .three, .visual, .intro, .panel, .chart, .table-panel, .note { width: 100%; max-width: 100%; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .topbar > div { max-width: 100%; overflow-wrap: anywhere; }
      .intro { padding: 22px; }
      h1 { font-size: clamp(2.15rem, 11vw, 2.95rem); }
      p { font-size: 0.98rem; line-height: 1.56; }
      .signature { font-size: 0.92rem; overflow-wrap: anywhere; }
      .live-grid, .controls { grid-template-columns: 1fr; }
      .map-panel { min-height: 360px; }
      .map-panel svg { min-height: 350px; }
      .chart { min-height: 330px; overflow: hidden; }
      .chart svg { min-height: 310px; }
      .queue { font-size: 0.86rem; }
      .queue th:nth-child(3), .queue td:nth-child(3) { display: none; }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>Roman Mirosenski | heat risk data project</div>
      <nav>
        <a href="#contract">contract lab</a>
        <a href="#forward">forward stress</a>
        <a id="repo-link" href="#">code</a>
      </nav>
    </div>

    <section class="hero">
      <div class="intro">
        <div>
          <h1>Heat Stress Reinsurance Workbench</h1>
          <p>I wanted this to feel like something a reinsurance team could actually open during a heat event. Not a notebook screenshot, not a toy chart. Live weather comes in, each region gets compared with its own climate history, then the page asks the insurance question: where is the stress, and does the trigger really catch it?</p>
          <p>This is a portfolio project, so the numbers are scenario values. Still, the workflow is real enough to show the thinking: data quality, thresholds, basis risk, payout sensitivity and future heat stress.</p>
        </div>
        <div>
          <div class="live-grid">
            <div class="metric"><span>highest region</span><strong id="metric-region">-</strong><small id="metric-driver">-</small></div>
            <div class="metric"><span>active triggers</span><strong id="metric-triggers">-</strong><small>under current wording</small></div>
            <div class="metric"><span>scenario payout</span><strong id="metric-payout">-</strong><small>selected notional</small></div>
            <div class="metric"><span>basis-risk flags</span><strong id="metric-basis">-</strong><small>stress that wording may miss</small></div>
          </div>
          <p class="signature">Data date <span id="analysis-date">-</span>. Source cached Open Meteo weather.</p>
        </div>
      </div>

      <div class="visual">
        <div class="controls">
          <div class="control"><label>trigger percentile</label><select id="pct"><option value="95">p95</option><option value="98" selected>p98</option><option value="99">p99</option></select></div>
          <div class="control"><label>required streak</label><input id="streak" type="range" min="2" max="6" step="1" value="3"><b id="streak-label">3 days</b></div>
          <div class="control"><label>notional per region</label><input id="notional" type="range" min="5" max="100" step="5" value="25"><b id="notional-label">CHF 25m</b></div>
          <div class="control"><label>payout rate</label><input id="rate" type="range" min="0.02" max="0.20" step="0.01" value="0.08"><b id="rate-label">0.08</b></div>
          <div class="control"><label>cap</label><input id="cap" type="range" min="0.05" max="0.50" step="0.05" value="0.25"><b id="cap-label">25%</b></div>
        </div>
        <div class="panel map-panel"><div id="map"></div></div>
      </div>
    </section>

    <section class="section three">
      <div class="table-panel">
        <div style="padding: 20px 20px 0;">
          <h2>Underwriting queue</h2>
          <p>Click a row to move the city drilldown. This is the part I would want first if I had to decide where to look before reading a long report.</p>
        </div>
        <table class="queue">
          <thead><tr><th>Region</th><th>Score</th><th>Main driver</th><th>Trigger</th><th>Payout</th></tr></thead>
          <tbody id="queue-body"></tbody>
        </table>
      </div>
      <div class="chart"><div id="drivers"></div></div>
    </section>

    <section class="section two" id="contract">
      <div class="chart"><div id="sensitivity"></div></div>
      <div class="chart"><div id="basis"></div></div>
    </section>

    <section class="section two">
      <div class="chart"><div id="timeline"></div></div>
      <div class="note">
        <h2 id="focus-title">City read</h2>
        <p id="focus-copy">-</p>
        <p>I kept the model transparent on purpose. A black-box score would look fancier, but for insurance work I think it is more useful to see which assumption is doing the work.</p>
      </div>
    </section>

    <section class="section two" id="forward">
      <div class="note">
        <h2>Forward stress</h2>
        <p id="forward-copy">-</p>
        <p>This is not a market loss forecast. It is a way to ask whether the same wording still behaves reasonably when future summers are hotter in the climate model data.</p>
      </div>
      <div class="chart"><div id="forward-chart"></div></div>
    </section>

    <p class="footer">Full Streamlit app, Python source and methodology notes are in the repository. The browser page is intentionally easier to open first, because nobody should need to install Python just to see the idea.</p>
  </main>

  <script id="payload" type="application/json">__DATA__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("payload").textContent);
    const state = { pct: "98", streak: 3, notional: 25, rate: 0.08, cap: 0.25, focus: DATA.regions[0].city };
    const colors = {
      red: "#dc2626",
      teal: "#0f766e",
      amber: "#f59e0b",
      blue: "#2563eb",
      slate: "#64748b",
      violet: "#7c3aed"
    };
    const config = { displayModeBar: false, responsive: true };

    document.getElementById("repo-link").href = DATA.repo_url;
    document.getElementById("analysis-date").textContent = DATA.analysis_date;

    function money(value) {
      if (value >= 10) return `CHF ${Math.round(value)}m`;
      return `CHF ${value.toFixed(1)}m`;
    }

    function driver(row) {
      return Object.entries(row.stress).sort((a, b) => b[1] - a[1])[0][0];
    }

    function scenario(row, overrides = {}) {
      const pct = overrides.pct || state.pct;
      const streak = overrides.streak || state.streak;
      const notional = overrides.notional || state.notional;
      const rate = overrides.rate || state.rate;
      const cap = overrides.cap || state.cap;
      const active = row.streaks[pct] >= streak;
      const raw = row.event_degree_days[pct] * rate * notional;
      const payout = active ? Math.min(raw, cap * notional) : 0;
      return { ...row, active, payout, activeText: active ? "Active" : "Not active", threshold: row.thresholds[pct], eventDegreeDays: row.event_degree_days[pct] };
    }

    function currentRows() {
      return DATA.regions.map(row => scenario(row)).sort((a, b) => b.score - a.score);
    }

    function updateLabels() {
      document.getElementById("streak-label").textContent = `${state.streak} days`;
      document.getElementById("notional-label").textContent = `CHF ${state.notional}m`;
      document.getElementById("rate-label").textContent = state.rate.toFixed(2);
      document.getElementById("cap-label").textContent = `${Math.round(state.cap * 100)}%`;
    }

    function updateMetrics(rows) {
      const leader = rows[0];
      const active = rows.filter(row => row.active).length;
      const total = rows.reduce((sum, row) => sum + row.payout, 0);
      const median = rows.map(row => row.score).sort((a, b) => a - b)[Math.floor(rows.length / 2)];
      const basis = rows.filter(row => row.score >= median && !row.active).length;
      document.getElementById("metric-region").textContent = leader.city;
      document.getElementById("metric-driver").textContent = `${leader.score.toFixed(1)} score, main driver ${driver(leader).toLowerCase()}`;
      document.getElementById("metric-triggers").textContent = active;
      document.getElementById("metric-payout").textContent = money(total);
      document.getElementById("metric-basis").textContent = basis;
    }

    function renderQueue(rows) {
      const body = document.getElementById("queue-body");
      body.innerHTML = rows.slice(0, 9).map(row => `
        <tr class="region-row" data-city="${row.city}">
          <td>${row.city}, ${row.country}</td>
          <td>${row.score.toFixed(1)}</td>
          <td>${driver(row)}</td>
          <td><span class="badge ${row.active ? "" : "off"}">${row.activeText}</span></td>
          <td>${money(row.payout)}</td>
        </tr>
      `).join("");
      body.querySelectorAll("tr").forEach(row => row.addEventListener("click", () => {
        state.focus = row.dataset.city;
        renderTimeline();
        renderFocusCopy();
      }));
    }

    function scale(value, min, max, a, b) {
      if (max === min) return (a + b) / 2;
      return a + ((value - min) / (max - min)) * (b - a);
    }

    function heatColor(value) {
      if (value >= 60) return "#b91c1c";
      if (value >= 45) return "#ef4444";
      if (value >= 25) return "#f59e0b";
      if (value >= 12) return "#0f766e";
      return "#64748b";
    }

    function cellColor(value, max) {
      const t = max <= 0 ? 0 : value / max;
      if (t > 0.75) return "#b91c1c";
      if (t > 0.50) return "#f97316";
      if (t > 0.25) return "#facc15";
      if (t > 0.05) return "#c7f9e8";
      return "#f8fafc";
    }

    function polyline(points) {
      return points.map(point => `${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
    }

    function areaPath(xs, high, low) {
      const top = xs.map((x, i) => `${x.toFixed(1)},${high[i].toFixed(1)}`).join(" L ");
      const bottom = [...xs].reverse().map((x, idx) => {
        const i = xs.length - 1 - idx;
        return `${x.toFixed(1)},${low[i].toFixed(1)}`;
      }).join(" L ");
      return `M ${top} L ${bottom} Z`;
    }

    function renderMap(rows) {
      const lonMin = -11;
      const lonMax = 25;
      const latMin = 36;
      const latMax = 55;
      const dots = rows.map(row => {
        const x = scale(row.lon, lonMin, lonMax, 70, 720);
        const y = scale(row.lat, latMin, latMax, 420, 80);
        const radius = Math.max(7, row.score / 3.1);
        return `<g class="city-dot" data-city="${row.city}" style="cursor:pointer">
          <circle cx="${x}" cy="${y}" r="${radius}" fill="${heatColor(row.score)}" stroke="#fff" stroke-width="2"/>
          <text x="${x + radius + 5}" y="${y + 4}" class="chart-small">${row.city}</text>
          <title>${row.city}, ${row.country} | score ${row.score.toFixed(1)} | ${row.activeText} | ${money(row.payout)}</title>
        </g>`;
      }).join("");
      document.getElementById("map").innerHTML = `
        <svg viewBox="0 0 780 520" role="img" aria-label="Current heat stress map">
          <rect x="0" y="0" width="780" height="520" fill="#fff"/>
          <text x="26" y="34" class="chart-title">Current heat stress across monitored Europe</text>
          <text x="26" y="58" class="chart-label">Dot size and color follow the current composite risk score. Click a city to update the drilldown.</text>
          <rect x="42" y="78" width="700" height="365" rx="8" fill="#edf6fb" stroke="#dbe4ea"/>
          <path d="M118,150 C174,102 250,102 304,140 C360,96 454,111 512,154 C582,158 666,210 684,292 C636,354 530,385 440,358 C361,410 250,383 216,326 C154,320 98,268 118,150 Z" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
          <line x1="42" x2="742" y1="260" y2="260" class="gridline"/>
          <line x1="390" x2="390" y1="78" y2="443" class="gridline"/>
          ${dots}
          <text x="42" y="480" class="chart-label">cooler / lower stress</text>
          <rect x="180" y="468" width="52" height="12" fill="#64748b"/><rect x="232" y="468" width="52" height="12" fill="#0f766e"/><rect x="284" y="468" width="52" height="12" fill="#f59e0b"/><rect x="336" y="468" width="52" height="12" fill="#ef4444"/><rect x="388" y="468" width="52" height="12" fill="#b91c1c"/>
          <text x="452" y="480" class="chart-label">higher stress</text>
        </svg>`;
      document.querySelectorAll(".city-dot").forEach(el => el.addEventListener("click", () => {
        state.focus = el.dataset.city;
        renderTimeline();
        renderFocusCopy();
      }));
    }

    function renderDrivers(rows) {
      const top = rows.slice(0, 8).reverse();
      const modules = ["Life and health", "Agriculture", "Energy", "Infrastructure", "Business interruption"];
      const moduleColors = [colors.blue, "#65a30d", colors.amber, colors.slate, colors.violet];
      const maxTotal = Math.max(...top.map(row => modules.reduce((sum, name) => sum + row.stress[name], 0)));
      const bars = top.map((row, idx) => {
        let x = 152;
        const y = 318 - idx * 34;
        const parts = modules.map((name, moduleIdx) => {
          const w = scale(row.stress[name], 0, maxTotal, 0, 520);
          const rect = `<rect x="${x}" y="${y - 17}" width="${w}" height="20" rx="3" fill="${moduleColors[moduleIdx]}"><title>${row.city} | ${name} ${row.stress[name].toFixed(1)}</title></rect>`;
          x += w;
          return rect;
        }).join("");
        return `<text x="24" y="${y}" class="chart-label">${row.city}</text>${parts}<text x="${x + 8}" y="${y}" class="chart-small">${row.score.toFixed(1)}</text>`;
      }).join("");
      const legend = modules.map((name, idx) => `<rect x="${26 + idx * 132}" y="364" width="12" height="12" fill="${moduleColors[idx]}"/><text x="${44 + idx * 132}" y="375" class="chart-label">${name}</text>`).join("");
      document.getElementById("drivers").innerHTML = `
        <svg viewBox="0 0 760 410" role="img" aria-label="Stress driver bars">
          <text x="24" y="34" class="chart-title">What is driving the top regions</text>
          <text x="24" y="58" class="chart-label">The score is decomposed into insurance-relevant pressure points, not just temperature.</text>
          ${bars}
          ${legend}
        </svg>`;
    }

    function renderSensitivity() {
      const percentiles = ["95", "98", "99"];
      const streaks = [2, 3, 4, 5, 6];
      const matrix = percentiles.map(pct => streaks.map(streak => DATA.regions.reduce((sum, row) => sum + scenario(row, { pct, streak }).payout, 0)));
      const max = Math.max(...matrix.flat(), 1);
      const cells = matrix.map((row, r) => row.map((value, c) => {
        const x = 155 + c * 104;
        const y = 112 + r * 64;
        return `<g class="heat-cell" data-pct="${percentiles[r]}" data-streak="${streaks[c]}" style="cursor:pointer">
          <rect x="${x}" y="${y}" width="96" height="52" rx="6" fill="${cellColor(value, max)}" stroke="#dbe4ea"/>
          <text x="${x + 48}" y="${y + 31}" text-anchor="middle" class="chart-small">${money(value)}</text>
          <title>p${percentiles[r]}, ${streaks[c]} days, ${money(value)}</title>
        </g>`;
      }).join("")).join("");
      const xLabels = streaks.map((streak, i) => `<text x="${203 + i * 104}" y="94" text-anchor="middle" class="chart-label">${streak}d</text>`).join("");
      const yLabels = percentiles.map((pct, i) => `<text x="104" y="${145 + i * 64}" class="chart-label">p${pct}</text>`).join("");
      document.getElementById("sensitivity").innerHTML = `
        <svg viewBox="0 0 760 360" role="img" aria-label="Trigger sensitivity heatmap">
          <text x="24" y="34" class="chart-title">Trigger wording sensitivity</text>
          <text x="24" y="58" class="chart-label">Click a cell to apply that wording to the whole page.</text>
          ${xLabels}${yLabels}${cells}
        </svg>`;
      document.querySelectorAll(".heat-cell").forEach(cell => cell.addEventListener("click", () => {
        state.pct = cell.dataset.pct;
        state.streak = Number(cell.dataset.streak);
        document.getElementById("pct").value = state.pct;
        document.getElementById("streak").value = state.streak;
        renderAll();
      }));
    }

    function renderBasis(rows) {
      const maxPayout = Math.max(...rows.map(row => row.payout), 1);
      const dots = rows.map(row => {
        const x = scale(row.score, 0, 100, 78, 690);
        const y = scale(row.payout, 0, maxPayout, 304, 82);
        const radius = Math.max(6, 5 + row.eventDegreeDays * 0.4);
        return `<circle cx="${x}" cy="${y}" r="${radius}" fill="${row.active ? colors.red : colors.slate}" opacity="0.85" stroke="#fff" stroke-width="1.5">
          <title>${row.city} | score ${row.score.toFixed(1)} | ${money(row.payout)} | ${row.activeText}</title>
        </circle>`;
      }).join("");
      document.getElementById("basis").innerHTML = `
        <svg viewBox="0 0 760 360" role="img" aria-label="Basis risk scatter">
          <text x="24" y="34" class="chart-title">Basis risk check</text>
          <text x="24" y="58" class="chart-label">High stress with no payout is where I would read the wording twice.</text>
          <line x1="78" x2="690" y1="304" y2="304" class="axis"/><line x1="78" x2="78" y1="82" y2="304" class="axis"/>
          <text x="585" y="334" class="chart-label">risk score</text><text x="20" y="92" class="chart-label">payout</text>
          ${dots}
          <circle cx="540" cy="58" r="6" fill="${colors.red}"/><text x="552" y="63" class="chart-label">active</text>
          <circle cx="620" cy="58" r="6" fill="${colors.slate}"/><text x="632" y="63" class="chart-label">not active</text>
        </svg>`;
    }

    function renderTimeline() {
      const rows = DATA.timeline[state.focus] || DATA.timeline[DATA.regions[0].city];
      const values = rows.flatMap(row => [row.temp, row.normal, row[`p${state.pct}`]]);
      const minY = Math.floor(Math.min(...values) - 2);
      const maxY = Math.ceil(Math.max(...values) + 2);
      const xFor = (idx) => scale(idx, 0, rows.length - 1, 70, 700);
      const yFor = (value) => scale(value, minY, maxY, 304, 84);
      const tempPts = rows.map((row, i) => [xFor(i), yFor(row.temp)]);
      const normalPts = rows.map((row, i) => [xFor(i), yFor(row.normal)]);
      const thresholdPts = rows.map((row, i) => [xFor(i), yFor(row[`p${state.pct}`])]);
      const circles = tempPts.map((point, i) => `<circle cx="${point[0]}" cy="${point[1]}" r="4" fill="${colors.red}"><title>${rows[i].date} | ${rows[i].temp} deg C</title></circle>`).join("");
      document.getElementById("timeline").innerHTML = `
        <svg viewBox="0 0 760 380" role="img" aria-label="City timeline">
          <text x="24" y="34" class="chart-title">${state.focus} against local history</text>
          <text x="24" y="58" class="chart-label">Daily max temperature, normal summer level and selected trigger threshold.</text>
          <line x1="70" x2="700" y1="304" y2="304" class="axis"/><line x1="70" x2="70" y1="84" y2="304" class="axis"/>
          <text x="18" y="90" class="chart-label">${maxY} deg C</text><text x="18" y="306" class="chart-label">${minY} deg C</text>
          <polyline points="${polyline(normalPts)}" fill="none" stroke="${colors.slate}" stroke-width="2" stroke-dasharray="5 5"/>
          <polyline points="${polyline(thresholdPts)}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="8 5"/>
          <polyline points="${polyline(tempPts)}" fill="none" stroke="${colors.red}" stroke-width="3"/>
          ${circles}
          <text x="72" y="345" class="chart-label">${rows[0].date}</text><text x="610" y="345" class="chart-label">${rows[rows.length - 1].date}</text>
          <rect x="420" y="28" width="12" height="12" fill="${colors.red}"/><text x="438" y="39" class="chart-label">daily max</text>
          <rect x="520" y="28" width="12" height="12" fill="${colors.slate}"/><text x="538" y="39" class="chart-label">normal</text>
          <rect x="610" y="28" width="12" height="12" fill="#111827"/><text x="628" y="39" class="chart-label">p${state.pct}</text>
        </svg>`;
    }

    function renderFocusCopy() {
      const row = scenario(DATA.regions.find(item => item.city === state.focus) || DATA.regions[0]);
      document.getElementById("focus-title").textContent = `${row.city} read`;
      document.getElementById("focus-copy").textContent =
        `${row.city} is at ${row.temp.toFixed(1)} deg C with a ${row.anomaly.toFixed(1)} deg C anomaly versus its local summer normal. Under the selected p${state.pct} trigger it is ${row.active ? "active" : "not active"}, with ${row.eventDegreeDays.toFixed(1)} heat degree days in the event window and ${money(row.payout)} of scenario payout. The main stress driver is ${driver(row).toLowerCase()}.`;
    }

    function renderForward() {
      const byYear = {};
      DATA.projection.forEach(row => {
        byYear[row.year] ||= [];
        byYear[row.year].push(row.extreme_days);
      });
      const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);
      const mean = years.map(year => byYear[year].reduce((a, b) => a + b, 0) / byYear[year].length);
      const low = years.map(year => quantile(byYear[year], 0.1));
      const high = years.map(year => quantile(byYear[year], 0.9));
      const activeRate = DATA.projection.filter(row => row.max_streak >= 3).length / DATA.projection.length;
      const payouts = DATA.projection.map(row => row.payout).sort((a, b) => a - b);
      document.getElementById("forward-copy").textContent =
        `For the leading region, the forward view uses ${DATA.projection_source}. The current p98 wording activates in ${(activeRate * 100).toFixed(0)}% of projected model summers. The p95 scenario payout is ${money(quantile(payouts, 0.95))}.`;
      const maxY = Math.ceil(Math.max(...high) + 3);
      const xFor = (year) => scale(year, years[0], years[years.length - 1], 74, 700);
      const yFor = (value) => scale(value, 0, maxY, 306, 84);
      const xs = years.map(xFor);
      const highPts = high.map(yFor);
      const lowPts = low.map(yFor);
      const meanPts = years.map((year, i) => [xFor(year), yFor(mean[i])]);
      const markers = meanPts.map((point, i) => `<circle cx="${point[0]}" cy="${point[1]}" r="4.5" fill="${colors.teal}"><title>${years[i]} | ${mean[i].toFixed(1)} days</title></circle>`).join("");
      document.getElementById("forward-chart").innerHTML = `
        <svg viewBox="0 0 760 380" role="img" aria-label="Forward heat stress chart">
          <text x="24" y="34" class="chart-title">${DATA.regions[0].city} projected days above today's trigger</text>
          <text x="24" y="58" class="chart-label">The band shows model spread, the line shows the model average.</text>
          <line x1="74" x2="700" y1="306" y2="306" class="axis"/><line x1="74" x2="74" y1="84" y2="306" class="axis"/>
          <text x="28" y="91" class="chart-label">${maxY} days</text><text x="38" y="309" class="chart-label">0</text>
          <path d="${areaPath(xs, highPts, lowPts)}" fill="rgba(14,165,233,0.18)" stroke="none"/>
          <polyline points="${polyline(meanPts)}" fill="none" stroke="${colors.teal}" stroke-width="3"/>
          ${markers}
          <text x="76" y="345" class="chart-label">${years[0]}</text><text x="660" y="345" class="chart-label">${years[years.length - 1]}</text>
        </svg>`;
    }

    function quantile(values, q) {
      const sorted = [...values].sort((a, b) => a - b);
      const pos = (sorted.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (sorted[base + 1] !== undefined) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
      return sorted[base] || 0;
    }

    function renderAll() {
      updateLabels();
      const rows = currentRows();
      updateMetrics(rows);
      renderQueue(rows);
      renderMap(rows);
      renderDrivers(rows);
      renderSensitivity();
      renderBasis(rows);
      renderTimeline();
      renderFocusCopy();
      renderForward();
    }

    document.getElementById("pct").addEventListener("change", event => { state.pct = event.target.value; renderAll(); });
    document.getElementById("streak").addEventListener("input", event => { state.streak = Number(event.target.value); renderAll(); });
    document.getElementById("notional").addEventListener("input", event => { state.notional = Number(event.target.value); renderAll(); });
    document.getElementById("rate").addEventListener("input", event => { state.rate = Number(event.target.value); renderAll(); });
    document.getElementById("cap").addEventListener("input", event => { state.cap = Number(event.target.value); renderAll(); });

    renderAll();
  </script>
</body>
</html>
'''


def main() -> None:
    payload = build_payload()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    assets = docs / "assets"
    assets.mkdir(exist_ok=True)
    (docs / "index.html").write_text(
        HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built docs/index.html at {payload['generated_at']}")


if __name__ == "__main__":
    main()
