from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heatwave_risk.climate import fallback_projection_from_baseline, summarize_projection  # noqa: E402
from heatwave_risk.data import build_weather_dataset, load_regions  # noqa: E402
from heatwave_risk.risk import DEFAULT_LINE_WEIGHTS, compute_risk_scores  # noqa: E402


MODULES = {
    "life_health_stress": "Life and health",
    "agriculture_stress": "Agriculture",
    "energy_stress": "Energy",
    "property_infra_stress": "Infrastructure",
    "business_interruption_stress": "Business interruption",
}

COUNTRIES = {
    "France",
    "Spain",
    "Portugal",
    "Italy",
    "Greece",
    "Switzerland",
    "Germany",
    "Austria",
    "Poland",
    "United Kingdom",
    "Ireland",
    "Belgium",
    "Netherlands",
    "Luxembourg",
    "Czechia",
    "Slovakia",
    "Slovenia",
    "Croatia",
    "Hungary",
    "Denmark",
    "Norway",
    "Sweden",
}

MAP = {
    "lon_min": -12.5,
    "lon_max": 25.8,
    "lat_min": 35.4,
    "lat_max": 55.8,
    "width": 900,
    "height": 560,
}


def clean_float(value: object, digits: int = 2) -> float:
    if pd.isna(value):
        return 0.0
    return round(float(value), digits)


def scenario_rows(scored: pd.DataFrame, selected_date: pd.Timestamp) -> list[dict[str, Any]]:
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

    rows: list[dict[str, Any]] = []
    for _, row in current.iterrows():
        region_id = str(row["region_id"])
        rows.append(
            {
                "region_id": region_id,
                "city": str(row["city"]),
                "country": str(row["country"]),
                "lat": clean_float(row["latitude"], 4),
                "lon": clean_float(row["longitude"], 4),
                "population_million": clean_float(row["population_million"], 2),
                "elderly_share": clean_float(row["elderly_share"], 3),
                "agriculture_exposure": clean_float(row["agriculture_exposure"], 3),
                "energy_exposure": clean_float(row["energy_exposure"], 3),
                "wildfire_exposure": clean_float(row["wildfire_exposure"], 3),
                "infrastructure_exposure": clean_float(row["infrastructure_exposure"], 3),
                "temp": clean_float(row["temperature_2m_max"], 1),
                "mean_temp": clean_float(row["temperature_2m_mean"], 1),
                "precipitation": clean_float(row.get("precipitation_sum", 0), 1),
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


def timeline_rows(scored: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for city, frame in scored.sort_values("date").groupby("city"):
        out[str(city)] = [
            {
                "date": str(row["date"].date()),
                "temp": clean_float(row["temperature_2m_max"], 1),
                "mean_temp": clean_float(row["temperature_2m_mean"], 1),
                "precipitation": clean_float(row.get("precipitation_sum", 0), 1),
                "normal": clean_float(row["clim_mean"], 1),
                "p95": clean_float(row["clim_p95"], 1),
                "p98": clean_float(row["clim_p98"], 1),
                "p99": clean_float(row["clim_p99"], 1),
            }
            for _, row in frame.iterrows()
        ]
    return out


def baseline_lookup(regions: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    lookup: dict[str, dict[str, dict[str, float]]] = {}
    for _, region in regions.iterrows():
        region_id = str(region["region_id"])
        cache_file = ROOT / "data" / "cache" / "open_meteo_baseline" / f"{region_id}_1991_2020.json"
        if not cache_file.exists():
            continue
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["daily"])
        frame["date"] = pd.to_datetime(frame.pop("time"))
        frame = frame[frame["date"].dt.year >= 2001].copy()
        frame["month_day"] = frame["date"].dt.strftime("%m-%d")
        grouped = frame.groupby("month_day")["temperature_2m_max"]
        stats = grouped.agg(
            min="min",
            p10=lambda values: values.quantile(0.10),
            p25=lambda values: values.quantile(0.25),
            mean="mean",
            p50=lambda values: values.quantile(0.50),
            p75=lambda values: values.quantile(0.75),
            p90=lambda values: values.quantile(0.90),
            p95=lambda values: values.quantile(0.95),
            p98=lambda values: values.quantile(0.98),
            p99=lambda values: values.quantile(0.99),
            max="max",
        )
        lookup[region_id] = {
            str(month_day): {
                "min": clean_float(row["min"], 2),
                "p10": clean_float(row["p10"], 2),
                "p25": clean_float(row["p25"], 2),
                "mean": clean_float(row["mean"], 2),
                "p50": clean_float(row["p50"], 2),
                "p75": clean_float(row["p75"], 2),
                "p90": clean_float(row["p90"], 2),
                "p95": clean_float(row["p95"], 2),
                "p98": clean_float(row["p98"], 2),
                "p99": clean_float(row["p99"], 2),
                "max": clean_float(row["max"], 2),
            }
            for month_day, row in stats.iterrows()
        }
    return lookup


def projection_payload(
    regions: pd.DataFrame,
    baselines: dict[str, dict[str, dict[str, float]]],
    selected_date: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    month_day = selected_date.strftime("%m-%d")
    for _, region in regions.iterrows():
        region_id = str(region["region_id"])
        daily_baseline = baselines.get(region_id, {})
        threshold = daily_baseline.get(month_day, {}).get("p98")
        if threshold is None and daily_baseline:
            threshold = float(np.median([item["p98"] for item in daily_baseline.values()]))
        if threshold is None:
            continue
        try:
            projection = fallback_projection_from_baseline(
                region_id=region_id,
                cache_dir=ROOT / "data" / "cache",
                end_year=2035,
            )
            annual = summarize_projection(projection, float(threshold))
        except Exception:
            continue
        out[region_id] = {
            "threshold": clean_float(threshold, 1),
            "source": "local trend stress test from the 1991 to 2020 archive cache",
            "rows": [
                {
                    "model": str(row["model"]),
                    "year": int(row["year"]),
                    "extreme_days": clean_float(row["local_extreme_days"], 1),
                    "max_streak": int(row["max_local_extreme_streak"]),
                    "local_hdd": clean_float(row["local_heat_degree_days"], 1),
                }
                for _, row in annual.iterrows()
            ],
        }
    return out


def _project(lon: float, lat: float) -> tuple[float, float]:
    x = (lon - MAP["lon_min"]) / (MAP["lon_max"] - MAP["lon_min"]) * MAP["width"]
    y = (MAP["lat_max"] - lat) / (MAP["lat_max"] - MAP["lat_min"]) * MAP["height"]
    return x, y


def _ring_to_path(ring: list[list[float]]) -> str | None:
    if not ring:
        return None
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    if max(lons) < MAP["lon_min"] or min(lons) > MAP["lon_max"]:
        return None
    if max(lats) < MAP["lat_min"] or min(lats) > MAP["lat_max"]:
        return None
    step = max(1, math.ceil(len(ring) / 160))
    points = ring[::step]
    if ring[-1] not in points:
        points.append(ring[-1])
    coords = []
    for lon, lat in points:
        x, y = _project(float(lon), float(lat))
        coords.append(f"{x:.1f},{y:.1f}")
    if len(coords) < 3:
        return None
    return "M" + " L".join(coords) + " Z"


def europe_map_markup() -> str:
    fallback = (
        '<path d="M98,162 C160,85 265,92 338,145 C410,92 535,107 624,175 '
        'C742,195 822,287 837,371 C742,450 588,464 472,421 C370,487 219,443 '
        '177,355 C102,345 62,246 98,162 Z" />'
    )
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    try:
        payload = requests.get(url, timeout=20).json()
    except Exception:
        return fallback

    paths: list[str] = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        name = props.get("ADMIN") or props.get("name")
        if name not in COUNTRIES:
            continue
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            polygons = [geom.get("coordinates", [])]
        elif geom.get("type") == "MultiPolygon":
            polygons = geom.get("coordinates", [])
        else:
            polygons = []
        for polygon in polygons:
            if not polygon:
                continue
            path = _ring_to_path(polygon[0])
            if path:
                paths.append(f'<path d="{path}" />')
    return "\n".join(paths) if len(paths) >= 8 else fallback


def write_preview_svg(rows: list[dict[str, Any]], selected_date: pd.Timestamp) -> None:
    leader = rows[0]
    bars = []
    y = 330
    for row in rows[:5]:
        width = int(row["score"] * 4.7)
        bars.append(
            f'<text x="84" y="{y}" class="city">{row["city"]}</text>'
            f'<rect x="255" y="{y - 18}" width="{width}" height="18" rx="4" fill="#0f766e"/>'
            f'<text x="{270 + width}" y="{y}" class="note">{row["score"]:.1f}</text>'
        )
        y += 43

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">
<style>
.bg {{ fill: #f3f6f8; }}
.panel {{ fill: #fff; stroke: #d8e1e8; stroke-width: 1; }}
.title {{ font: 700 50px Arial, sans-serif; fill: #102a43; }}
.copy {{ font: 400 22px Arial, sans-serif; fill: #425466; }}
.metric {{ font: 700 48px Arial, sans-serif; fill: #172033; }}
.note {{ font: 400 18px Arial, sans-serif; fill: #425466; }}
.city {{ font: 700 20px Arial, sans-serif; fill: #172033; }}
</style>
<rect class="bg" width="1200" height="675"/>
<rect class="panel" x="44" y="38" width="1112" height="598" rx="8"/>
<text x="80" y="112" class="title">Live Heat Stress Risk Monitor</text>
<text x="80" y="154" class="copy">A browser-ready heat stress, trigger and basis-risk workbench.</text>
<rect class="panel" x="80" y="195" width="245" height="98" rx="7"/>
<text x="102" y="232" class="note">Highest region</text>
<text x="102" y="275" class="metric">{leader["city"]}</text>
<rect class="panel" x="350" y="195" width="245" height="98" rx="7"/>
<text x="372" y="232" class="note">Risk score</text>
<text x="372" y="275" class="metric">{leader["score"]:.1f}</text>
<rect class="panel" x="620" y="195" width="245" height="98" rx="7"/>
<text x="642" y="232" class="note">Fallback data date</text>
<text x="642" y="275" class="metric">{selected_date.date()}</text>
<text x="80" y="364" class="copy">Current watchlist ranking</text>
{''.join(bars)}
</svg>'''
    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "preview.svg").write_text(svg, encoding="utf-8")


def build_payload() -> dict[str, Any]:
    regions = load_regions(ROOT / "data" / "regions.csv")
    daily, _regions = build_weather_dataset(ROOT / "data" / "regions.csv", ROOT / "data" / "cache", refresh=False)
    scored = compute_risk_scores(daily, DEFAULT_LINE_WEIGHTS)
    available_dates = sorted(scored["date"].dt.date.unique())
    selected_date_value = date.today()
    if selected_date_value not in available_dates:
        selected_date_value = available_dates[-1]
    selected_date = pd.Timestamp(selected_date_value)
    rows = scenario_rows(scored, selected_date)
    baselines = baseline_lookup(regions)
    projections = projection_payload(regions, baselines, selected_date)
    write_preview_svg(rows, selected_date)
    region_meta = [
        {
            key: clean_float(row[key], 4)
            if key in {"latitude", "longitude"}
            else clean_float(row[key], 3)
            if key not in {"region_id", "city", "country"}
            else str(row[key])
            for key in [
                "region_id",
                "city",
                "country",
                "latitude",
                "longitude",
                "population_million",
                "elderly_share",
                "agriculture_exposure",
                "energy_exposure",
                "wildfire_exposure",
                "infrastructure_exposure",
            ]
        }
        for _, row in regions.iterrows()
    ]
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "analysis_date": str(selected_date.date()),
        "regions": rows,
        "region_meta": region_meta,
        "timeline": timeline_rows(scored),
        "baselines": baselines,
        "projections": projections,
        "map_markup": europe_map_markup(),
        "repo_url": "https://github.com/RomanMski/europe-heatwave-reinsurance-risk-monitor",
    }


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Heat Stress Risk Monitor</title>
  <meta name="description" content="Browser-ready heat stress, trigger and basis-risk monitor by Roman Mirosenski.">
  <style>
    :root {
      --ink: #132238;
      --muted: #526273;
      --line: #d7e0e7;
      --paper: #ffffff;
      --soft: #f3f6f8;
      --teal: #0f766e;
      --green: #65a30d;
      --blue: #2563eb;
      --amber: #f59e0b;
      --red: #dc2626;
      --purple: #7c3aed;
      --slate: #64748b;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      color: var(--ink);
      background: var(--soft);
    }
    a { color: var(--teal); font-weight: 700; text-decoration-thickness: 1px; }
    button, select, input { font: inherit; }
    main { max-width: 1320px; margin: 0 auto; padding: 22px 18px 54px; }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 18px;
      font-size: 0.95rem;
      color: var(--muted);
    }
    .topbar nav { display: flex; gap: 14px; flex-wrap: wrap; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-weight: 700;
      white-space: nowrap;
    }
    .dot { width: 9px; height: 9px; border-radius: 999px; background: var(--amber); display: inline-block; }
    .status.live .dot { background: var(--teal); }
    .status.cached .dot { background: var(--amber); }
    .hero {
      display: grid;
      grid-template-columns: minmax(370px, 0.72fr) minmax(620px, 1.28fr);
      gap: 18px;
      align-items: start;
    }
    .hero > *, .intro, .workbench, .panel, .chart, .table-panel, .note { min-width: 0; }
    .intro, .panel, .metric, .control, .chart, .note, .table-panel {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .intro { padding: 26px 28px; display: grid; gap: 20px; }
    h1 {
      margin: 0 0 15px;
      font-size: clamp(2.2rem, 3.8vw, 4.25rem);
      line-height: 1.02;
      letter-spacing: 0;
      color: #102a43;
    }
    h2 { margin: 0 0 12px; font-size: 1.45rem; color: #102a43; }
    p { margin: 0 0 13px; color: var(--muted); line-height: 1.58; font-size: 1rem; }
    .intro p { font-size: 1.02rem; }
    .source-note { margin: 0; color: var(--ink); font-weight: 700; }
    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .metric { padding: 14px 16px; min-height: 96px; }
    .metric span { display: block; color: var(--muted); font-size: 0.86rem; margin-bottom: 9px; }
    .metric strong { display: block; font-size: clamp(1.8rem, 3vw, 2.65rem); line-height: 1; }
    .metric small { display: block; color: var(--teal); margin-top: 8px; font-weight: 700; font-size: 0.88rem; }
    .workbench { display: grid; gap: 12px; }
    .controls {
      display: grid;
      grid-template-columns: 1.1fr 1fr 1.1fr 1fr 1fr 1.2fr auto;
      gap: 10px;
    }
    .control { padding: 12px; min-height: 92px; }
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
    .control b { display: block; margin-top: 5px; font-size: 1.08rem; }
    .refresh {
      min-height: 92px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #102a43;
      color: #fff;
      padding: 0 16px;
      font-weight: 800;
      cursor: pointer;
    }
    .refresh:hover { background: #193b5d; }
    .map-panel { min-height: 560px; padding: 12px; position: relative; overflow: hidden; }
    .section { margin-top: 18px; }
    .two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    .three { display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 18px; align-items: start; }
    .chart { min-height: 390px; padding: 14px; overflow: hidden; }
    .chart svg, .map-panel svg { display: block; width: 100%; max-width: 100%; height: 100%; min-height: 360px; }
    .map-panel svg { min-height: 535px; }
    .map-land path { fill: #f7fafc; stroke: #c9d5df; stroke-width: 1.1; vector-effect: non-scaling-stroke; }
    .gridline { stroke: #e7eef4; stroke-width: 1; }
    .axis { stroke: #cbd5e1; stroke-width: 1; }
    .chart-title { font: 700 20px Inter, Arial, sans-serif; fill: #102a43; }
    .chart-label { font: 400 13px Inter, Arial, sans-serif; fill: #526273; }
    .chart-small { font: 700 12px Inter, Arial, sans-serif; fill: #132238; }
    .chart-tiny { font: 400 11px Inter, Arial, sans-serif; fill: #526273; }
    .city-dot { cursor: pointer; }
    .city-dot text { pointer-events: none; }
    .table-panel { overflow: hidden; }
    .table-head { padding: 20px 20px 0; }
    .queue { width: 100%; border-collapse: collapse; font-size: 0.94rem; }
    .queue th, .queue td { padding: 12px 10px; border-bottom: 1px solid #edf2f7; text-align: left; }
    .queue th { color: #102a43; }
    .region-row { cursor: pointer; }
    .region-row:hover td { background: #f7fafc; }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 0.8rem;
      font-weight: 800;
      background: #eef7f5;
      color: #0f766e;
    }
    .badge.off { background: #f1f5f9; color: #475569; }
    .note { padding: 22px; }
    .note p:last-child { margin-bottom: 0; }
    .context-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 18px;
    }
    .context-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #f8fafc;
    }
    .context-item span {
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      margin-bottom: 7px;
    }
    .context-item strong {
      display: block;
      color: var(--ink);
      font-size: 1.34rem;
      line-height: 1.05;
    }
    .context-item small {
      display: block;
      color: var(--teal);
      font-weight: 700;
      margin-top: 7px;
      line-height: 1.3;
    }
    .footer { color: var(--muted); font-size: 0.95rem; margin-top: 20px; }
    @media (max-width: 1120px) {
      .hero, .two, .three { grid-template-columns: minmax(0, 1fr); }
      .controls { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .refresh { min-height: 64px; }
    }
    @media (max-width: 680px) {
      body { overflow-x: hidden; }
      main { width: 100%; max-width: 100%; padding: 14px; overflow-x: hidden; }
      .hero, .intro, .workbench, .panel, .control, .chart, .table-panel, .note { width: 100%; max-width: 100%; min-width: 0; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .topbar > div { max-width: 100%; overflow-wrap: anywhere; }
      .intro { padding: 22px; }
      h1 { font-size: clamp(2.05rem, 10.5vw, 2.8rem); }
      p { font-size: 0.98rem; }
      .intro p, .source-note { overflow-wrap: anywhere; }
      .metrics, .controls { grid-template-columns: minmax(0, 1fr); }
      .context-grid { grid-template-columns: minmax(0, 1fr); }
      .map-panel { min-height: 420px; }
      .map-panel svg { min-height: 395px; }
      .chart { min-height: 340px; }
      .chart svg { min-height: 315px; }
      .queue { font-size: 0.86rem; }
      .queue th:nth-child(3), .queue td:nth-child(3) { display: none; }
      .queue th:nth-child(5), .queue td:nth-child(5) { display: none; }
    }
    @media (max-width: 520px) {
      main { max-width: 390px; margin: 0; }
    }
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>Roman Mirosenski | live heat risk monitor</div>
      <nav>
        <a href="#contract">contract</a>
        <a href="#drilldown">city read</a>
        <a href="#forward">forward stress</a>
        <a id="repo-link" href="#">code</a>
      </nav>
      <div id="data-status" class="status cached"><span class="dot"></span><span>opening cached run</span></div>
    </div>

    <section class="hero">
      <div class="intro">
        <div>
          <h1>Live Heat Stress Risk Monitor</h1>
          <p>This is a small browser workbench for a heat event. It opens from GitHub Pages, pulls the latest Open-Meteo forecast if the browser allows it, and keeps a cached run inside the page so it never opens blank.</p>
          <p>The point is simple: compare each city with its own climate history, see where stress is building, then test whether a parametric trigger would actually respond.</p>
        </div>
        <div class="metrics">
          <div class="metric"><span>first region to open</span><strong id="metric-region">-</strong><small id="metric-driver">-</small></div>
          <div class="metric"><span>active triggers</span><strong id="metric-triggers">-</strong><small>current wording</small></div>
          <div class="metric"><span>scenario payout</span><strong id="metric-payout">-</strong><small>selected notional</small></div>
          <div class="metric"><span>basis-risk flags</span><strong id="metric-basis">-</strong><small>stress missed by wording</small></div>
        </div>
        <p class="source-note" id="source-note">Data date -</p>
      </div>

      <div class="workbench">
        <div class="controls">
          <div class="control"><label>trigger threshold</label><select id="pct"><option value="95">p95</option><option value="98" selected>p98</option><option value="99">p99</option></select></div>
          <div class="control"><label>required streak</label><input id="streak" type="range" min="2" max="6" step="1" value="3"><b id="streak-label">3 days</b></div>
          <div class="control"><label>notional per city</label><input id="notional" type="range" min="5" max="100" step="5" value="25"><b id="notional-label">CHF 25m</b></div>
          <div class="control"><label>payout rate</label><input id="rate" type="range" min="0.02" max="0.20" step="0.01" value="0.08"><b id="rate-label">0.08</b></div>
          <div class="control"><label>cap</label><input id="cap" type="range" min="0.05" max="0.50" step="0.05" value="0.25"><b id="cap-label">25%</b></div>
          <div class="control"><label>map color</label><select id="mapMetric"><option value="score" selected>risk</option><option value="temp">temperature</option><option value="payout">payout</option><option value="energy">energy</option></select></div>
          <button id="refresh" class="refresh" type="button">Refresh live</button>
        </div>
        <div class="panel map-panel"><div id="map"></div></div>
      </div>
    </section>

    <section class="section three">
      <div class="table-panel">
        <div class="table-head">
          <h2>Watchlist</h2>
          <p>Click a row or a city on the map. I would use this first to decide where to spend attention before reading a longer report.</p>
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

    <section class="section two" id="drilldown">
      <div class="chart"><div id="timeline"></div></div>
      <div class="note">
        <h2 id="focus-title">City read</h2>
        <p id="focus-copy">-</p>
        <p id="focus-next">-</p>
        <div id="climate-context" class="context-grid"></div>
      </div>
    </section>

    <section class="section two" id="forward">
      <div class="note">
        <h2>Forward stress</h2>
        <p id="forward-copy">-</p>
        <p>This part is deliberately not sold as a loss forecast. It is a wording stress test: if summers keep getting hotter, does the trigger still behave in a way that makes sense?</p>
      </div>
      <div class="chart"><div id="forward-chart"></div></div>
    </section>

    <p class="footer">The full Python app is still in the repository. This page is the recruiter-friendly version: open it, move the controls, see the logic without installing anything.</p>
  </main>

  <script id="payload" type="application/json">__DATA__</script>
  <script>
    const CACHE = JSON.parse(document.getElementById("payload").textContent);
    const COLORS = { red: "#dc2626", teal: "#0f766e", green: "#65a30d", amber: "#f59e0b", blue: "#2563eb", purple: "#7c3aed", slate: "#64748b" };
    const MAP = { lonMin: -12.5, lonMax: 25.8, latMin: 35.4, latMax: 55.8, width: 900, height: 560 };
    const MODULES = ["Life and health", "Agriculture", "Energy", "Infrastructure", "Business interruption"];
    const MODULE_COLORS = [COLORS.blue, COLORS.green, COLORS.amber, COLORS.slate, COLORS.purple];

    let view = {
      regions: CACHE.regions,
      timeline: CACHE.timeline,
      analysisDate: CACHE.analysis_date,
      source: "cached Open-Meteo run",
      live: false
    };

    const state = {
      pct: "98",
      streak: 3,
      notional: 25,
      rate: 0.08,
      cap: 0.25,
      mapMetric: "score",
      focus: CACHE.regions[0]?.city || ""
    };

    document.getElementById("repo-link").href = CACHE.repo_url;

    function clamp(value, min = 0, max = 1) { return Math.max(min, Math.min(max, value)); }
    function bounded(value, upper) { return clamp((Number(value) || 0) / upper, 0, 1); }
    function round(value, digits = 1) { const p = 10 ** digits; return Math.round((Number(value) || 0) * p) / p; }
    function money(value) { return value >= 10 ? `CHF ${Math.round(value)}m` : `CHF ${value.toFixed(1)}m`; }
    function isoShift(dateText, days) {
      const dt = new Date(`${dateText}T12:00:00Z`);
      dt.setUTCDate(dt.getUTCDate() + days);
      return dt.toISOString().slice(0, 10);
    }
    function monthDay(dateText) { return dateText.slice(5); }
    function quantile(values, q) {
      const sorted = values.filter(v => Number.isFinite(v)).sort((a, b) => a - b);
      if (!sorted.length) return 0;
      const pos = (sorted.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
    }
    function avg(values) {
      const clean = values.filter(v => Number.isFinite(v));
      return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
    }
    function signed(value) {
      return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
    }
    function focusRegionMeta() {
      return CACHE.region_meta.find(row => row.city === state.focus) || CACHE.region_meta[0];
    }
    function scale(value, min, max, a, b) {
      if (max === min) return (a + b) / 2;
      return a + ((value - min) / (max - min)) * (b - a);
    }
    function project(lon, lat) {
      return {
        x: scale(lon, MAP.lonMin, MAP.lonMax, 0, MAP.width),
        y: scale(lat, MAP.latMin, MAP.latMax, MAP.height, 0)
      };
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
      return view.regions.map(row => scenario(row)).sort((a, b) => b.score - a.score);
    }
    function setStatus(text, mode) {
      const el = document.getElementById("data-status");
      el.className = `status ${mode || "cached"}`;
      el.querySelector("span:last-child").textContent = text;
    }
    function updateLabels() {
      document.getElementById("streak-label").textContent = `${state.streak} days`;
      document.getElementById("notional-label").textContent = `CHF ${state.notional}m`;
      document.getElementById("rate-label").textContent = state.rate.toFixed(2);
      document.getElementById("cap-label").textContent = `${Math.round(state.cap * 100)}%`;
      document.getElementById("source-note").textContent = `${view.live ? "Live" : "Cached"} data date ${view.analysisDate}. Source ${view.source}.`;
    }
    function updateMetrics(rows) {
      const leader = rows[0];
      const active = rows.filter(row => row.active).length;
      const total = rows.reduce((sum, row) => sum + row.payout, 0);
      const median = quantile(rows.map(row => row.score), 0.5);
      const basis = rows.filter(row => row.score >= median && !row.active).length;
      document.getElementById("metric-region").textContent = leader.city;
      document.getElementById("metric-driver").textContent = `${leader.score.toFixed(1)} score, ${driver(leader).toLowerCase()}`;
      document.getElementById("metric-triggers").textContent = active;
      document.getElementById("metric-payout").textContent = money(total);
      document.getElementById("metric-basis").textContent = basis;
      if (!rows.some(row => row.city === state.focus)) state.focus = leader.city;
    }
    function valueForMap(row) {
      if (state.mapMetric === "temp") return row.temp;
      if (state.mapMetric === "payout") return row.payout;
      if (state.mapMetric === "energy") return row.stress["Energy"];
      return row.score;
    }
    function colorFor(row) {
      const value = valueForMap(row);
      if (state.mapMetric === "temp") {
        if (value >= 38) return "#991b1b";
        if (value >= 34) return COLORS.red;
        if (value >= 30) return COLORS.amber;
        if (value >= 25) return COLORS.teal;
        return COLORS.slate;
      }
      if (state.mapMetric === "payout") {
        if (value >= 8) return "#991b1b";
        if (value >= 4) return COLORS.red;
        if (value > 0) return COLORS.amber;
        return COLORS.slate;
      }
      if (value >= 60) return "#991b1b";
      if (value >= 45) return COLORS.red;
      if (value >= 25) return COLORS.amber;
      if (value >= 12) return COLORS.teal;
      return COLORS.slate;
    }
    function renderMap(rows) {
      const maxMetric = Math.max(...rows.map(valueForMap), 1);
      const dots = rows.map(row => {
        const { x, y } = project(row.lon, row.lat);
        const value = valueForMap(row);
        const radius = 7 + Math.sqrt(Math.max(value, 0) / maxMetric) * 18;
        const selected = row.city === state.focus;
        const labelLeft = x > 760;
        const labelX = labelLeft ? x - radius - 7 : x + radius + 7;
        const labelAnchor = labelLeft ? "end" : "start";
        return `<g class="city-dot" data-city="${row.city}">
          <circle cx="${x}" cy="${y}" r="${radius + (selected ? 4 : 0)}" fill="${selected ? "#102a43" : "#fff"}" opacity="${selected ? 0.95 : 0.65}"/>
          <circle cx="${x}" cy="${y}" r="${radius}" fill="${colorFor(row)}" stroke="#fff" stroke-width="2"/>
          <text x="${labelX}" y="${y + 4}" text-anchor="${labelAnchor}" class="chart-small">${row.city}</text>
          <title>${row.city}, ${row.country} | score ${row.score.toFixed(1)} | temp ${row.temp.toFixed(1)} deg C | ${row.activeText} | ${money(row.payout)}</title>
        </g>`;
      }).join("");
      const legend = state.mapMetric === "temp" ? "cooler to hotter daily max" : state.mapMetric === "payout" ? "lower to higher scenario payout" : state.mapMetric === "energy" ? "lower to higher energy stress" : "lower to higher composite risk";
      document.getElementById("map").innerHTML = `
        <svg viewBox="0 0 940 610" role="img" aria-label="Interactive heat stress map">
          <rect x="0" y="0" width="940" height="610" fill="#ffffff"/>
          <text x="24" y="34" class="chart-title">Live watch map</text>
          <text x="24" y="58" class="chart-label">Color follows ${legend}. Click a city to move the drilldown.</text>
          <defs><clipPath id="map-clip"><rect x="0" y="0" width="900" height="560" rx="8"/></clipPath></defs>
          <g transform="translate(20,78)">
            <rect x="0" y="0" width="900" height="560" rx="8" fill="#eaf4f8" stroke="#d8e1e8"/>
            <g clip-path="url(#map-clip)">
              <line x1="0" x2="900" y1="280" y2="280" class="gridline"/>
              <line x1="450" x2="450" y1="0" y2="560" class="gridline"/>
              <g class="map-land">${CACHE.map_markup}</g>
              ${dots}
            </g>
            <text x="18" y="532" class="chart-label">${legend}</text>
            <rect x="222" y="520" width="54" height="12" fill="#64748b"/><rect x="276" y="520" width="54" height="12" fill="#0f766e"/><rect x="330" y="520" width="54" height="12" fill="#f59e0b"/><rect x="384" y="520" width="54" height="12" fill="#dc2626"/><rect x="438" y="520" width="54" height="12" fill="#991b1b"/>
          </g>
        </svg>`;
      document.querySelectorAll(".city-dot").forEach(el => el.addEventListener("click", () => {
        state.focus = el.dataset.city;
        renderAll();
      }));
    }
    function renderQueue(rows) {
      const body = document.getElementById("queue-body");
      body.innerHTML = rows.slice(0, 10).map(row => `
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
        renderAll();
      }));
    }
    function polyline(points) { return points.map(point => `${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" "); }
    function areaPath(xs, high, low) {
      const top = xs.map((x, i) => `${x.toFixed(1)},${high[i].toFixed(1)}`).join(" L ");
      const bottom = [...xs].reverse().map((x, idx) => {
        const i = xs.length - 1 - idx;
        return `${x.toFixed(1)},${low[i].toFixed(1)}`;
      }).join(" L ");
      return `M ${top} L ${bottom} Z`;
    }
    function cellColor(value, max) {
      const t = max <= 0 ? 0 : value / max;
      if (t > 0.75) return "#b91c1c";
      if (t > 0.50) return "#f97316";
      if (t > 0.25) return "#facc15";
      if (t > 0.05) return "#c7f9e8";
      return "#f8fafc";
    }
    function renderDrivers(rows) {
      const top = rows.slice(0, 8).reverse();
      const maxTotal = Math.max(...top.map(row => MODULES.reduce((sum, name) => sum + row.stress[name], 0)), 1);
      const bars = top.map((row, idx) => {
        let x = 152;
        const y = 320 - idx * 34;
        const parts = MODULES.map((name, moduleIdx) => {
          const w = scale(row.stress[name], 0, maxTotal, 0, 520);
          const rect = `<rect x="${x}" y="${y - 17}" width="${w}" height="20" rx="3" fill="${MODULE_COLORS[moduleIdx]}"><title>${row.city} | ${name} ${row.stress[name].toFixed(1)}</title></rect>`;
          x += w;
          return rect;
        }).join("");
        return `<text x="24" y="${y}" class="chart-label">${row.city}</text>${parts}<text x="${Math.min(x + 8, 724)}" y="${y}" class="chart-small">${row.score.toFixed(1)}</text>`;
      }).join("");
      const legend = MODULES.map((name, idx) => `<rect x="${24 + idx * 136}" y="364" width="12" height="12" fill="${MODULE_COLORS[idx]}"/><text x="${42 + idx * 136}" y="375" class="chart-tiny">${name}</text>`).join("");
      document.getElementById("drivers").innerHTML = `
        <svg viewBox="0 0 760 410" role="img" aria-label="Stress driver bars">
          <text x="24" y="34" class="chart-title">What is actually driving the score</text>
          <text x="24" y="58" class="chart-label">The score is split into insurance and operations pressure points, not only temperature.</text>
          ${bars}
          ${legend}
        </svg>`;
    }
    function renderSensitivity() {
      const percentiles = ["95", "98", "99"];
      const streaks = [2, 3, 4, 5, 6];
      const matrix = percentiles.map(pct => streaks.map(streak => view.regions.reduce((sum, row) => sum + scenario(row, { pct, streak }).payout, 0)));
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
          <text x="24" y="34" class="chart-title">Contract sensitivity</text>
          <text x="24" y="58" class="chart-label">Click a cell to apply that wording to the whole dashboard.</text>
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
      const scores = rows.map(row => row.score);
      const minScore = Math.max(0, Math.floor(Math.min(...scores) - 5));
      const maxScore = Math.ceil(Math.max(...scores) + 5);
      const scoreCut = quantile(scores, 0.5);
      const xFor = score => scale(score, minScore, maxScore, 92, 690);
      const yFor = payout => scale(payout, 0, maxPayout, 306, 92);
      const cutX = xFor(scoreCut);
      const xLabels = [minScore, Math.round(scoreCut), maxScore].map(score =>
        `<text x="${xFor(score)}" y="333" text-anchor="middle" class="chart-label">${score}</text>`
      ).join("");
      const dots = rows.map(row => {
        const x = xFor(row.score);
        const y = yFor(row.payout);
        const radius = Math.max(6, 5 + row.eventDegreeDays * 0.38);
        return `<g data-city="${row.city}" class="basis-dot" style="cursor:pointer">
          <circle cx="${x}" cy="${y}" r="${radius}" fill="${row.active ? COLORS.red : COLORS.slate}" opacity="0.86" stroke="#fff" stroke-width="1.5"/>
          <title>${row.city} | score ${row.score.toFixed(1)} | ${money(row.payout)} | ${row.activeText}</title>
        </g>`;
      }).join("");
      const activeLabels = rows
        .filter(row => row.active)
        .sort((a, b) => b.score - a.score)
        .map(row => `<text x="${xFor(row.score)}" y="78" text-anchor="middle" class="chart-small">${row.city}</text>`)
        .join("");
      const missed = rows.filter(row => !row.active).sort((a, b) => b.score - a.score)[0];
      const missedLabel = missed
        ? `<line x1="${xFor(missed.score)}" x2="${xFor(missed.score) + 20}" y1="${yFor(missed.payout)}" y2="${yFor(missed.payout) - 18}" stroke="#94a3b8"/>
           <text x="${xFor(missed.score) + 24}" y="${yFor(missed.payout) - 21}" class="chart-small">${missed.city}</text>`
        : "";
      document.getElementById("basis").innerHTML = `
        <svg viewBox="0 0 760 360" role="img" aria-label="Basis risk scatter">
          <text x="24" y="34" class="chart-title">Basis risk check</text>
          <text x="24" y="58" class="chart-label">Red dots paid. Grey dots in the watch area show stress the wording does not catch.</text>
          <rect x="${cutX}" y="224" width="${690 - cutX}" height="82" fill="#fff7ed" stroke="none"/>
          <rect x="${cutX}" y="92" width="${690 - cutX}" height="132" fill="rgba(239,68,68,0.06)" stroke="none"/>
          <line x1="${cutX}" x2="${cutX}" y1="92" y2="306" stroke="#94a3b8" stroke-dasharray="5 5"/>
          <line x1="92" x2="690" y1="306" y2="306" class="axis"/><line x1="92" x2="92" y1="92" y2="306" class="axis"/>
          <text x="584" y="248" class="chart-label">watch wording</text>
          <text x="584" y="112" class="chart-label">triggered cover</text>
          <text x="594" y="350" class="chart-label">risk score</text><text x="24" y="84" class="chart-label">payout</text>
          <text x="50" y="310" class="chart-label">0</text><text x="28" y="106" class="chart-label">${money(maxPayout)}</text>
          ${xLabels}
          ${dots}
          ${activeLabels}
          ${missedLabel}
          <circle cx="540" cy="58" r="6" fill="${COLORS.red}"/><text x="552" y="63" class="chart-label">payout</text>
          <circle cx="620" cy="58" r="6" fill="${COLORS.slate}"/><text x="632" y="63" class="chart-label">no payout</text>
        </svg>`;
      document.querySelectorAll(".basis-dot").forEach(el => el.addEventListener("click", () => {
        state.focus = el.dataset.city;
        renderAll();
      }));
    }
    function renderTimeline() {
      const rows = view.timeline[state.focus] || view.timeline[view.regions[0].city] || [];
      if (!rows.length) return;
      const meta = focusRegionMeta();
      const baselines = rows.map(row => baselineFor(meta.region_id, row.date));
      const values = rows.flatMap((row, i) => [row.temp, baselines[i].p10, baselines[i].p90, baselines[i][`p${state.pct}`]]);
      const minY = Math.floor(Math.min(...values) - 2);
      const maxY = Math.ceil(Math.max(...values) + 2);
      const xFor = idx => scale(idx, 0, Math.max(rows.length - 1, 1), 70, 700);
      const yFor = value => scale(value, minY, maxY, 304, 84);
      const tempPts = rows.map((row, i) => [xFor(i), yFor(row.temp)]);
      const normalPts = rows.map((row, i) => [xFor(i), yFor(baselines[i].mean)]);
      const lowPts = rows.map((row, i) => yFor(baselines[i].p10));
      const highPts = rows.map((row, i) => yFor(baselines[i].p90));
      const xs = rows.map((row, i) => xFor(i));
      const thresholdPts = rows.map((row, i) => [xFor(i), yFor(baselines[i][`p${state.pct}`])]);
      const circles = tempPts.map((point, i) => `<circle cx="${point[0]}" cy="${point[1]}" r="4" fill="${COLORS.red}"><title>${rows[i].date} | ${rows[i].temp} deg C</title></circle>`).join("");
      document.getElementById("timeline").innerHTML = `
        <svg viewBox="0 0 760 380" role="img" aria-label="City heat timeline">
          <text x="24" y="34" class="chart-title">${state.focus}: current weather vs 20-year range</text>
          <text x="24" y="58" class="chart-label">Red line is current weather. Blue band is the 2001 to 2020 p10-p90 range for the same calendar days.</text>
          <line x1="70" x2="700" y1="304" y2="304" class="axis"/><line x1="70" x2="70" y1="84" y2="304" class="axis"/>
          <text x="18" y="90" class="chart-label">${maxY} deg C</text><text x="18" y="306" class="chart-label">${minY} deg C</text>
          <path d="${areaPath(xs, highPts, lowPts)}" fill="rgba(37,99,235,0.12)" stroke="none"/>
          <polyline points="${polyline(normalPts)}" fill="none" stroke="${COLORS.slate}" stroke-width="2" stroke-dasharray="5 5"/>
          <polyline points="${polyline(thresholdPts)}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="8 5"/>
          <polyline points="${polyline(tempPts)}" fill="none" stroke="${COLORS.red}" stroke-width="3"/>
          ${circles}
          <text x="72" y="345" class="chart-label">${rows[0].date}</text><text x="610" y="345" class="chart-label">${rows[rows.length - 1].date}</text>
          <rect x="372" y="28" width="12" height="12" fill="rgba(37,99,235,0.22)"/><text x="390" y="39" class="chart-label">20y p10-p90</text>
          <rect x="492" y="28" width="12" height="12" fill="${COLORS.red}"/><text x="510" y="39" class="chart-label">current</text>
          <rect x="582" y="28" width="12" height="12" fill="#111827"/><text x="600" y="39" class="chart-label">p${state.pct}</text>
        </svg>`;
    }
    function renderFocusCopy() {
      const row = scenario(view.regions.find(item => item.city === state.focus) || view.regions[0]);
      const missed = !row.active && row.score >= quantile(view.regions.map(item => item.score), 0.5);
      document.getElementById("focus-title").textContent = `${row.city} read`;
      document.getElementById("focus-copy").textContent =
        `${row.city} is at ${row.temp.toFixed(1)} deg C, ${row.anomaly.toFixed(1)} deg C above its local summer normal. Under the selected p${state.pct} trigger it is ${row.active ? "active" : "not active"}, with ${row.eventDegreeDays.toFixed(1)} heat degree days in the event window and ${money(row.payout)} of scenario payout.`;
      document.getElementById("focus-next").textContent = missed
        ? `This is the uncomfortable case: meaningful stress, but no trigger response. I would check attachment points, waiting periods and whether the chosen weather station matches the exposure.`
        : `Main driver right now: ${driver(row).toLowerCase()}. I would compare that with actual exposure data before treating the score as more than a triage signal.`;
    }
    function renderClimateContext() {
      const meta = focusRegionMeta();
      const rows = view.timeline[state.focus] || view.timeline[view.regions[0].city] || [];
      if (!rows.length || !meta) return;
      const pastRows = rows.filter(row => row.date <= view.analysisDate);
      const windowRows = (pastRows.length ? pastRows : rows).slice(-10);
      const baselineRows = windowRows.map(row => baselineFor(meta.region_id, row.date));
      const currentAvg = avg(windowRows.map(row => row.temp));
      const currentLow = Math.min(...windowRows.map(row => row.temp));
      const currentHigh = Math.max(...windowRows.map(row => row.temp));
      const histAvg = avg(baselineRows.map(row => row.mean));
      const histLow = avg(baselineRows.map(row => row.p10));
      const histHigh = avg(baselineRows.map(row => row.p90));
      const todayRow = windowRows[windowRows.length - 1];
      const todayBaseline = baselineFor(meta.region_id, todayRow.date);
      const triggerGap = todayRow.temp - todayBaseline[`p${state.pct}`];
      document.getElementById("climate-context").innerHTML = `
        <div class="context-item"><span>10-day current avg</span><strong>${currentAvg.toFixed(1)} deg C</strong><small>${signed(currentAvg - histAvg)} deg C vs 2001-2020 average</small></div>
        <div class="context-item"><span>2001-2020 typical range</span><strong>${histLow.toFixed(1)}-${histHigh.toFixed(1)} deg C</strong><small>p10-p90 for the same calendar days</small></div>
        <div class="context-item"><span>current 10-day range</span><strong>${currentLow.toFixed(1)}-${currentHigh.toFixed(1)} deg C</strong><small>actual observed/forecast window</small></div>
        <div class="context-item"><span>trigger distance today</span><strong>${signed(triggerGap)} deg C</strong><small>against local p${state.pct}</small></div>
      `;
    }
    function renderForward() {
      const focus = view.regions.find(row => row.city === state.focus) || view.regions[0];
      const projection = CACHE.projections[focus.region_id] || CACHE.projections[view.regions[0].region_id];
      if (!projection) return;
      const byYear = {};
      projection.rows.forEach(row => {
        byYear[row.year] ||= [];
        byYear[row.year].push(row);
      });
      const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);
      const yearStats = years.map(year => {
        const rows = byYear[year];
        const extremeDays = rows.map(row => row.extreme_days);
        const payouts = rows.map(row => row.max_streak >= state.streak ? Math.min(row.local_hdd * state.rate * state.notional, state.cap * state.notional) : 0);
        return {
          year,
          mean: avg(extremeDays),
          low: quantile(extremeDays, 0.1),
          high: quantile(extremeDays, 0.9),
          activeShare: rows.filter(row => row.max_streak >= state.streak).length / rows.length,
          payout95: quantile(payouts, 0.95)
        };
      });
      const payouts = projection.rows.map(row => row.max_streak >= state.streak ? Math.min(row.local_hdd * state.rate * state.notional, state.cap * state.notional) : 0);
      const activeRate = projection.rows.filter(row => row.max_streak >= state.streak).length / projection.rows.length;
      document.getElementById("forward-copy").textContent =
        `${focus.city} uses a local p98 threshold of ${projection.threshold.toFixed(1)} deg C in the forward stress panel. In these local trend scenarios, the current streak wording activates in ${(activeRate * 100).toFixed(0)}% of model summers. The high-end scenario payout is ${money(quantile(payouts, 0.95))}.`;
      const maxY = Math.ceil(Math.max(...yearStats.map(row => row.high)) + 3);
      const maxPayout = Math.max(...yearStats.map(row => row.payout95), 1);
      const xFor = index => scale(index, 0, Math.max(yearStats.length - 1, 1), 86, 690);
      const yFor = value => scale(value, 0, maxY, 306, 84);
      const bars = yearStats.map((row, i) => {
        const x = xFor(i);
        const barWidth = 34;
        const yMean = yFor(row.mean);
        const yLow = yFor(row.low);
        const yHigh = yFor(row.high);
        const fill = row.activeShare >= 0.65 ? COLORS.red : row.activeShare >= 0.35 ? COLORS.amber : COLORS.teal;
        const payoutHeight = scale(row.payout95, 0, maxPayout, 0, 54);
        return `
          <line x1="${x}" x2="${x}" y1="${yHigh}" y2="${yLow}" stroke="#475569" stroke-width="2"/>
          <rect x="${x - barWidth / 2}" y="${yMean}" width="${barWidth}" height="${306 - yMean}" rx="4" fill="${fill}" opacity="0.82"/>
          <circle cx="${x}" cy="${yMean}" r="4.5" fill="#102a43"><title>${row.year} | avg ${row.mean.toFixed(1)} days | p10-p90 ${row.low.toFixed(1)}-${row.high.toFixed(1)} | ${money(row.payout95)}</title></circle>
          <rect x="${x - barWidth / 2}" y="${330 - payoutHeight}" width="${barWidth}" height="${payoutHeight}" rx="3" fill="${COLORS.purple}" opacity="0.55"/>
          <text x="${x}" y="352" text-anchor="middle" class="chart-tiny">${String(row.year).slice(2)}</text>
        `;
      }).join("");
      document.getElementById("forward-chart").innerHTML = `
        <svg viewBox="0 0 760 380" role="img" aria-label="Forward heat stress chart">
          <text x="24" y="34" class="chart-title">${focus.city}: forward stress by summer</text>
          <text x="24" y="58" class="chart-label">Bars show average days above local p98. Vertical pins show scenario spread. Purple bars show high-end payout.</text>
          <line x1="74" x2="700" y1="306" y2="306" class="axis"/><line x1="74" x2="74" y1="84" y2="306" class="axis"/>
          <text x="28" y="91" class="chart-label">${maxY} days</text><text x="38" y="309" class="chart-label">0</text>
          ${bars}
          <rect x="410" y="28" width="12" height="12" fill="${COLORS.red}" opacity="0.82"/><text x="428" y="39" class="chart-label">higher activation</text>
          <rect x="550" y="28" width="12" height="12" fill="${COLORS.purple}" opacity="0.55"/><text x="568" y="39" class="chart-label">p95 payout</text>
        </svg>`;
    }
    function renderAll() {
      updateLabels();
      const rows = currentRows();
      updateMetrics(rows);
      renderMap(rows);
      renderQueue(rows);
      renderDrivers(rows);
      renderSensitivity();
      renderBasis(rows);
      renderTimeline();
      renderFocusCopy();
      renderClimateContext();
      renderForward();
    }
    function baselineFor(regionId, dateText) {
      const fallback = { min: 22, p10: 24, p25: 26, mean: 28, p50: 28, p75: 31, p90: 33, p95: 34, p98: 35, p99: 36, max: 38 };
      const table = CACHE.baselines[regionId] || {};
      const direct = table[monthDay(dateText)];
      if (direct) return { ...fallback, ...direct };
      const values = Object.values(table);
      if (!values.length) return fallback;
      return { ...fallback, ...values[Math.floor(values.length / 2)] };
    }
    function scoreFrameRow(row, popMax) {
      const populationIndex = clamp(row.population_million / Math.max(popMax, 1), 0, 1);
      const p95Excess = Math.max(row.temp - row.thresholds["95"], 0);
      const heatSeverityScore = 100 * (
        0.30 * bounded(row.anomaly, 10)
        + 0.25 * bounded(p95Excess, 8)
        + 0.25 * bounded(row.streaks["95"], 5)
        + 0.20 * bounded(row.hdd30, 10)
      );
      const dryFactor = clamp(0.65 + 0.35 * bounded(row.dry_streak, 7), 0, 1.2);
      const streakFactor = clamp(0.60 + 0.40 * bounded(row.streaks["95"], 5), 0, 1.2);
      row.stress = {
        "Life and health": clamp(heatSeverityScore * (0.45 + row.elderly_share * 2.0) * (0.70 + 0.60 * populationIndex), 0, 100),
        "Agriculture": clamp(heatSeverityScore * (0.40 + row.agriculture_exposure * 0.80) * dryFactor, 0, 100),
        "Energy": clamp(heatSeverityScore * (0.40 + row.energy_exposure * 0.80) * (0.35 + 0.65 * bounded(row.cdd22, 10)), 0, 100),
        "Infrastructure": clamp(heatSeverityScore * (0.45 * (0.35 + row.wildfire_exposure * 0.75) * dryFactor + 0.55 * (0.35 + row.infrastructure_exposure * 0.75) * streakFactor), 0, 100),
        "Business interruption": clamp(heatSeverityScore * (0.45 + 0.35 * row.infrastructure_exposure + 0.20 * row.energy_exposure) * streakFactor, 0, 100)
      };
      row.score = round(
        0.30 * row.stress["Life and health"]
        + 0.22 * row.stress["Agriculture"]
        + 0.20 * row.stress["Energy"]
        + 0.18 * row.stress["Infrastructure"]
        + 0.10 * row.stress["Business interruption"],
        1
      );
      for (const key of Object.keys(row.stress)) row.stress[key] = round(row.stress[key], 1);
      return row;
    }
    function buildRegionFrame(region, payload, popMax) {
      const daily = payload.daily || {};
      const times = daily.time || [];
      const rows = times.map((dateText, idx) => {
        const b = baselineFor(region.region_id, dateText);
        const temp = Number(daily.temperature_2m_max?.[idx]);
        const mean = Number(daily.temperature_2m_mean?.[idx]);
        const precipitation = Number(daily.precipitation_sum?.[idx] || 0);
        return {
          region_id: region.region_id,
          city: region.city,
          country: region.country,
          lat: region.latitude,
          lon: region.longitude,
          population_million: region.population_million,
          elderly_share: region.elderly_share,
          agriculture_exposure: region.agriculture_exposure,
          energy_exposure: region.energy_exposure,
          wildfire_exposure: region.wildfire_exposure,
          infrastructure_exposure: region.infrastructure_exposure,
          date: dateText,
          temp: Number.isFinite(temp) ? temp : b.mean,
          mean_temp: Number.isFinite(mean) ? mean : b.mean - 6,
          precipitation,
          normal: b.mean,
          thresholds: { "95": b.p95, "98": b.p98, "99": b.p99 }
        };
      });
      const streaks = { "95": 0, "98": 0, "99": 0 };
      let dryStreak = 0;
      for (const row of rows) {
        row.anomaly = row.temp - row.normal;
        row.hdd30 = Math.max(row.temp - 30, 0);
        row.cdd22 = Math.max(row.mean_temp - 22, 0);
        dryStreak = row.precipitation < 1 ? dryStreak + 1 : 0;
        row.dry_streak = dryStreak;
        for (const pct of ["95", "98", "99"]) {
          streaks[pct] = row.temp > row.thresholds[pct] ? streaks[pct] + 1 : 0;
        }
        row.streaks = { ...streaks };
        scoreFrameRow(row, popMax);
      }
      return rows;
    }
    function rowsFromLiveFrames(frames, selectedDate) {
      const startDate = isoShift(selectedDate, -9);
      const rows = [];
      const timeline = {};
      for (const [regionId, frame] of Object.entries(frames)) {
        const current = frame.find(row => row.date === selectedDate);
        if (!current) continue;
        const windowRows = frame.filter(row => row.date >= startDate && row.date <= selectedDate);
        const eventDegreeDays = {};
        for (const pct of ["95", "98", "99"]) {
          eventDegreeDays[pct] = round(windowRows.reduce((sum, row) => sum + Math.max(row.temp - row.thresholds[pct], 0), 0), 2);
        }
        rows.push({
          region_id: current.region_id,
          city: current.city,
          country: current.country,
          lat: current.lat,
          lon: current.lon,
          population_million: current.population_million,
          elderly_share: current.elderly_share,
          agriculture_exposure: current.agriculture_exposure,
          energy_exposure: current.energy_exposure,
          wildfire_exposure: current.wildfire_exposure,
          infrastructure_exposure: current.infrastructure_exposure,
          temp: round(current.temp, 1),
          mean_temp: round(current.mean_temp, 1),
          precipitation: round(current.precipitation, 1),
          anomaly: round(current.anomaly, 1),
          score: round(current.score, 1),
          hdd30: round(current.hdd30, 1),
          stress: current.stress,
          thresholds: {
            "95": round(current.thresholds["95"], 1),
            "98": round(current.thresholds["98"], 1),
            "99": round(current.thresholds["99"], 1)
          },
          streaks: current.streaks,
          event_degree_days: eventDegreeDays
        });
        timeline[current.city] = frame.map(row => ({
          date: row.date,
          temp: round(row.temp, 1),
          mean_temp: round(row.mean_temp, 1),
          precipitation: round(row.precipitation, 1),
          normal: round(row.normal, 1),
          p95: round(row.thresholds["95"], 1),
          p98: round(row.thresholds["98"], 1),
          p99: round(row.thresholds["99"], 1)
        }));
      }
      return { regions: rows.sort((a, b) => b.score - a.score), timeline };
    }
    async function fetchLiveWeather() {
      const regions = CACHE.region_meta;
      const latitudes = regions.map(row => row.latitude).join(",");
      const longitudes = regions.map(row => row.longitude).join(",");
      const url = `https://api.open-meteo.com/v1/forecast?latitude=${latitudes}&longitude=${longitudes}&daily=temperature_2m_max,temperature_2m_mean,precipitation_sum&past_days=10&forecast_days=5&timezone=auto`;
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`Open-Meteo returned ${response.status}`);
      const payload = await response.json();
      const list = Array.isArray(payload) ? payload : [payload];
      const firstTimes = list[0]?.daily?.time || [];
      const today = new Date().toISOString().slice(0, 10);
      const pastOrToday = firstTimes.filter(item => item <= today);
      const selectedDate = pastOrToday.includes(today) ? today : pastOrToday[pastOrToday.length - 1] || firstTimes[firstTimes.length - 1];
      if (!selectedDate) throw new Error("No live dates returned");
      const popMax = Math.max(...regions.map(row => row.population_million), 1);
      const frames = {};
      regions.forEach((region, idx) => {
        if (list[idx]?.daily?.time?.length) frames[region.region_id] = buildRegionFrame(region, list[idx], popMax);
      });
      const liveRows = rowsFromLiveFrames(frames, selectedDate);
      if (!liveRows.regions.length) throw new Error("No live city rows built");
      return {
        regions: liveRows.regions,
        timeline: liveRows.timeline,
        analysisDate: selectedDate,
        source: "Open-Meteo forecast API, refreshed in this browser",
        live: true
      };
    }
    async function refreshLive(manual = false) {
      setStatus(manual ? "refreshing live data" : "checking live data", "cached");
      try {
        const live = await fetchLiveWeather();
        view = live;
        state.focus = live.regions[0].city;
        setStatus("live weather loaded", "live");
        renderAll();
      } catch (error) {
        setStatus("using cached fallback", "cached");
        if (manual) alert("Live refresh failed, so the cached run stayed loaded.");
      }
    }

    document.getElementById("pct").addEventListener("change", event => { state.pct = event.target.value; renderAll(); });
    document.getElementById("streak").addEventListener("input", event => { state.streak = Number(event.target.value); renderAll(); });
    document.getElementById("notional").addEventListener("input", event => { state.notional = Number(event.target.value); renderAll(); });
    document.getElementById("rate").addEventListener("input", event => { state.rate = Number(event.target.value); renderAll(); });
    document.getElementById("cap").addEventListener("input", event => { state.cap = Number(event.target.value); renderAll(); });
    document.getElementById("mapMetric").addEventListener("change", event => { state.mapMetric = event.target.value; renderAll(); });
    document.getElementById("refresh").addEventListener("click", () => refreshLive(true));

    renderAll();
    refreshLive(false);
  </script>
</body>
</html>
'''


def main() -> None:
    payload = build_payload()
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "index.html").write_text(
        HTML.replace("__DATA__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built docs/index.html at {payload['generated_at']}")


if __name__ == "__main__":
    main()
