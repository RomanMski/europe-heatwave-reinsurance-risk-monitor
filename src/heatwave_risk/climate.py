from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
DEFAULT_MODELS = ["EC_Earth3P_HR", "MPI_ESM1_2_XR", "MRI_AGCM3_2_S"]
DAILY_FIELDS = ["temperature_2m_max", "temperature_2m_mean", "precipitation_sum"]


def _read_or_fetch(url: str, params: dict, cache_file: Path, refresh: bool) -> dict:
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def fetch_climate_projection(
    region_id: str,
    latitude: float,
    longitude: float,
    cache_dir: Path,
    refresh: bool = False,
    start_year: int = 2026,
    end_year: int = 2035,
    models: list[str] | None = None,
) -> pd.DataFrame:
    models = models or DEFAULT_MODELS
    cache_file = cache_dir / "open_meteo_climate" / f"{region_id}_{start_year}_{end_year}_{'_'.join(models)}.json"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{start_year}-06-01",
        "end_date": f"{end_year}-08-31",
        "daily": ",".join(DAILY_FIELDS),
        "models": ",".join(models),
        "timezone": "auto",
    }
    payload = _read_or_fetch(CLIMATE_URL, params, cache_file, refresh)
    daily = payload["daily"]
    wide = pd.DataFrame(daily)
    wide["date"] = pd.to_datetime(wide["time"])
    wide = wide.drop(columns=["time"])

    rows = []
    for model in models:
        model_frame = pd.DataFrame(
            {
                "date": wide["date"],
                "model": model,
                "temperature_2m_max": wide[f"temperature_2m_max_{model}"],
                "temperature_2m_mean": wide[f"temperature_2m_mean_{model}"],
                "precipitation_sum": wide[f"precipitation_sum_{model}"],
            }
        )
        rows.append(model_frame)
    projection = pd.concat(rows, ignore_index=True)
    projection["region_id"] = region_id
    projection["year"] = projection["date"].dt.year
    projection["hdd_30"] = (projection["temperature_2m_max"] - 30).clip(lower=0)
    projection["heat_day_30"] = projection["temperature_2m_max"] >= 30
    projection["heat_day_35"] = projection["temperature_2m_max"] >= 35
    projection["cooling_degree_day_22"] = (projection["temperature_2m_mean"] - 22).clip(lower=0)
    projection["projection_source"] = "Open Meteo CMIP6 climate API"
    return projection


def fallback_projection_from_baseline(
    region_id: str,
    cache_dir: Path,
    start_year: int = 2026,
    end_year: int = 2035,
) -> pd.DataFrame:
    cache_file = cache_dir / "open_meteo_baseline" / f"{region_id}_1991_2020.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"No historical baseline cache found for {region_id}")

    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    baseline = pd.DataFrame(payload["daily"])
    baseline["date"] = pd.to_datetime(baseline.pop("time"))
    baseline["month_day"] = baseline["date"].dt.strftime("%m-%d")
    baseline = baseline[baseline["date"].dt.month.between(6, 8)].copy()

    daily_quantiles = (
        baseline.groupby("month_day")
        .agg(
            max_p50=("temperature_2m_max", lambda values: values.quantile(0.50)),
            max_p75=("temperature_2m_max", lambda values: values.quantile(0.75)),
            max_p90=("temperature_2m_max", lambda values: values.quantile(0.90)),
            mean_p50=("temperature_2m_mean", lambda values: values.quantile(0.50)),
            rain_p50=("precipitation_sum", lambda values: values.quantile(0.50)),
        )
        .reset_index()
    )

    scenarios = [
        ("Local trend low", "max_p50", 0.20),
        ("Local trend mid", "max_p75", 0.35),
        ("Local heat stress", "max_p90", 0.55),
    ]
    rows = []
    future_dates = pd.date_range(f"{start_year}-06-01", f"{end_year}-08-31", freq="D")
    future_dates = future_dates[future_dates.month.isin([6, 7, 8])]
    for model, max_col, warming_per_decade in scenarios:
        for future_date in future_dates:
            month_day = future_date.strftime("%m-%d")
            base = daily_quantiles[daily_quantiles["month_day"] == month_day]
            if base.empty:
                continue
            base = base.iloc[0]
            year_offset = int(future_date.year) - start_year
            trend = year_offset * warming_per_decade / 10
            seasonal_noise = 0.35 * np.sin((future_date.dayofyear + len(model)) / 9.0)
            max_temp = float(base[max_col] + trend + seasonal_noise)
            mean_temp = float(base["mean_p50"] + trend + seasonal_noise * 0.4)
            rows.append(
                {
                    "date": future_date,
                    "model": model,
                    "temperature_2m_max": max_temp,
                    "temperature_2m_mean": mean_temp,
                    "precipitation_sum": float(base["rain_p50"]),
                    "region_id": region_id,
                }
            )

    projection = pd.DataFrame(rows)
    projection["year"] = projection["date"].dt.year
    projection["hdd_30"] = (projection["temperature_2m_max"] - 30).clip(lower=0)
    projection["heat_day_30"] = projection["temperature_2m_max"] >= 30
    projection["heat_day_35"] = projection["temperature_2m_max"] >= 35
    projection["cooling_degree_day_22"] = (projection["temperature_2m_mean"] - 22).clip(lower=0)
    projection["projection_source"] = "Fallback local trend scenario from historical archive cache"
    return projection


def projection_period(year: int) -> str:
    if year <= 2030:
        return "2026 to 2030"
    if year <= 2035:
        return "2031 to 2035"
    if year <= 2040:
        return "2036 to 2040"
    if year <= 2045:
        return "2041 to 2045"
    return "2046 to 2050"


def _max_streak(values: pd.Series) -> int:
    count = 0
    best = 0
    for value in values.fillna(False):
        count = count + 1 if bool(value) else 0
        best = max(best, count)
    return best


def summarize_projection(projection: pd.DataFrame, local_threshold_c: float) -> pd.DataFrame:
    data = projection.copy()
    data["above_local_threshold"] = data["temperature_2m_max"] >= local_threshold_c
    data["local_threshold_excess"] = (data["temperature_2m_max"] - local_threshold_c).clip(lower=0)

    streaks = (
        data.sort_values(["model", "year", "date"])
        .groupby(["model", "year"])["above_local_threshold"]
        .apply(_max_streak)
        .reset_index(name="max_local_extreme_streak")
    )

    annual = (
        data.groupby(["model", "year"])
        .agg(
            summer_max_c=("temperature_2m_max", "max"),
            heat_days_30=("heat_day_30", "sum"),
            heat_days_35=("heat_day_35", "sum"),
            local_extreme_days=("above_local_threshold", "sum"),
            local_heat_degree_days=("local_threshold_excess", "sum"),
            heat_degree_days_30=("hdd_30", "sum"),
            cooling_degree_days_22=("cooling_degree_day_22", "sum"),
            dry_days=("precipitation_sum", lambda values: int((values < 1.0).sum())),
        )
        .reset_index()
        .merge(streaks, on=["model", "year"], how="left")
    )
    annual["summer_max_c"] = annual["summer_max_c"].round(1)
    annual["local_heat_degree_days"] = annual["local_heat_degree_days"].round(1)
    annual["heat_degree_days_30"] = annual["heat_degree_days_30"].round(1)
    annual["cooling_degree_days_22"] = annual["cooling_degree_days_22"].round(1)
    annual["period"] = annual["year"].map(projection_period)
    return annual


def projection_decade_summary(annual: pd.DataFrame) -> pd.DataFrame:
    data = annual.copy()
    data["period"] = data["year"].map(projection_period)
    summary = (
        data.groupby("period", observed=True)
        .agg(
            avg_heat_days_30=("heat_days_30", "mean"),
            avg_heat_days_35=("heat_days_35", "mean"),
            avg_local_extreme_days=("local_extreme_days", "mean"),
            avg_local_hdd=("local_heat_degree_days", "mean"),
            avg_max_streak=("max_local_extreme_streak", "mean"),
            avg_hdd_30=("heat_degree_days_30", "mean"),
            model_count=("model", "nunique"),
        )
        .reset_index()
    )
    numeric = [
        "avg_heat_days_30",
        "avg_heat_days_35",
        "avg_local_extreme_days",
        "avg_local_hdd",
        "avg_max_streak",
        "avg_hdd_30",
    ]
    summary[numeric] = summary[numeric].round(1)
    return summary
