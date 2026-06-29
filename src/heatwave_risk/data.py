from __future__ import annotations

import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "precipitation_sum",
]


def load_regions(path: Path) -> pd.DataFrame:
    regions = pd.read_csv(path)
    expected = {
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
    }
    missing = expected - set(regions.columns)
    if missing:
        raise ValueError(f"Missing region columns: {sorted(missing)}")
    return regions


def _request_json(url: str, params: dict, cache_file: Path | None, refresh: bool) -> dict:
    if cache_file and cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _fallback_summer_baseline(region: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    """Deterministic fallback used when historical API access is temporarily rate limited."""
    all_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    dates = all_dates[(all_dates.month >= 6) & (all_dates.month <= 8)]

    country_peak = {
        "Spain": 35.0,
        "Portugal": 32.0,
        "France": 31.0,
        "Italy": 33.0,
        "Greece": 34.5,
        "Switzerland": 28.5,
        "Germany": 29.0,
        "Austria": 29.5,
        "Poland": 28.5,
        "United Kingdom": 25.0,
        "Ireland": 22.0,
    }
    base_peak = country_peak.get(str(region["country"]), 29.0)
    latitude_adjustment = max(0, (48 - float(region["latitude"])) * 0.18)
    peak = base_peak + latitude_adjustment
    seasonal = []
    for dt in dates:
        day = dt.dayofyear
        wave = 0.5 + 0.5 * math.sin(((day - 172) / 55) * math.pi)
        max_temp = peak - 5.0 + 7.0 * wave
        mean_temp = max_temp - 7.5
        seasonal.append(
            {
                "date": dt,
                "temperature_2m_max": max_temp,
                "temperature_2m_mean": mean_temp,
                "apparent_temperature_max": max_temp + 1.2,
                "precipitation_sum": 1.2,
                "region_id": region["region_id"],
                "city": region["city"],
                "country": region["country"],
                "baseline_source": "fallback_climatology",
            }
        )
    return pd.DataFrame(seasonal)


def _daily_frame(payload: dict, region: pd.Series) -> pd.DataFrame:
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        raise ValueError(f"No daily weather data returned for {region['city']}")

    frame = pd.DataFrame(daily)
    frame["date"] = pd.to_datetime(frame.pop("time"))
    frame["region_id"] = region["region_id"]
    frame["city"] = region["city"]
    frame["country"] = region["country"]
    return frame


def fetch_recent_weather(region: pd.Series, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    cache_file = cache_dir / "open_meteo_recent" / f"{region['region_id']}.json"
    params = {
        "latitude": float(region["latitude"]),
        "longitude": float(region["longitude"]),
        "daily": ",".join(DAILY_FIELDS),
        "past_days": 10,
        "forecast_days": 5,
        "timezone": "auto",
    }
    payload = _request_json(FORECAST_URL, params, cache_file, refresh)
    return _daily_frame(payload, region)


def fetch_summer_baseline(
    region: pd.Series,
    cache_dir: Path,
    refresh: bool,
    start_year: int = 1991,
    end_year: int = 2020,
) -> pd.DataFrame:
    cache_file = cache_dir / "open_meteo_baseline" / f"{region['region_id']}_{start_year}_{end_year}.json"
    params = {
        "latitude": float(region["latitude"]),
        "longitude": float(region["longitude"]),
        "start_date": f"{start_year}-06-01",
        "end_date": f"{end_year}-08-31",
        "daily": ",".join(DAILY_FIELDS),
        "timezone": "auto",
    }
    try:
        payload = _request_json(ARCHIVE_URL, params, cache_file, refresh)
        frame = _daily_frame(payload, region)
        frame["baseline_source"] = "open_meteo_archive"
        return frame
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            return _fallback_summer_baseline(region, start_year, end_year)
        raise


def _baseline_stats(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame.copy()
    baseline["month_day"] = baseline["date"].dt.strftime("%m-%d")

    grouped = baseline.groupby(["region_id", "month_day"])["temperature_2m_max"]
    stats = grouped.agg(
        clim_mean="mean",
        clim_p95=lambda values: values.quantile(0.95),
        clim_p98=lambda values: values.quantile(0.98),
        clim_p99=lambda values: values.quantile(0.99),
    ).reset_index()
    if "baseline_source" in baseline.columns:
        source = baseline.groupby(["region_id", "month_day"])["baseline_source"].agg("first").reset_index()
        stats = stats.merge(source, on=["region_id", "month_day"], how="left")
    else:
        stats["baseline_source"] = "open_meteo_archive"
    return stats


def build_weather_dataset(
    regions_path: Path,
    cache_dir: Path,
    refresh: bool = False,
    limit_regions: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = load_regions(regions_path)
    if limit_regions:
        region_ids = set(limit_regions)
        regions = regions[regions["region_id"].isin(region_ids)].copy()

    recent_frames = []
    baseline_frames = []
    for _, region in regions.iterrows():
        recent_frames.append(fetch_recent_weather(region, cache_dir, refresh))
        time.sleep(0.05)
        baseline_frames.append(fetch_summer_baseline(region, cache_dir, refresh))
        time.sleep(0.05)

    recent = pd.concat(recent_frames, ignore_index=True)
    baseline = pd.concat(baseline_frames, ignore_index=True)
    stats = _baseline_stats(baseline)

    recent["month_day"] = recent["date"].dt.strftime("%m-%d")
    daily = recent.merge(stats, on=["region_id", "month_day"], how="left")
    daily = daily.merge(regions, on=["region_id", "city", "country"], how="left")

    today = pd.Timestamp(date.today())
    daily["is_forecast"] = daily["date"] > today
    daily["temp_anomaly"] = daily["temperature_2m_max"] - daily["clim_mean"]
    daily["above_p95"] = daily["temperature_2m_max"] > daily["clim_p95"]
    daily["above_p98"] = daily["temperature_2m_max"] > daily["clim_p98"]
    daily["above_p99"] = daily["temperature_2m_max"] > daily["clim_p99"]
    daily["hdd_30"] = (daily["temperature_2m_max"] - 30).clip(lower=0)
    daily["cdd_22"] = (daily["temperature_2m_mean"] - 22).clip(lower=0)
    daily["dry_day"] = daily["precipitation_sum"].fillna(0) < 1.0

    return daily.sort_values(["region_id", "date"]).reset_index(drop=True), regions
