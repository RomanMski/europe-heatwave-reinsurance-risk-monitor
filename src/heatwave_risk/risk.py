from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_LINE_WEIGHTS = {
    "life_health": 0.30,
    "agriculture": 0.22,
    "energy": 0.20,
    "property_infra": 0.18,
    "business_interruption": 0.10,
}


def _bounded(series: pd.Series, upper: float) -> pd.Series:
    return (series.fillna(0) / upper).clip(lower=0, upper=1)


def _streak(values: pd.Series) -> pd.Series:
    count = 0
    out = []
    for value in values.fillna(False):
        count = count + 1 if bool(value) else 0
        out.append(count)
    return pd.Series(out, index=values.index)


def add_event_streaks(daily: pd.DataFrame) -> pd.DataFrame:
    scored = daily.sort_values(["region_id", "date"]).copy()
    scored["p95_streak"] = scored.groupby("region_id", group_keys=False)["above_p95"].apply(_streak)
    scored["p98_streak"] = scored.groupby("region_id", group_keys=False)["above_p98"].apply(_streak)
    scored["p99_streak"] = scored.groupby("region_id", group_keys=False)["above_p99"].apply(_streak)
    scored["dry_streak"] = scored.groupby("region_id", group_keys=False)["dry_day"].apply(_streak)
    return scored


def compute_risk_scores(daily: pd.DataFrame, line_weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = line_weights or DEFAULT_LINE_WEIGHTS
    weight_sum = sum(weights.values())
    weights = {key: value / weight_sum for key, value in weights.items()}

    scored = add_event_streaks(daily)
    population_scale = max(scored["population_million"].max(), 1)
    population_index = (scored["population_million"] / population_scale).clip(0, 1)

    p95_excess = (scored["temperature_2m_max"] - scored["clim_p95"]).clip(lower=0)
    heat_severity = (
        0.30 * _bounded(scored["temp_anomaly"], 10)
        + 0.25 * _bounded(p95_excess, 8)
        + 0.25 * _bounded(scored["p95_streak"], 5)
        + 0.20 * _bounded(scored["hdd_30"], 10)
    )
    scored["heat_severity_score"] = (100 * heat_severity).round(1)

    dry_factor = (0.65 + 0.35 * _bounded(scored["dry_streak"], 7)).clip(0, 1.2)
    streak_factor = (0.60 + 0.40 * _bounded(scored["p95_streak"], 5)).clip(0, 1.2)

    scored["life_health_stress"] = (
        scored["heat_severity_score"]
        * (0.45 + scored["elderly_share"] * 2.0)
        * (0.70 + 0.60 * population_index)
    ).clip(0, 100)

    scored["agriculture_stress"] = (
        scored["heat_severity_score"]
        * (0.40 + scored["agriculture_exposure"] * 0.80)
        * dry_factor
    ).clip(0, 100)

    scored["energy_stress"] = (
        scored["heat_severity_score"]
        * (0.40 + scored["energy_exposure"] * 0.80)
        * (0.35 + 0.65 * _bounded(scored["cdd_22"], 10))
    ).clip(0, 100)

    scored["property_infra_stress"] = (
        scored["heat_severity_score"]
        * (
            0.45 * (0.35 + scored["wildfire_exposure"] * 0.75) * dry_factor
            + 0.55 * (0.35 + scored["infrastructure_exposure"] * 0.75) * streak_factor
        )
    ).clip(0, 100)

    scored["business_interruption_stress"] = (
        scored["heat_severity_score"]
        * (0.45 + 0.35 * scored["infrastructure_exposure"] + 0.20 * scored["energy_exposure"])
        * streak_factor
    ).clip(0, 100)

    scored["composite_risk_score"] = (
        weights["life_health"] * scored["life_health_stress"]
        + weights["agriculture"] * scored["agriculture_stress"]
        + weights["energy"] * scored["energy_stress"]
        + weights["property_infra"] * scored["property_infra_stress"]
        + weights["business_interruption"] * scored["business_interruption_stress"]
    ).round(1)

    stress_cols = [
        "life_health_stress",
        "agriculture_stress",
        "energy_stress",
        "property_infra_stress",
        "business_interruption_stress",
        "composite_risk_score",
    ]
    scored[stress_cols] = scored[stress_cols].round(1)
    return scored


def selected_day_view(
    scored: pd.DataFrame,
    selected_date: pd.Timestamp,
    threshold_percentile: str,
    trigger_days: int,
    notional_chf_m: float,
    payout_per_degree_day: float,
    cap_pct: float,
    lookback_days: int = 10,
) -> pd.DataFrame:
    threshold_col = f"clim_p{threshold_percentile}"
    streak_col = f"p{threshold_percentile}_streak"
    if threshold_col not in scored.columns:
        raise ValueError(f"Unsupported trigger threshold: {threshold_percentile}")

    selected_date = pd.Timestamp(selected_date)
    start_date = selected_date - pd.Timedelta(days=lookback_days - 1)
    window = scored[(scored["date"] >= start_date) & (scored["date"] <= selected_date)].copy()
    window["threshold_excess"] = (window["temperature_2m_max"] - window[threshold_col]).clip(lower=0)

    event = window.groupby("region_id").agg(
        event_days_above=("threshold_excess", lambda values: int((values > 0).sum())),
        event_degree_days=("threshold_excess", "sum"),
    )

    current = scored[scored["date"] == selected_date].copy()
    current = current.merge(event, on="region_id", how="left")
    current["event_days_above"] = current["event_days_above"].fillna(0).astype(int)
    current["event_degree_days"] = current["event_degree_days"].fillna(0).round(2)
    current["trigger_active"] = current[streak_col] >= trigger_days

    raw_payout = current["event_degree_days"] * payout_per_degree_day * notional_chf_m
    cap = cap_pct * notional_chf_m
    current["modeled_payout_chf_m"] = np.where(current["trigger_active"], raw_payout.clip(upper=cap), 0).round(2)
    current["trigger_threshold_c"] = current[threshold_col].round(1)
    return current.sort_values("composite_risk_score", ascending=False).reset_index(drop=True)

