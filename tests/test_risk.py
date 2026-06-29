import pandas as pd

from heatwave_risk.risk import compute_risk_scores, selected_day_view


def _sample_daily():
    dates = pd.date_range("2026-06-25", periods=4)
    return pd.DataFrame(
        {
            "region_id": ["x"] * 4,
            "city": ["Test City"] * 4,
            "country": ["Testland"] * 4,
            "date": dates,
            "temperature_2m_max": [31, 34, 35, 36],
            "temperature_2m_mean": [25, 27, 28, 29],
            "apparent_temperature_max": [33, 36, 37, 39],
            "precipitation_sum": [0, 0, 0, 0],
            "clim_mean": [25, 25, 25, 25],
            "clim_p95": [30, 30, 30, 30],
            "clim_p98": [32, 32, 32, 32],
            "clim_p99": [34, 34, 34, 34],
            "temp_anomaly": [6, 9, 10, 11],
            "above_p95": [True, True, True, True],
            "above_p98": [False, True, True, True],
            "above_p99": [False, False, True, True],
            "hdd_30": [1, 4, 5, 6],
            "cdd_22": [3, 5, 6, 7],
            "dry_day": [True, True, True, True],
            "latitude": [0.0] * 4,
            "longitude": [0.0] * 4,
            "population_million": [2.0] * 4,
            "elderly_share": [0.2] * 4,
            "agriculture_exposure": [0.6] * 4,
            "energy_exposure": [0.7] * 4,
            "wildfire_exposure": [0.5] * 4,
            "infrastructure_exposure": [0.6] * 4,
            "is_forecast": [False] * 4,
        }
    )


def test_scores_increase_with_heat_streak():
    scored = compute_risk_scores(_sample_daily())
    assert scored["heat_severity_score"].iloc[-1] > scored["heat_severity_score"].iloc[0]
    assert scored["composite_risk_score"].between(0, 100).all()


def test_parametric_trigger_pays_when_streak_is_met():
    scored = compute_risk_scores(_sample_daily())
    day = selected_day_view(
        scored,
        pd.Timestamp("2026-06-28"),
        threshold_percentile="98",
        trigger_days=3,
        notional_chf_m=10,
        payout_per_degree_day=0.08,
        cap_pct=0.25,
    )
    assert bool(day["trigger_active"].iloc[0])
    assert day["modeled_payout_chf_m"].iloc[0] > 0
