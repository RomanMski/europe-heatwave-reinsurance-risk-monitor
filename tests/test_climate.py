import pandas as pd

from heatwave_risk.climate import summarize_projection


def test_projection_summary_tracks_local_trigger_streak_and_degree_days():
    projection = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-01", periods=5),
            "model": ["model_a"] * 5,
            "year": [2026] * 5,
            "temperature_2m_max": [29.0, 32.0, 33.0, 34.0, 28.0],
            "temperature_2m_mean": [22.0, 24.0, 25.0, 26.0, 21.0],
            "precipitation_sum": [0.0, 0.0, 0.0, 1.2, 2.0],
            "heat_day_30": [False, True, True, True, False],
            "heat_day_35": [False, False, False, False, False],
            "hdd_30": [0.0, 2.0, 3.0, 4.0, 0.0],
            "cooling_degree_day_22": [0.0, 2.0, 3.0, 4.0, 0.0],
        }
    )

    annual = summarize_projection(projection, local_threshold_c=31.0)

    assert annual["local_extreme_days"].iloc[0] == 3
    assert annual["max_local_extreme_streak"].iloc[0] == 3
    assert annual["local_heat_degree_days"].iloc[0] == 6.0
