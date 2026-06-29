# Heat Stress Reinsurance Workbench

A live reinsurance analytics workbench for monitoring heatwave accumulation, exposure stress, basis risk and parametric trigger activation across major European regions.

The project was built around a simple question:

**If a severe European heatwave is unfolding today, where should a reinsurer look first and would the contract respond sensibly?**

It combines live weather data, local climatology, CMIP6 climate projection data, line of business stress modules and a transparent parametric trigger simulator. The output is an interactive Streamlit dashboard designed for both non technical and technical reviewers.

## Why This Matters

Heatwaves are correlated stress events. They can affect health, agriculture, energy demand, infrastructure, transport and business interruption at the same time. For insurers and reinsurers, the interesting part is not only the temperature. It is the accumulation of exposure across lines of business.

This workbench turns live weather data into a practical monitoring view:

- Which European regions are currently under the highest heat stress
- Which insurance lines are most exposed
- Where a simple parametric heat trigger would activate
- How sensitive scenario payouts are to trigger design
- Where basis risk appears, for example high stress without trigger activation
- How the same trigger design behaves across future climate model summers
- How future summers could shift heat stress under climate model scenarios
- What assumptions drive the result

## Dashboard Features

- Interactive Europe map with regional heatwave risk scores
- Underwriting queue ranked by composite stress
- Trigger design sensitivity matrix
- Basis risk scatter and basis risk flags
- Line of business stress matrix for life and health, agriculture, energy, property and infrastructure, and business interruption
- Parametric payout simulator with adjustable percentile, streak, notional, payout rate and cap
- City drilldown showing current temperatures versus local 1991 to 2020 climatology
- Climate forward view using downscaled CMIP6 model output
- Forward trigger stress test with projected activation rate, p75 payout, p95 payout and exceedance curve
- Reviewer brief with plain English event, contract, basis risk and model logic

## Tech Stack

- Python
- pandas
- DuckDB SQL
- Plotly
- Streamlit
- Open Meteo forecast, archive and climate APIs

## Quick Start

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The first run downloads weather data from Open Meteo and caches it under `data/cache/`. If the historical archive API rate limits a city, the dashboard flags that city and uses a simple fallback baseline only for that missing archive response.

## Methodology

The model uses a transparent proxy framework rather than a black box model.

Heat severity combines:

- Maximum temperature anomaly versus local 1991 to 2020 summer normal
- Exceedance above a local historical percentile threshold
- Consecutive heatwave days
- Heat degree days above 30 degrees Celsius

Line of business modules combine heat severity with simple exposure proxies:

- Life and health: elderly share and population concentration
- Agriculture: agricultural exposure and dry day streak
- Energy: cooling degree days and energy exposure
- Property and infrastructure: wildfire, dry streak and infrastructure exposure
- Business interruption: infrastructure and energy stress with heat persistence

The parametric trigger activates after a selected number of consecutive days above the local percentile threshold. Scenario payout is based on cumulative heat degree days above the trigger threshold and capped as a share of user selected notional.

Climate forward projections use downscaled CMIP6 model output from the Open Meteo climate API. The dashboard counts future summer days above the selected local heat threshold and shows the trend across model years. It also applies the same parametric trigger design to each projected model summer and reports activation rate, p75 payout, p95 payout and an exceedance curve. These are climate stress scenarios, not forecasts for specific future summers.

## Data Sources

- Open Meteo forecast API: https://open-meteo.com/en/docs
- Open Meteo historical weather API: https://open-meteo.com/en/docs/historical-weather-api
- Open Meteo climate API: https://open-meteo.com/en/docs/climate-api
- Swiss Re SONAR extreme heat insurance framing: https://www.swissre.com/institute/research/sonar/sonar2025/extreme-heat-insurance-fallouts.html
- World Meteorological Organization heatwave context: https://wmo.int/media/news/records-fall-extreme-heat-grips-europe
- WHO heat and health background: https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health

## Benchmarking

See `docs/benchmark_notes.md` for the short review of public disaster risk, catastrophe risk and insurance analytics projects that shaped the dashboard design.

## Important Limitations

This is not a production catastrophe model and not underwriting advice.

It does not include real insured values, actual policy wording, claims, medical data, crop yield models, grid topology, wildfire fuel moisture, insurer accumulation data or treaty terms. The CHF payout number is a scenario value based on user selected notional, not an estimate of market losses. Those inputs would be required before commercial use.

The goal is to demonstrate how a current event can be translated into a disciplined analytics workflow that is relevant to insurance and reinsurance.
