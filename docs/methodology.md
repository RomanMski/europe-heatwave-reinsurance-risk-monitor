# Methodology Notes

## Analytical Goal

The dashboard is designed as a transparent workbench for heatwave accumulation monitoring from a reinsurance perspective. It does not predict insured losses. It shows how live heat data, local climate thresholds, climate projection scenarios, basis risk checks and parametric trigger logic can produce a useful first level risk view.

## Weather Data

The app uses Open Meteo forecast data for the current event window and Open Meteo historical archive data for a 1991 to 2020 summer baseline when the archive API is available. If the archive API rate limits, the dashboard flags the affected city and uses a deterministic fallback only for that missing baseline.

For each region, the model pulls:

- Daily maximum temperature
- Daily mean temperature
- Daily maximum apparent temperature
- Daily precipitation

The baseline is calculated by month and day. For example, the June 29 threshold in Madrid is compared with historical June 29 observations between 1991 and 2020.

## Heat Severity

Heat severity is scaled from 0 to 100 and combines four components:

- Local temperature anomaly versus the 1991 to 2020 normal
- Exceedance above the local p95 threshold
- Consecutive days above p95
- Heat degree days above 30 degrees Celsius

This avoids treating 35 degrees Celsius in every region the same. A temperature that is normal in Seville can be extreme in London or Zurich.

## Exposure Proxies

The first version uses transparent hand curated regional exposure proxies stored in `data/regions.csv`.

These are intentionally easy to replace. A production version would use Eurostat population data, EEA land use data, crop exposure, insured values, company portfolio data and real accumulation views.

## Line of Business Stress

The dashboard models five insurance relevant stress modules:

- Life and health stress
- Agriculture stress
- Energy stress
- Property and infrastructure stress
- Business interruption stress

Each module combines heat severity with exposure proxies. The composite risk score is a weighted average of the modules. Users can adjust weights in the sidebar to test portfolio sensitivity.

## Parametric Trigger

The parametric trigger is a simplified example of how a heat product could be monitored.

The user selects:

- Trigger percentile
- Required consecutive trigger days
- Notional per region
- Payout rate per heat degree day
- Payout cap

The trigger activates when a region has enough consecutive days above the local threshold. Payout is linked to cumulative heat degree days above that threshold and capped as a share of notional.

## Climate Forward

The climate forward tab uses the Open Meteo climate API, which serves downscaled CMIP6 climate model data. The app counts projected summer days above the selected local heat threshold and shows the model average trend.

The same tab also applies the selected trigger design to each projected model summer. For each model year, it calculates:

- Days above the local trigger threshold
- Heat degree days above the local trigger threshold
- Maximum consecutive day streak above the trigger threshold
- Whether the selected trigger would activate
- Scenario payout after the selected cap

The output is summarized by period using average payout, p75 payout, p95 payout and trigger activation rate.

This is not a forecast for a specific summer. It is a climate stress scenario used to ask whether today's extreme threshold could become more common over time.

## What Would Improve The Model

- Actual insured exposure by line of business
- Claims data
- Mortality and hospital admission data
- Crop yield and soil moisture data
- Grid load and outage data
- Wildfire fuel moisture and wind data
- Real treaty terms and attachment points
- NUTS level regional granularity
