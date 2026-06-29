# Benchmark Notes

This project was shaped by a quick review of public climate risk, disaster risk and insurance analytics projects. The goal was not to copy their design. The goal was to identify what makes a portfolio project feel credible for insurance and reinsurance work.

## What Good Public Projects Tend To Have

### Real time signal

Public disaster dashboards often look stronger when they use live or frequently refreshed data instead of a static CSV. The useful part is not only freshness. It is the ability to show how a current event changes a monitoring view.

Comparable example:
https://github.com/logitechsoumili/Disaster_Risk_Prediction

What this project uses:

- Live Open Meteo forecast data
- Historical local climate baseline
- Current event ranking
- Action queue for follow up review

### Tail risk layer

Insurance analytics projects become more relevant when they include tail risk metrics, not just averages. A project can be technically simple and still feel actuarial if it shows stress, distribution and downside.

Comparable example:
https://github.com/the-irritater/cat-risk-ab-testing

What this project uses:

- Parametric heat trigger design
- Scenario payout by region
- Forward projected payout distribution
- p75, p95 and trigger activation rate by period

### Hazard, exposure and model logic separation

Serious catastrophe and climate models separate hazard, exposure, vulnerability and financial terms. Public frameworks such as Oasis LMF and CLIMADA are far larger than this project, but they are useful references for structure.

Comparable examples:
https://github.com/oasislmf
https://github.com/CLIMADA-project/climada_python

What this project uses:

- Weather hazard data
- Local threshold and heat severity logic
- Simple exposure proxies
- Line of business stress modules
- Separate scenario payout logic
- Reviewer brief with visible assumptions and limits

## Industry Framing Used

Swiss Re frames extreme heat as relevant for property, specialty, life and health, energy, transport and infrastructure risk.

https://www.swissre.com/institute/research/sonar/sonar2025/extreme-heat-insurance-fallouts.html

Munich Re describes parametric weather insurance as trigger based, using pre-agreed thresholds and independent data.

https://www.munichre.com/en/solutions/for-industry-clients/adverse-weather.html

World Weather Attribution highlights mortality, cooling demand, wildfire risk, infrastructure services and daily life impacts during European heatwaves.

https://www.worldweatherattribution.org/fossil-fuel-emissions-have-rapidly-worsened-european-heatwaves-in-just-a-few-decades/

Open Meteo provides the forecast, historical archive and CMIP6 climate projection data used in the prototype.

https://open-meteo.com/en/docs
https://open-meteo.com/en/docs/historical-weather-api
https://open-meteo.com/en/docs/climate-api

## Design Decision

The strongest direction is not to make a generic climate dashboard. The stronger direction is a small reinsurance analytics workbench:

- What is happening today?
- Which line of business is driving the risk?
- Would a heat trigger activate?
- How would the same trigger behave under future climate model summers?
- Which assumptions are real data, and which assumptions are scenario inputs?
