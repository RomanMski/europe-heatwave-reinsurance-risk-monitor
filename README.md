# Heat Stress Reinsurance Workbench

![Preview](docs/assets/preview.svg)

I built this because a heatwave is not just a weather story for an insurer. It can hit health, agriculture, energy demand, infrastructure and business interruption at the same time. The interesting question is not only "how hot is it?" but "where does the portfolio start to behave badly, and would the contract actually respond?"

The browser preview is meant to be the first click:

[Open the interactive preview](https://romanmski.github.io/europe-heatwave-reinsurance-risk-monitor/)

The preview has scenario controls, a Europe risk map, an underwriting queue, a trigger sensitivity view, a basis-risk scatter and a forward climate stress chart. The full Streamlit version in `app.py` goes deeper and can refresh the live weather data.

This is not pretending to be a production catastrophe model. It is a portfolio project showing the workflow: fetch current weather, build local thresholds from 1991 to 2020 climatology, score heat stress by line of business, test a parametric trigger, then ask where basis risk appears.

To run the full app locally on Windows, double-click `run_local.bat`. From a terminal you can also run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The model is deliberately transparent. Heat severity uses anomaly, local percentile exceedance, heat streaks and heat degree days. The stress modules then apply simple exposure proxies for life and health, agriculture, energy, infrastructure and business interruption. Scenario payouts are based on a user-selected notional, payout rate and cap.

Important caveat: this is not underwriting advice and it does not use real insured values, claims, policy wording, treaty terms or insurer accumulation data. Those would be needed before anything commercial. The goal here is to show that I can take a current event and turn it into a useful analytical product rather than just another notebook.
