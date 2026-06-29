# Heat Stress Reinsurance Workbench

[Open the working browser preview](https://romanmski.github.io/europe-heatwave-reinsurance-risk-monitor/)

![Browser preview](docs/assets/browser-preview.png)

I built this because a heatwave is not just a weather story for an insurer. It can hit health, agriculture, energy demand, infrastructure and business interruption at the same time. The interesting question is not only how hot it is. The more useful question is where a portfolio starts to behave badly, and whether the contract wording would actually respond.

The browser preview is the first thing to open. It loads as a static page, so nobody has to install Python just to understand the idea. You can move the trigger percentile, streak length, notional, payout rate and cap, then watch the risk map, underwriting queue, payout sensitivity, basis risk view and forward heat stress chart update together.

The full Streamlit version in `app.py` goes deeper and can refresh the weather data. The static preview uses cached Open Meteo data so the public link stays fast and easy to open from GitHub.

This is not pretending to be a production catastrophe model. It is a portfolio project showing the workflow in a transparent way: fetch current weather, compare each city with its own 1991 to 2020 climate history, score heat stress by line of business, test a parametric trigger, then ask where basis risk appears.

To run the full app locally on Windows, double-click `run_local.bat`. From a terminal you can also run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The model is deliberately transparent. Heat severity uses anomaly, local percentile exceedance, heat streaks and heat degree days. The stress modules then apply simple exposure proxies for life and health, agriculture, energy, infrastructure and business interruption. Scenario payouts are based on a user-selected notional, payout rate and cap.

Important caveat: this is not underwriting advice and it does not use real insured values, claims, policy wording, treaty terms or insurer accumulation data. Those would be needed before anything commercial. The goal here is to show that I can take a current event and turn it into a useful analytical product rather than just another notebook.
