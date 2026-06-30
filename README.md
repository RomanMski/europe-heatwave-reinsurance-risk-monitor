# Live Heat Stress Risk Monitor

[Open the working browser preview](https://romanmski.github.io/europe-heatwave-reinsurance-risk-monitor/)

![Browser preview](docs/assets/browser-preview.png)

I built this because a heatwave is not just a weather story for an insurer or an energy desk. It can hit health, agriculture, power demand, infrastructure and business interruption at the same time. The useful question is not only how hot it is. It is where stress is building, and whether the trigger or risk process actually catches it.

The browser preview is the first thing to open. It runs directly from GitHub Pages, tries to refresh the latest Open-Meteo forecast inside the browser, and keeps a cached run embedded so the page still opens if the live API is blocked. You can move the trigger percentile, streak length, notional, payout rate, cap and map coloring, then watch the map, watchlist, contract sensitivity, basis-risk view and forward stress chart update together.

The full Streamlit version in `app.py` goes deeper for local analysis. The public page is the fast version for a recruiter or hiring manager: open it, interact with it, and see the logic without installing anything.

This is not pretending to be a production catastrophe model. It is a portfolio project showing the workflow in a transparent way: fetch current weather, compare each city with its own 1991 to 2020 climate history, score heat stress by line of business, test a parametric trigger, then ask where basis risk appears.

To run the full app locally on Windows, double-click `run_local.bat`. From a terminal you can also run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The model is deliberately transparent. Heat severity uses anomaly, local percentile exceedance, heat streaks and heat degree days. The stress modules then apply simple exposure proxies for life and health, agriculture, energy, infrastructure and business interruption. Scenario payouts are based on a user-selected notional, payout rate and cap.

Important caveat: this is not underwriting advice and it does not use real insured values, claims, policy wording, treaty terms or insurer accumulation data. Those would be needed before anything commercial. The goal here is to show that I can take a current event and turn it into a useful analytical product rather than just another notebook.
