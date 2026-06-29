# Heat Stress Reinsurance Workbench

## Executive Summary

This workbench translates a live European heatwave into a reinsurance risk monitoring view. It combines current temperature data, local climatology, exposure proxies, line of business stress modules, basis risk checks and a simple parametric trigger simulator.

The purpose is not to estimate actual insured losses. The purpose is to show how a reinsurer could structure an early warning view for correlated heat stress across multiple lines of business.

## Why Heatwaves Matter For Reinsurance

Heatwaves are not single line events. The same period of extreme heat can stress life and health, agriculture, energy, infrastructure, transport and business interruption. This creates accumulation risk across a portfolio.

For example, a region with high elderly exposure, strong cooling demand, dry conditions and infrastructure sensitivity can become more relevant than a region that only has a high temperature reading.

## Prototype Output

The dashboard provides:

- A map of regional heatwave stress
- A ranked underwriting queue
- Basis risk flags that compare heat stress with contract response
- A stress matrix across insurance lines
- A parametric heat trigger simulator
- A city level drilldown against local historical thresholds
- A climate forward view using downscaled CMIP6 climate model data

## Interpretation

The most important output is not the exact score. It is the ranking and decomposition. A reviewer can see whether a region is high risk because of heat severity alone or because multiple lines of business become stressed at the same time.

The CHF amount shown in the dashboard is a parametric treaty scenario. It comes from the selected notional, trigger threshold, heat degree days and payout cap. It is deliberately labelled as scenario payout, not as an estimate of insured market loss.

The climate forward tab answers a separate question. It tests whether today's local heat threshold could become more common over the next summers under climate model scenarios. This helps connect an unfolding event with forward looking portfolio and product questions.

The forward trigger stress test then applies the same parametric cover design to projected model summers. This creates a simple distribution of scenario payout outcomes by period, including p75, p95 and activation rate. It is still not a pricing model, but it is closer to how an insurance reviewer would think about trigger behaviour and tail exposure.

## Limitations

The model is intentionally transparent and simple. It should not be used for underwriting, pricing or capital decisions without real exposure, claims, policy and treaty data.

The value of the prototype is in the structure of the workflow: live data, documented assumptions, explainable metrics, scenario controls and a professional monitoring surface.
