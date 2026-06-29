# Integrating AI Models for Processing Financial-Accounting Data and Forecasting Budget Revenues of Public Institutions

Source code and reproducible Jupyter notebooks for the research article **"Integrating AI Models for Processing Financial-Accounting Data and Forecasting Budget Revenues of Public Institutions."**

This is the **budget-revenue (income)** counterpart of our public-expenditure forecasting study; it reuses the same modeling framework and extends the analysis horizon to **2025**.

**Affiliation:** Valahia University of Târgoviște, Romania
**Code license:** GNU Affero General Public License v3.0 (AGPL-3.0) — see [LICENSE](LICENSE).

## Authors

| # | Author | Affiliation | E-mail |
|---|--------|-------------|--------|
| | Bogdan Drăgulin | 1 | bogdan.dragulin@ats.com.ro |
| | Veronica Ștefan | 1 | veronica.stefan@valahia.ro |
| | Alina-Iuliana Tăbîrcă | 2 | alina.tabirca@valahia.ro |
| | Valentin Radu \* | 2 | valentin.radu@valahia.ro |
| | Angela-Nicoleta Cozorici | 3 | angela.cozorici@usm.ro |
| | Marilena-Roxana Zuca | 4 | marilena.roxana.zuca@rau.ro |

\* Correspondence: valentin.radu@valahia.ro

**Affiliations**

1. Doctoral School of Economics and Humanities, Valahia University of Târgoviște, 130004 Târgoviște, Romania
2. Faculty of Economics, Valahia University of Târgoviște, 130004 Târgoviște, Romania
3. Faculty of Economic Sciences and Public Administration, Ștefan cel Mare University of Suceava, 720225 Suceava, Romania
4. School of Domestic and International Business, Banking and Finance, Romanian American University, 012101 Bucharest, Romania

## Overview

This repository contains a time-series forecasting framework that applies statistical, econometric, and deep-learning models to the **monthly budget revenues of public institutions**. The pipeline automates preprocessing, exploratory analysis (seasonality, ACF/PACF), train/test evaluation, and comparison across univariate and multivariate approaches, exporting both metrics and visualizations.

## Data

The notebooks read **bundled CSV files** (no network/API dependency), covering **2016–2025** at monthly resolution:

- `venituri_total.csv` — total budget revenues (`Valoare`), the forecasting target.
- `cheltuieli_total.csv` — total budget expenditures, used by the cointegration / ECM analysis.
- `indicatori_total.csv` — macro indicators: GDP (`PIB`), average gross monthly earnings (`CastigSalarial`), number of employees (`NumarSalariati`), resident population (`Populatie`), registered unemployment (`Somaj`).

**Provenance:** revenue and expenditure series are aggregated from the public budget-execution database (ANAF filings); the macro indicators are sourced from the **Romanian National Institute of Statistics (INS) TEMPO-Online** (series `CON104Q`, `FOM107E`, `FOM104D`, `POP105A`, `SOM101E`) and aligned to monthly resolution. The CSVs were exported once so the analysis is fully reproducible offline.

> Note on coverage: revenues, expenditures, GDP, unemployment, and population reach 2025; the annual labour-market series (employees, earnings) are published by INS with a lag, so their 2025 values are carried forward from 2024 until the official figures are released.

## Project structure

```
.
├── models.py                          # Forecasting model implementations
├── model_processor.py                 # Preprocessing, evaluation, and visualization pipeline
├── analiza-total-venituri.ipynb       # Univariate analysis (seasonality, ACF/PACF, SARIMA)
├── analiza-var-total-venituri.ipynb   # Vector AutoRegression (revenues ↔ GDP)
├── analiza-ecm-total-venituri.ipynb   # Cointegration / VECM (revenues ↔ expenditures)
├── prognoza-total-venituri.ipynb      # Full model comparison and forecasting
├── venituri_total.csv                 # Budget revenues (target), 2016–2025
├── cheltuieli_total.csv               # Budget expenditures, 2016–2025
├── indicatori_total.csv               # Macro indicators (GDP, earnings, employment, population, unemployment)
├── requirements.txt                   # Python dependencies
└── LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

or, with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

### Main dependencies

- TensorFlow / Keras — neural networks (LSTM, GRU, TCN)
- statsmodels — SARIMA, VAR, VECM
- Prophet — automated forecasting
- scikit-learn — metrics and preprocessing
- pandas, numpy, scipy — data manipulation
- matplotlib — visualizations

## Models

**Univariate:** Linear Regression, SARIMA, ETS, Prophet, Seasonal Naïve, LSTM, GRU, TCN
**Multivariate:** Linear Regression, SARIMA (with exogenous regressors), VAR, ECM (VECM), LSTM, GRU, TCN

Evaluation metrics: RMSE, MAPE, R², MAE. Default train/test split is 80/20; series are scaled with `RobustScaler` (or `StandardScaler` for global runs).

## Notebooks

- **analiza-total-venituri.ipynb** — Exploratory analysis of the revenue series: seasonal decomposition, ACF/PACF, and SARIMA parameter search.
- **analiza-var-total-venituri.ipynb** — Vector AutoRegression of revenues and GDP, including impulse-response analysis.
- **analiza-ecm-total-venituri.ipynb** — Cointegration / error-correction model of the long-run equilibrium between revenues and expenditures.
- **prognoza-total-venituri.ipynb** — End-to-end comparison of all univariate and multivariate models, with performance charts and loss curves.

## Results

Running `prognoza-total-venituri.ipynb` evaluates all models and writes a `results_*.json` file with per-model predictions and metrics, alongside the inline charts. Because the monthly revenue series is highly volatile, predictive accuracy is modest and is discussed in the article.

## Reproducibility notes

- The pins in `requirements.txt` reproduce the environment used for the published results, including `scipy==1.15.3` (required by `statsmodels==0.14.4`).
- **Prophet on Windows:** if you encounter `'Prophet' object has no attribute 'stan_backend'`, the bundled CmdStan stub is missing a `makefile`. Either run
  `python -c "import cmdstanpy; cmdstanpy.install_cmdstan(overwrite=True)"`,
  or create an empty `makefile` inside `…/site-packages/prophet/stan_model/cmdstan-*/` (Prophet uses the precompiled model and never invokes `make`).

## Publication

- Academic Editor: *Firstname Lastname*
- Received: *date* · Revised: *date* · Accepted: *date* · Published: *date*
- Citation: *To be added by editorial staff during production.*

## Citation

If you use this code in your research, please cite:

```
Bogdan Drăgulin, Veronica Ștefan, Alina-Iuliana Tăbîrcă, Valentin Radu,
Angela-Nicoleta Cozorici, and Marilena-Roxana Zuca. "Integrating AI Models for
Processing Financial-Accounting Data and Forecasting Budget Revenues of Public
Institutions." 2025.
```

## License

The source code in this repository is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see [LICENSE](LICENSE).

The accompanying article is © 2025 by the authors, submitted for possible open-access publication under the terms of the **Creative Commons Attribution (CC BY 4.0)** license (https://creativecommons.org/licenses/by/4.0/).
