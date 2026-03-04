# Data

## AAPL_prices.csv — Apple stock price (2022–2024)

| Field | Value |
|---|---|
| **Source** | Yahoo Finance via `yfinance` |
| **Frequency** | Daily (trading days) |
| **Period** | 2022-01-03 → 2024-12-30 |
| **Rows** | 752 days |
| **Columns** | `Date`, `Close` (adjusted closing price, USD) |
| **Licence** | Public financial data |
| **Usage example** | `examples/stock_prediction.py` |

---

## france_conso_elec.csv — French national electricity consumption (2020–2024)

| Field | Value |
|---|---|
| **Source** | **RTE** (Réseau de Transport d'Électricité) |
| **Portal** | [Open Data Réseaux Energies (ODRE)](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def) |
| **API** | `https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/records` |
| **Original frequency** | Hourly, aggregated into **daily mean (MW)** |
| **Period** | 2020-01-01 → 2024-12-31 |
| **Rows** | 1 827 days |
| **Columns** | `Date`, `Conso_MW` (daily mean consumption in MW) |
| **Licence** | [Licence Ouverte Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence) |
| **Citation** | RTE, Éco2mix national — consommation consolidée et définitive, ODRE, 2024 |
| **Usage example** | `examples/energy_prediction.py` |

**Notes:**
- Strong annual seasonality: ~80 GW in winter, ~36 GW in summer.
- Weekly seasonality: weekdays > weekends.
- Recommended modelling: STL decomposition before Kalman-EM.

---

## sncf_tgv_mensuel.csv — National monthly TGV traffic (2018–2025)

| Field | Value |
|---|---|
| **Source** | **SNCF Voyageurs** |
| **Portal** | [SNCF open data](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst) |
| **API** | `https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/regularite-mensuelle-tgv-aqst/records` |
| **Frequency** | Monthly (aggregation of all national TGV routes) |
| **Period** | 2018-01 → 2025-12 |
| **Rows** | 96 months |
| **Licence** | [Licence Ouverte Etalab 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence) |
| **Citation** | SNCF Voyageurs, Régularité mensuelle TGV, données ouvertes SNCF, 2025 |
| **Usage example** | `examples/transport_prediction.py` |

**Columns:**

| Column | Description | Unit |
|---|---|---|
| `Date` | First day of the month | YYYY-MM-DD |
| `trains_prevus` | Total number of scheduled TGV trains | trains/month |
| `annulations` | Number of cancelled trains | trains/month |
| `retards_15` | Trains arriving with > 15 min delay | trains/month |
| `ponctualite_pct` | Punctuality rate = (scheduled − late) / scheduled × 100 | % |

**Notes:**
- Sharp drop in 2020 (COVID-19): from ~36,000 to ~3,000 trains/month in April 2020.
- December 2019 strike visible in cancellation data.
- Moderate seasonality: summer peak (July–August) and February trough.

---

## sunspots_monthly.csv — Monthly sunspot numbers (1749–2026)

| Field | Value |
|---|---|
| **Source** | **WDC-SILSO** — World Data Center for the production, preservation and dissemination of the international sunspot number |
| **Provider** | Royal Observatory of Belgium, Brussels |
| **Portal** | [sidc.be/silso](https://www.sidc.be/silso/datafiles) |
| **URL** | `https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.csv` |
| **Frequency** | Monthly |
| **Period** | 1749-01 → 2026-03 (3 326 observations) |
| **Columns** | `Date`, `Sunspots` (monthly mean international sunspot number) |
| **Licence** | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for non-commercial use with attribution |
| **Citation** | SILSO World Data Center, *The International Sunspot Number*, Royal Observatory of Belgium, Brussels, 1749–present |

**Notes:**
- Quasi-periodic ~11-year solar cycle (Schwabe cycle), but period varies between 9 and 14 years.
- Amplitude strongly variable between cycles: some maxima exceed 200, others barely reach 80.
- Right-skewed distribution: sunspots ≥ 0 with occasional very high spikes.
- Contains the **Maunder Minimum** (1645–1715) and **Dalton Minimum** (1790–1830) — prolonged periods of very low activity.
- Cycle 25 peak (2024–2025) is visible at the end of the series.

**Recommended Streamlit settings:**
- Column: `Sunspots`
- STL: **disabled** (the cycle is quasi-periodic, not fixed-period)
- Latent dim `d`: **4** (captures the ~11-year oscillation and its harmonics)
- EM iterations: **200**

---

## airline_passengers.csv — Monthly international airline passengers (1949–1960)

| Field | Value |
|---|---|
| **Source** | Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (1976) *Time Series Analysis, Forecasting and Control*, 3rd ed., Holden-Day — **Series G** |
| **Available via** | R `datasets::AirPassengers` / `statsmodels.datasets.get_rdataset("AirPassengers")` |
| **Frequency** | Monthly |
| **Period** | 1949-01 → 1960-12 (144 observations) |
| **Columns** | `Date`, `Passengers` (monthly total international airline passengers, thousands) |
| **Licence** | Public domain — reproduced from the Box-Jenkins textbook, widely redistributed as a statistical benchmark |
| **Citation** | Box, G. E. P., Jenkins, G. M. & Reinsel, G. C. (1976). *Time Series Analysis: Forecasting and Control* (3rd ed.). Holden-Day. |

**Notes:**
- Canonical benchmark for time series analysis, originally used to illustrate SARIMA models.
- Strong multiplicative trend × annual seasonality: passenger numbers nearly tripled over the period.
- Variance increases with level → log transform recommended before Kalman-EM.
- Recommended Streamlit settings: enable STL (period = **12**), latent dim `d` = **2**, EM iterations ≥ 100.

---

## paris_temperature_daily.csv — Daily mean temperature in Paris (2020–2024)

| Field | Value |
|---|---|
| **Source** | [Open-Meteo](https://open-meteo.com/) — Historical Weather API |
| **API** | `https://archive-api.open-meteo.com/v1/archive?latitude=48.85&longitude=2.35&daily=temperature_2m_mean` |
| **Station** | Paris, France (48.85°N, 2.35°E) |
| **Frequency** | Daily |
| **Period** | 2020-01-01 → 2024-12-31 |
| **Rows** | 1 827 days |
| **Columns** | `Date`, `Temp_C` (daily mean air temperature at 2 m, °C) |
| **Licence** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — Open-Meteo open data |
| **Usage example** | Streamlit app or `from kalman_em import KalmanEM` |

**Notes:**
- Very strong annual seasonality: ~3 °C in January, ~25 °C in July.
- No weekly seasonality (temperature is not affected by human activity cycles).
- Recommended STL period: **365** (annual cycle).
- Ideal Streamlit demo: upload → enable STL (period=365) → Run Analysis.
