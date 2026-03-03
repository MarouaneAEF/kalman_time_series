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
