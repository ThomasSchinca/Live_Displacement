## 📈 Displacement Forecasting Script

### 1. Overview

This script automates the retrieval, processing, and forecasting of internal displacement event data, using historical and real-time inputs. It executes in two major phases:

* **Data Update**:
  The script loads existing monthly historical data (`Hist.csv`) and appends newly retrieved events from the IDMC API (`https://helix-tools-api.idmcdb.org`). It filters events that occurred within the same or consecutive months, aggregates figures by country (`iso3`), and ensures consistency before appending them to the historical series.

* **Forecast Generation**:
  For each country in the dataset, the ShapeFinder is applied to forecast displacement trends for the next 12 months. These predictions are then saved for downstream usage.
  
---

### 2. Outputs

The script generates three key output files:

#### `Hist.csv`

* **Type**: CSV
* **Content**: Historical monthly internal displacement figures by country (ISO3 codes).
* **Structure**:

  * **Index**: Monthly periods (`YYYY-MM`)
  * **Columns**: ISO3 country codes
  * **Values**: Aggregated displacement figures per month per country

#### `Predictions.csv`

* **Type**: CSV
* **Content**: Forecasted displacement figures for the next 12 months for each country.
* **Structure**:

  * **Index**: Future monthly timestamps
  * **Columns**: ISO3 country codes
  * **Values**: Forecasted displacements

#### `Scenarios.pkl`

* **Type**: Python Pickle
* **Content**: Dictionary containing alternative forecast scenarios for each country.
* **Structure**:

  ```python
  {
    'USA': [list_of_matching_sequences with their distances, scenario_dataframe (index is probablity and columns the date)],
    'FRA': [...],
    ...
  }
  ```

  * The `scenario_dataframe` includes 12-month alternative forecast series per country.
  * Useful for uncertainty analysis or ensemble visualization.
