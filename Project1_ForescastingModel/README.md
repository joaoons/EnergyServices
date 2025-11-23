# Forecasting Model

This folder contains the Python code, datasets, and trained models for forecasting electricity consumption in the **North Tower** building of IST.

## Files Overview

- **Data Files**
  - `IST_North_Tower_2017.csv` – Historical electricity consumption (2017)
  - `IST_North_Tower_2018.csv` – Historical electricity consumption (2018)
  - `IST_North_Tower_2019_raw.csv` – Raw 2019 electricity consumption data
  - `IST_North_Tower_Clean.csv` – Cleaned and processed historical data
  - `testData_2019_NorthTower.csv` – Test dataset for 2019 forecasts
  - `IST_meteo_data_2017_2018_2019.csv` – Weather data for the corresponding years

- **Models**
  - `BT_model.sav` – Trained Bagging model
  - `NN_model.sav` – Trained Neural Network model
  - `XGB_model.pkl` / `XGB_model.sav` – Trained XGBoost model

- **Scripts / Notebooks**
  - `forecasting.ipynb` – Main notebook: data cleaning, feature selection, model training, evaluation
