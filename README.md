# Store-Level Time Series Forecasting of Restaurant Visitors Using SARIMAX

## Overview

This project forecasts daily restaurant visitors at the individual store level using the **Recruit Restaurant Visitor Forecasting** dataset.

The modeling target is:

- Forecast unit: `air_store_id x visit_date`
- Target variable: `visitors`
- Baseline: `Seasonal Naive (lag = 7)`
- Benchmark: `SARIMA(1,1,1)(1,1,1)[7]`
- Final model: `SARIMAX(1,1,1)(1,1,1)[7]` with `is_holiday`
- Main final run tag: `main250cv`

This is a **store-level forecasting** project, not a single aggregated restaurant time series.

Dataset reference:
[Recruit Restaurant Visitor Forecasting (Kaggle)](https://www.kaggle.com/code/headsortails/be-my-guest-recruit-restaurant-eda/input)

## Final Results

Main `main250cv` run:

- Total AIR stores in raw data: `814`
- Eligible stores after filtering: `812`
- Stores evaluated in the main run: `250`
- Holdout window: `2017-03-15 -> 2017-04-22`
- Holdout rows evaluated: `9,721`

Holdout `RMSLE`:

- Seasonal Naive: `0.9314`
- SARIMA no exog: `0.7255`
- SARIMAX holiday only: `0.7098`

Improvement:

- SARIMA vs Naive: `22.1%`
- SARIMAX vs Naive: `23.8%`
- SARIMAX vs SARIMA: `2.17%`

Store-level win rates:

- SARIMA beats Naive on `87.2%` of stores
- SARIMAX beats Naive on `89.2%` of stores
- SARIMAX beats SARIMA on `67.2%` of stores

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- src/
|   |-- 01_data_understanding_eda.py
|   |-- 02_data_cleaning_feature_engineering.py
|   `-- 03_final_store_level_forecasting.py
|-- notebooks/
|   |-- 01_data_understanding_eda.ipynb
|   |-- 02_data_cleaning_feature_engineering.ipynb
|   `-- 03_final_store_level_forecasting.ipynb
|-- report/
|   |-- final_report.pdf
|   |-- store_level_results.md
|   `-- figures/
`-- output/
```

If you want the key project files first, read:

1. `README.md`
2. `report/store_level_results.md`
3. `report/final_report.pdf`
4. `src/03_final_store_level_forecasting.py`
5. `output/store_level_summary_main250cv.json`
6. `output/store_level_metrics_main250cv.csv`

## Source Files

- `src/01_data_understanding_eda.py`
  Data understanding and exploratory analysis.
- `src/02_data_cleaning_feature_engineering.py`
  Data cleaning, duplicate handling, outlier treatment, and feature preparation.
- `src/03_final_store_level_forecasting.py`
  Final model comparison, rolling-origin validation, and holdout evaluation.

## Data Requirements

The repository does **not** include these data folders in version control:

- `notebooks/raw_Data/`
- `notebooks/processed_data/`

They are excluded in `.gitignore`, so a fresh clone will not run end-to-end until the data is added locally.

Expected layout:

```text
notebooks/
|-- raw_Data/
|   |-- air_visit_data.csv
|   |-- air_reserve.csv
|   |-- hpg_reserve.csv
|   |-- air_store_info.csv
|   |-- hpg_store_info.csv
|   |-- date_info.csv
|   |-- store_id_relation.csv
|   `-- sample_submission.csv
`-- processed_data/
    |-- air_visits.csv
    |-- air_reserve.csv
    |-- hpg_reserve.csv
    |-- air_store.csv
    |-- holidays.csv
    `-- store_ids.csv
```

Notes:

- `src/01` and `src/02` read from `notebooks/raw_Data/`
- `src/03` reads from `notebooks/processed_data/`
- if processed data already exists, you can run only the final forecasting script

## How To Run

Run commands from the repository root.

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the scripts in order:

```bash
python src/01_data_understanding_eda.py
python src/02_data_cleaning_feature_engineering.py
python src/03_final_store_level_forecasting.py
```

If processed data already exists, you can run only:

```bash
python src/03_final_store_level_forecasting.py
```

## Final Modeling Setup

Main evaluation setup:

- Main store sample: `250`
- Holdout window: `2017-03-15 -> 2017-04-22`
- Rolling-origin folds:
  - `CV1`: `2017-01-15 -> 2017-01-28`
  - `CV2`: `2017-02-01 -> 2017-02-14`
  - `CV3`: `2017-02-15 -> 2017-02-28`
- Primary evaluation metric: `RMSLE`

Modeling choices in the final script:

- no synthetic pre-history padding
- each store is modeled only on its observed date span
- missing days inside the observed span are filled with `0 visitors`
- final comparison includes `Seasonal Naive`, `SARIMA no exog`, and `SARIMAX holiday only`

## Output Files

Main outputs in `output/`:

- `store_level_summary_main250cv.json`
- `store_level_metrics_main250cv.csv`
- `store_level_holdout_predictions_main250cv.csv`
- `store_level_cv_summary_main250cv.csv`
- `store_level_ablation_metrics_main250cv.csv`

Additional outputs:

- `output/store_level_summary_robust400holdout.json`
- `output/store_level_metrics_robust400holdout.csv`
- `output/representative_residual_diagnostics.csv`
- `output/archive/` for older experiment results

## Requirements

Core libraries used by the project:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `folium`
- `statsmodels`
- `scikit-learn`

`jupyter`, `notebook`, and `ipykernel` remain in `requirements.txt` for notebook compatibility.

## Recommended Submission Files

For grading, review, or handoff, prioritize:

- `README.md`
- `report/store_level_results.md`
- `report/final_report.pdf`
- `src/01_data_understanding_eda.py`
- `src/02_data_cleaning_feature_engineering.py`
- `src/03_final_store_level_forecasting.py`
- `output/*main250cv*`
