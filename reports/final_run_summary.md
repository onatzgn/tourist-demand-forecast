Final run summary

This run completed end-to-end on local environment using synthetic data generation, SQLite loading, weekly feature engineering, model training, model evaluation, and API smoke testing.

Artifacts produced:
- /Users/onatozgen/Desktop/tourism-demand-forecast/data/raw/visits.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/data/raw/landmark_meta.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/data/raw/weather_daily.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/data/raw/events.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/data/db.sqlite
- /Users/onatozgen/Desktop/tourism-demand-forecast/data/processed/weekly_features.parquet
- /Users/onatozgen/Desktop/tourism-demand-forecast/models/model.pkl
- /Users/onatozgen/Desktop/tourism-demand-forecast/reports/metrics_summary.json
- /Users/onatozgen/Desktop/tourism-demand-forecast/reports/predictions_test.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/reports/metrics_by_city.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/reports/metrics_by_landmark.csv
- /Users/onatozgen/Desktop/tourism-demand-forecast/reports/figures/*.png

Evaluation summary (test cutoff: 2024-10-07):
- baseline: MAE 7.0056, RMSE 9.2895, MAPE 48.9026
- random_forest: MAE 6.4083, RMSE 8.4497, MAPE 49.5400
- sarimax: MAE 8.3368, RMSE 11.0398, MAPE 65.3615

Model selection note:
- RandomForest performed best on MAE and RMSE in this run.
- MAPE is less stable for low count targets, so MAE/RMSE are primary for selection.

API smoke test:
- Endpoint: POST /predict
- Input: {"city":"Barcelona","landmark_id":"casa_batllo","week_start_date":"2023-01-09"}
- Output: predicted_next_week_visit_count = 22.749
- Status: success

Test status:
- Unit tests executed after path fix (tests/conftest.py) to ensure `src` imports in pytest.

Operational note:
- Environment used Python 3.13, so dependency setup required non-pinned install path during execution.
