Forecast tourist demand for historical landmarks using synthetic behavior, seasonality, weather, ticket pricing, and event effects. The pipeline generates raw travel data, builds weekly city-landmark features, trains forecasting models, compares baseline vs machine learning vs time-series forecasts, and produces metrics and plots.

Outputs include raw CSV files, a SQLite database, processed weekly features, a trained model bundle, test predictions, city and landmark metric breakdowns, and actual-vs-predicted charts in `reports/figures`. Metrics are MAE, RMSE, and MAPE (with low-count protection to limit instability when denominators are near zero).

Latest local run summary (`rows=50000`): cutoff `2024-10-07`, baseline MAE/RMSE `7.01/9.29`, random forest MAE/RMSE `6.41/8.45`, SARIMAX MAE/RMSE `8.34/11.04`; API smoke test passed for `POST /predict` and returned `predicted_next_week_visit_count=22.749` for (`Barcelona`, `casa_batllo`, `2023-01-09`).
