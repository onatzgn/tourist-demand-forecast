This is a small side data project inspired by TravelGuideApp, a travel and discovery app. The goal is to simulate user visits to historical landmarks and build a forecasting pipeline that estimates weekly demand per location using behavioral, seasonal, weather, pricing, and event-related features.

The project includes a synthetic dataset generator, basic data processing steps, SQL based analysis, and a few forecasting approaches. It was initially developed as part of an internal analysis for exploring B2B use cases and demand patterns across different cities and seasons. The pipeline generates raw travel data, builds weekly city–landmark features, trains multiple forecasting models, and compares baseline, machine learning, and time-series approaches.

All data in the project is synthetic and generated locally so the full pipeline can be reproduced without external dependencies.

Outputs include raw CSV files, a SQLite database, processed weekly features, a trained model bundle, test predictions, city and landmark metric breakdowns, and actual-vs-predicted charts in `reports/figures`. Metrics are MAE, RMSE, and MAPE (with low-count protection to limit instability when denominators are near zero).

Latest local run summary (`rows=50000`): cutoff `2024-10-07`, baseline MAE/RMSE `7.01/9.29`, random forest MAE/RMSE `6.41/8.45`, SARIMAX MAE/RMSE `8.34/11.04`; API smoke test passed for `POST /predict` and returned `predicted_next_week_visit_count=22.749` for (`Barcelona`, `casa_batllo`, `2023-01-09`).
