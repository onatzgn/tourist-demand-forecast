PYTHON=python

install:
	$(PYTHON) -m pip install -r requirements.txt

generate:
	$(PYTHON) -m src.data_generation.generate --rows 500000 --output_dir data/raw

sqlite:
	$(PYTHON) -m src.etl.build_sqlite --input_dir data/raw --db_path data/db.sqlite

features:
	$(PYTHON) -m src.etl.build_features --db_path data/db.sqlite --output_path data/processed/weekly_features.parquet

train:
	$(PYTHON) -m src.modeling.train --data_path data/processed/weekly_features.parquet --model_out models/model.pkl

evaluate:
	$(PYTHON) -m src.modeling.evaluate --data_path data/processed/weekly_features.parquet --model_path models/model.pkl

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest -q
