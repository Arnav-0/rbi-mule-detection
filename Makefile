.PHONY: setup data features features-fast train train-quick evaluate explain fairness temporal \
        serve dashboard submit test test-data test-features lint clean all \
        docker-build docker-up docker-down

setup:
	pip install -e . --break-system-packages

data:
	python -m src.data.loader && python -m src.data.validator

features:
	python -m src.features.pipeline

features-fast:
	python -m src.features.pipeline --skip-graph

train:
	python -m src.models.trainer --all-models

train-quick:
	python -m src.models.trainer --model lightgbm --optuna-trials 20

evaluate:
	python -m src.models.evaluator --all-models

explain:
	python -m src.explainability.shap_explainer

fairness:
	python -m src.explainability.fairness

temporal:
	python -m src.temporal.window_detector

serve:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run frontend/app.py

submit:
	python -m src.models.trainer --predict-test --output outputs/predictions/submission.csv

test:
	pytest tests/ -v --tb=short

test-data:
	pytest tests/test_data/ -v

test-features:
	pytest tests/test_features/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf data/processed/* outputs/* __pycache__

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

all: setup data features train evaluate explain fairness temporal submit
