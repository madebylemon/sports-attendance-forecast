.PHONY: help backfill refresh features train forecast pipeline test test-coverage dashboard render docker-build docker-run mlflow-ui format lint clean

help:
	@echo "sports-attendance-forecast — available commands:"
	@echo ""
	@echo "Data pipeline:"
	@echo "  make backfill          Backfill 5 seasons of historical data"
	@echo "  make refresh           Daily incremental data refresh"
	@echo "  make features          Engineer features from raw data"
	@echo "  make train             Train/update model and log to MLflow"
	@echo "  make forecast          Generate predictions for upcoming fixtures"
	@echo "  make pipeline          Run refresh → features → train → forecast"
	@echo ""
	@echo "Testing & quality:"
	@echo "  make test              Run pytest on all tests"
	@echo "  make test-coverage     Run pytest with coverage report (HTML)"
	@echo "  make lint              Run ruff checks"
	@echo "  make format            Format code with ruff"
	@echo ""
	@echo "Dashboard & visualization:"
	@echo "  make dashboard         Preview Quarto dashboard locally"
	@echo "  make render            Render Quarto dashboard to static HTML"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make docker-build      Build Docker image"
	@echo "  make docker-run        Run pipeline in Docker container"
	@echo "  make mlflow-ui         Start MLflow UI (http://localhost:5000)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             Remove __pycache__, .pytest_cache, etc."

backfill:
	python -m pipeline.backfill

refresh:
	python -m pipeline.refresh

features:
	python -m pipeline.features

train:
	python -m pipeline.train

forecast:
	python -m pipeline.forecast

pipeline: refresh features train forecast

test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ -v --cov=pipeline --cov-report=html --cov-report=term-missing

dashboard:
	quarto preview dashboard/

render:
	quarto render dashboard/

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-run:
	docker compose -f docker/docker-compose.yml run pipeline bash

mlflow-ui:
	mlflow ui --backend-store-uri mlruns/ --port 5000

format:
	ruff format pipeline/ tests/

lint:
	ruff check pipeline/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/ .coverage
