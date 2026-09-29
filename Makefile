.PHONY: install coverage-scan ingest features replay predict predict-test eval race-chart dashboard-db dashboard test lint fmt clean

install:
	uv sync --all-extras

coverage-scan:
	uv run fairlap-scan-coverage --seasons 2023 2024 2025 2026 --out data/coverage.csv

ingest:
	uv run fairlap-ingest-openf1 --seasons 2023 2024 2025 2026
	uv run fairlap-ingest-polymarket --seasons 2023 2024 2025 2026

features: ingest
	uv run fairlap-features

MODEL ?= gbm
replay:
	uv run fairlap-replay --session-key $(SESSION_KEY) --model $(MODEL) $(if $(SPEED),--speed $(SPEED))

predict:
	uv run fairlap-predict

# The single final evaluation: fit on 2023-24, score 2025-26. Phase 4 only.
predict-test:
	uv run fairlap-predict --holdout

eval: predict-test
	uv run fairlap-eval --backtest --out data

CHART_KEY = $(or $(SESSION_KEY),9947)
race-chart:
	uv run python -m fairlap.eval.race_chart --session-key $(CHART_KEY) --out data/race_$(CHART_KEY).png

# Replays every held-out race, then writes the small DB the hosted dashboard uses.
dashboard-db:
	uv run fairlap-dashboard-db

dashboard:
	uv run streamlit run dashboard/app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests dashboard experiments

fmt:
	uv run ruff format src tests dashboard experiments

clean:
	rm -rf data/duckdb/fairlap.duckdb
