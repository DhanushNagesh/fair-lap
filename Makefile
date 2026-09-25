.PHONY: install coverage-scan ingest features replay eval dashboard test lint fmt clean

install:
	uv sync --all-extras

coverage-scan:
	uv run fairlap-scan-coverage --seasons 2023 2024 2025 2026 --out data/coverage.csv

ingest:
	uv run fairlap-ingest-openf1 --seasons 2023 2024 2025 2026
	uv run fairlap-ingest-polymarket --seasons 2023 2024 2025 2026

features: ingest
	uv run fairlap-features

replay:
	uv run fairlap-replay --session-key $(SESSION_KEY)

eval:
	uv run fairlap-eval

dashboard:
	uv run streamlit run dashboard/app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests dashboard

fmt:
	uv run ruff format src tests dashboard

clean:
	rm -rf data/duckdb/fairlap.duckdb
