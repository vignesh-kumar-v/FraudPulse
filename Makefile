# FraudPulse — one entry point per phase. Every target is idempotent.
PY := .venv/bin/python
PIP := .venv/bin/pip
VENV := .venv

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps smoke data prepare test lint fmt clean \
        topic produce land features parity train serve loadtest drift nuke \
        build-offline hpo-compare verify e2e status

help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- environment
setup: ## create .venv and install everything
	python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip setuptools wheel
	$(PIP) install -q -e ".[dev,stretch]"
	@echo "ready: source $(VENV)/bin/activate"

up: ## start redpanda + redis + mlflow + console
	docker compose up -d
	@docker compose ps --format "table {{.Name}}\t{{.Status}}"

down: ## stop the stack (keeps volumes)
	docker compose down

nuke: ## stop the stack and delete all volumes
	docker compose down -v

ps: ## service status
	@docker compose ps

logs: ## tail all service logs
	docker compose logs -f --tail=50

smoke: ## PHASE 0 verify: kafka round-trip from the host
	$(PY) scripts/smoke_kafka.py

# ---------------------------------------------------------------------- data
data: ## download IEEE-CIS train_transaction.csv from kaggle
	$(PY) -m kaggle competitions download -c ieee-fraud-detection \
	  -f train_transaction.csv -p data/raw --force
	cd data/raw && unzip -o train_transaction.csv.zip && rm -f train_transaction.csv.zip

prepare: ## build data/processed/events.parquet from the raw csv
	$(PY) -m fraudpulse.cli prepare

# ----------------------------------------------------------------- streaming
topic: ## (re)create the transactions topic
	$(PY) -m fraudpulse.cli topic

produce: ## PHASE 1: replay the dataset onto kafka
	$(PY) -m fraudpulse.cli produce

land: ## PHASE 1: consume raw events into the parquet landing zone
	$(PY) -m fraudpulse.cli land

features: ## PHASE 2: consume events, compute online features, push to redis
	$(PY) -m fraudpulse.cli features

# ------------------------------------------------------------------ features
build-offline: ## PHASE 2: compute offline features and register the feast repo
	$(PY) -m fraudpulse.cli build-offline

parity: ## PHASE 2 verify: offline vs online feature parity, in-process and e2e
	$(PY) -m fraudpulse.cli parity
	$(PY) scripts/verify_parity.py

# ------------------------------------------------------------------ modelling
train: ## PHASE 3: build training set, tune, train, register in mlflow
	$(PY) -m fraudpulse.cli train

serve: ## PHASE 3: run the inference API on :8000
	$(PY) -m uvicorn fraudpulse.serving.app:app --host 0.0.0.0 --port 8000

loadtest: ## PHASE 3 verify: measure p50/p95/p99 inference latency
	$(PY) -m fraudpulse.cli loadtest --n 2000 --concurrency 1
	$(PY) -m fraudpulse.cli loadtest --n 8000 --concurrency 8 --processes 4
	$(PY) -m fraudpulse.cli loadtest --n 2000 --concurrency 1 --explain

status: ## show what exists so far
	$(PY) -m fraudpulse.cli status

# ----------------------------------------------------------------- monitoring
drift: ## PHASE 4 verify: run the drift monitor against an injected shift
	$(PY) -m fraudpulse.cli drift

hpo-compare: ## PHASE 5 verify (stretch): optuna vs ray tune wall-clock
	$(PY) -m fraudpulse.cli hpo-compare

verify: ## run every phase's verification gate and print a pass/fail table
	$(PY) -m fraudpulse.cli verify-all

e2e: up topic prepare produce land build-offline features parity train verify ## full pipeline from nothing

# ----------------------------------------------------------------------- dev
test: ## run the test suite (no docker, no dataset required)
	$(PY) -m pytest -q

lint: ## ruff check
	$(VENV)/bin/ruff check src tests scripts

fmt: ## ruff format + fix
	$(VENV)/bin/ruff format src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

clean: ## remove generated artifacts (keeps raw data)
	rm -rf data/landing/* data/processed/* reports/*.json reports/*.html \
	       feature_repo/data/* .pytest_cache .ruff_cache
