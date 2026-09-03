# FraudPulse — one entry point per phase. Every target is idempotent.
PY := .venv/bin/python
PIP := .venv/bin/pip
VENV := .venv

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps smoke data prepare test lint fmt clean \
        topic produce land features parity train serve loadtest drift nuke \
        build-offline hpo-compare verify e2e status rebuild-state serve-bg \
        serve-stop reset-stream

help: ## show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
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
topic: ## create the transactions topic if absent
	$(PY) -m fraudpulse.cli topic

reset-stream: ## wipe the topic, the landing zone and the online store
	$(PY) -m fraudpulse.cli topic --recreate
	rm -rf data/landing/*
	-@docker exec fp-redis redis-cli flushall

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

rebuild-state: ## PHASE 2 verify: rebuild the streaming state from the landing zone
	$(PY) scripts/rebuild_state.py

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

# `e2e` used to stop at `train`, which meant `verify` had no fresh latency or
# drift report to read and fell back to whatever was committed. It now runs the
# steps that produce those reports, including starting and stopping the API,
# so a full pass is earned on this machine rather than inherited from the repo.
e2e: ## full pipeline from nothing, ending in a verify that measured everything itself
	$(MAKE) up prepare
	$(MAKE) reset-stream
	$(MAKE) produce land build-offline features
	$(MAKE) parity rebuild-state train
	$(MAKE) serve-bg
	$(MAKE) loadtest || ($(MAKE) serve-stop && false)
	$(MAKE) serve-stop
	$(MAKE) drift
	$(MAKE) verify

serve-bg: ## start the API in the background and wait for /health
	@$(PY) -m uvicorn fraudpulse.serving.app:app --host 0.0.0.0 --port 8000 \
	  --workers 4 --log-level warning & echo $$! > .uvicorn.pid
	@echo "waiting for the api ..."
	@for i in $$(seq 1 40); do \
	  curl -sf http://localhost:8000/health >/dev/null 2>&1 && \
	    { echo "api up"; exit 0; }; \
	  sleep 3; \
	done; echo "api did not become healthy"; exit 1

serve-stop: ## stop the background API
	-@pkill -f "uvicorn fraudpulse.serving.app" 2>/dev/null || true
	-@rm -f .uvicorn.pid

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
