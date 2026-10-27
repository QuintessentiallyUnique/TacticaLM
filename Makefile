PYTHON ?= python
TRAIN_DATA ?= data/evaluation/train_pilot.json
ADAPTER_OUTPUT ?= models/adapters/tacticalm-v2
EVAL_DATA ?= data/evaluation/dev/dev_eval.json
EVAL_OUTPUT ?= reports/evaluation
BASE_MODEL ?= Qwen/Qwen3-4B-Instruct-2507
ADAPTER ?= models/adapters/tacticalm-v2

.PHONY: install install-dev test lint format type-check clean download-data train evaluate serve-api serve-app

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest tests/

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache build/ dist/ *.egg-info

download-data:
	$(PYTHON) scripts/data_pipeline.py

train:
	$(PYTHON) scripts/run_training.py --data $(TRAIN_DATA) --output $(ADAPTER_OUTPUT) --model $(BASE_MODEL)

evaluate:
	$(PYTHON) scripts/run_evaluation.py --dataset $(EVAL_DATA) --base-model $(BASE_MODEL) --adapter $(ADAPTER) --output-dir $(EVAL_OUTPUT)

serve-api:
	uvicorn tacticalm.api.main:app --reload

serve-app:
	streamlit run app/streamlit_app.py
