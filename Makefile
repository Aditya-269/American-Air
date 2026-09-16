PYTHON ?= python3

.PHONY: help install test benchmark ingest index golden clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install python dependencies"
	@echo "  make test       - Run all unit and integration tests"
	@echo "  make benchmark  - Run full benchmark evaluation across all 3 systems"
	@echo "  make ingest     - Ingest AmericanAir threads from data/raw"
	@echo "  make index      - Rebuild intent centroids and grounding index"
	@echo "  make golden     - Rebuild golden set and human audit set"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/test_pipeline.py -v

benchmark:
	$(PYTHON) -m src.eval_harness

ingest:
	$(PYTHON) src/ingest.py

index:
	$(PYTHON) src/intents.py
	$(PYTHON) src/retrieval.py

golden:
	$(PYTHON) scripts/build_golden_set.py
	$(PYTHON) scripts/create_human_audit.py

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
