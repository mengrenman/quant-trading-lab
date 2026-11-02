# Common tasks. Run them from an environment with the project installed (make install),
# or point PYTHON at one: make check PYTHON=/path/to/env/bin/python

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
NB_TIMEOUT ?= 1800
NOTEBOOKS ?= $(wildcard notebooks/*.ipynb)

.PHONY: help install lint test notebooks-check notebooks check serve

help:
	@echo "make install          install the package with all extras (editable)"
	@echo "make lint             ruff"
	@echo "make test             pytest"
	@echo "make notebooks-check  execute every notebook into a temp dir (stored outputs untouched)"
	@echo "make notebooks        re-execute every notebook in place, refreshing stored outputs"
	@echo "make check            lint + test + notebooks-check"
	@echo "make serve            run the dashboard on $(HOST):$(PORT)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

notebooks-check:
	@tmp=$$(mktemp -d); status=0; \
	for nb in $(NOTEBOOKS); do \
		echo "executing $$nb"; \
		$(PYTHON) -m nbconvert --to notebook --execute --output-dir "$$tmp" \
			--ExecutePreprocessor.timeout=$(NB_TIMEOUT) "$$nb" > "$$tmp/log.txt" 2>&1 \
			|| { echo "FAILED: $$nb"; tail -20 "$$tmp/log.txt"; status=1; break; }; \
	done; rm -rf "$$tmp"; exit $$status

notebooks:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) $(NOTEBOOKS)

check: lint test notebooks-check

# Bound to localhost by default; the API has no authentication. The graceful-shutdown
# timeout stops an open dashboard tab (a live event stream) from blocking shutdown.
serve:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --timeout-graceful-shutdown 5
