# Common tasks. Run them from an environment with the project installed (make install),
# or point PYTHON at one: make check PYTHON=/path/to/env/bin/python

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
NB_TIMEOUT ?= 1800
NOTEBOOKS ?= $(wildcard notebooks/*.ipynb)
TAILWIND = TAILWINDCSS_VERSION=v3.4.17 $(PYTHON) -m pytailwindcss -c tailwind.config.js \
	-i app/static/css/tailwind.input.css --minify

.PHONY: help install lint test test-browser notebooks-check notebooks css css-check check serve

help:
	@echo "make install          install the package with all extras (editable)"
	@echo "make lint             ruff"
	@echo "make test             pytest"
	@echo "make test-browser     drive the dashboard in Chrome (Playwright; not part of make check)"
	@echo "make notebooks-check  execute every notebook into a temp dir (stored outputs untouched)"
	@echo "make notebooks        re-execute every notebook in place, refreshing stored outputs"
	@echo "make css              rebuild app/static/css/app.css after changing classes in the templates"
	@echo "make css-check        fail if app.css is out of date with the templates"
	@echo "make check            lint + css-check + test + notebooks-check"
	@echo "make serve            run the dashboard on $(HOST):$(PORT)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

# Uses the installed Google Chrome, or Playwright's Chromium (python -m playwright install chromium).
test-browser:
	$(PYTHON) -m pytest -m browser tests/browser

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

# The dashboard's stylesheet is built from the classes used in app/templates (Tailwind CSS v3).
css:
	$(PYTHON) scripts/build_themes.py
	$(TAILWIND) -o app/static/css/app.css

css-check:
	@$(PYTHON) -c "import runpy, pathlib, sys; m = runpy.run_path('scripts/build_themes.py'); \
		sys.exit(0 if m['build']() == pathlib.Path('app/static/css/themes.css').read_text() else \
		print('app/static/css/themes.css is out of date: run make css') or 1)"
	@tmp=$$(mktemp); \
	$(TAILWIND) -o "$$tmp" > /dev/null 2>&1 || { rm -f "$$tmp"; echo "Tailwind build failed"; exit 1; }; \
	if cmp -s "$$tmp" app/static/css/app.css; then rm -f "$$tmp"; echo "app.css is up to date"; \
	else rm -f "$$tmp"; echo "app/static/css/app.css is out of date: run make css"; exit 1; fi

check: lint css-check test notebooks-check

# Bound to localhost by default; the API has no authentication. The graceful-shutdown
# timeout stops an open dashboard tab (a live event stream) from blocking shutdown.
serve:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --timeout-graceful-shutdown 5
