# Common tasks. Run them from an environment with the project installed (make install),
# or point PYTHON at one: make check PYTHON=/path/to/env/bin/python

PYTHON ?= python
HOST ?= 127.0.0.1
PORT ?= 8000
NB_TIMEOUT ?= 3600
NOTEBOOKS ?= $(wildcard notebooks/*.ipynb)
# Notebook 18 reads Nasdaq files (data/itch/, ignored by git) and takes about 50 minutes on five workers to run on them
# (a worker has reached 8 GB, so allow 40 GB), so it
# is never in `notebooks` (which re-executes in place: on a machine without the data that would replace its stored
# outputs with a line for each cell, and on one with it, the reported stocks' result outputs would be refreshed by a
# stray `make notebooks`). A real run of it is `make notebook-18`. `notebooks-check` leaves it out where the data is
# present and runs it (it finishes without error) where it is not.
NB18 = notebooks/18_real_books.ipynb
NB18_DATA ?= $(wildcard data/itch/derived/*/frames)
NB18_TIMEOUT ?= 10800
CHECKED = $(if $(NB18_DATA),$(filter-out $(NB18),$(NOTEBOOKS)),$(NOTEBOOKS))
# quant-plots, the backtest figures notebook 10 draws (a sibling repository, installed editable)
QUANT_PLOTS ?= ../../work/quant-plots
TAILWIND = TAILWINDCSS_VERSION=v3.4.17 $(PYTHON) -m pytailwindcss -c tailwind.config.js \
	-i app/static/css/tailwind.input.css --minify

.PHONY: help install lint test test-browser test-browsers notebooks-check notebooks notebook-18 css css-check check serve

help:
	@echo "make install          install the package with all extras, and quant-plots from QUANT_PLOTS (both editable)"
	@echo "make lint             ruff"
	@echo "make test             pytest"
	@echo "make test-browser     drive the dashboard in Chrome (Playwright; not part of make check)"
	@echo "make test-browsers    the same in Chrome, Firefox and WebKit (Safari's engine)"
	@echo "make notebooks-check  execute every notebook into a temp dir (stored outputs untouched)"
	@echo "make notebooks        re-execute every notebook but notebook 18 in place, refreshing stored outputs"
	@echo "make notebook-18      re-execute notebook 18 in place (needs data/itch/; about 50 minutes, five workers)"
	@echo "make css              rebuild app/static/css/app.css after changing classes in the templates"
	@echo "make css-check        fail if app.css is out of date with the templates"
	@echo "make check            lint + css-check + test + notebooks-check"
	@echo "make serve            run the dashboard on $(HOST):$(PORT)"

install:
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pip install -e "$(QUANT_PLOTS)"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

# Uses the installed Google Chrome, or Playwright's Chromium (python -m playwright install chromium).
test-browser:
	$(PYTHON) -m pytest -m browser tests/browser

test-browsers:
	LOB_BROWSERS=chrome,firefox,webkit $(PYTHON) -m pytest -m browser tests/browser

notebooks-check:
	@tmp=$$(mktemp -d); status=0; \
	for nb in $(CHECKED); do \
		echo "executing $$nb"; \
		$(PYTHON) -m nbconvert --to notebook --execute --output-dir "$$tmp" \
			--ExecutePreprocessor.timeout=$(NB_TIMEOUT) "$$nb" > "$$tmp/log.txt" 2>&1 \
			|| { echo "FAILED: $$nb"; tail -20 "$$tmp/log.txt"; status=1; break; }; \
	done; rm -rf "$$tmp"; exit $$status

notebooks:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT) $(filter-out $(NB18),$(NOTEBOOKS))

# A real run of notebook 18, in place: every cell is kept with its output, an error in a cell is recorded in the
# notebook and the run goes on (--allow-errors), and a cell may take three hours. Five workers, one thread each, as
# the runs whose outputs the notebook holds were made.
notebook-18:
	LOKY_MAX_CPU_COUNT=5 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 \
		$(PYTHON) -m nbconvert --to notebook --execute --inplace --allow-errors \
		--ExecutePreprocessor.timeout=$(NB18_TIMEOUT) $(NB18)

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
# timeout stops an open dashboard tab (a live event stream) from blocking shutdown. The
# keep-alive is longer than Safari keeps an idle connection (30 s): at uvicorn's default
# of 5 s Safari loses a request sent just as the server closes the connection.
serve:
	$(PYTHON) -m uvicorn app.main:app --host $(HOST) --port $(PORT) --timeout-graceful-shutdown 5 --timeout-keep-alive 75
