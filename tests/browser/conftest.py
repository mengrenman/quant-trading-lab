"""Fixtures for the browser tests: a dashboard server and Chrome driven by Playwright.

These tests are marked ``browser`` and left out of ``make test``; run them with
``make test-browser``. They use the installed Google Chrome, or Playwright's own
Chromium if Chrome is missing (``python -m playwright install chromium``), and
skip when neither is available.

``LOB_BROWSERS`` picks other engines, comma-separated: ``chrome``, ``firefox``
and ``webkit`` (Safari's engine), each from ``python -m playwright install``.
``make test-browsers`` runs all three.
"""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url():
    """A dashboard server of its own, on a free port, for the whole test session."""
    port = _free_port()
    proc = subprocess.Popen(
        # The keep-alive is `make serve`'s: at uvicorn's default of 5 seconds Safari's engine loses a request that it
        # sends just as the server closes the idle connection (it does not send it again, as Chrome and Firefox do).
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--timeout-graceful-shutdown", "2", "--timeout-keep-alive", "75", "--log-level", "warning"],
        cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while True:
        try:
            urllib.request.urlopen(url + "/api/snapshot", timeout=1)
            break
        except OSError:
            if proc.poll() is not None or time.time() > deadline:
                proc.kill()
                pytest.fail("the dashboard server did not start")
            time.sleep(0.2)
    yield url
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


ENGINES = ("chrome", "firefox", "webkit")


def pytest_generate_tests(metafunc):
    """Run each browser test once per engine named in LOB_BROWSERS (Chrome by default)."""
    if "engine" in metafunc.fixturenames:
        engines = [e.strip() for e in os.environ.get("LOB_BROWSERS", "chrome").split(",") if e.strip()]
        unknown = set(engines) - set(ENGINES)
        if unknown:
            raise pytest.UsageError(f"LOB_BROWSERS: unknown engines {sorted(unknown)}; choose from {ENGINES}")
        metafunc.parametrize("engine", engines, scope="session")


@pytest.fixture(scope="session")
def playwright():
    sync_api = pytest.importorskip("playwright.sync_api", reason="install playwright (make install)")
    instance = sync_api.sync_playwright().start()
    yield instance
    instance.stop()


@pytest.fixture(scope="session")
def browser(playwright, engine):
    try:
        if engine == "chrome":
            try:
                browser = playwright.chromium.launch(channel="chrome")
            except Exception:
                browser = playwright.chromium.launch()
        else:
            browser = getattr(playwright, engine).launch()
    except Exception as exc:
        pytest.skip(f"no {engine} for Playwright ({exc}); see python -m playwright install")
    yield browser
    browser.close()


@pytest.fixture
def context(browser, engine):
    """A fresh browser profile, so a fresh dashboard session; fails the test on any page error."""
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    errors = []

    def on_console(msg):
        # A 422 is the server rejecting input, which the page reports in place (each engine words it its own way).
        if msg.type == "error" and "422" not in msg.text:
            errors.append(msg.text)

    context.on("console", on_console)
    context.on("weberror", lambda err: errors.append(str(err.error)))
    yield context
    context.close()
    assert not errors, f"errors in the page: {errors}"


@pytest.fixture
def page(context):
    return context.new_page()
