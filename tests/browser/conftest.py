"""Fixtures for the browser tests: a dashboard server and Chrome driven by Playwright.

These tests are marked ``browser`` and left out of ``make test``; run them with
``make test-browser``. They use the installed Google Chrome, or Playwright's own
Chromium if Chrome is missing (``python -m playwright install chromium``), and
skip when neither is available.
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
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port),
         "--timeout-graceful-shutdown", "2", "--log-level", "warning"],
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


@pytest.fixture(scope="session")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api", reason="install playwright (make install)")
    playwright = sync_api.sync_playwright().start()
    try:
        try:
            browser = playwright.chromium.launch(channel="chrome")
        except Exception:
            try:
                browser = playwright.chromium.launch()
            except Exception as exc:
                pytest.skip(f"no Chrome or Chromium for Playwright: {exc}")
        yield browser
        browser.close()
    finally:
        playwright.stop()


@pytest.fixture
def context(browser):
    """A fresh browser profile, so a fresh dashboard session; fails the test on any page error."""
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    errors = []

    def on_console(msg):
        # A 422 is the server rejecting input, which the page reports in place.
        if msg.type == "error" and "status of 422" not in msg.text:
            errors.append(msg.text)

    context.on("console", on_console)
    context.on("weberror", lambda err: errors.append(str(err.error)))
    yield context
    context.close()
    assert not errors, f"errors in the page: {errors}"


@pytest.fixture
def page(context):
    return context.new_page()
