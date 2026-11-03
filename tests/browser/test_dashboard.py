"""Drive the dashboard in Chrome: what the server-side tests cannot see (the page's own scripts).

Run with ``make test-browser``. Each test gets a fresh browser profile, so a fresh simulation.
"""

import re

import pytest

pytestmark = pytest.mark.browser

LIVE = "document.getElementById('connection-status').textContent === 'Live'"
EVENTS = "Number(document.getElementById('sim-stats-badge').textContent.match(/Events\\s+(\\d+)/)[1])"
RUNNING = "document.getElementById('sim-stats-badge').dataset.running === 'true'"
SPEED_STATE = """() => ({
    badge: document.getElementById('speed-display-badge').textContent.trim(),
    slider: Number(document.getElementById('sim-speed-slider').value),
    active: [...document.querySelectorAll('[data-speed-preset]')]
        .filter(b => b.getAttribute('aria-pressed') === 'true' || b.classList.contains('border-cyan-500'))
        .map(b => b.dataset.speedPreset),
})"""


def open_dashboard(page, base_url, width=1440, height=900):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(base_url + "/")
    page.wait_for_function(LIVE)
    return page


def settle(page):
    """Wait until htmx has finished swapping (new content is bound only after its settle delay)."""
    page.wait_for_function("!document.querySelector('.htmx-added, .htmx-settling, .htmx-swapping, .htmx-request')")


def run_for(page, ms, speed=500):
    page.evaluate(f"setSpeedPreset({speed})")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_timeout(ms)
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    page.wait_for_timeout(300)  # the last panel push


def test_page_works_offline_and_is_styled(page, base_url):
    requests = []
    page.on("request", lambda r: requests.append(r.url))
    open_dashboard(page, base_url)
    page.evaluate("document.fonts.ready.then(() => true)")
    external = [u for u in requests if not u.startswith((base_url, "data:"))]
    assert external == []
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(9, 13, 22)"  # app.css loaded
    assert page.evaluate("document.fonts.check('12px Inter') && document.fonts.check('12px \"JetBrains Mono\"')")
    assert page.evaluate("typeof htmx === 'object' && typeof LightweightCharts === 'object'")


def test_playback_buttons_and_keyboard_shortcuts(page, base_url):
    open_dashboard(page, base_url)
    assert page.evaluate(EVENTS) == 0
    page.keyboard.press("ArrowRight")
    page.wait_for_function(f"{EVENTS} === 1")
    page.keyboard.press("Shift+ArrowRight")
    page.wait_for_function(f"{EVENTS} === 11")

    page.keyboard.press("Space")
    page.wait_for_function(RUNNING)
    assert page.inner_text("#sim-state-label") == "Running"
    page.keyboard.press("Space")
    page.wait_for_function(f"!({RUNNING})")
    assert page.evaluate("scrollY") == 0  # Space did not scroll the page

    # Shortcuts are off while typing in a field.
    page.click("#order-price")
    before = page.evaluate(EVENTS)
    page.keyboard.press("ArrowRight")
    page.keyboard.press("Space")
    page.wait_for_timeout(500)
    assert page.evaluate(EVENTS) == before
    assert not page.evaluate(RUNNING)

    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    assert page.text_content("#sim-play-text").strip() == "Pause Sim"
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    page.click("button:has-text('+10')")
    page.wait_for_function(f"{EVENTS} > {before}")


def test_speed_slider_is_logarithmic_and_highlights_the_preset(page, base_url):
    open_dashboard(page, base_url)
    assert page.evaluate(SPEED_STATE) == {"badge": "50 events/s", "slider": 515, "active": ["50"]}
    assert page.evaluate("[0, 250, 500, 750, 1000].map(sliderToSpeed)") == [1, 7, 45, 299, 2000]

    page.click('[data-speed-preset="500"]')
    assert page.evaluate(SPEED_STATE) == {"badge": "500 events/s", "slider": 818, "active": ["500"]}

    page.eval_on_selector("#sim-speed-slider", """s => {
        s.value = 300;
        s.dispatchEvent(new Event('input'));
        s.dispatchEvent(new Event('change'));
    }""")
    # Position 300 is 10 events/s; on release the slider snaps to that speed's own position.
    assert page.evaluate(SPEED_STATE) == {"badge": "10 events/s", "slider": 303, "active": ["10"]}

    # The server has the new speed: another tab of the session shows it.
    other = open_dashboard(page.context.new_page(), base_url)
    assert other.evaluate(SPEED_STATE)["badge"] == "10 events/s"


def test_quick_prices_follow_the_book_and_orders_report_results(page, base_url):
    open_dashboard(page, base_url)
    run_for(page, 2000)
    bid, mid, ask = page.evaluate(
        "(() => { const d = document.getElementById('metrics-container').dataset;"
        " return [d.bestBid, d.mid, d.bestAsk].map(Number); })()")
    labels = page.locator("[data-quick] span").all_inner_texts()
    assert labels == [f"{bid:g}", f"{mid:g}", f"{ask:g}"]

    page.click('[data-quick="bid"]')
    assert page.input_value("#order-price") == f"{bid:g}"
    page.check("#limit-order-form input[value=ask]", force=True)
    page.click('[data-quick="mid"]')
    assert int(page.input_value("#order-price")) == -int(-mid // 1)  # a sell rounds the mid up

    # A resting buy at the best bid is reported and marked in the order book.
    page.check("#limit-order-form input[value=bid]", force=True)
    page.fill("#order-price", f"{bid:g}")
    page.fill("#limit-order-size", "7")
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('is resting in the book')")
    page.wait_for_function("document.getElementById('ladder-container').textContent.includes('you 7')")

    page.click("#tab-btn-market")
    page.click("#market-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('Bought 5 lots at')")

    page.click("#tab-btn-limit")
    page.eval_on_selector("#order-price", "i => { i.removeAttribute('min'); i.value = 0; }")
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('Order rejected')")

    # Canceling the order removes its marker.
    page.click("#active-orders-container button:has-text('Cancel')")
    page.wait_for_function("!document.getElementById('ladder-container').textContent.includes('you 7')")


def test_controls_stay_in_sync_across_tabs(page, base_url):
    a = open_dashboard(page, base_url)
    b = open_dashboard(page.context.new_page(), base_url)  # same profile, so the same session
    panel = "document.getElementById('model-params')"

    a.select_option("select[name=model]", "hawkes_zi")
    b.wait_for_function("document.querySelector('select[name=model]').value === 'hawkes_zi'"
                        f" && !!{panel}.querySelector('input[name=branching]')")

    settle(a)
    a.evaluate(f"{panel}.open = true")
    a.fill("#model-params input[name=mu]", "12")
    a.click("#model-params button[type=submit]")
    b.wait_for_function(f"{panel}.querySelector('input[name=mu]').value === '12.0'")
    settle(a)
    assert "Applied" in a.inner_text("#model-params summary")  # the tab that applied keeps its message
    assert not b.evaluate(f"{panel}.open")  # the other tab's panel stays closed

    a.click("#engine-toggle")
    b.wait_for_function("document.getElementById('engine-toggle').dataset.engine === 'pro_rata'")
    a.click('[data-speed-preset="1000"]')
    b.wait_for_function("document.getElementById('speed-display-badge').textContent.trim() === '1000 events/s'")


def test_a_parameter_form_being_edited_updates_once_focus_leaves(page, base_url):
    a = open_dashboard(page, base_url)
    b = open_dashboard(page.context.new_page(), base_url)
    panel = "document.getElementById('model-params')"
    b.evaluate(f"{panel}.open = true")
    b.fill("#model-params input[name=lam]", "3.5")  # B is typing

    settle(a)
    a.evaluate(f"{panel}.open = true")
    a.fill("#model-params input[name=delta]", "0.05")
    a.click("#model-params button[type=submit]")
    a.wait_for_function(f"{panel}.textContent.includes('Applied')")
    b.wait_for_timeout(500)
    assert b.input_value("#model-params input[name=lam]") == "3.5"  # B's typing is kept

    b.locator("#model-params input[name=lam]").blur()
    b.wait_for_function(f"{panel}.querySelector('input[name=delta]').value === '0.05'")


def test_fit_recenters_the_chart_after_dragging_the_price_axis(page, base_url):
    open_dashboard(page, base_url, height=1200)
    run_for(page, 3000)
    margins = """() => {
        const ts = priceChart.timeScale(), r = ts.getVisibleLogicalRange(), data = candleSeries.data();
        const from = Math.max(0, Math.ceil(r.from)), to = Math.min(data.length - 1, Math.floor(r.to));
        let hi = -Infinity, lo = Infinity;
        for (let i = from; i <= to; i++) { hi = Math.max(hi, data[i].high); lo = Math.min(lo, data[i].low); }
        const pane = document.getElementById('price-chart-container').clientHeight - ts.height();
        return {top: candleSeries.priceToCoordinate(hi), bottom: pane - candleSeries.priceToCoordinate(lo),
                autoScale: priceChart.priceScale('right').options().autoScale};
    }"""
    box = page.locator("#price-chart-container").bounding_box()
    axis_x = box["x"] + box["width"] - page.evaluate("priceChart.priceScale('right').width()") / 2
    y = box["y"] + box["height"] * 0.45
    page.mouse.move(axis_x, y)
    page.mouse.down()
    page.mouse.move(axis_x, y + 120, steps=10)
    page.mouse.up()
    dragged = page.evaluate(margins)
    assert not dragged["autoScale"]

    page.click("#chart-controls-toolbar button:has-text('Fit')")
    page.wait_for_timeout(300)
    fitted = page.evaluate(margins)
    assert fitted["autoScale"]
    assert abs(fitted["top"] - fitted["bottom"]) <= 2  # centered
    assert fitted["top"] >= 0 and fitted["bottom"] >= 0


def test_layout_fits_a_laptop_screen_and_a_phone(page, base_url):
    open_dashboard(page, base_url, 1440, 900)
    chart_bottom = page.evaluate("document.getElementById('price-chart-container').getBoundingClientRect().bottom")
    assert chart_bottom <= 900  # the whole price chart is visible without scrolling

    open_dashboard(page, base_url, 390, 844)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow == 0
    assert re.search(r"Reset Book", page.inner_text("#sim-controls-container"))


def fact_calls(page):
    return page.evaluate(
        "Object.fromEntries([...document.querySelectorAll('#facts-container [data-fact]')]"
        ".map(r => [r.dataset.fact, r.dataset.call]))")


def test_stylized_facts_panel_collects_a_sample_and_reads_it(page, base_url):
    open_dashboard(page, base_url)
    assert "Warming up" in page.inner_text("#facts-container")
    assert set(fact_calls(page).values()) == {"Collecting", "No"}  # placement is by design

    page.click("#facts-collect")
    page.wait_for_selector("#facts-collecting")
    page.wait_for_selector("#facts-collect", timeout=30_000)  # the buttons come back when it ends
    page.wait_for_function("!document.getElementById('facts-container').textContent.includes('Collecting')")
    calls = fact_calls(page)
    assert calls["Bid-ask bounce"] == "Yes"
    assert calls["Power-law relative prices"] == "No"
    assert "Collecting" not in calls.values()
    assert not page.evaluate(RUNNING)
    assert page.evaluate("candleSeries.data().length") > 100  # the chart reloaded with the new bars

    page.select_option("select[name=model]", "mike_farmer_2008")  # a new model starts a new sample
    page.wait_for_function("document.getElementById('facts-container').textContent.includes('Warming up')")
    assert fact_calls(page)["Power-law relative prices"] == "Yes"


def test_notebook_values_fill_the_parameter_form(page, base_url):
    open_dashboard(page, base_url)
    page.select_option("select[name=model]", "hawkes_zi")
    page.wait_for_function("!!document.querySelector('#model-params input[name=branching]')")
    settle(page)
    page.evaluate("document.getElementById('model-params').open = true")
    page.click("#model-params button:has-text(\"Use notebook 05's values\")")
    values = page.evaluate("Object.fromEntries(['mu', 'lam', 'delta', 'price_range', 'lot_size', 'branching', 'decay']"
                           ".map(n => [n, Number(document.querySelector(`#model-params input[name=${n}]`).value)]))")
    assert values == {"mu": 1.0, "lam": 0.5, "delta": 0.1, "price_range": 50, "lot_size": 1,
                      "branching": 0.8, "decay": 0.5}
    page.click("#model-params button[type=submit]")
    page.wait_for_function("document.getElementById('model-params').textContent.includes('Applied')")
    page.wait_for_function("document.getElementById('facts-container').textContent.includes('Warming up')")


def test_your_resting_orders_survive_the_model(page, base_url):
    open_dashboard(page, base_url)
    page.fill("#order-price", "20")  # far below the market, so it will not fill
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('is resting in the book')")
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('1 open')")

    page.click("#facts-collect")  # 400,000 model events, during which the model cancels thousands of orders
    page.wait_for_selector("#facts-collecting")
    page.wait_for_selector("#facts-collect", timeout=30_000)
    page.wait_for_timeout(500)
    assert "1 open" in page.inner_text("#active-orders-container")
