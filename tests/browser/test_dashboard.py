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
    page.click("#limit-order-form label:has(input[value=ask])")  # its visible label
    page.click('[data-quick="mid"]')
    assert int(page.input_value("#order-price")) == -int(-mid // 1)  # a sell rounds the mid up

    # A resting buy at the best bid is reported and marked in the order book.
    page.click("#limit-order-form label:has(input[value=bid])")  # its visible label
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
    page.click("#active-orders-container button:text-is('Cancel')")
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


VISIBLE_VIEWS = "['trade', 'analyze', 'learn'].filter(v => !document.getElementById('view-' + v).hidden)"


def test_views_switch_by_tab_and_key_and_are_remembered(page, base_url):
    open_dashboard(page, base_url)
    assert page.evaluate(VISIBLE_VIEWS) == ["trade"]
    assert page.evaluate("location.hash") == ""  # a plain address stays plain
    page.click("#tab-analyze")
    assert page.evaluate(VISIBLE_VIEWS) == ["analyze"]
    assert page.get_attribute("#tab-analyze", "aria-selected") == "true"
    assert page.evaluate("location.hash") == "#analyze"
    page.keyboard.press("3")
    assert page.evaluate(VISIBLE_VIEWS) == ["learn"]
    # In the tab list the arrow keys move between tabs (wrapping around) and do not step the simulation.
    page.focus("#tab-learn")
    page.keyboard.press("ArrowRight")
    assert page.evaluate(VISIBLE_VIEWS) == ["trade"]
    assert page.evaluate("document.activeElement.id") == "tab-trade"
    assert page.evaluate(EVENTS) == 0

    # This browser remembers the view; the chart, loaded while hidden, sizes itself once shown.
    page.click("#tab-learn")
    open_dashboard(page, base_url)
    assert page.evaluate(VISIBLE_VIEWS) == ["learn"]
    page.click("#tab-trade")
    page.wait_for_function("priceChart.options().width > 0 && priceChart.options().width"
                           " === document.getElementById('price-chart-container').clientWidth")
    # A link to a view opens it.
    page.goto(base_url + "/#analyze")
    page.wait_for_function(LIVE)
    assert page.evaluate(VISIBLE_VIEWS) == ["analyze"]


PANEL_BOXES = """() => Object.fromEntries(['chart-panel', 'order-form-container', 'active-orders-container',
    'ladder-container', 'trades-container'].map(id => {
        const r = document.getElementById(id).getBoundingClientRect();
        return [id, {left: r.left, right: r.right, top: r.top, bottom: r.bottom}];
    }))"""


def test_layout_uses_the_width_it_has(page, base_url):
    # A laptop: order entry and your orders in a column left of the chart, the book and the tape right of it.
    open_dashboard(page, base_url, 1440, 900)
    box = page.evaluate(PANEL_BOXES)
    assert box["order-form-container"]["right"] < box["chart-panel"]["left"] < box["chart-panel"]["right"] \
        < box["ladder-container"]["left"]
    assert box["active-orders-container"]["top"] > box["order-form-container"]["bottom"]
    assert box["trades-container"]["top"] > box["ladder-container"]["bottom"]  # the tape under the book
    assert box["chart-panel"]["top"] == box["order-form-container"]["top"] == box["ladder-container"]["top"]
    chart_bottom = page.evaluate("document.getElementById('volume-chart-container').getBoundingClientRect().bottom")
    assert chart_bottom <= 900  # both charts are visible without scrolling

    # A wide screen: the tape gets a column of its own.
    open_dashboard(page, base_url, 1728, 1000)
    box = page.evaluate(PANEL_BOXES)
    assert box["trades-container"]["left"] > box["ladder-container"]["right"]
    assert box["trades-container"]["top"] == box["ladder-container"]["top"]

    # Narrower: order entry goes under the chart.
    open_dashboard(page, base_url, 1100, 800)
    box = page.evaluate(PANEL_BOXES)
    assert box["order-form-container"]["top"] > box["chart-panel"]["bottom"]

    open_dashboard(page, base_url, 390, 844)
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow == 0
    assert re.search(r"Reset Book", page.inner_text("#sim-controls-container"))


def test_text_size_scales_the_page_and_is_remembered(page, base_url):
    open_dashboard(page, base_url)
    root = "getComputedStyle(document.documentElement).fontSize"
    chart_font = "priceChart.options().layout.fontSize"
    assert page.evaluate(root) == "12px" and page.evaluate(chart_font) == 8  # Small, the default
    page.select_option("#text-select", "large")
    assert page.evaluate(root) == "16px" and page.evaluate(chart_font) == 11
    price_box = "document.getElementById('price-chart-container')"
    page.wait_for_function(f"priceChart.options().height === {price_box}.clientHeight"
                           f" && {price_box}.clientHeight === 480")  # 30 rem
    open_dashboard(page, base_url)
    assert page.input_value("#text-select") == "large" and page.evaluate(root) == "16px"
    page.select_option("#text-select", "medium")
    assert page.evaluate(root) == "14px"


def fact_calls(page):
    return page.evaluate(
        "Object.fromEntries([...document.querySelectorAll('#facts-container [data-fact]')]"
        ".map(r => [r.dataset.fact, r.dataset.call]))")


def test_stylized_facts_panel_collects_a_sample_and_reads_it(page, base_url):
    open_dashboard(page, base_url)
    page.click("#tab-analyze")
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
    # The chart's session VWAP line follows the trades.
    assert page.evaluate("vwapSeries.data().length") > 100
    assert page.inner_text("#hud-vwap") != "—"
    # The same sample feeds the order-flow card: its scatter and fitted slope.
    page.wait_for_function("document.querySelectorAll('#ofi-container circle').length > 50")
    assert "ticks per lot" in page.inner_text("#ofi-container")

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

    page.click("#tab-analyze")
    page.click("#facts-collect")  # 400,000 model events, during which the model cancels thousands of orders
    page.wait_for_selector("#facts-collecting")
    page.wait_for_selector("#facts-collect", timeout=30_000)
    page.wait_for_timeout(500)
    page.click("#tab-trade")
    assert "1 open" in page.inner_text("#active-orders-container")


def test_a_large_market_order_walks_its_protection_and_the_rest_waits(page, base_url):
    open_dashboard(page, base_url)
    best_ask = int(page.evaluate("document.getElementById('metrics-container').dataset.bestAsk"))
    page.click("#tab-btn-market")
    page.fill("#market-order-size", "1000000")
    page.fill("#market-protection", "3")
    page.click("#market-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('Partly filled')")
    result = page.inner_text("#order-result")
    assert f"rest as a buy order at {best_ask + 3}, the protection limit" in result
    assert "of 1,000,000 lots" in result
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('1 open')")
    # The rest is now the best bid, at the top of the bid side, marked as yours.
    page.wait_for_function(f"document.getElementById('metrics-container').dataset.bestBid === '{best_ask + 3}'")
    assert "you " in page.inner_text("#dom-bids-section > div:first-child")


def test_an_iceberg_from_the_form_and_the_downloads(page, base_url):
    open_dashboard(page, base_url)
    best_bid = page.evaluate("document.getElementById('metrics-container').dataset.bestBid")
    page.fill("#order-price", best_bid)
    page.fill("#limit-order-size", "100")
    page.fill("#limit-order-display", "10")
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('showing 10 at a time')")
    page.wait_for_function("document.getElementById('ladder-container').textContent.includes('+90 hidden')")
    assert "+90" in page.inner_text("#active-orders-container")

    with page.expect_download() as download:
        page.click("a[href='/api/export/trades.csv']")
    assert download.value.suggested_filename == "lob-trades.csv"
    with page.expect_download() as download:
        page.click("a[href='/api/export/bars.parquet']")
    assert download.value.suggested_filename == "lob-bars-2s.parquet"


def test_a_fill_of_your_resting_order_shows_a_notice(page, base_url):
    open_dashboard(page, base_url)
    inside = int(page.evaluate("document.getElementById('metrics-container').dataset.bestBid")) + 1
    page.fill("#order-price", str(inside))  # a bid inside the spread is the first a sell meets
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('1 open')")
    page.click("#tab-btn-market")
    page.click("#market-order-form label:has(input[value=ask])")  # its visible label
    page.click("#market-order-form button[type=submit]")  # 5 lots, against your own bid
    page.wait_for_function("document.getElementById('toasts').textContent.includes('now filled in full')")
    assert f"filled 5 lots at {inside}" in page.inner_text("#toasts")
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('Recent fills')")


def test_a_split_order_runs_and_reports_from_the_page(page, base_url):
    open_dashboard(page, base_url)
    page.click("#tab-btn-market")
    page.fill("#market-order-size", "60")
    page.fill("#market-slices", "3")
    page.fill("#market-interval", "1")
    page.click("#market-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('order-result').textContent.includes('Splitting a buy of 60 lots')")
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('1 sent')")
    page.evaluate("setSpeedPreset(500)")
    page.click("#sim-play-btn")
    page.wait_for_function("document.getElementById('toasts').textContent.includes('Split buy done')", timeout=15_000)
    page.click("#sim-play-btn")
    assert "One market order at the start would have bought" in page.inner_text("#toasts")
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('Split buy done')")


def test_themes_recolor_the_page_and_the_chart_and_are_remembered(page, base_url):
    open_dashboard(page, base_url)
    background = "getComputedStyle(document.body).backgroundColor"
    up = "candleSeries.options().upColor"
    assert page.evaluate(background) == "rgb(9, 13, 22)"
    assert page.evaluate(up) == "rgb(16, 185, 129)"  # emerald-500
    page.select_option("#theme-select", "daylight")
    assert page.evaluate(background) == "rgb(241, 245, 249)"
    assert page.evaluate(up) == "rgb(4, 120, 87)"  # darker on a light background (emerald-700)
    page.click("#palette-toggle")
    assert page.get_attribute("#palette-toggle", "aria-pressed") == "true"
    assert page.evaluate(up) == "rgb(29, 78, 216)"  # buying in blue (blue-700)

    open_dashboard(page, base_url)  # remembered by this browser, and applied before the page paints
    assert page.evaluate("[document.documentElement.dataset.theme, document.documentElement.dataset.palette]") == [
        "daylight", "colorblind"]
    assert page.input_value("#theme-select") == "daylight"
    assert page.evaluate(up) == "rgb(29, 78, 216)"
    page.select_option("#theme-select", "terminal")
    page.click("#palette-toggle")
    assert page.evaluate(background) == "rgb(0, 0, 0)"
    family = page.evaluate("getComputedStyle(document.body).fontFamily")
    assert family.replace('"', '').startswith("JetBrains Mono")  # WebKit leaves the quotes out


def test_your_orders_and_fills_show_on_the_chart(page, base_url):
    open_dashboard(page, base_url)
    lines = "[...ownLines.values()].map(l => l.options().title)"
    markers = "candleSeries.markers().map(m => m.text)"
    legend_hidden = "document.getElementById('own-legend').classList.contains('hidden')"
    assert page.evaluate(lines) == [] and page.evaluate(markers) == [] and page.evaluate(legend_hidden)

    quotes = "document.getElementById('metrics-container').dataset"
    bid, ask = page.evaluate(f"[{quotes}.bestBid, {quotes}.bestAsk].map(Number)")
    page.fill("#order-price", str(bid - 3))
    page.fill("#limit-order-size", "40")
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function(f"{lines}.includes('you buy 40')")
    assert page.evaluate("[...ownLines.values()][0].options().price") == bid - 3

    page.click("#tab-btn-market")
    page.click("#market-order-form button[type=submit]")  # 5 lots at the best ask
    page.wait_for_function(f"{markers}.includes('Buy 5 @ {ask}')")
    assert not page.evaluate(legend_hidden)

    # Canceling the order removes its line; the marker stays, since the fill happened.
    page.click("#active-orders-container button:text-is('Cancel')")
    page.wait_for_function(f"{lines}.length === 0")
    assert page.evaluate(markers) == [f"Buy 5 @ {ask}"]


BID_LEVELS = "[...document.querySelectorAll('#dom-bids-section [data-price]')].map(r => Number(r.dataset.price))"


def center_of(page, selector):
    box = page.locator(selector).first.bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def show_bids(page):
    """Scroll the bid levels to the middle of the window (the sticky header covers the top)."""
    page.evaluate("document.getElementById('dom-bids-section').scrollIntoView({block: 'center'})")


def test_one_click_trading_from_the_order_book(page, base_url):
    open_dashboard(page, base_url)
    page.fill("#limit-order-size", "12")
    show_bids(page)
    row = "#dom-bids-section [data-price='97']"
    page.mouse.click(*center_of(page, row))  # off: a click copies the price
    assert page.input_value("#order-price") == "97"
    assert "you " not in page.inner_text("#dom-bids-section")

    page.click("#one-click-toggle")
    assert page.get_attribute("#one-click-toggle", "aria-pressed") == "true"
    show_bids(page)
    page.evaluate("setSpeedPreset(500)")
    page.click("#sim-play-btn")  # levels come and go while you aim, but the book holds still
    page.wait_for_function(RUNNING)
    target = page.evaluate(BID_LEVELS)[3]
    x, y = center_of(page, f"#dom-bids-section [data-price='{target}']")
    page.mouse.move(x, y)
    page.wait_for_function("window.ladderHeld === true")
    assert page.inner_text("#ladder-hint") == f"Buy 12 at {target}"
    page.wait_for_timeout(600)  # several pushes later the same level is under the pointer
    assert page.evaluate(f"document.elementFromPoint({x}, {y}).closest('[data-price]').dataset.price") == str(target)
    page.mouse.click(x, y)
    page.wait_for_function(f"document.getElementById('toasts').textContent.includes('for 12 lots at {target}')")
    page.mouse.move(5, 400)  # off the book: it runs freely again
    page.wait_for_function("window.ladderHeld === false")
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    assert page.get_attribute("#one-click-toggle", "aria-pressed") == "true"  # kept across pushes
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('1 open')")


def test_dragging_your_badge_moves_your_orders(page, base_url):
    open_dashboard(page, base_url)
    page.fill("#order-price", "97")
    page.fill("#limit-order-size", "12")
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_selector("#dom-bids-section [data-own]")
    show_bids(page)
    levels = page.evaluate(BID_LEVELS)
    to_price = levels[levels.index(97) + 2]
    page.mouse.move(*center_of(page, "#dom-bids-section [data-own]"))
    page.mouse.down()
    # Over an ask level, the hint warns that the move would trade at once.
    best_ask = page.evaluate("document.getElementById('metrics-container').dataset.bestAsk")
    page.mouse.move(*center_of(page, f"#dom-asks-section [data-price='{best_ask}']"), steps=8)
    assert page.inner_text("#ladder-hint").endswith("crosses the spread and trades now")
    page.mouse.move(*center_of(page, f"#dom-bids-section [data-price='{to_price}']"), steps=8)
    assert page.inner_text("#ladder-hint") == f"Move your buy orders (12 lots) to {to_price}"
    assert page.evaluate("document.querySelectorAll('#ladder-container .drop-target').length") == 1
    page.mouse.up()
    page.wait_for_function("document.getElementById('toasts').textContent.includes("
                           f"'Moved your buy order (12 lots) from 97 to {to_price}')")
    page.wait_for_function(f"document.querySelector('#dom-bids-section [data-own]').closest('[data-price]')"
                           f".dataset.price === '{to_price}'")
    assert not page.evaluate("window.ladderHeld")
    assert page.evaluate("document.querySelectorAll('#ladder-container .drop-target').length") == 0


PAINTED = """() => {
    const c = document.getElementById('heatmap-canvas');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i] + d[i + 1] + d[i + 2] > 150) n++;
    return n;
}"""


def test_the_heatmap_draws_the_book_over_time(page, base_url):
    open_dashboard(page, base_url)
    run_for(page, 1500)
    assert page.evaluate("window.heatmapColumns().length") == 0  # loaded when first shown
    page.click("#tab-analyze")
    page.wait_for_function("window.heatmapColumns().length > 5")
    page.wait_for_function("document.getElementById('heatmap-span').textContent.startsWith('Time 0 to')")
    assert page.evaluate(PAINTED) > 1000

    n = page.evaluate("window.heatmapColumns().length")
    page.click("#sim-play-btn")  # the stream brings the new columns
    page.wait_for_function(f"window.heatmapColumns().length > {n + 3}")
    page.click("#sim-play-btn")
    times = page.evaluate("window.heatmapColumns().map(c => c.t)")
    assert times == sorted(set(times))

    page.click("#heatmap-hidden")
    assert page.get_attribute("#heatmap-hidden", "aria-pressed") == "true"
    box = page.locator("#heatmap-canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] / 2)
    page.wait_for_function("!document.getElementById('heatmap-tip').hidden")
    assert re.match(r"Time \d+, price \d+: ", page.inner_text("#heatmap-tip"))


STEP_DONE = "document.getElementById('lesson-next').dataset.done === 'true'"


def test_a_guided_lesson_runs_and_resumes(page, base_url):
    open_dashboard(page, base_url)
    page.click("#tab-learn")
    assert page.evaluate("[...document.querySelectorAll('#lesson-catalog [data-lesson]')].map(c => c.dataset.lesson)") \
        == ["impact", "walls", "icebergs", "ofi", "split", "market-making", "strategies"]
    page.click("[data-lesson='walls'] button")
    assert page.evaluate(VISIBLE_VIEWS) == ["trade"]  # the view its first step is about
    assert page.inner_text("#lesson-progress") == "Step 1 of 6"
    page.click("#lesson-do")  # pause
    page.wait_for_function(STEP_DONE)

    page.click("#lesson-next")
    bid = int(page.evaluate("document.getElementById('metrics-container').dataset.bestBid"))
    assert f"a buy of 300 lots at {bid - 3}" in page.inner_text("#lesson-text")
    page.click("#lesson-do")  # place the wall
    page.wait_for_function(STEP_DONE)
    assert "is resting in the book" in page.inner_text("#lesson-result")
    page.wait_for_function(f"document.querySelector('#dom-bids-section [data-price=\"{bid - 3}\"] [data-own]')")

    page.click("#lesson-next")
    page.click("#lesson-show")  # outlines the depth chart
    page.wait_for_function("!document.getElementById('lesson-spotlight').hidden")
    spot = page.locator("#lesson-spotlight").bounding_box()
    chart = page.locator("#depth-chart").bounding_box()
    assert abs(spot["x"] + 4 - chart["x"]) < 2 and abs(spot["width"] - 8 - chart["width"]) < 2

    open_dashboard(page, base_url)  # this browser keeps your place
    assert page.inner_text("#lesson-progress") == "Step 3 of 6"
    for _ in range(3):
        page.click("#lesson-next")
    assert page.inner_text("#lesson-progress") == "Step 6 of 6"
    page.click("#lesson-do")  # cancel the wall
    page.wait_for_function(STEP_DONE)
    assert page.inner_text("#lesson-result") == "Canceled 1 order."
    page.click("#lesson-next")  # Finish
    assert page.evaluate("document.getElementById('lesson-dock').hidden")
    page.click("#tab-learn")
    assert "✓ done" in page.inner_text("[data-lesson='walls']")
    assert page.inner_text("#lessons-done") == "1 of 7 done"


def test_the_heatmap_recovers_when_it_starts_over_during_its_load(page, base_url):
    open_dashboard(page, base_url)
    page.click("#tab-analyze")  # starts loading the history...
    # ...and a push from a heatmap that has started over arrives meanwhile.
    page.evaluate("document.dispatchEvent(new CustomEvent('lob:heatmap', {detail: {version: -1, columns: []}}))")
    page.wait_for_function("window.heatmapColumns().length >= 1")
    n = page.evaluate("window.heatmapColumns().length")
    page.evaluate("setSpeedPreset(500)")
    page.click("#sim-play-btn")
    page.wait_for_function(f"window.heatmapColumns().length > {n + 3}")  # still updating
    page.click("#sim-play-btn")
    times = page.evaluate("window.heatmapColumns().map(c => c.t)")
    assert times == list(range(times[0], times[0] + len(times)))  # one column per unit, no gaps


def test_the_iceberg_lesson_waits_for_the_iceberg_itself_to_trade(page, base_url):
    open_dashboard(page, base_url)
    inside = int(page.evaluate("document.getElementById('metrics-container').dataset.bestAsk")) - 1
    page.click("#limit-order-form label:has(input[value=ask])")  # its visible label
    page.fill("#order-price", str(inside))  # a sell inside the spread...
    page.click("#limit-order-form button[type=submit]")
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('1 open')")
    page.click("#tab-btn-market")
    page.click("#market-order-form button[type=submit]")  # ...that your own buy fills
    page.wait_for_function("document.getElementById('active-orders-container').textContent.includes('Your sell order')")

    page.evaluate("startLesson('icebergs')")
    page.click("#lesson-do")  # pause
    page.click("#lesson-next")
    page.click("#lesson-do")  # place the iceberg
    page.wait_for_function(STEP_DONE)
    page.click("#lesson-next")
    page.wait_for_timeout(1200)  # the old sell fill is in the list, but this step is not done
    assert page.evaluate("document.getElementById('lesson-next').dataset.done") == "false"


def test_your_pnl_on_the_page_and_under_the_chart(page, base_url):
    open_dashboard(page, base_url)
    assert page.evaluate("document.getElementById('pnl-chart-panel').classList.contains('hidden')")
    assert "No trades yet" in page.inner_text("#pnl-container")
    page.click("#tab-btn-market")
    page.click("#market-order-form button[type=submit]")  # buys 5 at the best ask
    page.wait_for_function("document.getElementById('pnl-container').textContent.includes('long 5')")
    assert "long 5" in page.inner_text("#metrics-pnl")
    page.wait_for_function("!document.getElementById('pnl-chart-panel').classList.contains('hidden')")
    run_for(page, 1500)
    charts = page.evaluate("""() => ({
        points: pnlSeries.data().length, bars: candleSeries.data().length,
        spacing: [priceChart, pnlChart].map(c => c.timeScale().options().barSpacing),
        scales: [priceChart, volumeChart, pnlChart].map(c => c.priceScale('right').width()),
        last: pnlSeries.data().slice(-1)[0].value, close: candleSeries.data().slice(-1)[0].close })""")
    assert charts["points"] == charts["bars"]  # a point for every bar, so the charts line up
    assert charts["spacing"][0] == charts["spacing"][1]
    assert len(set(charts["scales"])) == 1
    mark = page.evaluate("fetch('/api/candlesticks').then(r => r.json()).then(d => d.own_trades.slice(-1)[0])")
    assert charts["last"] == mark["cash"] + mark["position"] * charts["close"]


def test_the_tick_ladder_trades_at_any_price(page, base_url):
    open_dashboard(page, base_url)
    page.click("#ladder-container button:has-text('Ticks')")
    page.wait_for_function("window.ticksLadder.rows().size === 81")
    assert page.is_visible("#ticks-ladder") and not page.is_visible("#ladder-container")
    prices = page.evaluate(
        "[...document.querySelectorAll('#ticks-rows > [data-price]')].map(r => Number(r.dataset.price))")
    assert prices == list(range(140, 59, -1))  # every tick, 40 either side of the mid
    assert page.inner_text("#ticks-rows [data-price='99'][data-side='bid']") == "25"
    assert page.inner_text("#ticks-rows [data-price='100'][data-side='bid']") == ""  # the spread

    # One-click at an empty price inside the spread.
    page.fill("#limit-order-size", "7")
    page.click("#ticks-ladder .one-click-toggle")
    assert page.get_attribute("#one-click-toggle", "aria-pressed") == "true"  # both toggles, one state
    page.evaluate("document.querySelector('#ticks-rows [data-price=\"100\"]').scrollIntoView({block: 'center'})")
    page.mouse.click(*center_of(page, "#ticks-rows [data-price='100'][data-side='bid']"))
    page.wait_for_function("document.getElementById('toasts').textContent.includes('for 7 lots at 100')")
    page.wait_for_function("document.getElementById('metrics-container').dataset.bestBid === '100'")
    page.click("#ticks-ladder .one-click-toggle")

    # Drag your order to another empty price, below the book's best bid.
    page.wait_for_selector("#ticks-rows [data-price='100'] [data-own]")
    page.mouse.move(*center_of(page, "#ticks-rows [data-price='100'] [data-own]"))
    page.mouse.down()
    page.mouse.move(*center_of(page, "#ticks-rows [data-price='96'][data-side='bid']"), steps=8)
    assert page.inner_text("#ladder-hint") == "Move your buy orders (7 lots) to 96"
    page.mouse.up()
    page.wait_for_function("document.getElementById('toasts').textContent.includes('from 100 to 96')")
    page.wait_for_selector("#ticks-rows [data-price='96'] [data-own]")

    open_dashboard(page, base_url)  # the view is remembered
    page.wait_for_function("window.ticksLadder.rows().size === 81")
    assert page.is_visible("#ticks-ladder")
    page.click("#ticks-ladder button:has-text('Levels')")
    assert page.is_visible("#ladder-container") and not page.is_visible("#ticks-ladder")


BOARD_ROW = "#strategies-board tr[data-kind='{}']"


def row_data(kind, key):
    """A JavaScript expression: one of the leaderboard row's data attributes, for the strategy of this kind."""
    return f"document.querySelector(\"{BOARD_ROW.format(kind)}\").dataset.{key}"


def test_strategies_start_trade_and_rank_on_the_board(page, base_url):
    open_dashboard(page, base_url)
    fieldset = "#strategy-form fieldset[data-strategy-kind='{}']"
    assert page.is_visible(fieldset.format("market_making")) and not page.is_visible(fieldset.format("momentum"))
    page.click("#strategy-form button[type=submit]")  # a market maker, as set
    page.wait_for_selector(BOARD_ROW.format("market_making"))
    assert "Started Market making 1" in page.inner_text("#strategy-result")
    assert "bid 5 at 98 · ask 5 at 102" in page.inner_text(BOARD_ROW.format("market_making"))
    page.wait_for_function("document.querySelectorAll('#ladder-container [data-strategies]').length === 2")

    # Only the chosen strategy's settings show and are sent; one that contradicts another is refused.
    page.click("#strategy-form label:has-text('Momentum')")
    assert page.is_visible(fieldset.format("momentum")) and not page.is_visible(fieldset.format("market_making"))
    fast = fieldset.format("momentum") + " input[name='fast']"
    page.fill(fast, "40")
    page.click("#strategy-form button[type=submit]")
    page.wait_for_function("document.getElementById('strategy-result').textContent.includes('shorter than the slow')")
    page.fill(fast, "5")
    page.click("#strategy-form button[type=submit]")
    page.wait_for_selector(BOARD_ROW.format("momentum"))
    assert page.evaluate("document.getElementById('strategies-board').dataset.count") == "2"

    page.evaluate("setSpeedPreset(500)")
    page.click("#sim-play-btn")
    page.wait_for_function(f"Number({row_data('market_making', 'lots')}) >= 20", timeout=20_000)
    # Stop it while the board is pushed and reordered several times a second: the click lands on it.
    stop = BOARD_ROW.format("market_making") + " button[data-action='stop']"
    page.hover(stop)
    page.click(stop)
    page.wait_for_function(f"{row_data('market_making', 'running')} === 'false'")
    assert page.evaluate(row_data("momentum", "running")) == "true"
    page.click("#sim-play-btn")
    page.mouse.move(5, 5)  # off the board: it catches up
    page.wait_for_function("!document.querySelector('#ladder-container [data-strategies]')")  # quotes withdrawn
    assert "No trades yet" in page.inner_text("#pnl-container")  # their fills are not yours
    assert page.is_visible("#strategies-board tr[data-strategy-id='you']")
    page.click(BOARD_ROW.format("market_making") + " button[data-action='remove']")
    page.mouse.move(5, 5)
    page.wait_for_function(f"!document.querySelector(\"{BOARD_ROW.format('market_making')}\")")


def test_fitting_a_strategy_over_a_training_period(page, base_url):
    open_dashboard(page, base_url)
    page.click("#strategy-form label:has-text('Breakout')")
    page.click("#fit-panel summary")
    assert page.is_visible("fieldset[data-fit-kind='breakout']")
    assert not page.is_visible("fieldset[data-fit-kind='momentum']")
    page.fill("#fit-panel input[name='events']", "2000")
    page.click("#fit-btn")
    page.wait_for_selector("#fit-summary", timeout=60_000)
    assert page.evaluate("document.getElementById('fit-summary').dataset.candidates") == "12"  # 4 windows, 3 intervals
    second = "#fit-result tr[data-rank='2']"
    page.click(f"{second} button:has-text('Use')")  # its settings go in the form
    window = page.inner_text(f"{second} td:nth-child(2)")
    assert page.input_value("fieldset[data-strategy-kind='breakout'] input[name='window']") == window
    page.click("#fit-result tr[data-rank='1'] button:has-text('Start')")
    page.wait_for_selector(BOARD_ROW.format("breakout"))
    assert "fitted on a fresh run" in page.inner_text(BOARD_ROW.format("breakout"))
    page.fill("fieldset[data-fit-kind='breakout'] input[name='grid_window']", "10, ten")
    page.click("#fit-btn")
    page.wait_for_function("document.getElementById('fit-result').textContent.includes('enter a number')")


def test_fitting_many_settings_tries_a_random_sample_of_them(page, base_url):
    open_dashboard(page, base_url)
    page.click("#strategy-form label:has-text('Mean reversion')")
    page.click("#fit-panel summary")
    assert page.inner_text("#fit-count") == "27 combinations to try."  # window, entry and exit, three values each
    panel = "fieldset[data-fit-kind='mean_reversion']"
    page.check(f"{panel} input[name='fit_size']")
    page.check(f"{panel} input[name='fit_interval']")
    assert page.inner_text("#fit-count").startswith("243 combinations: a random 48 of them will be tried")
    page.fill("#fit-panel input[name='events']", "1000")
    page.click("#fit-btn")
    page.wait_for_selector("#fit-summary", timeout=60_000)
    assert page.evaluate("document.getElementById('fit-summary').dataset.sampled") == "true"
    assert "48 of 243 combinations, drawn at random (seed 1)" in page.inner_text("#fit-summary")
    page.uncheck(f"{panel} input[name='fit_entry']")
    assert page.inner_text("#fit-count") == "81 combinations: a random 48 of them will be tried, drawn from the seed " \
        "(fewer over a long period)."


def test_the_null_and_whether_the_flow_pays(page, base_url):
    open_dashboard(page, base_url)
    page.click("#strategy-form label:has-text('Penny jumping')")
    page.click("#strategy-form button[type=submit]")
    page.wait_for_selector(BOARD_ROW.format("penny_jumping"))
    assert page.is_visible("#add-null button")
    page.click("#add-null button")
    page.wait_for_selector(BOARD_ROW.format("quote_joiner"))
    page.wait_for_function("document.getElementById('add-null').hidden")
    run_for(page, 3000, speed=1000)
    page.wait_for_selector("#regime-line")
    assert page.inner_text("#regime-line").startswith("The order flow pays resting orders here.")
    assert "the quote joiner" in page.inner_text("#regime-line")
    vs_null = "td:nth-child(10)"  # the Vs null column
    assert page.inner_text(f"{BOARD_ROW.format('quote_joiner')} {vs_null}") == "the null"
    assert re.fullmatch(r"[+-]\d\.\d\d", page.inner_text(f"{BOARD_ROW.format('penny_jumping')} {vs_null}"))


def test_the_market_making_lesson_runs_a_strategy(page, base_url):
    open_dashboard(page, base_url)
    page.click("#tab-learn")
    page.click("[data-lesson='market-making'] button")
    page.click("#lesson-do")  # starts a market maker and the simulation
    page.wait_for_function(STEP_DONE)
    page.wait_for_selector(BOARD_ROW.format("market_making"))
    # Pause once the server says it runs: a panel push from just before the lesson's start can still flip
    # the button back for a moment, and a click then would start it instead.
    page.wait_for_function(RUNNING)
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    for _ in range(3):
        page.click("#lesson-next")
    page.click("#lesson-do")  # stops it, and starts one that requotes more often
    page.wait_for_function("document.getElementById('strategies-board').dataset.count === '2'")
    page.wait_for_function("document.getElementById('strategies-board').dataset.running === '1'")
    page.click("#lesson-next")
    page.click("#lesson-do")  # stops it
    page.wait_for_function(STEP_DONE)
    assert page.evaluate("document.getElementById('strategies-board').dataset.running") == "0"


BADGE = "document.getElementById('sim-stats-badge').dataset"


def test_recording_a_session_and_playing_it_back(page, base_url):
    open_dashboard(page, base_url)
    assert not page.is_visible("#playback-bar")
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'true'")
    run_for(page, 1500)
    assert page.inner_text("#record-text").startswith("Stop recording (")
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'false'")
    assert page.inner_text("#record-text") == "Record"

    # Keep it as a file, and play the file back too.
    page.click("#tab-analyze")
    page.wait_for_selector("#recordings-list [data-recording-id]")
    with page.expect_download() as download:
        page.click("#recordings-list a:has-text('Download')")
    assert download.value.suggested_filename == "lob-recording-1.npz"
    page.set_input_files("#recording-upload input[type=file]", download.value.path())
    page.click("#recording-upload button[type=submit]")
    page.wait_for_function("document.querySelectorAll('#recordings-list [data-recording-id]').length === 2")
    assert "Added Recording 1 (2)" in page.inner_text("#recording-upload-result")

    page.click("#recordings-list [data-recording-id]:first-child button[data-recording-action='play']")
    page.wait_for_function(f"{BADGE}.playback === 'true'")
    assert page.is_visible("#playback-bar") and page.inner_text("#sim-reset-text") == "Rewind"
    assert page.inner_text("#playback-name") == "Recording 1"
    page.click("#tab-trade")
    run_for(page, 800)
    assert int(page.evaluate(f"{BADGE}.pbCursor")) > 0
    # Drag the timeline to its end: the playback goes there and stops.
    page.evaluate("""() => {
        const timeline = document.getElementById('playback-timeline');
        timeline.value = timeline.max;
        timeline.dispatchEvent(new Event('change'));
    }""")
    page.wait_for_function(f"{BADGE}.pbEnded === 'true'")
    page.wait_for_function("document.getElementById('playback-note').textContent.startsWith('The end')")
    page.click("#sim-reset-btn")  # rewinds
    page.wait_for_function(f"{BADGE}.pbCursor === '0'")
    page.click("#playback-exit")  # live from here
    page.wait_for_function(f"{BADGE}.playback === 'false'")
    assert not page.is_visible("#playback-bar") and page.inner_text("#sim-reset-text") == "Reset Book"
    run_for(page, 500)  # the model again


KBD_CURSOR = "(() => { const el = document.querySelector('#ladder-wrapper .kbd-cursor');" \
             " return el && [Number(el.dataset.price), el.dataset.side]; })()"


def test_the_order_book_works_from_the_keyboard(page, base_url):
    open_dashboard(page, base_url)
    page.fill("#limit-order-size", "7")
    page.focus("#ladder-wrapper")
    assert page.evaluate(KBD_CURSOR) == [99, "bid"]  # it starts at the best bid
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")  # in the levels view: nothing, and the simulation does not step
    assert page.evaluate(KBD_CURSOR) == [98, "bid"] and page.evaluate(EVENTS) == 0
    assert page.inner_text("#ladder-hint").startswith("98: 30 lots bid. Enter: copy the price")
    page.keyboard.press("Enter")
    assert page.input_value("#order-price") == "98"

    page.click("#one-click-toggle")
    page.focus("#ladder-wrapper")
    assert page.inner_text("#ladder-hint").startswith("98: 30 lots bid. Enter: buy 7 at 98")
    page.keyboard.press("Enter")  # one-click from the keyboard
    page.wait_for_function("document.getElementById('toasts').textContent.includes('for 7 lots at 98')")
    page.wait_for_selector("#dom-bids-section [data-price='98'] [data-own]")

    page.keyboard.press("m")  # pick up your orders...
    assert page.inner_text("#ladder-hint").startswith("Your buy orders (7 lots) at 98")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    assert page.inner_text("#ladder-hint").startswith("Move your buy orders (7 lots) to 96")
    page.keyboard.press("Escape")  # ...or not: the cursor goes back to them
    assert page.evaluate(KBD_CURSOR) == [98, "bid"]
    page.keyboard.press("m")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")  # ...and move them down a level
    page.wait_for_function("document.getElementById('toasts').textContent.includes('from 98 to 97')")

    page.click("#ladder-container button:has-text('Ticks')")
    page.wait_for_function("window.ticksLadder.rows().size > 0")
    page.focus("#ladder-wrapper")
    best_bid = page.evaluate("Number(document.getElementById('metrics-container').dataset.bestBid)")
    assert page.evaluate(KBD_CURSOR) == [best_bid, "bid"]
    page.keyboard.press("ArrowRight")  # in the tick view, to the asks at the same price
    page.keyboard.press("ArrowUp")
    assert page.evaluate(KBD_CURSOR) == [best_bid + 1, "ask"]
    assert page.evaluate("document.getElementById('ladder-wrapper').getAttribute('aria-activedescendant')") \
        == f"book-tick-ask-{best_bid + 1}"
