"""Drive the dashboard in Chrome: what the server-side tests cannot see (the page's own scripts).

Run with ``make test-browser``. Each test gets a fresh browser profile, so a fresh simulation.
"""

import base64
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


def test_speeds_asked_for_together_reach_the_server_in_the_order_asked(page, base_url):
    # Two requests at once go on different connections and can arrive in either order. The page sends a speed only
    # when the one before it has been answered, so the last one asked for is the one the server keeps. Here the first
    # request is held back for a moment: sent at once, the second would overtake it and the server would end at 500.
    open_dashboard(page, base_url)
    log = page.evaluate("""async () => {
        const real = window.fetch, log = [];
        window.fetch = (url, options) => {
            if (url !== '/api/sim/config') return real(url, options);
            const speed = options.body.get('speed'), first = !log.length;
            log.push(['asked', speed]);
            const wait = first ? new Promise((r) => setTimeout(r, 400)) : Promise.resolve();
            return wait.then(() => { log.push(['sent', speed]); return real(url, options); })
                .then((answer) => { log.push(['answered', speed]); return answer; });
        };
        window.updateSimSpeed(500);
        await window.updateSimSpeed(10);
        window.fetch = real;
        return log;
    }""")
    assert log == [["asked", "500"], ["sent", "500"], ["answered", "500"],
                   ["asked", "10"], ["sent", "10"], ["answered", "10"]]
    assert page.evaluate(SPEED_STATE)["badge"] == "10 events/s"
    other = open_dashboard(page.context.new_page(), base_url)  # what the server has
    assert other.evaluate(SPEED_STATE)["badge"] == "10 events/s"
    # A request that fails does not hold up the next.
    assert page.evaluate("""async () => {
        const real = window.fetch;
        let failed = false;
        window.fetch = (url, options) => {
            if (url === '/api/sim/config' && !failed) { failed = true; return Promise.reject(new TypeError('lost')); }
            return real(url, options);
        };
        window.updateSimSpeed(1000).catch(() => {});
        const answer = await window.updateSimSpeed(100);
        window.fetch = real;
        return answer.ok;
    }""")
    other.reload()
    other.wait_for_function(LIVE)
    assert other.evaluate(SPEED_STATE)["badge"] == "100 events/s"


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


VISIBLE_VIEWS = ("['trade', 'analyze', 'learn', 'pairs', 'statarb']"
                 ".filter(v => !document.getElementById('view-' + v).hidden)")


def test_views_switch_by_tab_and_key_and_are_remembered(page, base_url):
    open_dashboard(page, base_url)
    assert page.evaluate(VISIBLE_VIEWS) == ["trade"]
    assert page.evaluate("location.hash") == ""  # a plain address stays plain
    page.click("#tab-analyze")
    assert page.evaluate(VISIBLE_VIEWS) == ["analyze"]
    assert page.get_attribute("#tab-analyze", "aria-selected") == "true"
    assert page.evaluate("location.hash") == "#analyze"
    page.keyboard.press("5")
    assert page.evaluate(VISIBLE_VIEWS) == ["statarb"]
    page.keyboard.press("3")
    assert page.evaluate(VISIBLE_VIEWS) == ["learn"]
    # In the tab list the arrow keys move between tabs (wrapping around) and do not step the simulation.
    page.focus("#tab-learn")
    page.keyboard.press("ArrowRight")
    assert page.evaluate(VISIBLE_VIEWS) == ["pairs"]
    page.keyboard.press("ArrowRight")
    assert page.evaluate(VISIBLE_VIEWS) == ["statarb"]
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
    page.keyboard.press("4")  # the dock stays on the Pairs view, whose stream is not this market's
    page.click("#lesson-do")  # starts a market maker and the simulation, back on the Trade view
    assert page.evaluate(VISIBLE_VIEWS) == ["trade"]
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


def test_bars_pushed_while_the_chart_reloads_wait_for_it(page, base_url):
    # After a reset the series holds the old bars until the reload answers, and it refuses bars older than those.
    # The fixture fails the test on the error that would log.
    open_dashboard(page, base_url)
    run_for(page, 1500)
    page.evaluate("""() => {
        const real = window.fetch;
        window.fetch = (url, ...rest) => (String(url).includes('/api/candlesticks')
            ? new Promise((resolve) => setTimeout(resolve, 1500)).then(() => real(url, ...rest))
            : real(url, ...rest));
    }""")
    page.click("#sim-reset-btn")
    page.wait_for_function(f"{EVENTS} === 0")
    page.wait_for_function("!document.getElementById('chart-loading').classList.contains('hidden')")
    page.evaluate("stepSim(300)")
    page.wait_for_function(f"{EVENTS} >= 300")
    page.wait_for_timeout(300)  # the bars' push, still ahead of the reload's answer
    assert page.is_visible("#chart-loading")
    page.wait_for_function("document.getElementById('chart-loading').classList.contains('hidden')", timeout=10_000)
    page.evaluate("stepSim(300)")  # and the chart carries on from what the reload brought
    page.wait_for_function(f"{EVENTS} >= 600")
    page.wait_for_timeout(300)


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


PAIRS_TIME = "Number(document.getElementById('pairs-status').dataset.time)"
PAIRS_ROW = "#pairs-board tr[data-pairs-id='{}']"


def test_the_pairs_view_runs_two_books_and_pairs_traders(page, base_url):
    open_dashboard(page, base_url)
    page.keyboard.press("4")
    assert page.evaluate(VISIBLE_VIEWS) == ["pairs"]
    assert not page.is_visible("#sim-controls-container")  # the Trade view's market: its controls hide here
    page.wait_for_function("window.lobPairs.chartState() && window.lobPairs.chartState().points === 1")

    # Run on by 500 units of time: one point a unit on the chart, which fills from the left.
    page.click("[data-pairs-step='100']")
    page.wait_for_function(f"{PAIRS_TIME} >= 100")
    for _ in range(4):
        page.click("[data-pairs-step='100']")
    page.wait_for_function("window.lobPairs.chartState().points === 501")
    state = page.evaluate("window.lobPairs.chartState()")
    assert state["ranges"][0]["from"] <= 0 and state["following"]
    # Each book's figures fit its half of a narrow column, at any price.
    assert page.evaluate("[...document.querySelectorAll('#pairs-books [data-book]')]"
                         ".every(book => book.scrollWidth <= book.clientWidth)")
    assert page.evaluate(EVENTS) == 0  # the Trade view's market did not move

    # A trader that learns the gap from the chart trades at once; a bad setting is refused in place.
    page.fill("#pairs-trader-form input[name=exit]", "3")
    page.click("#pairs-trader-start")
    page.wait_for_function("document.getElementById('pairs-trader-result').textContent.includes('below the entry')")
    assert page.get_attribute("#pairs-trader-form input[name=exit]", "aria-invalid") == "true"
    page.fill("#pairs-trader-form input[name=exit]", "0")
    page.fill("#pairs-trader-form input[name=entry]", "1")
    page.click("#pairs-trader-start")
    page.wait_for_selector(PAIRS_ROW.format(1))
    assert "learned from the chart's last 400 points" in page.inner_text("#pairs-trader-result")
    assert page.input_value("#pairs-bands-select") == "1"

    # Running, the chart follows the newest points, with the trader's bands, markers and P&L line.
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("Space")
    page.wait_for_function("document.getElementById('pairs-status').dataset.running === 'true'")
    page.click("[data-pairs-speed='200']")
    page.wait_for_function(f"Number(document.querySelector(\"{PAIRS_ROW.format(1)}\").dataset.lots) > 0",
                           timeout=20_000)
    page.wait_for_function("window.lobPairs.chartState().points > 700", timeout=20_000)
    state = page.evaluate("window.lobPairs.chartState()")
    assert state["following"] and state["ranges"][0]["to"] >= state["points"] - 1 and state["traders"] == ["1"]
    assert state["bands"] == "1" and state["markers"] > 0

    # Stop it while the board is pushed and reordered several times a second: the click lands on it.
    stop = PAIRS_ROW.format(1) + " button[data-pairs-action='stop']"
    page.hover(stop)
    page.click(stop)
    page.wait_for_function(f"document.querySelector(\"{PAIRS_ROW.format(1)}\").dataset.active === 'false'")
    page.mouse.move(5, 5)

    # New settings start the books over; another view closes the stream, and coming back reloads the chart.
    page.check("#pairs-config input[name=arbitrageur]")
    page.click("#pairs-config button[type=submit]")
    page.wait_for_function("document.getElementById('pairs-config-result').textContent.includes('from time 0')")
    page.wait_for_selector("#pairs-board tr[data-pairs-id='arbitrageur']")
    page.keyboard.press("1")
    assert page.is_visible("#sim-controls-container")
    page.keyboard.press("4")
    page.wait_for_function("window.lobPairs.chartState().points > 1")


CHART_STATE = "window.lobPairs.chartState()"


def open_pairs(page, base_url):
    page.goto(base_url + "/#pairs")
    page.wait_for_function(LIVE)
    page.wait_for_function(f"{CHART_STATE} && {CHART_STATE}.points > 0")


def chart_box(page, chart):
    """A chart's box, in the middle of the window: the sticky header covers the top, and a reload keeps the scroll."""
    page.evaluate(f"document.getElementById('{chart}').scrollIntoView({{block: 'center'}})")
    return page.locator(f"#{chart}").bounding_box()


def test_the_pairs_chart_holds_still_while_it_fills_and_follows_until_dragged_back(page, base_url):
    open_pairs(page, base_url)
    page.click("#pairs-play-btn")
    page.wait_for_function(f"{CHART_STATE}.points > 10")
    frames = page.evaluate("""() => new Promise(done => {
        const out = [], t0 = performance.now();
        (function tick() {
            const r = window.lobPairs.chartState().ranges[0];
            out.push([r.from, r.to]);
            if (performance.now() - t0 < 1000) requestAnimationFrame(tick); else done(out);
        })();
    })""")
    assert len(frames) > 10 and len({tuple(f) for f in frames}) == 1  # it holds still: no shimmer
    assert frames[0][0] <= 0 and frames[0][1] > 600  # filling a frame of 600 points from the left

    page.click("[data-pairs-speed='200']")
    page.wait_for_function(f"{CHART_STATE}.points > 800", timeout=20_000)
    page.wait_for_function(f"{CHART_STATE}.ranges[0].from > 100")  # past the frame: it scrolls on
    box = chart_box(page, "pairs-gap-chart")
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    # A click, or a zoom, leaves it following; only dragging back in time stops it. All three panes go along.
    page.mouse.click(x, y)
    page.mouse.move(x, y)
    page.mouse.wheel(0, -300)
    page.wait_for_function(f"{CHART_STATE}.span < 600")
    page.wait_for_function(f"new Set({CHART_STATE}.ranges.map(r => Math.round(r.to - r.from))).size === 1")
    before = page.evaluate(CHART_STATE)
    page.wait_for_function(f"{CHART_STATE}.points > {before['points'] + 20}")
    state = page.evaluate(CHART_STATE)
    assert state["following"] and state["span"] == before["span"]  # the zoom stays, and it scrolls on

    page.mouse.down()
    page.mouse.move(x + 300, y, steps=6)
    page.mouse.up()
    page.wait_for_function(f"!{CHART_STATE}.following")
    page.wait_for_timeout(100)  # the other panes take up the last move as they paint
    held = page.evaluate(CHART_STATE)
    page.wait_for_timeout(500)
    now = page.evaluate(CHART_STATE)
    assert now["points"] > held["points"] + 20  # it runs on, and what is shown stays where it was left
    assert now["times"] == held["times"] and now["ranges"][0] == now["ranges"][1] == now["ranges"][2]

    # Fit shows every point and follows again.
    page.click("#pairs-play-btn")
    page.wait_for_function("document.getElementById('pairs-status').dataset.running === 'false'")
    page.click("#pairs-fit-btn")
    page.wait_for_function(f"{CHART_STATE}.following && {CHART_STATE}.span > 800")

    def run_on(thousands):
        target = page.evaluate(PAIRS_TIME) + 1000 * thousands
        for _ in range(thousands):
            page.click("[data-pairs-step='1000']")
        page.wait_for_function(f"{PAIRS_TIME} >= {target} && {CHART_STATE}.last >= {int(target)}", timeout=30_000)

    run_on(4)
    assert page.evaluate(CHART_STATE)["following"]

    # When the chart drops its oldest points the times shown stay; once they have gone, the oldest left show.
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 300, y, steps=6)
    page.mouse.up()
    page.wait_for_function(f"!{CHART_STATE}.following")
    page.wait_for_timeout(100)
    looking = page.evaluate(CHART_STATE)
    run_on(2)
    now = page.evaluate(CHART_STATE)
    assert now["first"] > looking["first"] and now["points"] < looking["points"]  # some have been dropped
    assert now["times"] == looking["times"] and not now["following"]
    run_on(4)
    now = page.evaluate(CHART_STATE)
    assert now["first"] > looking["times"]["to"] and now["times"]["from"] == now["first"] and not now["following"]
    assert now["span"] == looking["span"]


def test_a_tab_on_the_pairs_view_holds_one_stream_so_many_can(page, base_url):
    """Browsers allow about six connections per host: two streams a tab froze the page at three tabs."""
    tabs = [page] + [page.context.new_page() for _ in range(3)]
    for tab in tabs:
        open_pairs(tab, base_url)
    tabs[3].click("[data-pairs-step='100']")
    for tab in tabs:
        tab.wait_for_function(f"{PAIRS_TIME} >= 100")  # every tab hears, and each can still be answered


BARS_UP_TO_DATE = """(() => {
    const data = candleSeries.data();
    const clock = Number(document.getElementById('sim-stats-badge').textContent.match(/Time\\s+([0-9.]+)s/)[1]);
    const whole = data.every((bar, i) => !i || bar.time - data[i - 1].time === currentInterval);
    return whole && data.length > 2 && data[data.length - 1].time >= clock - 2 * currentInterval;
})()"""


def test_the_trade_view_catches_up_after_the_pairs_view(page, base_url):
    open_dashboard(page, base_url)
    inside = int(page.evaluate("document.getElementById('metrics-container').dataset.bestBid")) + 1
    post = "window.postForm('/api/{}', {}).then(r => r.status)"
    page.keyboard.press("4")
    page.wait_for_function(LIVE)

    # A fill of your resting order shows on this view too, though /events is closed here.
    assert page.evaluate(post.format("order/limit", f"{{price: {inside}, size: 5, side: 'bid'}}")) == 200
    assert page.evaluate(post.format("order/market", "{size: 5, side: 'ask'}")) == 200
    page.wait_for_function("document.getElementById('toasts').textContent.includes('now filled in full')")

    # The Trade view's market runs on meanwhile: back there, its stream takes over and the chart has every bar.
    assert page.evaluate(post.format("sim/config", "{speed: 500}")) == 204
    page.evaluate(post.format("sim/control", "{action: 'play'}"))
    page.wait_for_timeout(1000)
    assert not page.evaluate(RUNNING)  # the hidden badge waits for the stream
    page.keyboard.press("1")
    page.wait_for_function(RUNNING)
    page.wait_for_function(BARS_UP_TO_DATE)

    # So does a page that opened on the Pairs view, whose chart loaded long before it connected.
    page.keyboard.press("4")
    page.reload()  # going to the address alone would only change its hash
    page.wait_for_function(LIVE)
    assert page.evaluate(VISIBLE_VIEWS) == ["pairs"]
    page.wait_for_timeout(1000)
    page.keyboard.press("1")
    page.wait_for_function(RUNNING)
    page.wait_for_function(BARS_UP_TO_DATE)
    page.click("#sim-play-btn")


def test_the_pairs_view_runs_each_kind_of_trader(page, base_url):
    open_pairs(page, base_url)
    form = "#pairs-trader-form"
    assert page.is_visible(f"{form} input[name=hedge]") and not page.is_visible(f"{form} input[name=memory]")
    page.check("#pairs-config input[name=drifting]")
    page.click("#pairs-config button[type=submit]")
    page.wait_for_function("document.getElementById('pairs-config-result').textContent.includes('from time 0')")
    page.click("[data-pairs-step='1000']")
    page.wait_for_function(f"{CHART_STATE}.points === 1001", timeout=20_000)

    # A kind's own settings show, and only they are sent; a mistake is marked in the kind in force.
    page.click(f"{form} label:has-text('Kalman')")
    assert page.is_visible(f"{form} input[name=memory]") and not page.is_visible(f"{form} input[name=trading]")
    page.fill(f"{form} fieldset[data-pairs-kind=kalman] input[name=memory]", "1")
    page.click("#pairs-trader-start")
    page.wait_for_function("document.getElementById('pairs-trader-result').textContent.includes('Memory')")
    assert page.get_attribute(f"{form} fieldset[data-pairs-kind=kalman] input[name=memory]", "aria-invalid") == "true"
    page.fill(f"{form} fieldset[data-pairs-kind=kalman] input[name=memory]", "300")
    page.click("#pairs-trader-start")
    page.wait_for_selector(PAIRS_ROW.format(1))
    assert "Kalman 1 learned" in page.inner_text("#pairs-trader-result")

    page.click(f"{form} label:has-text('Copula')")
    assert page.is_visible(f"{form} input[name=trading]") and not page.is_visible(f"{form} input[name=memory]")
    page.fill(f"{form} fieldset[data-pairs-kind=copula] input[name=window]", "100")
    page.fill(f"{form} fieldset[data-pairs-kind=copula] input[name=trading]", "50")
    page.click("#pairs-trader-start")
    page.wait_for_selector(PAIRS_ROW.format(2))

    # The chart shows the first trader's filter, then the other's flags, with each one's levels.
    page.click("[data-pairs-step='1000']")
    page.wait_for_function(f"{CHART_STATE}.points === 2001", timeout=20_000)
    assert page.input_value("#pairs-bands-select") == "1"
    assert "filter's error" in page.inner_text("#pairs-signal-legend")
    assert page.evaluate(CHART_STATE)["levels"] == 2
    assert "hedge " in page.inner_text(PAIRS_ROW.format(1) + " [data-pairs-detail]")
    page.select_option("#pairs-bands-select", "2")
    legend = page.inner_text("#pairs-signal-legend")
    assert "A's flag" in legend and "B's flag" in legend
    assert " / " in page.inner_text(PAIRS_ROW.format(2))  # both flags on its row
    assert "fit " in page.inner_text(PAIRS_ROW.format(2) + " [data-pairs-detail]")
    assert "flags " in page.inner_text("#pairs-hud")


PAIRS_FORM = "#pairs-trader-form"
FIT_SET = PAIRS_FORM + " fieldset[data-pairs-fit-kind={}]"
FIT_SUMMARY = "#pairs-fit-summary"


def set_quick_fit(page):
    """A small fit: windows and entries short enough for a run to take moments."""
    page.fill(PAIRS_FORM + " fieldset[data-pairs-kind=zscore] input[name=window]", "20")
    page.click("#pairs-fit-panel summary")
    page.fill(FIT_SET.format("zscore") + " input[name=grid_window]", "10, 20")
    page.fill(FIT_SET.format("zscore") + " input[name=grid_entry]", "1.5, 2")
    page.fill(PAIRS_FORM + " input[name=span]", "500")
    page.fill(PAIRS_FORM + " input[name=runs]", "2")


def test_a_pairs_trader_is_fitted_over_training_runs_and_started(page, base_url):
    open_pairs(page, base_url)
    # Each kind has its own settings to try, and the panel counts what they make before anything runs.
    page.click("#pairs-fit-panel summary")
    assert page.is_visible(FIT_SET.format("zscore")) and not page.is_visible(FIT_SET.format("copula"))
    assert "16 combinations, and your settings" in page.inner_text("#pairs-fit-count")
    page.click(PAIRS_FORM + " label:has-text('Copula')")
    assert page.is_visible(FIT_SET.format("copula")) and not page.is_visible(FIT_SET.format("zscore"))
    assert "15 combinations" in page.inner_text("#pairs-fit-count")
    page.click(PAIRS_FORM + " label:has-text('Z-score')")
    page.click("#pairs-fit-panel summary")

    # What is wrong is said against its field, and nothing runs.
    set_quick_fit(page)
    page.fill(FIT_SET.format("zscore") + " input[name=grid_window]", "10, nope")
    page.click("#pairs-fit-run")
    page.wait_for_function("document.getElementById('pairs-fit-status').textContent.includes('enter a number')")
    assert page.get_attribute(FIT_SET.format("zscore") + " input[name=grid_window]", "aria-invalid") == "true"
    assert page.get_attribute("#pairs-fit-run", "aria-disabled") == "false"

    # Enter in the panel fits rather than starts; the results come with the user's settings marked.
    page.fill(FIT_SET.format("zscore") + " input[name=grid_window]", "10, 20")
    assert "4 combinations" in page.inner_text("#pairs-fit-count")
    page.press(FIT_SET.format("zscore") + " input[name=grid_window]", "Enter")
    page.wait_for_selector(FIT_SUMMARY, timeout=60_000)
    page.wait_for_function("document.getElementById('pairs-fit-status').textContent.includes('Tried 4 settings')")
    assert not page.is_visible(PAIRS_ROW.format(1), timeout=0)  # no trader was started
    assert page.locator(f"{FIT_SUMMARY} tr[data-rank]").count() == 4
    assert page.locator(f"{FIT_SUMMARY} tr[data-yours]").count() == 1
    assert page.get_attribute(FIT_SUMMARY, "data-verdict") in ("yours", "clearly", "unclear")
    assert page.inner_text(f"{FIT_SUMMARY} [data-pairs-fit-verdict]")
    assert page.get_attribute("#pairs-fit-run", "aria-disabled") == "false"

    # Use puts a candidate's settings in the form, and its Start puts it on the board, keeping its training result.
    second = f"{FIT_SUMMARY} tr[data-rank='2']"
    window = page.inner_text(f"{second} td:nth-child(2)")
    page.click(f"{second} [data-pairs-fit-use]")
    assert page.input_value(PAIRS_FORM + " fieldset[data-pairs-kind=zscore] input[name=window]") == window
    page.click(f"{second} [data-pairs-fit-start]")
    page.wait_for_selector(PAIRS_ROW.format(1) + " [data-pairs-fitted]")
    note = page.inner_text(PAIRS_ROW.format(1) + " [data-pairs-fitted]")
    assert "2nd of 4" in note and "in 2 training runs" in note and "nothing yet here" in note
    page.click("[data-pairs-step='100']")
    fitted = f"document.querySelector(\"{PAIRS_ROW.format(1)} [data-pairs-fitted]\").textContent"
    page.wait_for_function(f"{fitted}.includes('a unit here')")

    # The results stay with the session: a reload brings them back, through the stream.
    page.reload()
    page.wait_for_function(LIVE)
    page.wait_for_selector(FIT_SUMMARY)
    assert page.locator(f"{FIT_SUMMARY} tr[data-rank]").count() == 4
    # A push of the board does not take the results' buttons: nothing else replaces them while the market runs.
    page.evaluate(f"document.querySelector('{FIT_SUMMARY}').dataset.marked = 'yes'")
    page.click("#pairs-play-btn")
    page.wait_for_function("Number(document.getElementById('pairs-status').dataset.time) > 110", timeout=20_000)
    assert page.get_attribute(FIT_SUMMARY, "data-marked") == "yes"
    page.click("#pairs-play-btn")


def test_a_fit_shows_its_progress_in_every_tab_and_can_be_canceled(page, base_url):
    open_pairs(page, base_url)
    other = page.context.new_page()
    open_pairs(other, base_url)
    set_quick_fit(page)
    page.fill(PAIRS_FORM + " input[name=span]", "20000")
    page.fill(PAIRS_FORM + " input[name=runs]", "4")
    page.click("#pairs-fit-run")
    for tab in (page, other):
        tab.wait_for_function("document.getElementById('pairs-fit-progress').textContent.startsWith('Fitting')",
                              timeout=30_000)
        tab.wait_for_function("document.getElementById('pairs-status').dataset.fitting !== ''", timeout=30_000)
    assert page.get_attribute("#pairs-fit-run", "aria-disabled") == "true"
    page.click("#pairs-fit-cancel")
    page.wait_for_function("document.getElementById('pairs-fit-status').textContent.includes('Canceled')",
                           timeout=30_000)
    for tab in (page, other):
        tab.wait_for_function("document.getElementById('pairs-fit-progress').textContent === ''", timeout=30_000)
    assert page.get_attribute("#pairs-fit-run", "aria-disabled") == "false"
    assert page.locator(FIT_SUMMARY).count() == 0
    other.close()


def scale_of(page, pane):
    s = page.evaluate(CHART_STATE)["scale"][pane]
    return s["top"] - s["bottom"], s["zoom"], s["auto"]


def test_a_first_trader_started_late_leaves_the_pairs_chart_able_to_drop_its_oldest_points(page, base_url):
    # The P&L pane has only its zero line until then, and the library goes wrong if that is set again to the same times.
    open_pairs(page, base_url)
    for _ in range(3):
        time_before = page.evaluate(PAIRS_TIME)
        page.evaluate("window.lobPairs.step(5000)")
        page.wait_for_function(f"{PAIRS_TIME} >= {time_before + 5000}", timeout=60_000)
    page.wait_for_function(f"{CHART_STATE}.last >= 15000")
    page.click("#pairs-trader-start")
    page.wait_for_selector(PAIRS_ROW.format(1))
    page.wait_for_function(f"{CHART_STATE}.traders.length === 1")
    points = page.evaluate(f"{CHART_STATE}.points")
    assert points == 6000  # the next point is one too many
    page.evaluate("window.lobPairs.step(100)")
    page.wait_for_function(f"{CHART_STATE}.last >= 15100")
    state = page.evaluate(CHART_STATE)
    assert state["points"] < points and state["first"] > 7001  # the oldest have gone, and the page has no errors


def test_the_pairs_charts_scale_and_panes(page, base_url):
    open_pairs(page, base_url)
    for _ in range(2):
        page.click("[data-pairs-step='1000']")
    page.wait_for_function(f"{CHART_STATE}.points === 2001", timeout=20_000)
    assert page.get_attribute("[data-pairs-span='600']", "aria-pressed") == "true"
    assert page.inner_text("#pairs-hzoom-badge").startswith("600 points")

    # Horizontal zoom: steppers, presets and the slider, all three panes along, up to the newest point.
    page.click("#pairs-zoom-in")
    page.wait_for_function(f"{CHART_STATE}.span === 480")
    assert page.inner_text("#pairs-hzoom-badge").startswith("480 points")
    assert page.get_attribute("[data-pairs-span='600']", "aria-pressed") == "false"
    page.click("#pairs-zoom-out")
    page.wait_for_function(f"{CHART_STATE}.span === 600")
    page.click("[data-pairs-span='200']")
    page.wait_for_function(f"{CHART_STATE}.span === 200")
    state = page.evaluate(CHART_STATE)
    assert state["following"] and all(r == state["ranges"][0] for r in state["ranges"])
    page.evaluate("""() => { const s = document.getElementById('pairs-hzoom-slider'); s.value = 1000;
                             s.dispatchEvent(new Event('input', {bubbles: true})); }""")
    page.wait_for_function(f"{CHART_STATE}.span === 6000")
    assert page.evaluate(CHART_STATE)["filling"]  # more room than points: it fills from the left again
    page.click("#pairs-fit-btn")
    page.wait_for_function(f"{CHART_STATE}.span === 2000 && !{CHART_STATE}.filling")

    # Vertical zoom in one pane, around its middle, and centering it again.
    page.wait_for_timeout(100)
    full, zoom, auto = scale_of(page, "gap")
    assert zoom == 1 and auto
    page.click("[data-pairs-vzoom='gap'][data-by='1']")
    page.click("[data-pairs-vzoom='gap'][data-by='1']")
    page.wait_for_function(f"{CHART_STATE}.scale.gap.zoom > 1.5")
    page.wait_for_function(f"Math.abs({CHART_STATE}.scale.gap.top - {CHART_STATE}.scale.gap.bottom) < {full} * 0.75")
    assert page.evaluate(CHART_STATE)["scale"]["price"]["zoom"] == 1  # only that pane
    page.click("[data-pairs-center='gap']")
    page.wait_for_function(f"{CHART_STATE}.scale.gap.zoom === 1")
    page.wait_for_function(f"Math.abs({CHART_STATE}.scale.gap.top - {CHART_STATE}.scale.gap.bottom - {full})"
                           f" < {full} * 0.02")

    # Dragging a price axis takes a pane off auto-scale; Center (or C) puts it back, as Fit does on the Trade view.
    box = chart_box(page, "pairs-gap-chart")
    x, y = box["x"] + box["width"] - 15, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y + 50, steps=5)
    page.mouse.up()
    page.wait_for_function(f"!{CHART_STATE}.scale.gap.auto")
    page.mouse.move(5, 5)
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("c")
    page.wait_for_function(f"{CHART_STATE}.scale.gap.auto")

    # Panes: hiding one leaves the time axis on the lowest left, and a pane shown again takes up the range.
    assert page.evaluate(CHART_STATE)["axis"] == ["pnl"]
    page.click("[data-pairs-toggle='pnl']")
    assert not page.is_visible("#pairs-pnl-chart") and page.evaluate(CHART_STATE)["axis"] == ["hedge"]
    page.click("[data-pairs-span='600']")
    page.wait_for_function(f"{CHART_STATE}.span === 600")
    page.click("[data-pairs-toggle='pnl']")
    page.wait_for_function(f"{CHART_STATE}.axis[0] === 'pnl' && {CHART_STATE}.ranges.length === 5")
    page.wait_for_function(f"new Set({CHART_STATE}.ranges.map(r => r.from)).size === 1")
    for pane in ("price", "gap", "signal", "hedge"):
        page.click(f"[data-pairs-toggle='{pane}']")
    assert page.get_attribute("[data-pairs-toggle='pnl']", "aria-pressed") == "true"
    page.click("[data-pairs-toggle='pnl']")  # the last one stays
    assert page.is_visible("#pairs-pnl-chart")
    for pane in ("price", "gap", "signal", "hedge"):
        page.click(f"[data-pairs-toggle='{pane}']")

    # Height, kept in this browser with the panes.
    low = page.evaluate(CHART_STATE)["scale"]["gap"]["height"]
    page.click("[data-pairs-height='2']")
    page.wait_for_function(f"{CHART_STATE}.scale.gap.height > {low} * 1.8")
    page.click("[data-pairs-toggle='hedge']")
    page.reload()
    page.wait_for_function(LIVE)
    page.wait_for_function(f"{CHART_STATE} && {CHART_STATE}.points === 2001")
    assert page.get_attribute("[data-pairs-height='2']", "aria-pressed") == "true"
    assert not page.is_visible("#pairs-hedge-chart")
    assert page.evaluate(CHART_STATE)["scale"]["gap"]["height"] > low * 1.8

    # The readout: the newest point, or the one under the crosshair, and the newest again when the pointer leaves.
    newest = page.inner_text("#pairs-hud")
    assert newest.startswith("time 2,000")
    box = chart_box(page, "pairs-price-chart")
    # Outside Chromium the chart library ignores the mouse for half a second after a touch, and counts from time 0.
    page.wait_for_function("performance.now() > 700")
    page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + 20)  # it only notes the first event inside it
    page.mouse.move(box["x"] + box["width"] * 0.35, box["y"] + 30, steps=2)
    page.wait_for_function("!document.getElementById('pairs-hud').textContent.startsWith('time 2,000')")
    page.mouse.move(5, 5)
    page.wait_for_function("document.getElementById('pairs-hud').textContent.startsWith('time 2,000')")


# ---- The Stat-arb view ----

STATARB_STATE = "window.lobStatarb.chartState()"
STATARB_ROW = "tr[data-statarb-id='{}']"
STATARB_FORM = "#statarb-trader-form"
STATARB_TIME = "Number(document.getElementById('statarb-status').dataset.time)"


def open_statarb(page, base_url):
    page.goto(base_url + "/#statarb")
    page.wait_for_function(LIVE)
    page.wait_for_function(f"{STATARB_STATE} && {STATARB_STATE}.points > 0")


def statarb_step(page, units, until=None):
    time_before = page.evaluate(STATARB_TIME)
    page.click(f"[data-statarb-step='{units}']")
    page.wait_for_function(f"{STATARB_TIME} >= {time_before + units}", timeout=60_000)


def test_the_statarb_view_runs_many_books_and_traders_in_them(page, base_url):
    open_statarb(page, base_url)
    status = page.inner_text("#statarb-status")
    assert "12 books" in status and "3 factors" in status and "9 relations" in status
    assert page.locator("#statarb-books [data-statarb-book]").count() == 12
    assert page.evaluate(STATARB_STATE)["books"] == 12
    assert not page.is_visible("#sim-controls-wrapper")  # a market of its own: the single book's controls hide

    # A trader of many pairs: it says what it is waiting for, then chooses, and shows what it chose.
    page.fill(f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] input[name=formation]", "100")
    page.fill(f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] input[name=window]", "40")
    page.fill(f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] input[name=count]", "4")
    page.click("#statarb-trader-start")
    row = STATARB_ROW.format(1)
    page.wait_for_selector(row)
    assert "it chooses once it has seen 99 more" in page.inner_text("#statarb-trader-result")
    assert "learning 1/100" in page.inner_text(row)
    assert page.input_value("#statarb-shown-select") == "1"
    statarb_step(page, 1000)
    page.wait_for_selector(f"{row} [data-statarb-relations]", timeout=30_000)
    assert page.locator(f"{row} [data-statarb-relations] > span").count() == 4
    assert page.get_attribute(row, "data-relations") == "4"
    page.wait_for_function(f"Number(document.querySelector(\"{row}\").dataset.lots) > 0", timeout=30_000)
    state = page.evaluate(STATARB_STATE)
    assert state["traders"] == ["1"] and state["shown"] == "1" and state["levels"] == 2
    assert "P&L" in page.inner_text("#statarb-hud") and "z-score" in page.inner_text("#statarb-signal-legend")

    # What the tests see, on the status panel.
    page.click("#statarb-test-btn")
    page.wait_for_selector("[data-statarb-test]", timeout=30_000)
    assert "of 9 relations" in page.inner_text("[data-statarb-test]")

    # The board's buttons.
    page.click(f"{row} button[data-statarb-action='stop']")
    page.wait_for_function(f"document.querySelector(\"{row}\").dataset.active === 'false'")
    page.click(f"{row} button[data-statarb-action='resume']")
    page.wait_for_function(f"document.querySelector(\"{row}\").dataset.active === 'true'")
    page.click(f"{row} button[data-statarb-action='remove']")
    page.mouse.move(5, 5)  # the board waits while the pointer is on its buttons
    page.wait_for_selector(row, state="detached")
    page.wait_for_function(f"{STATARB_STATE}.traders.length === 0")
    assert page.input_value("#statarb-shown-select") == ""


def test_the_statarb_view_draws_a_signal_line_a_book_and_no_levels_for_traders_that_score_every_book(
        page, base_url):
    open_statarb(page, base_url)
    page.click("#statarb-config label:has-text('Mixed')")
    page.fill("#statarb-config input[name=books]", "5")
    page.fill("#statarb-config input[name=factors]", "1")
    page.click("#statarb-config button[type=submit]")
    page.wait_for_function("document.getElementById('statarb-config-result').textContent.includes('5 books, 4 rel')")
    page.wait_for_function(f"{STATARB_STATE}.books === 5")

    def start(kind, **settings):
        page.click(f"{STATARB_FORM} input[name=kind][value={kind}] + span")
        for name, value in settings.items():
            page.fill(f"{STATARB_FORM} fieldset[data-statarb-kind={kind}] input[name={name}]", str(value))
        page.click("#statarb-trader-start")

    start("reversal", lookback=5, hold=5, factors=1, pca_window=30)
    page.wait_for_selector(STATARB_ROW.format(1))
    assert "learning" in page.inner_text(STATARB_ROW.format(1)) and "Reversal 1" in page.inner_text("#statarb-board")
    start("flow", hold=5, factors=1, pca_window=30)
    page.wait_for_selector(STATARB_ROW.format(2))
    page.wait_for_function("document.getElementById('statarb-trader-result').textContent.includes('Order flow 2')")
    assert "order flow only from now on" in page.inner_text("#statarb-trader-result")
    assert page.input_value(f"{STATARB_FORM} fieldset[data-statarb-kind=flow] select[name=measure]") == "queues"
    statarb_step(page, 1000)
    for number, kind, words in ((1, "reversal", "minus its residual move over the look back"),
                                (2, "flow", "queue imbalance")):
        page.select_option("#statarb-shown-select", str(number))
        page.wait_for_function(f"{STATARB_STATE}.shown === '{number}' && {STATARB_STATE}.signals === 5", timeout=30_000)
        state = page.evaluate(STATARB_STATE)
        assert state["levels"] == 0 and state["books"] == 5 and state["signals"] == 5  # a line a book, none at levels
        legend = page.inner_text("#statarb-signal-legend")
        assert words in legend and "before it is made neutral" in legend
        row = STATARB_ROW.format(number)
        page.wait_for_function(f"Number(document.querySelector(\"{row}\").dataset.lots) > 0", timeout=30_000)
        assert page.get_attribute(row, "data-kind") == kind and page.get_attribute(row, "data-relations") == "5"
        assert page.locator(f"{row} [data-statarb-relations] > span").count() == 5
        assert "z-score" not in legend
    # A trader of z-scores keeps its levels.
    start("pairs", count=1, formation=100, window=40)
    page.wait_for_selector(STATARB_ROW.format(3))
    page.select_option("#statarb-shown-select", "3")
    page.wait_for_function(f"{STATARB_STATE}.shown === '3' && {STATARB_STATE}.levels === 2")


def test_the_statarb_view_changes_its_market_and_says_what_is_wrong(page, base_url):
    open_statarb(page, base_url)
    # The sectors' fields show until another shape is chosen, and the chart takes up the new books.
    sectors, books = "#statarb-config input[name=sectors]", "#statarb-config input[name=books]"
    assert page.is_visible(sectors) and not page.is_visible(books)
    page.click("#statarb-config label:has-text('Mixed')")
    assert page.is_visible(books) and not page.is_visible(sectors)
    page.fill("#statarb-config input[name=books]", "3")
    page.fill("#statarb-config input[name=factors]", "3")
    page.click("#statarb-config button[type=submit]")
    page.wait_for_function("document.getElementById('statarb-config-result').textContent.includes('Fewer factors')")
    assert page.get_attribute("#statarb-config input[name=factors]", "aria-invalid") == "true"
    page.fill("#statarb-config input[name=books]", "6")
    page.fill("#statarb-config input[name=factors]", "2")
    page.click("#statarb-config button[type=submit]")
    page.wait_for_function("document.getElementById('statarb-config-result').textContent.includes('6 books, 4 rel')")
    page.wait_for_function(f"{STATARB_STATE}.books === 6")
    assert page.locator("#statarb-books [data-statarb-book]").count() == 6
    assert "every book" in page.inner_text("#statarb-books-legend")

    # A trader with a setting out of range: said against its field, and nothing starts.
    page.click(f"{STATARB_FORM} label:has-text('Residuals')")
    assert page.is_visible(f"{STATARB_FORM} input[name=pca_window]")
    open_at = f"{STATARB_FORM} fieldset[data-statarb-kind=residuals] input[name=open_at]"
    border = "(el) => getComputedStyle(el).borderTopColor"
    rose = page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--c-rose-500').trim()")
    assert page.eval_on_selector(open_at, border) != "rgb(%s)" % ", ".join(rose.split())
    page.fill(open_at, "99")
    page.click("#statarb-trader-start")
    page.wait_for_function("document.getElementById('statarb-trader-result').textContent.includes('between')")
    assert page.get_attribute(open_at, "aria-invalid") == "true"
    assert page.eval_on_selector(open_at, border) == "rgb(%s)" % ", ".join(rose.split())  # marked, not only flagged
    assert page.locator("tr[data-statarb-id]").count() == 0
    # And a choice that is not one of the options: the select is marked the same way.
    page.click(f"{STATARB_FORM} label:has-text('Order flow')")
    measure = f"{STATARB_FORM} fieldset[data-statarb-kind=flow] select[name=measure]"
    page.eval_on_selector(measure, "(el) => el.add(new Option('tea', 'tea'))")
    page.select_option(measure, "tea")
    page.click("#statarb-trader-start")
    page.wait_for_function("document.getElementById('statarb-trader-result').textContent.includes('Reads')")
    assert page.get_attribute(measure, "aria-invalid") == "true"
    assert page.eval_on_selector(measure, border) == "rgb(%s)" % ", ".join(rose.split())
    assert page.get_attribute(open_at, "aria-invalid") is None and page.locator("tr[data-statarb-id]").count() == 0


def test_a_first_trader_started_late_leaves_the_statarb_chart_able_to_drop_its_oldest_points(page, base_url):
    open_statarb(page, base_url)
    # Few books, so that it soon has a long history.
    page.click("#statarb-config label:has-text('Mixed')")
    page.fill("#statarb-config input[name=books]", "3")
    page.fill("#statarb-config input[name=factors]", "1")
    page.click("#statarb-config button[type=submit]")
    page.wait_for_function(f"{STATARB_STATE}.books === 3")
    for _ in range(3):
        statarb_step(page, 5000)
    page.wait_for_function(f"{STATARB_STATE}.last >= 15000")
    page.fill(f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] input[name=count]", "1")
    page.click("#statarb-trader-start")
    page.wait_for_selector(STATARB_ROW.format(1))
    page.wait_for_function(f"{STATARB_STATE}.traders.length === 1")
    points = page.evaluate(f"{STATARB_STATE}.points")
    assert points == 6000  # the next point is one too many
    statarb_step(page, 100)
    page.wait_for_function(f"{STATARB_STATE}.last >= 15100")
    state = page.evaluate(STATARB_STATE)
    assert state["points"] < points and state["first"] > 7001  # the oldest have gone, and the page has no errors


def test_the_statarb_chart_has_the_pairs_charts_controls(page, base_url):
    open_statarb(page, base_url)
    page.evaluate("document.getElementById('statarb-hzoom-badge').scrollIntoView({block: 'center'})")
    statarb_step(page, 1000)
    page.wait_for_function(f"{STATARB_STATE}.points === 1001", timeout=20_000)
    page.click("#statarb-zoom-in")
    page.wait_for_function(f"{STATARB_STATE}.span === 480")
    page.click("[data-statarb-span='200']")
    page.wait_for_function(f"{STATARB_STATE}.span === 200")
    page.click("#statarb-fit-btn")
    page.wait_for_function(f"{STATARB_STATE}.span >= 600")
    # A pane's vertical zoom, and Center.
    page.click("[data-statarb-vzoom='books'][data-by='1']")
    page.wait_for_function(f"{STATARB_STATE}.scale.books.zoom > 1")
    page.click("#statarb-center-btn")
    page.wait_for_function(f"{STATARB_STATE}.scale.books.zoom === 1")
    # Panes, the lowest with the time axis, and heights.
    page.click("[data-statarb-toggle='pnl']")
    page.wait_for_function(f"{STATARB_STATE}.axis.join() === 'held'")
    page.click("[data-statarb-toggle='pnl']")
    page.wait_for_function(f"{STATARB_STATE}.axis.join() === 'pnl'")
    tall = page.evaluate("document.getElementById('statarb-books-chart').clientHeight")
    page.click("[data-statarb-height='2']")
    page.wait_for_function(f"document.getElementById('statarb-books-chart').clientHeight > {tall * 1.5}")
    # The keyboard: F, C and the arrow act on these books too.
    before = page.evaluate(STATARB_TIME)
    page.keyboard.press("ArrowRight")
    page.wait_for_function(f"{STATARB_TIME} >= {before + 1}")


def test_a_tab_moves_between_the_views_holding_one_stream(page, base_url):
    open_statarb(page, base_url)
    page.click("#tab-pairs")
    page.wait_for_function(LIVE)
    page.wait_for_function(f"{CHART_STATE} && {CHART_STATE}.points > 0")
    assert page.is_visible("#view-pairs") and not page.is_visible("#view-statarb")
    page.keyboard.press("5")
    page.wait_for_function("document.getElementById('view-statarb').hidden === false")
    page.wait_for_function(LIVE)
    statarb_step(page, 10)
    page.click("#tab-trade")
    page.wait_for_function(LIVE)
    assert page.is_visible("#sim-controls-wrapper")
    before = page.evaluate(EVENTS)
    page.evaluate("stepSim(50)")
    page.wait_for_function(f"{EVENTS} >= {before + 50}")  # the page's own stream is back


STATARB_FIT = STATARB_FORM + " fieldset[data-statarb-fit-kind={}]"
STATARB_SUMMARY = "#statarb-fit-summary"


def set_quick_statarb_fit(page):
    """Three books and a small fit: a formation and windows short enough for a run to take moments."""
    page.click("#statarb-config label:has-text('Mixed')")
    page.fill("#statarb-config input[name=books]", "3")
    page.fill("#statarb-config input[name=factors]", "1")
    page.click("#statarb-config button[type=submit]")
    page.wait_for_function(f"{STATARB_STATE}.books === 3")
    pairs = f"{STATARB_FORM} fieldset[data-statarb-kind=pairs]"
    page.fill(pairs + " input[name=formation]", "60")
    page.fill(pairs + " input[name=window]", "20")
    page.fill(pairs + " input[name=count]", "2")
    page.click("#statarb-fit-panel summary")
    page.fill(STATARB_FIT.format("pairs") + " input[name=grid_window]", "10, 20")
    page.fill(STATARB_FIT.format("pairs") + " input[name=grid_entry]", "1.5, 2")
    page.fill(STATARB_FORM + " input[name=span]", "500")
    page.fill(STATARB_FORM + " input[name=runs]", "2")


def test_a_statarb_trader_is_fitted_over_training_runs_and_started(page, base_url):
    open_statarb(page, base_url)
    # Each kind has its own settings to try, and the panel counts what they make before anything runs.
    page.click("#statarb-fit-panel summary")
    assert page.is_visible(STATARB_FIT.format("pairs")) and not page.is_visible(STATARB_FIT.format("residuals"))
    assert "9 combinations, and your settings" in page.inner_text("#statarb-fit-count")
    page.click(f"{STATARB_FORM} label:has-text('Residuals')")
    assert page.is_visible(STATARB_FIT.format("residuals")) and not page.is_visible(STATARB_FIT.format("pairs"))
    assert "9 combinations" in page.inner_text("#statarb-fit-count")
    page.check(STATARB_FIT.format("residuals") + " input[name=fit_close_long]")
    page.fill(STATARB_FIT.format("residuals") + " input[name=grid_close_long]", "0, 0.5")
    assert "18 combinations" in page.inner_text("#statarb-fit-count")
    page.click(f"{STATARB_FORM} label:has-text('Pairs')")
    assert "9 combinations" in page.inner_text("#statarb-fit-count")  # the pairs trader's own: not what was ticked
    page.click("#statarb-fit-panel summary")

    # What is wrong is said against its field, and nothing runs.
    set_quick_statarb_fit(page)
    page.fill(STATARB_FIT.format("pairs") + " input[name=grid_window]", "10, nope")
    page.click("#statarb-fit-run")
    page.wait_for_function("document.getElementById('statarb-fit-status').textContent.includes('enter a number')")
    assert page.get_attribute(STATARB_FIT.format("pairs") + " input[name=grid_window]", "aria-invalid") == "true"
    assert page.get_attribute("#statarb-fit-run", "aria-disabled") == "false"

    # Enter in the panel fits rather than starts; the results come with the user's settings marked.
    page.fill(STATARB_FIT.format("pairs") + " input[name=grid_window]", "10, 20")
    assert "4 combinations" in page.inner_text("#statarb-fit-count")
    page.press(STATARB_FIT.format("pairs") + " input[name=grid_window]", "Enter")
    page.wait_for_selector(STATARB_SUMMARY, timeout=60_000)
    page.wait_for_function("document.getElementById('statarb-fit-status').textContent.includes('Tried 4 settings')")
    assert page.locator(STATARB_ROW.format(1)).count() == 0  # no trader was started
    assert page.locator(f"{STATARB_SUMMARY} tr[data-rank]").count() == 4
    assert page.locator(f"{STATARB_SUMMARY} tr[data-yours]").count() == 1
    assert page.get_attribute(STATARB_SUMMARY, "data-verdict") in ("yours", "clearly", "unclear")
    assert page.inner_text(f"{STATARB_SUMMARY} [data-statarb-fit-verdict]")
    assert page.get_attribute("#statarb-fit-run", "aria-disabled") == "false"

    # Use puts a candidate's settings in the form, and its Start puts it on the board, keeping its training result.
    second = f"{STATARB_SUMMARY} tr[data-rank='2']"
    window = page.inner_text(f"{second} td:nth-child(2)")
    page.fill(f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] input[name=window]", "33")
    page.click(f"{second} [data-statarb-fit-use]")
    assert page.input_value(f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] input[name=window]") == window
    assert "in the form now" in page.inner_text("#statarb-fit-reply")
    page.click(f"{second} [data-statarb-fit-start]")
    row = STATARB_ROW.format(1)
    page.wait_for_selector(f"{row} [data-statarb-fitted]")
    note = page.inner_text(f"{row} [data-statarb-fitted]")
    assert "2nd of 4" in note and "in 2 training runs" in note and "nothing yet here" in note
    assert "Pairs 1" in page.inner_text("#statarb-fit-reply")
    statarb_step(page, 1000)
    fitted = f"document.querySelector(\"{row} [data-statarb-fitted]\").textContent"
    page.wait_for_function(f"{fitted}.includes('a unit here')", timeout=30_000)

    # The results stay with the session: a reload brings them back, through the stream.
    page.reload()
    page.wait_for_function(LIVE)
    page.wait_for_selector(STATARB_SUMMARY)
    assert page.locator(f"{STATARB_SUMMARY} tr[data-rank]").count() == 4
    # Other books than the ones they were fitted on: the results say so.
    assert page.locator("[data-statarb-fit-stale]").count() == 0
    page.fill("#statarb-config input[name=half_life]", "20")
    page.click("#statarb-config button[type=submit]")
    page.wait_for_selector("[data-statarb-fit-stale]")


def test_a_statarb_fit_shows_its_progress_in_every_tab_and_can_be_canceled(page, base_url):
    open_statarb(page, base_url)
    other = page.context.new_page()
    open_statarb(other, base_url)
    set_quick_statarb_fit(page)
    page.fill(STATARB_FORM + " input[name=span]", "20000")
    page.fill(STATARB_FORM + " input[name=runs]", "4")
    page.click("#statarb-fit-run")
    for tab in (page, other):
        tab.wait_for_function("document.getElementById('statarb-fit-progress').textContent.startsWith('Fitting')",
                              timeout=30_000)
        tab.wait_for_function("document.getElementById('statarb-status').dataset.fitting !== ''", timeout=30_000)
    assert page.get_attribute("#statarb-fit-run", "aria-disabled") == "true"
    page.click("#statarb-fit-cancel")
    page.wait_for_function("document.getElementById('statarb-fit-status').textContent.includes('Canceled')",
                           timeout=30_000)
    for tab in (page, other):
        tab.wait_for_function("document.getElementById('statarb-fit-progress').textContent === ''", timeout=30_000)
        tab.wait_for_function("document.getElementById('statarb-status').dataset.fitting === ''", timeout=30_000)
    assert page.get_attribute("#statarb-fit-run", "aria-disabled") == "false"
    assert page.locator(STATARB_SUMMARY).count() == 0
    other.close()


# ---- What a narrow window, a step back, and results pushed again do to the fit panels ----

FIT_VIEWS = {  # what each view has under its own names, to run the same steps in both
    "pairs": {"open": open_pairs, "quick": set_quick_fit, "form": PAIRS_FORM, "summary": FIT_SUMMARY,
              "row": PAIRS_ROW, "config": "#pairs-config", "kind": "copula", "only": "trading",
              "fit": FIT_SET, "step": lambda page, units: page.click(f"[data-pairs-step='{units}']")},
    "statarb": {"open": open_statarb, "quick": set_quick_statarb_fit, "form": STATARB_FORM, "summary": STATARB_SUMMARY,
                "row": STATARB_ROW, "config": "#statarb-config", "kind": "baskets", "only": "legs",
                "fit": STATARB_FIT, "step": lambda page, units: statarb_step(page, units)},
}


def fit_once(page, view):
    """A small fit, and its results on the page."""
    v = FIT_VIEWS[view]
    v["quick"](page)
    page.click(f"#{view}-fit-run")
    page.wait_for_selector(v["summary"], timeout=60_000)
    page.wait_for_function(f"document.getElementById('{view}-fit-status').textContent.includes('Tried')")


@pytest.mark.parametrize("view", ["pairs", "statarb"])
def test_a_phone_does_not_scroll_sideways_for_fit_results_and_traders(page, base_url, view):
    v = FIT_VIEWS[view]
    page.set_viewport_size({"width": 390, "height": 800})
    v["open"](page, base_url)
    sideways = "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    assert page.evaluate(sideways) == 0
    fit_once(page, view)
    assert page.evaluate(sideways) == 0  # the results' table scrolls inside its box: its hidden header does not show
    page.click(f"{v['summary']} [data-{view}-fit-start='1']")
    page.wait_for_selector(v["row"].format(1))
    assert page.evaluate(sideways) == 0  # nor does the board's


@pytest.mark.parametrize("view", ["pairs", "statarb"])
@pytest.mark.parametrize("width", [360, 390, 430, 520, 640, 768, 1100])
def test_the_speed_presets_stay_inside_the_controls_at_every_width(page, base_url, view, width):
    page.set_viewport_size({"width": width, "height": 900})
    FIT_VIEWS[view]["open"](page, base_url)
    box = page.evaluate(f"""() => {{
        const card = document.getElementById('{view}-speed-slider').closest('.rounded-xl').getBoundingClientRect();
        const presets = [...document.querySelectorAll('[data-{view}-speed]')].map((b) => b.getBoundingClientRect());
        return {{card: [card.left, card.right], presets: [Math.min(...presets.map((r) => r.left)),
                                                           Math.max(...presets.map((r) => r.right))]}};
    }}""")
    assert box["card"][0] <= box["presets"][0] and box["presets"][1] <= box["card"][1] + 0.5, box


def test_no_setting_of_a_statarb_trader_has_its_label_cut_off(page, base_url):
    open_statarb(page, base_url)
    for width in (1440, 1100):
        page.set_viewport_size({"width": width, "height": 900})
        for kind in ("pairs", "baskets", "residuals", "copula", "reversal", "momentum", "leadlag", "flow"):
            page.click(f"{STATARB_FORM} input[name=kind][value={kind}] + span")
            cut = page.evaluate(f"""() => [...document.querySelectorAll(
                    '{STATARB_FORM} fieldset[data-statarb-kind={kind}] label > span:first-child')]
                .filter((s) => s.scrollWidth > s.clientWidth).map((s) => s.textContent.trim())""")
            assert cut == [], (width, kind, cut)
    page.click(f"{STATARB_FORM} input[name=kind][value=pairs] + span")
    refit = f"{STATARB_FORM} fieldset[data-statarb-kind=pairs] label:has(input[name=refit])"
    assert "0: never" in page.inner_text(refit)


@pytest.mark.parametrize("theme, size", [("daylight", "small"), ("terminal", "small"), ("terminal", "medium"),
                                         ("terminal", "large")])
def test_no_setting_of_a_statarb_trader_has_its_label_cut_off_in_another_theme_or_size(page, base_url, theme, size):
    open_statarb(page, base_url)
    page.select_option("#theme-select", theme)  # the Terminal's font is wider, and its labels wrap instead
    page.evaluate(f"setTextSize('{size}')")
    for width in (1440, 1100):
        page.set_viewport_size({"width": width, "height": 900})
        for kind in ("pairs", "baskets", "residuals", "copula", "reversal", "momentum", "leadlag", "flow"):
            page.click(f"{STATARB_FORM} input[name=kind][value={kind}] + span")
            cut = page.evaluate(f"""() => [...document.querySelectorAll(
                    '{STATARB_FORM} fieldset[data-statarb-kind={kind}] label > span:first-child')]
                .filter((s) => {{  // hidden where it clips, or past a pixel's rounding where it does not
                    const hides = getComputedStyle(s).overflow !== 'visible';
                    const over = Math.max(s.scrollWidth - s.clientWidth, s.scrollHeight - s.clientHeight);
                    return hides ? over > 0 : over > 1;
                }})
                .map((s) => s.textContent.trim())""")
            assert cut == [], (theme, size, width, kind, cut)


@pytest.mark.parametrize("view", ["pairs", "statarb"])
def test_stepping_back_to_the_page_keeps_the_chosen_kind_and_what_was_typed_for_it(page, base_url, view):
    v = FIT_VIEWS[view]
    v["open"](page, base_url)
    form, kind, only = v["form"], v["kind"], v["only"]
    page.click(f"{form} input[name=kind][value={kind}] + span")
    page.fill(f"{form} fieldset[data-{view}-kind={kind}] input[name=window]", "123")
    page.goto("about:blank")
    page.go_back()
    page.wait_for_function(LIVE)
    page.wait_for_selector(form)
    chosen = page.evaluate(f"""() => ({{
        checked: document.querySelector('{form} input[name=kind]:checked').value,
        settings: [...document.querySelectorAll('{form} fieldset[data-{view}-kind]')]
            .filter((f) => !f.hidden && !f.disabled).map((f) => f.dataset.{view}Kind),
        fits: [...document.querySelectorAll('{form} fieldset[data-{view}-fit-kind]')]
            .filter((f) => !f.hidden && !f.disabled).map((f) => f.dataset.{view}FitKind),
        sent: [...new FormData(document.querySelector('{form}')).keys()].includes('{only}'),
    }})""")
    assert chosen == {"checked": kind, "settings": [kind], "fits": [kind], "sent": True}  # not the first kind's
    assert page.input_value(f"{form} fieldset[data-{view}-kind={kind}] input[name=window]") == "123"
    page.click(f"#{view}-fit-panel summary")
    count = page.inner_text(f"#{view}-fit-count")
    page.evaluate(f"document.querySelector('{form} input[name=kind][value={kind}]').dispatchEvent("
                  "new Event('change', {bubbles: true}))")  # what choosing it says of its values to try
    assert count == page.inner_text(f"#{view}-fit-count") and "combinations" in count  # was said for it, not another


def test_the_statarb_count_of_combinations_is_the_count_of_numbers_not_of_ways_to_write_them(page, base_url):
    open_statarb(page, base_url)
    page.click("#statarb-fit-panel summary")
    fit = STATARB_FIT.format("pairs")
    page.fill(fit + " input[name=grid_window]", "10,20")
    page.fill(fit + " input[name=grid_entry]", "2, 2.0, 2.00")
    assert "2 combinations, and your settings" in page.inner_text("#statarb-fit-count")
    page.fill(fit + " input[name=grid_entry]", "2, 2.5, 02")
    assert "4 combinations, and your settings" in page.inner_text("#statarb-fit-count")


@pytest.mark.parametrize("view", ["pairs", "statarb"])
def test_results_that_come_above_the_window_stop_below_the_sticky_header(page, base_url, view):
    v = FIT_VIEWS[view]
    page.set_viewport_size({"width": 1440, "height": 500})
    v["open"](page, base_url)
    fit_once(page, view)
    for n in range(1, 3):  # a board of traders under the results
        page.click(f"#{view}-trader-start")
        page.wait_for_selector(v["row"].format(n))
    page.fill(f"{v['form']} input[name=span]", "5000")
    page.click(f"#{view}-fit-run")  # the next results will take the place of these
    page.wait_for_function(f"document.getElementById('{view}-fit-progress').textContent.startsWith('Fitting')",
                           timeout=30_000)
    result = f"document.getElementById('{view}-fit-result')"
    page.evaluate("document.body.style.paddingBottom = '3000px'")  # room to scroll on, past the results
    page.evaluate(f"window.scrollTo(0, {result}.getBoundingClientRect().bottom + window.scrollY + 20)")
    assert page.evaluate(f"{result}.getBoundingClientRect().bottom") < 0  # they are above the window now
    page.evaluate(f"""() => {{  // where the results are as the page takes the reader to them, not once things move
        const box = {result}, scroll = box.scrollIntoView.bind(box);
        window.landed = [];
        box.scrollIntoView = (options) => {{ scroll(options); window.landed.push(box.getBoundingClientRect().top); }};
    }}""")
    page.wait_for_function(f"{result}.querySelector('[data-fit-id]').dataset.fitId === '2'", timeout=60_000)
    landed, header = page.evaluate("[window.landed, document.querySelector('header').getBoundingClientRect().bottom]")
    assert landed and min(landed) >= header - 0.5, (landed, header)  # the first line of the results is not behind it


@pytest.mark.parametrize("view", ["pairs", "statarb"])
def test_the_same_results_pushed_again_do_not_take_the_page_or_clear_the_reply(page, base_url, view):
    v = FIT_VIEWS[view]
    page.set_viewport_size({"width": 1440, "height": 600})
    v["open"](page, base_url)
    fit_once(page, view)
    second = f"{v['summary']} tr[data-rank='2']"
    page.click(f"{second} [data-{view}-fit-use]")
    reply = f"#{view}-fit-reply"
    assert "in the form now" in page.inner_text(reply)
    # A longer fit is running; applying the books' settings meanwhile sends the old results again.
    page.fill(f"{v['form']} input[name=span]", "20000")
    page.fill(f"{v['form']} input[name=runs]", "4")
    page.click(f"#{view}-fit-run")
    page.wait_for_function(f"document.getElementById('{view}-fit-progress').textContent.startsWith('Fitting')",
                           timeout=30_000)
    page.evaluate("window.scrollTo(0, 0)")
    page.evaluate(f"document.querySelector('{v['summary']}').dataset.seen = 'yes'")
    page.click(f"{v['config']} button[type=submit]")
    page.wait_for_function(f"!document.querySelector('{v['summary']}') || "
                           f"document.querySelector('{v['summary']}').dataset.seen !== 'yes'", timeout=30_000)  # pushed
    page.wait_for_timeout(500)
    assert page.evaluate("window.scrollY") == 0  # it was not taken to results that are not the answer
    assert "in the form now" in page.inner_text(reply)  # and the reply to the Use is still there
    page.click(f"#{view}-fit-cancel")
    page.wait_for_function(f"document.getElementById('{view}-fit-status').textContent.includes('Canceled')",
                           timeout=30_000)


# ---- The microstructure studies band (static/js/lob-studies.js), the footprint (lob-footprint.js) and the tape ----

REPAINT = """() => new Promise(done => {
    lobFootprint.repaint(); requestAnimationFrame(() => requestAnimationFrame(done)); })"""
STUDY_KEYS = ["ofi", "cvd", "delta", "vpin", "otr", "imbalance", "spread", "resilience"]
TILE = """key => { const t = document.querySelector(`[data-tile=${key}]`);
    return {state: t.dataset.state, hidden: t.hidden, value: t.querySelector('[data-value]').textContent,
            lines: [...t.querySelector('[data-lines]').children].map(d => d.textContent)}; }"""
SPARK = """key => { const svg = document.querySelector(`[data-tile=${key}] [data-spark]`);
    const read = sel => svg.querySelector(sel).getAttribute('points').split(' ').filter(Boolean)
        .map(p => p.split(',').map(Number));
    return {mid: read('[data-mid]'), line: read('[data-line]'),
            polylines: svg.querySelectorAll('polyline').length}; }"""
STUDIES_LOADED = "window.lobStudies.state().loaded"


def studies_open(page, base_url, width=1440, height=900):
    open_dashboard(page, base_url, width, height)
    page.wait_for_function(STUDIES_LOADED)
    return page


def test_the_study_band_has_a_tile_for_each_study_that_warms_up_then_updates_in_place(page, base_url):
    studies_open(page, base_url)
    assert page.evaluate("[...document.querySelectorAll('#studies-tiles [data-tile]')].map(t => t.dataset.tile)") \
        == STUDY_KEYS
    # Fresh: the windows are not full, so the tiles say so and show nothing else in their place.
    for key in ("ofi", "delta", "vpin", "otr", "resilience"):
        page.wait_for_function(f"document.querySelector('[data-tile={key}]').dataset.state === 'warming'")
        assert page.evaluate(TILE, key)["value"] == "warming up", key
    assert page.evaluate(TILE, "cvd")["value"] == "0 lots"

    page.evaluate("window.__ofiTile = document.querySelector('[data-tile=ofi]')")
    run_for(page, 3000)
    page.wait_for_function("window.lobStudies.state().last > 20")
    page.wait_for_function("document.querySelector('[data-tile=ofi]').dataset.state === 'ok'")
    assert page.evaluate("document.querySelector('[data-tile=ofi]') === window.__ofiTile")  # filled in place
    ofi, vpin, otr = (page.evaluate(TILE, key) for key in ("ofi", "vpin", "otr"))
    assert re.fullmatch(r"[+-]?[\d,]+ lots", ofi["value"]) and ofi["lines"] == ["over 5 units"]
    assert vpin["value"] == "warming up"
    assert re.fullmatch(r"buckets \d+/50 of (\d+ lots|…)", vpin["lines"][1])  # sized by the first 200 fills
    assert re.fullmatch(r"\d+\.\d+ msgs/trade", otr["value"]) and otr["lines"][0].startswith("TTO 0.")  # OTR and TTO
    assert re.fullmatch(r"[+-]?\d\.\d{3}", page.evaluate(TILE, "imbalance")["value"])
    assert re.fullmatch(r"\d+ ticks?", page.evaluate(TILE, "spread")["value"])
    assert "unit" in page.inner_text("#studies-status")

    # The sparklines: two polylines each, x in the units of simulation time (the newest at the right edge, 120
    # units across), each on its own y scale.
    state = page.evaluate("({rows: window.lobStudies.rows(), keys: window.lobStudies.state().keys})")
    rows, last = state["rows"], state["rows"][-1][0]
    inside = [r for r in rows if r[0] > last - 120]
    for key in STUDY_KEYS:
        spark = page.evaluate(SPARK, key)
        assert spark["polylines"] == 2
        column = 2 + state["keys"].index(key)
        for points, index in ((spark["mid"], 1), (spark["line"], column)):
            usable = [r for r in inside if r[index] is not None]
            expected = [round(100 * (r[0] - (last - 120)) / 120, 2) for r in usable]
            assert [round(x, 2) for x, _ in points] == expected, key
            values = [r[index] for r in usable]
            signed = key in ("ofi", "cvd", "delta", "imbalance") and index == column  # their scale holds the zero too
            if points and max(values) > min(values) and not signed:
                ys = [y for _, y in points]  # up is a larger value, between 2.5 and 21.5
                assert min(ys) == pytest.approx(2.5, abs=0.01) and max(ys) == pytest.approx(21.5, abs=0.01), key
    zero = "document.querySelector('[data-tile={}] [data-zero]').getAttribute('visibility')"
    assert page.evaluate(zero.format("cvd")) == "visible" and page.evaluate(zero.format("vpin")) == "hidden"

    # The window is a setting of this browser, and the tiles follow it.
    page.select_option("#studies-window", "30")
    page.wait_for_function("window.lobStudies.state().window === 30")
    page.wait_for_function("document.querySelector('[data-tile=spread] [data-line]')"
                           ".getAttribute('points').split(' ').length <= 30")

    # Running, the values and the sparklines move, still in the same nodes.
    before = page.evaluate("window.lobStudies.state().last")
    page.evaluate("setSpeedPreset(200)")
    page.click("#sim-play-btn")
    page.wait_for_function(f"window.lobStudies.state().last >= {before} + 5")
    page.wait_for_function("document.getElementById('studies-status').textContent"
                           ".endsWith('unit ' + window.lobStudies.state().last)")
    assert page.evaluate("document.querySelector('[data-tile=ofi]') === window.__ofiTile")
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")


def test_a_study_setting_is_checked_applied_and_followed_by_the_other_tab(page, base_url):
    a = studies_open(page, base_url)
    b = studies_open(page.context.new_page(), base_url)  # the same session
    a.bring_to_front()
    settle(a)
    a.click("[data-tile=ofi] [data-open=settings]")
    assert a.is_visible("#studies-form fieldset[data-fields=ofi]")
    assert not a.is_visible("#studies-form fieldset[data-fields=delta]")
    assert a.input_value("#studies-form fieldset[data-fields=ofi] input[name=window]") == "5"
    assert a.is_checked("#studies-enabled")

    # A value out of range is refused, with the reason, in place (a 422 that the page swaps in).
    a.fill("#studies-form fieldset[data-fields=ofi] input[name=window]", "500")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("document.getElementById('studies-result').textContent.includes('must be between 0.5 and 100')")
    assert a.is_visible("#studies-result .text-rose-400")
    assert a.evaluate(TILE, "ofi")["lines"] == ["over 5 units"]

    a.fill("#studies-form fieldset[data-fields=ofi] input[name=window]", "8")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("document.getElementById('studies-result').textContent"
                        ".includes('Order flow imbalance: applied')")
    a.wait_for_function("window.lobStudies.state().settings.ofi.window === 8")
    assert a.evaluate(TILE, "ofi")["lines"] == ["over 8 units"]
    b.bring_to_front()
    b.wait_for_function("window.lobStudies.state().settings.ofi.window === 8")
    # the tile follows:
    b.wait_for_function("document.querySelector('[data-tile=ofi] [data-lines]').textContent === 'over 8 units'")
    assert b.evaluate("window.lobStudies.state().loaded")

    # The settings of the others are their own: delta by the quote rule at lag 0 says what that is worth.
    a.bring_to_front()
    a.click("[data-tile=delta] [data-open=settings]")
    a.select_option("#studies-form fieldset[data-fields=delta] select[name=side]", "quote")
    a.fill("#studies-form fieldset[data-fields=delta] input[name=quote_lag]", "0")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("document.querySelector('[data-tile=delta] [data-lines]').textContent"
                        ".includes('a check, not a result')")
    assert a.evaluate(TILE, "delta")["lines"][0] == "quote rule, 10 units"
    assert a.evaluate(TILE, "delta")["lines"][1].startswith("coverage")

    # What is typed in the form stays while the studies are pushed (and the page's other panels are replaced), and
    # follows another tab's change once it is applied, not before.
    a.click("[data-tile=ofi] [data-open=settings]")
    a.fill("#studies-form fieldset[data-fields=ofi] input[name=window]", "9")
    a.evaluate("setSpeedPreset(200)")
    a.click("#sim-play-btn")
    a.wait_for_function("window.lobStudies.state().last > 15")
    a.click("#sim-play-btn")
    a.wait_for_function(f"!({RUNNING})")
    assert a.input_value("#studies-form fieldset[data-fields=ofi] input[name=window]") == "9"
    a.evaluate("document.activeElement.blur(); document.getElementById('studies-panel-title').click()")
    a.wait_for_timeout(300)
    assert a.input_value("#studies-form fieldset[data-fields=ofi] input[name=window]") == "9"  # typed, not yet applied
    b.bring_to_front()
    b.click("[data-tile=ofi] [data-open=settings]")
    b.fill("#studies-form fieldset[data-fields=ofi] input[name=window]", "6")
    b.click("#studies-form button[type=submit]")
    b.wait_for_function("window.lobStudies.state().settings.ofi.window === 6")
    a.bring_to_front()
    a.wait_for_function("window.lobStudies.state().settings.ofi.window === 6")
    a.wait_for_function("document.querySelector('#studies-form fieldset[data-fields=ofi] input[name=window]')"
                        ".value === '6'")

    # Settings that contradict each other change nothing.
    a.click("[data-tile=resilience] [data-open=settings]")
    a.fill("#studies-form fieldset[data-fields=resilience] input[name=depleted]", "0.95")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("document.getElementById('studies-result').textContent.includes('must be above')")
    assert a.evaluate("window.lobStudies.state().settings.resilience.depleted") == 0.5

    # The switch: off in both tabs, and on again.
    a.click("[data-tile=cvd] [data-open=settings]")
    assert "No settings" in a.inner_text("#studies-form fieldset[data-fields=cvd]")
    a.uncheck("#studies-enabled")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("document.querySelector('[data-tile=cvd]').dataset.state === 'off'")
    b.bring_to_front()
    b.wait_for_function("document.querySelector('[data-tile=cvd]').dataset.state === 'off'")
    assert b.evaluate(TILE, "cvd")["value"] == "off"
    a.bring_to_front()
    a.check("#studies-enabled")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("document.querySelector('[data-tile=cvd]').dataset.state === 'ok'")


def test_text_typed_after_an_apply_is_not_replaced_when_focus_leaves_the_form(page, base_url):
    a = studies_open(page, base_url)
    settle(a)
    window = "#studies-form fieldset[data-fields=ofi] input[name=window]"
    a.evaluate("setSpeedPreset(200)")
    a.click("#sim-play-btn")
    a.wait_for_function(RUNNING)
    a.click("[data-tile=ofi] [data-open=settings]")
    a.fill(window, "12")
    a.click("#studies-form button[type=submit]")
    a.wait_for_function("window.lobStudies.state().settings.ofi.window === 12")
    a.wait_for_function("document.querySelector('[data-tile=ofi] [data-lines]').textContent === 'over 12 units'")
    a.wait_for_timeout(500)  # the pushes go on, the form having just applied the change itself
    # Typed after the Apply and not applied: it stays when focus leaves the form, while the studies are pushed.
    a.fill(window, "7")
    a.click("#studies-panel-title")
    a.wait_for_timeout(1500)
    assert a.input_value(window) == "7"
    assert a.evaluate("window.lobStudies.state().settings.ofi.window") == 12
    a.click("#sim-play-btn")
    a.wait_for_function(f"!({RUNNING})")

    # A change another tab makes still reaches the form once focus is out of it.
    b = studies_open(a.context.new_page(), base_url)
    b.click("[data-tile=ofi] [data-open=settings]")
    b.fill("#studies-form fieldset[data-fields=ofi] input[name=window]", "6")
    b.click("#studies-form button[type=submit]")
    b.wait_for_function("window.lobStudies.state().settings.ofi.window === 6")
    a.bring_to_front()
    a.wait_for_function("window.lobStudies.state().settings.ofi.window === 6")
    a.wait_for_function(f"document.querySelector('{window}').value === '6'")


def test_the_resilience_tile_says_which_side_is_depleted_while_its_score_is_undefined(page, base_url):
    studies_open(page, base_url)
    push = """([bid, ask, score]) => { const s = window.lobStudies.state();
        document.dispatchEvent(new CustomEvent('lob:studies', {detail: {version: s.version, failed: null,
            suspended: false, settings: s.settings, enabled: s.enabled, keys: s.keys, unit: s.unit, samples: [],
            values: {resilience: {score, baseline_bid: 20, baseline_ask: 20, depleted_bid: bid, depleted_ask: ask,
                median_recovery: score === null ? null : 12, abandoned_share: 0.5, started: 3, recovered: 1,
                abandoned: 1, quote_moves: 0, restored_share: null, median_restoration: null}}}})); }"""
    for args, value in (([5.5, None, None], "bid depleted"), ([None, 7.5, None], "ask depleted"),
                        ([5.5, 7.5, None], "both depleted"), ([5.5, 7.5, 0.31], "score 0.31"),
                        ([None, None, 1.0], "score 1.00")):
        page.evaluate(push, args)
        page.wait_for_function(
            f"document.querySelector('[data-tile=resilience] [data-value]').textContent === '{value}'")
        if value == "both depleted":
            assert page.evaluate(TILE, "resilience")["lines"][0] == "depleted: bid for 5.5u, ask for 7.5u"


def test_a_page_opened_on_the_pairs_view_has_the_studies_and_the_tape_of_now_on_the_trade_view(page, base_url):
    studies_open(page, base_url)
    page.keyboard.press("4")
    page.reload()  # it opens on the Pairs view: the band and the tape loaded before any connection
    page.wait_for_function(LIVE)
    assert page.evaluate(VISIBLE_VIEWS) == ["pairs"]
    page.wait_for_function(STUDIES_LOADED)
    post = "window.postForm('/api/{}', {}).then(r => r.status)"
    page.evaluate(post.format("sim/config", "{speed: 500}"))
    page.evaluate(post.format("sim/control", "{action: 'play'}"))
    page.wait_for_timeout(1500)
    page.evaluate(post.format("sim/control", "{action: 'pause'}"))
    page.wait_for_timeout(300)
    server = page.evaluate("Promise.all([fetch('/api/studies'), fetch('/api/tape')].map(f => f.then(r => r.json())))")
    assert server[0]["unit"] > 8 and server[1]["seq"] > 20
    page.keyboard.press("1")
    page.wait_for_function(f"window.lobStudies.state().last === {server[0]['unit']}")
    page.wait_for_function(f"window.lobTape.state().seq === {server[1]['seq']}")
    assert page.evaluate("window.lobStudies.rows().every((row, i, rows) => !i || row[0] - rows[i - 1][0] === 1)")
    assert page.evaluate("window.lobTape.state().shown[0].seq") == server[1]["seq"]


def test_a_tile_hidden_and_shown_is_remembered_and_so_is_the_sparkline_window(page, base_url):
    studies_open(page, base_url)
    assert page.is_visible("[data-tile=spread]")
    page.click("[data-chip=spread]")
    assert not page.is_visible("[data-tile=spread]")
    assert page.get_attribute("[data-chip=spread]", "aria-pressed") == "false"
    page.select_option("#studies-window", "300")
    saved = page.evaluate("JSON.parse(localStorage.getItem('lob-studies'))")
    assert saved == {"hidden": ["spread"], "window": 300}

    page.reload()  # a page of this browser comes back as it was left
    page.wait_for_function(LIVE)
    page.wait_for_function(STUDIES_LOADED)
    assert not page.is_visible("[data-tile=spread]") and page.is_visible("[data-tile=ofi]")
    assert page.input_value("#studies-window") == "300" and page.evaluate("window.lobStudies.state().window") == 300
    assert page.evaluate("window.lobStudies.state().hidden") == ["spread"]

    page.click("[data-chip=spread]")
    page.wait_for_selector("[data-tile=spread]", state="visible")
    # drawn at once, not at the next push:
    page.wait_for_function("document.querySelector('[data-tile=spread] [data-value]').textContent !== '…'")
    page.reload()
    page.wait_for_function(LIVE)
    assert page.is_visible("[data-tile=spread]")
    assert page.evaluate("JSON.parse(localStorage.getItem('lob-studies')).hidden") == []

    # A hidden tile whose panel is open closes it.
    page.click("[data-tile=vpin] [data-open=settings]")
    assert page.is_visible("#studies-panel")
    page.click("[data-chip=vpin]")
    assert not page.is_visible("#studies-panel")


HELP_SAYS = {
    "ofi": "Cont, Kukanov and Stoikov", "cvd": "from the start of a playback", "delta": "a check, not a result",
    "vpin": "near 0.5 by construction", "otr": "minus one", "imbalance": "VisualHFT's code",
    "spread": "VisualHFT: 2", "resilience": "never occurs",
}


def test_each_tile_has_a_help_that_says_what_it_measures_and_what_to_watch_for(page, base_url):
    studies_open(page, base_url)
    for key, phrase in HELP_SAYS.items():
        page.click(f"[data-tile={key}] [data-open=help]")
        block = f"#studies-help [data-help={key}]"
        assert page.is_visible(block) and phrase in page.inner_text(block)
        assert page.inner_text(f"#studies-help [data-help={key}]").startswith("What it measures.")
        assert not page.is_visible("#studies-form")  # help alone: no form
        assert page.get_attribute(f"[data-tile={key}]", "data-selected") == "true"
        assert page.evaluate("document.querySelectorAll('#studies-help [data-help]:not([hidden])').length") == 1
    page.click("[data-tile=vpin] [data-open=settings]")
    assert page.is_visible("#studies-form")
    assert page.is_visible("#studies-form fieldset[data-fields=vpin] select[name=side]")
    assert page.inner_text("#studies-help [data-help=vpin]").count("VisualHFT's website bands") == 1
    assert "Andersen and Bondarenko 2014" in page.inner_text("#studies-help [data-help=vpin]")
    page.click("[data-tile=vpin] [data-open=settings]")  # again: closed
    assert not page.is_visible("#studies-panel")
    page.click("[data-tile=otr] [data-open=help]")
    page.keyboard.press("Escape")
    assert not page.is_visible("#studies-panel")
    # The caveats that belong to a setting show on the tile too.
    page.evaluate("document.activeElement.blur()")
    page.click("[data-tile=vpin] [data-open=settings]")
    page.select_option("#studies-form fieldset[data-fields=vpin] select[name=side]", "bvc")
    page.click("#studies-form button[type=submit]")
    page.wait_for_function("document.querySelector('[data-tile=vpin] [data-lines]').textContent"
                           ".includes('near 0.5 by construction')")


def test_the_studies_reload_after_a_reset_a_stale_push_a_gap_and_the_pairs_view(page, base_url):
    studies_open(page, base_url)
    requests = []
    page.on("request", lambda r: requests.append(r.url) if r.url.split("?")[0].endswith("/api/studies") else None)
    run_for(page, 2000)
    page.wait_for_function("window.lobStudies.state().last > 15")
    version = page.evaluate("window.lobStudies.state().version")

    # A reset starts them over: a new version, a few samples, every window warming up.
    page.click("#sim-reset-btn")
    page.wait_for_function(f"window.lobStudies.state().version > {version} && window.lobStudies.state().last < 5")
    page.wait_for_function("document.querySelector('[data-tile=ofi]').dataset.state === 'warming'")

    # A push of another version is not merged: the page asks for the studies again. So does a push that leaves a gap.
    push = """([bump, ahead]) => { const s = window.lobStudies.state(), unit = s.unit + ahead;
        document.dispatchEvent(new CustomEvent('lob:studies', {detail: {version: s.version + bump, failed: null,
            suspended: false,
            settings: s.settings, enabled: s.enabled, values: {}, keys: s.keys, unit,
            samples: [[unit, 12345, 0, 0, 0, 0, 0, 0, 0, 0]]}})); }"""
    # The next unit of another version, and a unit far ahead of the last.
    for bump, ahead in ((-1, 1), (1, 1), (0, 400)):
        asked = len(requests)
        page.evaluate(push, [bump, ahead])
        page.wait_for_function(STUDIES_LOADED)
        page.wait_for_timeout(300)
        assert len(requests) == asked + 1, (bump, ahead, requests)
        assert page.evaluate("window.lobStudies.rows().every(row => row[1] !== 12345)")  # not drawn
    # The next unit of the same version is merged, with nothing asked for.
    asked = len(requests)
    page.evaluate(push, [0, 1])
    page.wait_for_function("window.lobStudies.rows().some(row => row[1] === 12345)")
    assert len(requests) == asked

    # The stream connecting again (after a drop, or back from Pairs) asks again. The chart reloading (an interval,
    # the footprint, which the studies do not depend on) does not: a reset changes the version, which is what asks.
    asked = len(requests)
    page.evaluate("document.dispatchEvent(new CustomEvent('lob:connection', {detail: 'live'}))")
    page.wait_for_function(STUDIES_LOADED)
    page.wait_for_timeout(300)
    assert len(requests) == asked + 1, requests
    asked = len(requests)
    page.evaluate("document.dispatchEvent(new CustomEvent('lob:chart-reset'))")
    page.click("#btn-interval-5")
    page.wait_for_function("loadedInterval === 5")
    page.wait_for_timeout(300)
    assert len(requests) == asked, requests

    # Back from the Pairs view, where the stream is closed: the market went on, and the tiles have all of it.
    post = "window.postForm('/api/{}', {}).then(r => r.status)"
    page.keyboard.press("4")
    page.wait_for_function(LIVE)
    page.evaluate(post.format("sim/config", "{speed: 500}"))
    page.evaluate(post.format("sim/control", "{action: 'play'}"))
    page.wait_for_timeout(1500)
    page.evaluate(post.format("sim/control", "{action: 'pause'}"))
    unit = page.evaluate("fetch('/api/studies').then(r => r.json()).then(d => d.unit)")
    assert unit > 8
    page.keyboard.press("1")
    page.wait_for_function(f"window.lobStudies.state().last === {unit}")
    assert page.evaluate("window.lobStudies.rows().every((row, i, rows) => !i || row[0] - rows[i - 1][0] === 1)")
    assert page.evaluate(TILE, "cvd")["state"] == "ok"


def test_the_studies_follow_a_playback_and_a_seek(page, base_url):
    studies_open(page, base_url)
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'true'")
    run_for(page, 1500)
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'false'")
    page.click("#tab-analyze")
    page.wait_for_selector("#recordings-list [data-recording-id]")
    version = page.evaluate("window.lobStudies.state().version")
    page.click("#recordings-list [data-recording-id]:first-child button[data-recording-action='play']")
    page.wait_for_function(f"{BADGE}.playback === 'true'")
    page.click("#tab-trade")
    # (afresh)
    page.wait_for_function(f"window.lobStudies.state().version > {version} && window.lobStudies.state().loaded")
    assert (page.evaluate("window.lobStudies.state().last") or 0) <= 1

    page.evaluate("""() => {
        const timeline = document.getElementById('playback-timeline');
        timeline.value = timeline.max;
        timeline.dispatchEvent(new Event('change'));
    }""")
    page.wait_for_function(f"{BADGE}.pbEnded === 'true'")
    server = page.evaluate("fetch('/api/studies').then(r => r.json()).then(d => ({unit: d.unit, version: d.version}))")
    assert server["unit"] > 5
    page.wait_for_function(f"window.lobStudies.state().last === {server['unit']}"
                           f" && window.lobStudies.state().version === {server['version']}")
    assert page.evaluate("window.lobStudies.rows().every((row, i, rows) => !i || row[0] - rows[i - 1][0] === 1)")
    page.click("#sim-reset-btn")  # rewinds: the studies start again from the recording's start
    page.wait_for_function(f"{BADGE}.pbCursor === '0'")
    page.wait_for_function("window.lobStudies.state().last <= 1")
    page.click("#playback-exit")
    page.wait_for_function(f"{BADGE}.playback === 'false'")


# ---- The tape ----

def tick_rule(prices):
    """The tick rule's sign for each price of a sequence (None until a price has changed), as the studies take it."""
    signs, previous, last = [], None, None
    for price in prices:
        if previous is not None:
            last = 1 if price > previous else -1 if price < previous else last
        signs.append(None if previous is None else last)
        previous = price
    return signs


def inject_tape(page, fills, bump=0):
    page.evaluate("""([fills, bump]) => { const s = window.lobTape.state();
        document.dispatchEvent(new CustomEvent('lob:tape', {detail: {version: s.version + bump,
            seq: fills[fills.length - 1][0],
                                                                      reload: true, fills}})); }""", [fills, bump])


def test_the_tape_filters_marks_the_tick_rule_and_computes_the_rate_from_the_fills_it_holds(page, base_url):
    open_dashboard(page, base_url)
    page.wait_for_function("window.lobTape.state().loaded")
    # A tape of known fills (the simulation is paused, so nothing else arrives):
    # a number, a time, a price, a size, a side.
    prices = [100, 101, 101, 100, 100, 100, 102, 101, 101, 103] * 3
    sides = [1, 1, -1, -1, 1, -1, 1, -1, -1, 1] * 3
    sizes = [5, 5, 10, 5, 20, 5, 5, 15, 5, 5] * 3
    fills = [[1000 + i, 100 + 0.5 * i, prices[i], sizes[i], sides[i]] for i in range(30)]
    inject_tape(page, fills)
    page.wait_for_function("window.lobTape.state().shown.length === 30")
    shown = page.evaluate("window.lobTape.state().shown")
    assert [r["seq"] for r in shown] == list(range(1029, 999, -1))  # newest first
    assert [r["price"] for r in shown] == [str(p) for p in prices[::-1]]
    assert [r["side"] for r in shown] == sides[::-1] and shown[0]["time"] == "114.50s"
    assert page.inner_text("#tape-count") == "30 shown of 30"
    assert page.evaluate("document.querySelector('#tape-rows [data-seq=\"1029\"] [data-side]').textContent") == "BUY"

    # The rate: the fills in the last 10 units (after 104.5) over 10.
    assert page.evaluate("window.lobTape.state().rate") == pytest.approx(20 / 10)
    assert page.inner_text("#tape-rate") == "2.0 fills/unit"
    page.fill("#tape-window", "20")  # more than the tape holds: all 30 fills over the 14.5 units they span
    page.wait_for_function("Math.abs(window.lobTape.state().rate - 30 / 14.5) < 1e-9")
    page.fill("#tape-window", "10")
    page.wait_for_function("window.lobTape.state().rate === 2")

    # Filters: side and minimum size, kept through what arrives (the filters are outside any pushed HTML).
    page.select_option("#tape-side", "buy")
    shown = page.evaluate("window.lobTape.state().shown")
    assert [r["seq"] for r in shown] == [1000 + i for i in range(29, -1, -1) if sides[i] == 1]
    page.select_option("#tape-side", "sell")
    assert all(r["side"] == -1 for r in page.evaluate("window.lobTape.state().shown"))
    page.select_option("#tape-side", "all")
    page.fill("#tape-min", "10")
    shown = page.evaluate("window.lobTape.state().shown")
    assert [r["seq"] for r in shown] == [1000 + i for i in range(29, -1, -1) if sizes[i] >= 10]
    assert page.inner_text("#tape-count") == f"{len(shown)} shown of 30"
    page.fill("#tape-min", "21")
    assert page.evaluate("window.lobTape.state().shown") == [] and "No fills match" in page.inner_text("#tape-empty")
    assert page.evaluate("window.lobTape.state().rate") == pytest.approx(2.0)  # the rate counts every fill
    page.fill("#tape-min", "1")

    # The tick rule from the price sequence the tape has seen: a mark where it differs from the true side.
    page.check("#tape-tick")
    signs = tick_rule(prices)
    wrong = [i for i in range(30) if signs[i] is not None and signs[i] != sides[i]]
    marked = page.evaluate("window.lobTape.state().shown.filter(r => r.tick).map(r => r.seq)")
    assert sorted(marked) == [1000 + i for i in wrong]
    window = [i for i in range(30) if 100 + 0.5 * i > 104.5 and signs[i] is not None]
    share = 100 * sum(1 for i in window if signs[i] != sides[i]) / len(window)
    assert page.inner_text("#tape-share") == f"tick rule wrong on {share:.1f}% of {len(window)} fills (10 units)"
    page.uncheck("#tape-tick")
    assert page.evaluate("window.lobTape.state().shown.filter(r => r.tick).length") == 0
    assert page.inner_text("#tape-share") == ""

    # The settings of this browser are kept.
    page.select_option("#tape-side", "buy")
    page.fill("#tape-window", "7")
    page.reload()
    page.wait_for_function("window.lobTape.state().loaded")
    assert page.input_value("#tape-side") == "buy" and page.input_value("#tape-window") == "7"


def test_the_tape_reloads_on_a_new_version_a_gap_and_when_it_falls_behind(page, base_url):
    open_dashboard(page, base_url)
    page.wait_for_function("window.lobTape.state().loaded")
    requests = []
    page.on("request", lambda r: requests.append(r.url) if r.url.split("?")[0].endswith("/api/tape") else None)
    inject_tape(page, [[500 + i, 1.0 + i, 100, 5, 1] for i in range(5)])
    page.wait_for_function("window.lobTape.state().held === 5")
    assert requests == []  # a push that says "reload" is the whole tape: nothing to ask for

    push = """detail => { const s = window.lobTape.state();
        document.dispatchEvent(new CustomEvent('lob:tape', {detail: {version: s.version + detail.bump,
            seq: detail.seq, reload: false, fills: detail.fills}})); }"""
    page.evaluate(push, {"bump": 0, "seq": 505, "fills": [[505, 6.0, 101, 5, 1]]})  # the next one: appended
    page.wait_for_function("window.lobTape.state().held === 6")
    assert requests == []
    for detail in ({"bump": 0, "seq": 600, "fills": [[600, 7.0, 101, 5, 1]]},  # numbers missing: a gap
                   {"bump": 1, "seq": 506, "fills": [[506, 7.0, 101, 5, 1]]}):  # another version, the next number
        asked = len(requests)
        page.evaluate(push, detail)
        page.wait_for_function("window.lobTape.state().loaded && window.lobTape.state().held !== 6")
        page.wait_for_timeout(200)
        assert len(requests) == asked + 1, detail
        # what the server has, not the fakes:
        assert page.evaluate("window.lobTape.state().shown.every(r => r.seq < 500)")
        inject_tape(page, [[500 + i, 1.0 + i, 100, 5, 1] for i in range(6)])
        page.wait_for_function("window.lobTape.state().held === 6")
    # The stream connecting again asks for all of it. The chart reloading (an interval, the footprint) does not: the
    # tape does not depend on the chart, and a reset is a new version.
    asked = len(requests)
    page.evaluate("document.dispatchEvent(new CustomEvent('lob:connection', {detail: 'live'}))")
    page.wait_for_function("window.lobTape.state().loaded && window.lobTape.state().held !== 6")
    page.wait_for_timeout(200)
    assert len(requests) == asked + 1
    inject_tape(page, [[500 + i, 1.0 + i, 100, 5, 1] for i in range(6)])
    page.wait_for_function("window.lobTape.state().held === 6")
    asked = len(requests)
    page.evaluate("document.dispatchEvent(new CustomEvent('lob:chart-reset'))")
    page.click("#btn-interval-5")
    page.wait_for_function("loadedInterval === 5")
    page.wait_for_timeout(300)
    assert len(requests) == asked and page.evaluate("window.lobTape.state().held") == 6


def test_a_paused_tape_and_its_button_stay_in_step_through_a_reload_a_reset_and_the_charts_reloads(page, base_url):
    open_dashboard(page, base_url)
    page.wait_for_function("window.lobTape.state().loaded")
    inject_tape(page, [[500 + i, 1.0 + i, 100, 5, 1] for i in range(60)])
    page.wait_for_function("window.lobTape.state().shown.length === 50")
    page.click("#tape-pause")
    assert page.evaluate("window.lobTape.state().paused") is True
    assert page.inner_text("#tape-pause") == "Resume" and page.get_attribute("#tape-pause", "aria-pressed") == "true"
    held = [r["seq"] for r in page.evaluate("window.lobTape.state().shown")]

    # The chart reloading (an interval) is not the tape's business: still paused, the same list, and nothing asked.
    requests = []
    page.on("request", lambda r: requests.append(r.url) if r.url.split("?")[0].endswith("/api/tape") else None)
    page.click("#btn-interval-5")
    page.wait_for_function("loadedInterval === 5")
    page.wait_for_timeout(300)
    assert page.evaluate("window.lobTape.state().paused") is True and requests == []
    assert page.inner_text("#tape-pause") == "Resume"
    assert [r["seq"] for r in page.evaluate("window.lobTape.state().shown")] == held

    # A reload of fills the list has moved past (a new market, or a tape that fell behind): it follows again, and the
    # button says so, so that its next click pauses.
    inject_tape(page, [[900 + i, 1.0 + i, 100, 5, 1] for i in range(60)])
    page.wait_for_function("!window.lobTape.state().paused")
    assert page.inner_text("#tape-pause") == "Pause" and page.get_attribute("#tape-pause", "aria-pressed") == "false"
    page.wait_for_selector("#tape-newest", state="hidden")
    assert [r["seq"] for r in page.evaluate("window.lobTape.state().shown")][0] == 959
    page.click("#tape-pause")
    assert page.evaluate("window.lobTape.state().paused") is True and page.inner_text("#tape-pause") == "Resume"

    # The same through a real Reset.
    page.click("#sim-reset-btn")
    page.wait_for_function("!window.lobTape.state().paused")
    assert page.inner_text("#tape-pause") == "Pause" and page.get_attribute("#tape-pause", "aria-pressed") == "false"
    assert page.evaluate("window.lobTape.state().held") == 0
    page.click("#tape-pause")
    assert page.evaluate("window.lobTape.state().paused") is True and page.inner_text("#tape-pause") == "Resume"
    page.click("#tape-pause")
    assert page.evaluate("window.lobTape.state().paused") is False and page.inner_text("#tape-pause") == "Pause"


def test_the_tape_fills_holds_still_when_paused_or_scrolled_and_its_rate_is_the_exports(page, base_url):
    open_dashboard(page, base_url)
    page.evaluate("setSpeedPreset(200)")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_function("window.lobTape.state().shown.length === 50")
    state = page.evaluate("window.lobTape.state()")
    seqs = [r["seq"] for r in state["shown"]]
    assert seqs == sorted(seqs, reverse=True) and len(set(seqs)) == 50  # newest first, once each
    assert all(r["side"] in (1, -1) and re.fullmatch(r"\d+\.\d\ds", r["time"]) for r in state["shown"])
    assert page.inner_text("#tape-count").endswith("of 2000") or page.evaluate("window.lobTape.state().held") < 2000

    # A filter typed while it runs is still there after the pushes.
    page.fill("#tape-min", "5")
    page.wait_for_timeout(600)
    assert page.input_value("#tape-min") == "5"
    page.fill("#tape-min", "1")

    # Paused, the list holds still while the fills go on arriving, and says how many came.
    page.click("#tape-pause")
    assert page.get_attribute("#tape-pause", "aria-pressed") == "true" and page.is_visible("#tape-newest")
    held = [r["seq"] for r in page.evaluate("window.lobTape.state().shown")]
    page.wait_for_function(f"window.lobTape.state().seq > {held[0]} + 3")
    assert [r["seq"] for r in page.evaluate("window.lobTape.state().shown")] == held
    assert re.fullmatch(r"Jump to newest \(\+\d+\+?\)", page.inner_text("#tape-newest"))
    page.click("#tape-newest")
    assert not page.is_visible("#tape-newest") and page.get_attribute("#tape-pause", "aria-pressed") == "false"
    page.wait_for_function(f"window.lobTape.state().shown[0].seq > {held[0]} + 3")

    # Scrolled away from the top it holds still too, and returns when scrolled back.
    page.evaluate("document.getElementById('tape-list').scrollTop = 90")
    page.wait_for_function("window.lobTape.state().frozen")
    held = [r["seq"] for r in page.evaluate("window.lobTape.state().shown")]
    page.wait_for_function(f"window.lobTape.state().seq > {held[0]} + 3")
    assert [r["seq"] for r in page.evaluate("window.lobTape.state().shown")] == held
    assert page.evaluate("document.getElementById('tape-list').scrollTop") == 90  # and its scroll is not reset
    page.evaluate("document.getElementById('tape-list').scrollTop = 0")
    page.wait_for_function(f"window.lobTape.state().shown[0].seq > {held[0]}")

    # Stopped: the rate is the export's count of the last 10 units over 10, to the fill.
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    page.wait_for_timeout(500)
    state = page.evaluate("window.lobTape.state()")
    export = page.evaluate("fetch('/api/export/trades.csv').then(r => r.text())")
    times = [float(line.split(",")[0]) for line in export.strip().split("\n")[1:]]
    assert state["held"] == len(times) or state["held"] == 2000
    last = times[-1]
    assert state["rate"] == pytest.approx(sum(1 for t in times if t > last - 10) / 10)
    assert page.inner_text("#tape-rate") == f"{state['rate']:.1f} fills/unit"
    live = page.evaluate("fetch('/api/tape').then(r => r.json())")
    assert state["seq"] == live["seq"] and state["shown"][0]["seq"] == live["fills"][-1][0]


def test_the_tick_rule_marks_of_a_live_tape_are_the_ones_the_price_sequence_gives(page, base_url):
    open_dashboard(page, base_url)
    run_for(page, 2500)
    page.wait_for_function("window.lobTape.state().shown.length === 50")
    page.check("#tape-tick")
    live = page.evaluate("fetch('/api/tape').then(r => r.json())")["fills"]
    signs = tick_rule([f[2] for f in live])
    wrong = {f[0] for f, sign in zip(live, signs, strict=True) if sign is not None and sign != f[4]}
    marked = page.evaluate("window.lobTape.state().shown.filter(r => r.tick).map(r => r.seq)")
    shown = {r["seq"] for r in page.evaluate("window.lobTape.state().shown")}
    assert wrong & shown == set(marked)
    assert 0 < len(wrong) < len(live) / 2  # the tick rule is mostly right, never all
    assert re.fullmatch(r"tick rule wrong on \d+\.\d% of \d+ fills \(10 units\)", page.inner_text("#tape-share"))


# ---- The footprint ----

FOOT = "document.getElementById('price-chart-container').dataset"
FOOT_ON = f"{FOOT}.footprint === 'on'"
FOOT_OFF = f"{FOOT}.footprint === 'off'"
TRANSPARENT = "rgba(0, 0, 0, 0)"


def footprint_reference(cells, ratio, minimum):
    """The point of control and the diagonal imbalances of one bar's cells [price, sold, bought], by the definition."""
    sold = {price: s for price, s, _ in cells}
    bought = {price: b for price, _, b in cells}
    busiest = max(s + b for _, s, b in cells)
    return {
        "poc": min(price for price, s, b in cells if s + b == busiest),  # the lowest of equals
        "buy": sorted(p for p, _, b in cells if b > 0 and b >= minimum and b >= ratio * sold.get(p - 1, 0)),
        "sell": sorted(p for p, s, _ in cells if s > 0 and s >= minimum and s >= ratio * bought.get(p + 1, 0)),
    }


def trade_cells(page, interval):
    """From the export, each bucket's fills by price: {bucket: {price: [sold into the bid, bought from the ask]}}."""
    text = page.evaluate("fetch('/api/export/trades.csv').then(r => r.text())")
    cells = {}
    for line in text.strip().split("\n")[1:]:
        time, price, size, aggressor = line.split(",")[:4]
        cell = cells.setdefault(int(float(time) // interval * interval), {}).setdefault(int(price), [0, 0])
        cell[aggressor == "buy"] += int(size)
    return cells


# The cells to look at on the screen: for each, a point a fifth of a bar either side of the bar's center, in its row.
PROBES = """() => { const ts = priceChart.timeScale(), out = [];
    for (const time of lobFootprint.state().last.bars.map(b => b.time)) {
        const x = ts.timeToCoordinate(time), poc = lobFootprint.derived(time).poc, q = ts.options().barSpacing / 5;
        for (const [price, sold, bought] of lobFootprint.cells(time)) {
            const y = Math.round(candleSeries.priceToCoordinate(price));
            if (price !== poc && x > 6) {
                out.push({sold, bought, left: [Math.floor(x - q), y], right: [Math.floor(x + q), y]});
            }
        }
    }
    return out; }"""
# The colors of a PNG (base64) at points: [r, g, b] each.
PIXELS = """async ([base64, points]) => { const image = new Image();
    image.src = 'data:image/png;base64,' + base64;
    await image.decode(); const canvas = document.createElement('canvas');
    canvas.width = image.width; canvas.height = image.height;
    const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
    return points.map(([x, y]) => Array.from(ctx.getImageData(x, y, 1, 1).data.slice(0, 3))); }"""


def test_the_footprint_draws_each_bars_cells_and_they_add_up_to_the_bars(page, base_url):
    open_dashboard(page, base_url)
    page.click("#btn-interval-15")
    page.wait_for_function("loadedInterval === 15")
    assert page.evaluate(FOOT_OFF) and page.get_attribute("#footprint-toggle", "aria-pressed") == "false"
    assert not page.is_visible("#footprint-ratio") and page.evaluate("candleSeries.options().upColor") != TRANSPARENT
    assert "15 units or more" in page.inner_text("#footprint-hint")
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    assert page.get_attribute("#footprint-toggle", "aria-pressed") == "true" and page.is_visible("#footprint-ratio")
    # The candle bodies are transparent (not hidden: the price scale still autoscales to them); the primitive is on.
    assert page.evaluate("candleSeries.options().upColor") == TRANSPARENT
    assert page.evaluate("candleSeries.options().downColor") == TRANSPARENT
    assert page.evaluate("candleSeries.options().visible") is True and page.evaluate("lobFootprint.state().attached")

    run_for(page, 8000, speed=1000)
    state = page.evaluate("lobFootprint.state()")
    assert state["error"] is None and state["withCells"] >= 5 and state["bad"] == []
    assert page.evaluate(f"{FOOT}.footprintCheck") == "ok"
    assert int(page.evaluate(f"{FOOT}.footprintBars")) == state["withCells"]
    candles = page.evaluate("candleSeries.data().map(c => c.time)")
    assert state["times"] == candles and all(t % 15 == 0 for t in candles)  # the bars of the candles, no others

    # The cells are the fills of the export, price by price, and add up to the bar:
    # buys to its buy volume, all to its volume.
    exported = trade_cells(page, 15)
    bars = page.evaluate("fetch('/api/candlesticks').then(r => r.json()).then(d => d.candles)")
    for bar in bars:
        cells = page.evaluate("t => lobFootprint.cells(t)", bar["time"])
        assert cells == [[p, s, b] for p, (s, b) in sorted(exported.get(bar["time"], {}).items())], bar["time"]
        assert sum(b for _, _, b in cells) == bar["buy_volume"] and sum(s + b for _, s, b in cells) == bar["volume"]

    # Switched on at an interval where it reads, the chart was widened for it (from the usual few pixels a bar).
    assert page.evaluate("priceChart.timeScale().options().barSpacing") >= 40
    page.wait_for_function("lobFootprint.state().last.numbers")
    # Back to the usual zoom, where the cells are colors only.
    page.evaluate("""() => [priceChart, volumeChart].forEach(c => {
        c.timeScale().applyOptions({barSpacing: 6}); c.timeScale().scrollToRealTime(); })""")
    page.wait_for_function("lobFootprint.state().last.spacing < 10")

    # At the usual zoom the cells are colors only. Look at what is on the screen: in each cell's row, a quarter of a bar
    # either side of the bar's center, the half is the color of its side where there was volume (red sold, green bought)
    # and the background where there was none. The candles' outlines and the VWAP line, which cross the cells, are
    # hidden for the look (and the point of control, which is outlined, left out).
    assert page.evaluate("lobFootprint.state().last.numbers") is False
    assert page.evaluate("lobFootprint.state().last.spacing") < 40
    page.evaluate("""() => { const none = 'rgba(0, 0, 0, 0)';
        candleSeries.applyOptions({borderUpColor: none, borderDownColor: none, wickUpColor: none, wickDownColor: none});
        vwapSeries.applyOptions({visible: false}); }""")
    found = []
    for _ in range(5):  # the chart may still be settling its layout (the axis widths): look again if it moved meanwhile
        page.wait_for_timeout(300)
        before = page.evaluate(PROBES)
        png = page.locator("#price-chart-container").screenshot()
        if page.evaluate(PROBES) != before:
            continue
        colors = page.evaluate(PIXELS, [base64.b64encode(png).decode(),
                                        [p for probe in before for p in (probe["left"], probe["right"])]])
        right = 0
        for i, probe in enumerate(before):
            (lr, lg, _), (rr, rg, _) = colors[2 * i], colors[2 * i + 1]
            right += ((lr > lg + 8) == (probe["sold"] > 0)) and ((rg > rr + 8) == (probe["bought"] > 0))
        found.append((right, len(before)))
        if len(before) >= 20 and right >= 0.95 * len(before):
            break
    page.evaluate("applyChartTheme(); vwapSeries.applyOptions({visible: true})")
    assert found and found[-1][1] >= 20 and found[-1][0] >= 0.95 * found[-1][1], found

    # Widened, the numbers fit. What was drawn is where the chart puts it.
    page.click("#footprint-wide")
    page.wait_for_function("lobFootprint.state().last.spacing >= 40 && lobFootprint.state().last.numbers")
    page.wait_for_timeout(300)
    drawn = page.evaluate("""() => { const f = lobFootprint.state(), ts = priceChart.timeScale();
        return {spacing: f.last.spacing, rowHeight: f.last.rowHeight, bars: f.last.bars.map(b => {
            const d = lobFootprint.derived(b.time);
            return {time: b.time, poc: b.poc === d.poc, dx: Math.abs(ts.timeToCoordinate(b.time) - b.x),
                    dtop: Math.abs(candleSeries.priceToCoordinate(d.top) - b.top),
                    dbottom: Math.abs(candleSeries.priceToCoordinate(d.bottom) - b.bottom),
                    row: Math.abs(candleSeries.priceToCoordinate(d.top - 1)
                                  - candleSeries.priceToCoordinate(d.top))}; })}; }""")
    context = page.evaluate("""() => { const ts = priceChart.timeScale(), f = lobFootprint.state();
        return {times: f.times, drawn: f.last.bars.map(b => b.time), range: ts.getVisibleLogicalRange(),
                visible: ts.getVisibleRange(), spacing: ts.options().barSpacing, width: ts.width()}; }""")
    assert len(drawn["bars"]) >= 5 and drawn["rowHeight"] >= 9, context
    for bar in drawn["bars"]:
        assert bar["dx"] < 0.5 and bar["dtop"] < 0.5 and bar["dbottom"] < 0.5 and bar["poc"], bar
        assert bar["row"] == pytest.approx(drawn["rowHeight"], abs=0.5)
    # A bar is drawn where the candle is: the center of its time, between the price of its top cell and its bottom one.
    assert all(abs(b["x"] - page.evaluate("t => priceChart.timeScale().timeToCoordinate(t)", b["time"])) < 0.5
               for b in page.evaluate("lobFootprint.state().last.bars"))

    # A finished bar's column is drawn once and copied while nothing about it changes.
    page.evaluate(REPAINT)
    before = page.evaluate("lobFootprint.state()")
    for _ in range(3):
        page.evaluate(REPAINT)
    after = page.evaluate("lobFootprint.state()")
    assert after["paints"] > before["paints"] and after["columns"] == before["columns"]
    assert after["copies"] > before["copies"]

    # The point of control and the imbalances follow the settings (a ratio and the lots an imbalance needs: they are
    # this browser's).
    for ratio, minimum in ((3, 10), (1, 1), (2, 20)):
        page.fill("#footprint-ratio", str(ratio))
        page.fill("#footprint-min", str(minimum))
        page.wait_for_function(f"lobFootprint.state().settings.ratio === {ratio}"
                               f" && lobFootprint.state().settings.minLots === {minimum}")
        for bar in bars[-6:]:
            cells = page.evaluate("t => lobFootprint.cells(t)", bar["time"])
            if cells:
                got = page.evaluate("t => lobFootprint.derived(t)", bar["time"])
                reference = footprint_reference(cells, ratio, minimum)
                assert (got["poc"], sorted(got["buyImbalance"]), sorted(got["sellImbalance"])) == (
                    reference["poc"], reference["buy"], reference["sell"])
    page.fill("#footprint-ratio", "3")
    page.fill("#footprint-min", "10")
    assert page.evaluate("JSON.parse(localStorage.getItem('lob-footprint'))") == {"ratio": 3, "minLots": 10}


def test_switching_the_footprint_on_widens_the_chart_once_where_it_reads_and_leaves_a_zoom_alone(page, base_url):
    open_dashboard(page, base_url)
    spacing = "priceChart.timeScale().options().barSpacing"
    run_for(page, 8000, speed=1000)

    # At an interval too short to read it is left as it is (the hint says so).
    page.click("#btn-interval-5")
    page.wait_for_function("loadedInterval === 5")
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    page.wait_for_timeout(600)
    assert page.evaluate(spacing) < 24
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_OFF)

    # At one where it reads, switching on widens the bars to where the numbers show.
    page.click("#btn-interval-30")
    page.wait_for_function("loadedInterval === 30")
    assert page.evaluate(spacing) < 24
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    page.wait_for_function(f"{spacing} >= 40")
    page.wait_for_function("lobFootprint.state().withCells >= 1 && lobFootprint.state().last.numbers")
    assert page.evaluate("lobFootprint.state().error") is None

    # Switched off and on again at a zoom of the user's own, it leaves that alone; so does every later reload.
    page.evaluate("setHorizontalZoom(30)")
    mine = page.evaluate(spacing)
    assert 20 < mine < 40
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_OFF)
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    page.wait_for_timeout(600)
    assert page.evaluate(spacing) == pytest.approx(mine, abs=1.0)
    page.click("#btn-interval-15")
    page.wait_for_function("loadedInterval === 15 && lobFootprint.state().interval === 15")
    page.wait_for_timeout(300)
    assert page.evaluate(spacing) == pytest.approx(mine, abs=1.0)


def test_the_footprint_keeps_the_last_2000_bars_and_counts_those_with_cells_as_they_come_and_go(page, base_url):
    open_dashboard(page, base_url)
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    result = page.evaluate("""() => {
        const bar = (i, ok) => ({time: 100000 + 2 * i, volume: 5, buy_volume: i % 7 === 0 ? 5 : 0,
            cells: [[100, i % 7 === 0 ? 0 : 5, i % 7 === 0 ? 5 : 0]].concat(ok ? [] : [[101, 1, 0]])});
        lobFootprint.bar(bar(0, false));  // the oldest disagrees with its volume: it is let go with the rest
        const flagged = lobFootprint.state().bad.length;
        for (let i = 1; i < 2600; i++) lobFootprint.bar(bar(i, i !== 2590));
        const s = lobFootprint.state();
        return {flagged, bars: s.bars, withCells: s.withCells, times: s.times, bad: s.bad,
                attribute: Number(document.getElementById('price-chart-container').dataset.footprintBars),
                badAttribute: document.getElementById('price-chart-container').dataset.footprintBad,
                check: document.getElementById('price-chart-container').dataset.footprintCheck,
                gone: lobFootprint.cells(100000), kept: lobFootprint.cells(100000 + 2 * 2599)}; }""")
    assert result["flagged"] == 1
    assert result["bars"] == 2000 and len(result["times"]) == 2000
    assert result["times"][0] == 100000 + 2 * 600 and result["times"][-1] == 100000 + 2 * 2599
    assert result["times"] == sorted(result["times"])
    assert result["gone"] is None and result["kept"] is not None
    assert result["withCells"] == 2000 and result["attribute"] == 2000  # what is counted as bars go equals a count
    assert result["bad"] == [100000 + 2 * 2590] and result["badAttribute"] == "1" and result["check"] == "bad"
    # A bar pushed again, changed, is neither counted twice nor lost.
    page.evaluate("""() => {
        lobFootprint.bar({time: 100000 + 2 * 2590, volume: 5, buy_volume: 0, cells: [[100, 5, 0]]});
        lobFootprint.bar({time: 100000 + 2 * 2599, volume: 5, buy_volume: 5, cells: null}); }""")
    page.wait_for_function("lobFootprint.state().bad.length === 0")
    state = page.evaluate("lobFootprint.state()")
    assert state["bars"] == 2000 and state["withCells"] == 1999
    assert int(page.evaluate(f"{FOOT}.footprintBars")) == 1999 and page.evaluate(f"{FOOT}.footprintCheck") == "ok"
    assert state["error"] is None


def test_a_bar_whose_cells_do_not_add_up_is_flagged_for_the_tests_and_not_in_the_console(page, base_url):
    open_dashboard(page, base_url)
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    run_for(page, 2500, speed=1000)
    page.wait_for_function("lobFootprint.state().withCells >= 1")
    assert page.evaluate(f"{FOOT}.footprintCheck") == "ok"
    # A pushed bar whose cells are 5 lots short of its volume (and one whose buy cells are not its buy volume).
    broken = page.evaluate("""() => { const c = {...candleSeries.data().slice(-1)[0], volume: 25,
            buy_volume: 10, sell_volume: 15,
            cells: [[100, 10, 5], [101, 5, 0]]};  // 20 lots in all, 5 bought
        lobFootprint.bar(c); return lobFootprint.state().bad; }""")
    assert len(broken) == 1 and page.evaluate(f"{FOOT}.footprintCheck") == "bad"
    assert page.evaluate(f"{FOOT}.footprintBad") == "1"
    page.evaluate("""() => { const c = {...candleSeries.data().slice(-1)[0], volume: 20, buy_volume: 5, sell_volume: 15,
            cells: [[100, 10, 5], [101, 5, 0]]}; lobFootprint.bar(c); }""")
    assert page.evaluate(f"{FOOT}.footprintCheck") == "ok" and page.evaluate("lobFootprint.state().bad") == []
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_OFF)


def test_the_footprint_follows_the_interval_a_reset_another_tab_and_turns_off(page, base_url):
    a = open_dashboard(page, base_url)
    a.click("#btn-interval-15")
    a.wait_for_function("loadedInterval === 15")
    a.click("#footprint-toggle")
    a.wait_for_function(FOOT_ON)
    run_for(a, 5000, speed=1000)
    a.wait_for_function("lobFootprint.state().withCells >= 3")

    # Another tab of the session shows it too: the setting is the session's, and the tab's chart reloads with cells.
    b = open_dashboard(a.context.new_page(), base_url)
    b.wait_for_function(FOOT_ON)
    b.wait_for_function("lobFootprint.state().withCells >= 3 && lobFootprint.state().attached")
    assert b.evaluate("candleSeries.options().upColor") == TRANSPARENT
    assert b.get_attribute("#footprint-toggle", "aria-pressed") == "true"
    a.bring_to_front()

    # Another interval: the bars are rebuilt at it, with their cells, and the check holds.
    a.click("#btn-interval-30")
    a.wait_for_function("lobFootprint.state().interval === 30 && currentInterval === 30")
    a.wait_for_function("lobFootprint.state().times.length > 0 && lobFootprint.state().times.every(t => t % 30 === 0)")
    state = a.evaluate("lobFootprint.state()")
    assert state["times"] == a.evaluate("candleSeries.data().map(c => c.time)") and state["bad"] == []
    assert a.evaluate(f"{FOOT}.footprintCheck") == "ok"
    exported = trade_cells(a, 30)
    for time in state["times"]:
        cells = a.evaluate("t => lobFootprint.cells(t)", time)
        assert cells == [[p, s, b] for p, (s, b) in sorted(exported.get(time, {}).items())], time
    assert "text-amber-400" not in a.get_attribute("#footprint-hint", "class")
    a.click("#btn-interval-5")  # too short a bar to read: the hint says so
    a.wait_for_function("loadedInterval === 5 && lobFootprint.state().interval === 5")
    assert "text-amber-400" in a.get_attribute("#footprint-hint", "class")
    a.click("#btn-interval-15")
    a.wait_for_function("lobFootprint.state().interval === 15")

    # A reset: the cells start over with the bars, the footprint stays on, and what comes is checked again.
    a.click("#sim-reset-btn")
    a.wait_for_function("lobFootprint.state().times.length <= 1")
    assert a.evaluate(FOOT_ON) and a.evaluate("lobFootprint.state().attached")
    run_for(a, 4000, speed=1000)
    a.wait_for_function("lobFootprint.state().withCells >= 2")
    assert a.evaluate(f"{FOOT}.footprintCheck") == "ok" and a.evaluate("lobFootprint.state().bad") == []
    assert a.evaluate("lobFootprint.state().times") == a.evaluate("candleSeries.data().map(c => c.time)")

    # The zoom goes as wide as the numbers need while it is on (and no wider than before when it is not).
    spacing = "priceChart.timeScale().options().barSpacing"
    a.evaluate("setHorizontalZoom(60)")
    assert a.evaluate(spacing) > 40 and a.get_attribute("#hzoom-slider", "max") == "72"

    # Off: no cells in the data, the candles opaque again, the primitive gone, and the other tab follows.
    a.click("#footprint-toggle")
    a.wait_for_function(FOOT_OFF)
    a.evaluate("setHorizontalZoom(60)")
    assert a.evaluate(spacing) <= 36.5 and a.get_attribute("#hzoom-slider", "max") == "32"
    assert a.evaluate("candleSeries.options().upColor") != TRANSPARENT
    assert not a.evaluate("lobFootprint.state().attached")
    assert a.evaluate("fetch('/api/candlesticks').then(r => r.json())"
                      ".then(d => d.footprint === false && !('cells' in d.candles[0]))")
    assert not a.is_visible("#footprint-ratio") and a.evaluate(f"{FOOT}.footprintCheck") == "none"
    b.bring_to_front()
    b.wait_for_function(FOOT_OFF)
    assert b.evaluate("candleSeries.options().upColor") != TRANSPARENT
    assert not b.evaluate("lobFootprint.state().attached")


def test_the_footprint_is_redrawn_with_the_theme_and_survives_the_chart_being_hidden_and_shown(page, base_url):
    open_dashboard(page, base_url)
    page.click("#btn-interval-15")
    page.wait_for_function("loadedInterval === 15")
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    run_for(page, 4000, speed=1000)
    page.wait_for_function("lobFootprint.state().withCells >= 2")
    page.click("#footprint-wide")
    page.wait_for_function("lobFootprint.state().last.numbers")
    columns = page.evaluate("lobFootprint.state().columns")
    # (the chart's theme step makes the bodies opaque: not while the footprint is on)
    page.select_option("#theme-select", "daylight")
    # the cached columns are drawn again in the new colors
    page.wait_for_function(f"lobFootprint.state().columns > {columns}")
    assert page.evaluate("candleSeries.options().upColor") == TRANSPARENT
    page.click("#palette-toggle")
    page.wait_for_function(f"lobFootprint.state().columns > {columns} + 1")
    assert page.evaluate("candleSeries.options().upColor") == TRANSPARENT
    # The Analyze view and back: the chart is as it was, still drawn.
    page.click("#tab-analyze")
    page.click("#tab-trade")
    paints = page.evaluate("lobFootprint.state().paints")
    page.evaluate("lobFootprint.repaint()")
    page.wait_for_function(f"lobFootprint.state().paints > {paints}")
    assert page.evaluate("lobFootprint.state().error") is None
    assert page.evaluate("lobFootprint.state().last.bars.length") >= 2


def test_the_study_band_sits_below_the_trade_grid_across_the_width_and_wraps_on_a_phone(page, base_url):
    boxes = """() => { const box = id => { const r = document.getElementById(id).getBoundingClientRect();
        return {left: r.left, right: r.right, top: r.top, bottom: r.bottom}; };
        const tiles = [...document.querySelectorAll('#studies-tiles [data-tile]')];
        const tops = tiles.map(t => Math.round(t.getBoundingClientRect().top));
        return {band: box('studies-band'), chart: box('chart-panel'), trades: box('trades-container'),
                ladder: box('ladder-container'),
                strategies: box('strategies-card'), form: box('order-form-container'), pnl: box('pnl-wrapper'),
                perRow: tops.filter(t => t === tops[0]).length,
                tilesWide: tiles.every(t => t.scrollWidth <= t.clientWidth + 1),
                overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth}; }"""
    for width, height, per_row in ((1440, 900, 8), (1728, 1000, 8), (1100, 800, 4), (390, 844, 2)):
        open_dashboard(page, base_url, width, height)
        page.wait_for_function(STUDIES_LOADED)
        box = page.evaluate(boxes)
        grid_bottom = max(box[k]["bottom"] for k in ("chart", "trades", "ladder", "form", "pnl"))
        assert box["band"]["top"] >= grid_bottom, (width, box)  # below the grid, never above it
        assert box["band"]["bottom"] <= box["strategies"]["top"]  # and above the strategies card
        for side in ("left", "right"):  # the width of the strategies card
            assert box["band"][side] == pytest.approx(box["strategies"][side], abs=1)
        assert box["perRow"] == per_row, (width, box["perRow"])
        assert box["tilesWide"] and box["overflow"] == 0, width
    # The chart keeps its room at 1440x900 (the band is not above it), and the text size scales the band too.
    open_dashboard(page, base_url, 1440, 900)
    assert page.evaluate("document.getElementById('volume-chart-container').getBoundingClientRect().bottom") <= 900
    small = page.evaluate("document.getElementById('studies-band').getBoundingClientRect().height")
    page.select_option("#text-select", "large")
    assert page.evaluate("document.getElementById('studies-band').getBoundingClientRect().height") > small


def test_the_band_and_the_tape_are_drawn_only_while_the_trade_view_is_shown_and_catch_up_when_it_is(page, base_url):
    open_dashboard(page, base_url)
    page.evaluate("setSpeedPreset(200)")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_function("window.lobStudies.state().last > 5 && window.lobTape.state().shown.length > 5")
    page.click("#tab-analyze")  # the stream stays open here: the data go on arriving
    page.wait_for_timeout(300)
    unit = page.evaluate("window.lobStudies.state().last")
    seq = page.evaluate("window.lobTape.state().seq")
    status = page.evaluate("document.getElementById('studies-status').textContent")
    newest = page.evaluate("window.lobTape.state().shown[0].seq")
    page.wait_for_function(f"window.lobStudies.state().last > {unit} + 3 && window.lobTape.state().seq > {seq} + 3")
    assert page.evaluate("document.getElementById('studies-status').textContent") == status  # nothing was drawn
    assert page.evaluate("window.lobTape.state().shown[0].seq") == newest
    page.click("#tab-trade")
    page.wait_for_function(f"document.getElementById('studies-status').textContent !== {status!r}")
    page.wait_for_function(f"window.lobTape.state().shown[0].seq > {newest}")
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")


def test_the_footprint_draws_only_the_bars_in_view(page, base_url):
    open_dashboard(page, base_url)
    page.click("#btn-interval-5")
    page.wait_for_function("loadedInterval === 5")
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    run_for(page, 7000, speed=1000)
    page.wait_for_function("lobFootprint.state().withCells >= 24")
    page.click("#footprint-wide")
    page.wait_for_function("lobFootprint.state().last.spacing >= 40")
    page.wait_for_timeout(300)
    state = page.evaluate("lobFootprint.state()")
    visible = state["last"]["bars"]
    assert 8 <= len(visible) < state["withCells"] - 4  # the newest bars that fit, not all that there are
    assert visible[-1]["time"] == state["times"][-1]
    assert [b["time"] for b in visible] == [t for t in state["times"] if t >= visible[0]["time"]]  # in order, no holes
    width = page.evaluate("priceChart.timeScale().width()")
    spacing = state["last"]["spacing"]  # at most a bar either side of the window is drawn
    assert all(-spacing - 1 < b["x"] < width + spacing + 1 for b in visible)
    # Scrolled back to the first bars, those are drawn and the newest are not.
    page.evaluate("() => { const ts = priceChart.timeScale();"
                  " ts.setVisibleLogicalRange({from: -0.5, to: ts.width() / 48 - 0.5}); }")
    page.wait_for_function("lobFootprint.state().last.bars.length > 0 && lobFootprint.state().last.bars[0].time === 0")
    first = page.evaluate("lobFootprint.state().last.bars")
    assert first[-1]["time"] < state["times"][-1] and len(first) <= width / 48 + 3  # the window's bars, not all of them
    assert all(-spacing - 1 < b["x"] < width + spacing + 1 for b in first)


def test_the_trade_view_stays_responsive_at_2000_events_a_second_with_everything_on(page, base_url, engine):
    if engine != "chrome":
        pytest.skip("long tasks are measured in Chromium")
    open_dashboard(page, base_url)
    page.click("#btn-interval-15")
    page.wait_for_function("loadedInterval === 15")
    page.click("#footprint-toggle")
    page.wait_for_function(FOOT_ON)
    page.evaluate("""() => { window.__long = []; new PerformanceObserver(list => {
        for (const e of list.getEntries()) window.__long.push(e.duration);
    }).observe({type: 'longtask', buffered: true}); }""")
    page.evaluate("setSpeedPreset(2000)")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_function("window.lobStudies.state().last > 20 && window.lobFootprint.state().withCells > 2")
    frames = page.evaluate("""() => new Promise(done => { const gaps = [], t0 = performance.now(); let last = t0;
        (function tick(now) { gaps.push(now - last); last = now;
            if (now - t0 < 3000) requestAnimationFrame(tick); else done(gaps); })(t0); })""")
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    frames.sort()
    long_tasks = page.evaluate("window.__long")
    assert len(frames) > 30
    assert frames[len(frames) // 2] < 50 and frames[-1] < 500, (frames[len(frames) // 2], frames[-1])
    assert not [d for d in long_tasks if d > 250], long_tasks
    assert page.evaluate("lobFootprint.state().error") is None and page.evaluate(f"{FOOT}.footprintCheck") == "ok"


# ---- Triggers (static/js/lob-triggers.js): rules on the studies' values ----

TRIGGER_ROW = "#triggers-rows tr[data-rule-id='{}']"
TRIGGERS_READY = "window.lobTriggers && document.getElementById('trigger-form')"
TRIGGER_RULES = "fetch('/api/triggers').then(r => r.json()).then(s => s.rules)"
TOASTS = "document.getElementById('toasts').textContent"


def triggers_open(page, base_url, width=1440, height=900):
    open_dashboard(page, base_url, width, height)
    page.wait_for_function(TRIGGERS_READY)
    return page


def post_rule(page, **fields):
    """Make a rule the way the editor does: the fields of the form, a list for each that repeats (``action``)."""
    return page.evaluate("""fields => { const body = new FormData();
        Object.entries(fields).forEach(([k, v]) => [].concat(v).forEach(x => body.append(k, x)));
        return fetch('/api/triggers', {method: 'POST', body}).then(r => r.status); }""", fields)


def to_middle(page, selector):
    """Scroll an element to the middle of the window: the sticky header would cover one parked at the top."""
    page.evaluate("s => document.querySelector(s).scrollIntoView({block: 'center'})", selector)


def trigger_row(page, rule_id, key):
    return page.evaluate(f"""() => {{ const tr = document.querySelector("{TRIGGER_ROW.format(rule_id)}");
        return tr && ({{state: tr.dataset.state, armed: tr.dataset.armed, count: tr.dataset.count,
            name: tr.querySelector('[data-name]').textContent, text: tr.querySelector('[data-text]').textContent,
            how: tr.querySelector('[data-how]').textContent, fired: tr.querySelector('[data-fired]').textContent,
            values: tr.querySelector('[data-values]').textContent,
            toggle: tr.querySelector('[data-action=arm], [data-action=disarm]').textContent}})["{key}"]; }}""")


def test_a_rule_on_ofi_crossing_a_level_pauses_the_run_and_the_card_follows_it_through_edit_disarm_delete(
        page, base_url):
    triggers_open(page, base_url)
    assert page.is_visible("#triggers-empty") and not page.is_visible("#triggers-table")
    assert page.evaluate("window.lobTriggers.state().rules.length") == 0

    # The editor: OFI crosses above 0, and pause (which makes the rule one-shot unless told otherwise).
    to_middle(page, "#trigger-form")
    assert page.input_value("#trigger-form select[name=value]") == "ofi:ofi"
    page.fill("#trigger-form input[name=name]", "OFI burst")
    page.select_option("#trigger-form select[name=op]", "crosses_above")
    page.fill("#trigger-form input[name=threshold]", "0")
    assert "true only on the event it happens" in page.inner_text("#trigger-conditions [data-edge]")
    assert not page.is_checked("#trigger-one-shot")
    page.check("#trigger-form input[name=action][value=pause]")
    assert page.is_checked("#trigger-one-shot")
    page.click("#trigger-submit")
    page.wait_for_function("document.getElementById('triggers-result').textContent.includes('Added OFI burst')")
    page.wait_for_selector(TRIGGER_ROW.format(1))
    assert trigger_row(page, 1, "name") == "OFI burst" and trigger_row(page, 1, "text") == "OFI crosses above 0 lots"
    assert trigger_row(page, 1, "how") == "notice, pause · cooldown 10 · one-shot"
    assert trigger_row(page, 1, "toggle") == "Disarm" and trigger_row(page, 1, "count") == "0"
    assert page.is_visible("#triggers-table") and not page.is_visible("#triggers-empty")
    assert page.inner_text("#triggers-status") == "1 rule, 1 armed"

    # Run: the rule pauses the simulation by itself, the row says it fired and disarmed, the toast says what it did.
    page.evaluate("setSpeedPreset(500)")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_function(f"!({RUNNING})", timeout=40_000)
    page.wait_for_function(f"document.querySelector(\"{TRIGGER_ROW.format(1)}\").dataset.state === 'disarmed'")
    assert trigger_row(page, 1, "count") == "1" and trigger_row(page, 1, "fired").startswith("t ")
    assert trigger_row(page, 1, "toggle") == "Arm"
    page.wait_for_function(f"{TOASTS}.includes(\"Trigger 'OFI burst' at time \")")
    assert "OFI" in page.evaluate(TOASTS) and "crossed above 0. Paused." in page.evaluate(TOASTS)
    events = page.evaluate(EVENTS)
    page.wait_for_timeout(600)
    assert page.evaluate(EVENTS) == events and page.inner_text("#sim-state-label") != "Running"  # it stays paused
    (rule,) = page.evaluate(TRIGGER_RULES)
    assert (rule["count"], rule["armed"], rule["state"]) == (1, False, "disarmed")
    # The firing marks the chart, once, with the rule's name.
    page.wait_for_function("candleSeries.markers().some(m => m.text === 'OFI burst')")
    assert page.evaluate("candleSeries.markers().filter(m => m.text === 'OFI burst').length") == 1

    # Edit: the editor is filled from the rule; saving clears what it had seen and the chart's mark goes.
    to_middle(page, TRIGGER_ROW.format(1))
    page.click(TRIGGER_ROW.format(1) + " button[data-action=edit]")
    assert page.input_value("#trigger-form input[name=name]") == "OFI burst"
    assert page.input_value("#trigger-form select[name=op]") == "crosses_above"
    assert page.input_value("#trigger-form input[name=threshold]") == "0"
    assert page.is_checked("#trigger-form input[name=action][value=pause]") and page.is_checked("#trigger-one-shot")
    assert page.inner_text("#trigger-submit") == "Save changes" and page.is_visible("#trigger-editing")
    page.fill("#trigger-form input[name=name]", "OFI burst 2")
    page.fill("#trigger-form input[name=threshold]", "25")
    page.click("#trigger-submit")
    page.wait_for_function("document.getElementById('triggers-result').textContent.includes('Saved OFI burst 2')")
    assert page.inner_text("#trigger-submit") == "Add the rule" and not page.is_visible("#trigger-editing")
    page.mouse.move(5, 5)
    page.wait_for_function(f"document.querySelector(\"{TRIGGER_ROW.format(1)}\").querySelector('[data-name]').textContent"
                           " === 'OFI burst 2'")
    assert trigger_row(page, 1, "text") == "OFI crosses above 25 lots" and trigger_row(page, 1, "fired") == "—"
    assert trigger_row(page, 1, "state") == "disarmed"  # an edit leaves it as it was
    page.wait_for_function("!candleSeries.markers().some(m => m.text === 'OFI burst')")

    # Arm, then disarm: the row follows at once, and the server agrees.
    page.click(TRIGGER_ROW.format(1) + " button[data-action=arm]")
    page.wait_for_function(f"document.querySelector(\"{TRIGGER_ROW.format(1)}\").dataset.armed === 'true'")
    assert trigger_row(page, 1, "toggle") == "Disarm"
    assert page.evaluate(TRIGGER_RULES + ".then(r => r[0].armed)") is True
    page.click(TRIGGER_ROW.format(1) + " button[data-action=disarm]")
    page.wait_for_function(f"document.querySelector(\"{TRIGGER_ROW.format(1)}\").dataset.armed === 'false'")
    assert page.evaluate(TRIGGER_RULES + ".then(r => r[0].armed)") is False
    page.mouse.move(5, 5)

    # Delete: the row goes (when the pointer has left), and the empty note is back.
    to_middle(page, TRIGGER_ROW.format(1))
    page.click(TRIGGER_ROW.format(1) + " button[data-action=delete]")
    page.mouse.move(5, 5)
    page.wait_for_function("!document.querySelector(\"#triggers-rows tr[data-rule-id='1']\")")
    assert page.is_visible("#triggers-empty") and page.evaluate(TRIGGER_RULES + ".then(r => r.length)") == 0


def test_the_editor_adds_and_removes_conditions_states_the_edge_rule_and_shows_the_servers_reasons(page, base_url):
    triggers_open(page, base_url)
    to_middle(page, "#trigger-form")
    conditions = "#trigger-conditions [data-condition]"
    assert page.locator(conditions).count() == 1 and not page.is_visible("#trigger-combine")
    assert page.get_attribute("#trigger-conditions [data-remove-condition]", "aria-disabled") == "true"  # the last one
    edge = lambda i: page.locator(f"{conditions} [data-edge]").nth(i).inner_text()  # noqa: E731
    assert edge(0) == "true at every event while it holds"
    page.select_option("#trigger-form select[name=op]", "crosses_below")
    assert "true only on the event it happens" in edge(0) and "at or above the level" in edge(0)
    page.select_option("#trigger-form select[name=op]", "crosses_above")
    assert "at or below the level" in edge(0)

    for _ in range(3):
        page.click("#trigger-add-condition")
    assert page.locator(conditions).count() == 4
    assert page.get_attribute("#trigger-add-condition", "aria-disabled") == "true"
    page.dispatch_event("#trigger-add-condition", "click")  # no fifth
    assert page.locator(conditions).count() == 4 and page.is_visible("#trigger-combine")
    page.locator(f"{conditions} [data-remove-condition]").nth(3).click()
    page.locator(f"{conditions} [data-remove-condition]").nth(2).click()
    assert page.locator(conditions).count() == 2
    assert page.get_attribute("#trigger-add-condition", "aria-disabled") is None

    # Two crossings in an all-of rule: the editor says so, and the server refuses it in the same words.
    page.locator(f"{conditions} select[name=op]").nth(1).select_option("crosses_below")
    page.locator(f"{conditions} select[name=value]").nth(1).select_option("cvd:cvd")
    assert page.is_visible("#triggers-result [data-hint=crossings]")
    page.click("#trigger-submit")
    page.wait_for_selector("#triggers-result .text-rose-400")  # the server's answer replaces the editor's hint
    assert "only one crossing" in page.inner_text("#triggers-result") and not page.is_visible("#triggers-table")
    assert not page.is_visible("#triggers-result [data-hint]")
    page.check("#trigger-form input[name=combine][value=any]")
    assert not page.is_visible("#triggers-result [data-hint=crossings]") or page.evaluate(
        "!document.querySelector('#triggers-result [data-hint]')")

    page.fill("#trigger-form input[name=cooldown]", "-1")
    page.click("#trigger-submit")
    page.wait_for_function(
        "document.getElementById('triggers-result').textContent.includes('cooldown must be between')")
    page.fill("#trigger-form input[name=cooldown]", "3")
    page.uncheck("#trigger-form input[name=action][value=notice]")
    page.click("#trigger-submit")
    page.wait_for_function("document.getElementById('triggers-result').textContent.includes('at least one action')")
    page.check("#trigger-form input[name=action][value=notice]")
    page.click("#trigger-submit")
    page.wait_for_function("document.getElementById('triggers-result').textContent.includes('Added')")
    page.wait_for_selector(TRIGGER_ROW.format(1))
    assert trigger_row(page, 1, "text") == "any of: OFI crosses above 0 lots or CVD crosses below 0 lots"
    assert trigger_row(page, 1, "name") == "OFI crosses above 0 lots or CVD crosses"  # made from them, 40 characters
    assert trigger_row(page, 1, "how") == "notice · cooldown 3"
    assert page.evaluate("document.querySelectorAll('#triggers-rows [data-values] > div').length") == 2
    page.mouse.move(5, 5)


def test_the_list_waits_while_the_pointer_is_on_a_rules_buttons_and_catches_up_when_it_leaves(page, base_url):
    triggers_open(page, base_url)
    page.evaluate("""() => window.postForm('/api/triggers', {name: 'every step', value: 'cvd:cvd', op: 'above',
        threshold: -1e9, action: 'notice', cooldown: 0})""")
    page.wait_for_selector(TRIGGER_ROW.format(1))
    page.evaluate("setSpeedPreset(100)")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_function("window.lobTriggers.state().rules[0].count > 20")
    to_middle(page, TRIGGER_ROW.format(1))
    page.hover(TRIGGER_ROW.format(1) + " button[data-action=disarm]")
    page.wait_for_function("window.lobTriggers.state().held")
    shown = trigger_row(page, 1, "count")
    page.wait_for_function(f"window.lobTriggers.state().rules[0].count > Number({shown}) + 50")  # it went on
    assert page.evaluate("window.lobTriggers.state().queued") is True
    assert trigger_row(page, 1, "count") == shown  # and the list did not move under the pointer
    page.click("#sim-play-btn")  # the click is on the play button: the pointer has left the list
    page.wait_for_function(f"!({RUNNING})")
    page.wait_for_function("!window.lobTriggers.state().held")
    latest = page.evaluate("window.lobTriggers.state().rules[0].count")
    page.wait_for_function(f"document.querySelector(\"{TRIGGER_ROW.format(1)}\").dataset.count === '{latest}'")


def test_a_firing_is_marked_on_the_chart_beside_the_fills_of_the_user_and_neither_replaces_the_other(page, base_url):
    triggers_open(page, base_url)
    page.evaluate("""() => window.postForm('/api/triggers', {name: 'tick', value: 'cvd:cvd', op: 'above',
        threshold: -1e9, action: 'notice', cooldown: 4})""")
    page.wait_for_selector(TRIGGER_ROW.format(1))
    run_for(page, 2500, speed=200)
    page.evaluate("window.postForm('/api/order/market', {size: 6, side: 'bid'})")
    page.wait_for_function("candleSeries.markers().some(m => m.text.startsWith('Buy 6'))")
    run_for(page, 1500, speed=200)
    markers = page.evaluate("candleSeries.markers().map(m => ({time: m.time, text: m.text, shape: m.shape}))")
    fired = [m for m in markers if m["shape"] == "circle"]
    fills = [m for m in markers if m["text"].startswith("Buy")]
    assert len(fills) == 1 and len(fired) >= 2
    assert [m["time"] for m in markers] == sorted(m["time"] for m in markers)  # one list, in time order
    assert len({m["time"] for m in fired}) == len(fired)  # one a bar
    # Only the newest firing carries the rule's name: a name on each would print over its neighbors'.
    assert [m["text"] for m in fired[:-1]] == [""] * (len(fired) - 1) and fired[-1]["text"].startswith("tick")
    # They follow the chart's interval: every mark sits on a bar of it.
    bars = page.evaluate("candleSeries.data().map(b => b.time)")
    assert all(m["time"] in bars for m in markers)


FIRED_LABELS = """() => {
    // What is on the chart: the circles of the firings, and for those that carry a name how far apart the names sit
    // against how wide each is, measured here in the font the chart writes it in.
    const layout = priceChart.options().layout, ts = priceChart.timeScale();
    const context = document.createElement('canvas').getContext('2d');
    context.font = `${layout.fontSize}px ${layout.fontFamily}`;
    const circles = candleSeries.markers().filter(m => m.shape === 'circle');
    const labeled = circles.filter(m => m.text).map(m => ({time: m.time, text: m.text, x: ts.timeToCoordinate(m.time),
        width: context.measureText(m.text).width}));
    return {circles: circles.length, labeled, spacing: ts.options().barSpacing,
            clear: labeled.every((a, i) => i === 0 || a.x - labeled[i - 1].x >= (a.width + labeled[i - 1].width) / 2)};
}"""


def zoom_chart(page, spacing):
    page.evaluate("""s => [priceChart, volumeChart].forEach(c => {
        c.timeScale().applyOptions({barSpacing: s}); c.timeScale().scrollToRealTime(); })""", spacing)
    page.wait_for_function(f"Math.abs(priceChart.timeScale().options().barSpacing - {spacing}) < 0.01")


def test_only_the_newest_firing_of_a_rule_carries_its_name_and_names_too_close_to_read_share_a_label(page, base_url):
    triggers_open(page, base_url)
    run_for(page, 3000, speed=1000)
    page.wait_for_function("candleSeries.data().length >= 16")
    bars = page.evaluate("candleSeries.data().map(b => b.time)")
    alpha, bravo = "Alpha rule with a long name", "Bravo rule with a long name"
    # Firings, as the page takes them from the stream: Alpha four times, the last two in one bar, and Bravo ten bars
    # after that; Alpha's newest bar is 10 bars before the end of the chart.
    marks = [[bars[-14] + 0.1, 1, alpha], [bars[-13] + 0.1, 1, alpha], [bars[-11] + 0.1, 1, alpha],
             [bars[-11] + 0.4, 1, alpha], [bars[-1] + 0.1, 2, bravo]]
    page.evaluate("""marks => document.dispatchEvent(new CustomEvent('lob:triggers',
        {detail: {rules: [], marks, limit: 12, needs_model: false}}))""", marks)

    # Narrow bars: the two names are 10 bars, 60 pixels apart and each 150 wide: one label, on the later, naming both.
    zoom_chart(page, 6)
    page.wait_for_function("candleSeries.markers().filter(m => m.shape === 'circle').length === 4")
    got = page.evaluate(FIRED_LABELS)
    assert got["circles"] == 4 and got["clear"]
    assert [(m["time"], m["text"]) for m in got["labeled"]] == [(bars[-1], "Bravo rule with a lo +1")]

    # Wide bars: 10 of them are 360 pixels, the names fit, and each rule has one name, on its newest firing. The two
    # firings of Alpha in the one bar are counted there.
    zoom_chart(page, 36)
    page.wait_for_function("candleSeries.markers().filter(m => m.text).length === 2")
    got = page.evaluate(FIRED_LABELS)
    assert got["circles"] == 4 and got["clear"]
    assert [(m["time"], m["text"]) for m in got["labeled"]] == [
        (bars[-11], "Alpha rule with a lo ×2"), (bars[-1], "Bravo rule with a lo")]
    assert all(m["x"] is not None for m in got["labeled"]) and got["labeled"][1]["x"] - got["labeled"][0]["x"] > 300


def test_names_on_the_chart_from_rules_that_fire_every_few_bars_do_not_overprint_each_other(page, base_url):
    triggers_open(page, base_url)
    for name, cooldown in (("Alpha rule with a long name", 4), ("Bravo rule with a long name", 6)):
        assert post_rule(page, name=name, value="cvd:cvd", op="above", threshold=-1e9, action="notice",
                         cooldown=cooldown) == 200
    run_for(page, 3000, speed=1000)
    page.wait_for_function("candleSeries.markers().filter(m => m.shape === 'circle').length >= 6")
    got = page.evaluate(FIRED_LABELS)
    assert got["circles"] >= 6 and 1 <= len(got["labeled"]) <= 2 and got["clear"], got  # many circles, a name or two


def test_after_a_rule_is_added_the_editor_starts_clean_and_keeps_the_message(page, base_url):
    triggers_open(page, base_url)
    to_middle(page, "#trigger-form")
    page.fill("#trigger-form input[name=name]", "Twice over")
    page.click("#trigger-add-condition")
    second = "#trigger-conditions [data-condition]:nth-child(2) "
    page.select_option(second + "select[name=value]", "cvd:cvd")
    page.select_option(second + "select[name=op]", "crosses_below")
    page.fill(second + "input[name=threshold]", "-5")
    page.check("#trigger-form input[name=combine][value=any]")
    page.check("#trigger-form input[name=action][value=pause]")
    # A refusal leaves what was typed, to be put right.
    page.fill("#trigger-form input[name=cooldown]", "-1")
    page.click("#trigger-submit")
    page.wait_for_function(
        "document.getElementById('triggers-result').textContent.includes('cooldown must be between')")
    assert page.input_value("#trigger-form input[name=name]") == "Twice over"
    assert page.locator("#trigger-conditions [data-condition]").count() == 2
    assert page.input_value(second + "input[name=threshold]") == "-5"
    assert page.input_value("#trigger-form input[name=cooldown]") == "-1"
    assert page.is_checked("#trigger-form input[name=action][value=pause]")
    page.fill("#trigger-form input[name=cooldown]", "7")
    page.click("#trigger-submit")
    page.wait_for_function("document.getElementById('triggers-result').textContent.includes('Added Twice over')")
    page.wait_for_selector(TRIGGER_ROW.format(1))
    # The form is as a page that has just loaded has it; what was done is still said.
    assert page.input_value("#trigger-form input[name=name]") == ""
    assert page.locator("#trigger-conditions [data-condition]").count() == 1 and not page.is_visible("#trigger-combine")
    assert page.input_value("#trigger-form select[name=value]") == "ofi:ofi"
    assert page.input_value("#trigger-form select[name=op]") == "above"
    assert page.input_value("#trigger-form input[name=threshold]") == "0"
    assert page.input_value("#trigger-form input[name=cooldown]") == "10"
    assert page.is_checked("#trigger-form input[name=combine][value=all]")
    assert page.is_checked("#trigger-form input[name=action][value=notice]")
    assert not page.is_checked("#trigger-form input[name=action][value=pause]")
    assert not page.is_checked("#trigger-one-shot")
    assert page.inner_text("#trigger-submit") == "Add the rule" and not page.is_visible("#trigger-editing")
    assert "Added" in page.inner_text("#triggers-result")
    # Another press of Add makes a rule of what the editor holds now, not a copy of the one before.
    page.click("#trigger-submit")
    page.wait_for_selector(TRIGGER_ROW.format(2))
    first, again = page.evaluate(TRIGGER_RULES)
    assert (first["name"], first["combine"], first["actions"], first["cooldown"], first["one_shot"]) == (
        "Twice over", "any", ["notice", "pause"], 7.0, True)
    assert len(first["conditions"]) == 2 and first["conditions"][1]["op"] == "crosses_below"
    assert (again["text"], again["actions"], again["cooldown"], again["one_shot"]) == (
        "OFI above 0 lots", ["notice"], 10.0, False)


def test_the_rules_table_keeps_its_columns_together_at_any_width_and_never_widens_the_page(page, base_url):
    triggers_open(page, base_url, 1728, 1000)
    assert post_rule(page, name="OFI burst", value="ofi:ofi", op="crosses_above", threshold=50, action="notice") == 200
    long_name = "A name of the longest length a rule can."  # 40 characters: more than its cell holds at this width
    assert len(long_name) == 40
    assert post_rule(page, name=long_name, value="cvd:cvd", op="above", threshold=5, action=["notice", "pause"]) == 200
    page.wait_for_selector(TRIGGER_ROW.format(2))
    to_middle(page, "#triggers-table")
    columns = """() => { const table = document.getElementById('triggers-table');
        const box = (el) => el.getBoundingClientRect();
        const rem = parseFloat(getComputedStyle(document.documentElement).fontSize), t = box(table);
        const heads = [...table.querySelectorAll('thead th')].map(box);
        return {rem, width: t.width, widths: heads.map(h => h.width), now: heads[1].left - heads[0].left,
                fired: heads[2].left - heads[0].left,
                end: heads[3].right - t.left, wrap: [table.parentElement.clientWidth, table.parentElement.scrollWidth],
                tight: [...table.querySelectorAll('tbody td')].filter(td => td.scrollWidth > td.clientWidth + 1).length,
                page: [document.documentElement.scrollWidth, innerWidth]}; }"""
    got = page.evaluate(columns)
    # In the width of a wide window the table stops at 68 rem and its columns are fixed, so what a rule reads now
    # (which changes at every push) sits within 26 rem of the rule's name, not at the far end of the card.
    assert got["width"] <= 68 * got["rem"] + 1 and got["now"] <= 26 * got["rem"] and got["fired"] <= 50 * got["rem"]
    assert got["tight"] == 0 and got["page"][0] <= got["page"][1] and got["wrap"][0] == got["wrap"][1]
    # The widths are the ones the markup gives (a column that followed its content would move at every push).
    assert [round(w) for w in got["widths"][::2]] == [round(0.36 * got["width"]), round(8.5 * got["rem"])] and round(
        got["widths"][3]) == round(11 * got["rem"])
    # And they stay when what a rule reads is too long for its column (here a word of 70 letters, as the page takes it
    # from the stream): the text spills over, the table does not move.
    page.evaluate("""() => { const s = lobTriggers.state(); s.rules[0].conditions[0].label = 'W'.repeat(70);
        document.dispatchEvent(new CustomEvent('lob:triggers', {detail: s})); }""")
    page.wait_for_function("document.querySelector('#triggers-rows [data-values]').textContent.includes('WWWWWWWW')")
    assert page.evaluate(columns)["widths"] == got["widths"]
    # In a narrow one the table scrolls inside its box and the page stays as wide as the window (a visually hidden
    # label that is positioned would stick out of the box and widen the page).
    page.set_viewport_size({"width": 390, "height": 800})
    page.wait_for_function("innerWidth === 390")
    to_middle(page, "#triggers-table")
    got = page.evaluate(columns)
    assert got["wrap"][1] > got["wrap"][0] and got["page"][0] <= got["page"][1] == 390, got  # a scrollbar may narrow it


def test_the_card_says_when_triggers_need_a_running_model_and_follows_a_new_rule_in_another_tab(page, base_url):
    a = triggers_open(page, base_url)
    b = triggers_open(page.context.new_page(), base_url)
    a.evaluate("window.postForm('/api/sim/config', {model: 'manual'})")
    a.wait_for_function("window.lobTriggers.state().needsModel")
    assert "needs a running model" in a.inner_text("#triggers-status")
    a.evaluate("""() => window.postForm('/api/triggers', {name: 'from a script', value: 'cvd:cvd', op: 'above',
        threshold: 5, action: 'notice'})""")
    b.wait_for_selector(TRIGGER_ROW.format(1))  # the other tab, with no reload
    assert trigger_row(b, 1, "name") == "from a script" and "needs a running model" in b.inner_text("#triggers-status")
    assert post_rule(a, name="made disarmed", value="cvd:cvd", op="above", threshold=5, action="notice",
                     armed="off") == 200
    b.wait_for_selector(TRIGGER_ROW.format(2))  # a rule that starts disarmed has its Arm button, labeled and styled
    assert trigger_row(b, 2, "toggle") == "Arm" and trigger_row(b, 2, "state") == "disarmed"
    assert "emerald" in b.get_attribute(TRIGGER_ROW.format(2) + " button[data-action=arm]", "class")
    a.evaluate("window.postForm('/api/sim/config', {model: 'smith_zi'})")
    b.wait_for_function("!window.lobTriggers.state().needsModel")


def test_the_card_says_that_in_a_playback_only_the_notice_and_the_pause_act(page, base_url):
    triggers_open(page, base_url)
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'true'")
    run_for(page, 800)
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'false'")
    page.click("#tab-analyze")
    page.wait_for_selector("#recordings-list [data-recording-id]")
    page.click("#recordings-list [data-recording-id]:first-child button[data-recording-action='play']")
    page.wait_for_function(f"{BADGE}.playback === 'true'")
    page.click("#tab-trade")
    assert post_rule(page, name="in playback", value="cvd:cvd", op="above", threshold=-1e9,
                     action=["notice", "cancel", "flatten"], cooldown=100) == 200
    page.wait_for_selector(TRIGGER_ROW.format(1))
    page.wait_for_function("window.lobTriggers.state().playback")
    status = page.inner_text("#triggers-status")
    assert "playback" in status and "cancel and flatten are off" in status
    page.evaluate("stepSim(20)")
    page.wait_for_function("window.lobTriggers.state().rules[0].count >= 1")
    page.wait_for_function(f"{TOASTS}.includes('Cancel is off in a playback')")


# ---- The informed-traders model: its ground truth and Reveal ----

INFORMED = "window.lobInformed.state()"
NEEDS_MODEL = "needs the ZI with informed traders model"
NO_ORDERS_YET = "document.getElementById('informed-line').textContent === 'Informed traders: no market orders yet'"
TRUTH = re.compile(r"Informed traders: ([\d.]+)% of the model's taker lots \(([\d,]+) of ([\d,]+)\); "
                   r"makers lose (-?[\d.]+) ticks a lot to them, against the hidden value")


def choose_informed(page):
    """Switch to the model with informed traders; the switch is enabled once the server has said so."""
    page.select_option("select[name=model]", "informed_zi")
    settle(page)
    page.wait_for_function("!document.getElementById('reveal-toggle').disabled")


def reveal_matches_the_server(page):
    """Every point of the line is the `fair` of the candle at its time that /api/candlesticks gives, and the chart
    has no point where the server has no value."""
    page.wait_for_function("""() => fetch('/api/candlesticks').then(r => r.json()).then(d => {
        const drawn = lobInformed.data();
        const fair = d.candles.filter(c => c.fair != null);
        return fair.length === drawn.length && fair.every((c, i) =>
            drawn[i].time === c.time && Math.abs(drawn[i].value - c.fair) < 1e-9);
    })""")


def test_the_informed_model_reports_its_ground_truth_and_reveal_draws_the_hidden_value_and_turns_off(page, base_url):
    open_dashboard(page, base_url)
    # Another model: the line says none, and the switch is off with its reason (a click on it does nothing).
    assert page.inner_text("#informed-line") == "Informed traders: none in this model"
    assert page.is_disabled("#reveal-toggle") and page.get_attribute("#reveal-toggle", "aria-pressed") == "false"
    assert page.inner_text("#reveal-note") == NEEDS_MODEL
    assert NEEDS_MODEL in page.get_attribute("#reveal-toggle", "title")
    assert page.evaluate(INFORMED)["points"] == 0 and not page.is_visible("#fair-legend")
    assert page.is_visible("#model-params summary .truncate")  # the model's summary, in the bar

    choose_informed(page)
    assert page.locator("#model-params summary .truncate").count() == 0  # the line of the ground truth has the room
    assert page.inner_text("#informed-line") == "Informed traders: no market orders yet"
    assert page.inner_text("#reveal-note") == "" and not page.is_disabled("#reveal-toggle")
    assert page.get_attribute("#reveal-toggle", "aria-pressed") == "false" and not page.is_visible("#fair-legend")
    page.click("#reveal-toggle")
    page.wait_for_function(f"{INFORMED}.pressed && {INFORMED}.legend")
    assert page.is_visible("#fair-legend") and page.inner_text("#fair-legend").strip() == "hidden value (revealed)"
    assert page.get_attribute("#price-chart-container", "data-reveal") == "on"

    run_for(page, 5000, speed=1000)
    state = page.evaluate(INFORMED)
    assert state["error"] is None and state["points"] >= 20 and state["status"]["active"] is True
    reveal_matches_the_server(page)
    # A line of its own color: not the candles', the VWAP's or the order lines'.
    colors = page.evaluate("[themeColor('violet-400'), themeColor('emerald-500'), themeColor('rose-500'),"
                           " themeColor('amber-500')]")
    assert state["color"] == colors[0] and state["color"] not in colors[1:]
    assert page.evaluate("fetch('/api/candlesticks').then(r => r.json()).then(d => d.reveal.active)")

    # The ground truth, counted live: a share of the model's lots, in lots of the model's size (5), and a loss that
    # is positive (informed orders are sent only when the value lies beyond the quote they hit).
    text = page.inner_text("#informed-line")
    pct, informed_lots, taker_lots, loss = TRUTH.fullmatch(text).groups()
    informed_lots, taker_lots = int(informed_lots.replace(",", "")), int(taker_lots.replace(",", ""))
    assert informed_lots % 5 == 0 and taker_lots % 5 == 0 and 0 < informed_lots < taker_lots >= 100
    assert float(pct) == pytest.approx(100 * informed_lots / taker_lots, abs=0.051)
    assert 2 < float(pct) < 45 and 0 < float(loss) < 8
    assert page.get_attribute("#informed-line", "data-informed") == "yes"

    # Another tab of the session shows the line, the pressed switch and the value.
    b = open_dashboard(page.context.new_page(), base_url)
    b.wait_for_function(f"{INFORMED}.pressed && {INFORMED}.legend && {INFORMED}.points >= 20")
    reveal_matches_the_server(b)
    assert TRUTH.fullmatch(b.inner_text("#informed-line"))
    page.bring_to_front()

    # Off: no value in the data, no line, no legend; the other tab follows.
    page.click("#reveal-toggle")
    page.wait_for_function(f"!{INFORMED}.pressed && !{INFORMED}.legend && {INFORMED}.points === 0")
    assert page.get_attribute("#price-chart-container", "data-reveal") == "off"
    assert page.evaluate("fetch('/api/candlesticks').then(r => r.json())"
                         ".then(d => d.reveal.on === false && d.candles.every(c => !('fair' in c)))")
    b.bring_to_front()
    b.wait_for_function(f"!{INFORMED}.pressed && {INFORMED}.points === 0")
    b.close()
    page.bring_to_front()

    # Pushes after it is off carry no value: the line stays gone as the run goes on.
    run_for(page, 1500, speed=1000)
    assert page.evaluate(INFORMED)["points"] == 0

    # On again, then another model: the switch is off and disabled with its reason, the line says none, the value goes.
    page.click("#reveal-toggle")
    page.wait_for_function(f"{INFORMED}.pressed && {INFORMED}.points >= 1")
    page.select_option("select[name=model]", "smith_zi")
    settle(page)
    page.wait_for_function(
        f"{INFORMED}.disabled && !{INFORMED}.pressed && !{INFORMED}.legend && {INFORMED}.points === 0")
    assert page.inner_text("#reveal-note") == NEEDS_MODEL
    assert page.inner_text("#informed-line") == "Informed traders: none in this model"
    assert page.get_attribute("#informed-line", "data-informed") == "no"


def test_the_line_of_the_ground_truth_survives_the_parameter_panel_being_replaced_and_follows_a_reset(page, base_url):
    open_dashboard(page, base_url)
    choose_informed(page)
    run_for(page, 2500, speed=1000)
    page.wait_for_function("window.lobInformed.state().line.includes('% of the model')")
    before = TRUTH.fullmatch(page.inner_text("#informed-line")).groups()
    # The panel is replaced by new parameters: the line it comes back with is the server's, and goes on updating.
    page.click("#model-params summary")  # open it
    page.fill("#model-params input[name=kappa]", "4")
    page.click("#model-params button[type=submit]")
    settle(page)
    page.wait_for_function("document.querySelector('#model-params').textContent.includes('Applied')")
    # A model rebuilt on the book starts its counts again.
    page.wait_for_function(NO_ORDERS_YET)
    run_for(page, 1500, speed=1000)
    after = TRUTH.fullmatch(page.inner_text("#informed-line")).groups()
    assert after != before
    assert page.inner_text("#informed-line") == page.evaluate(INFORMED)["line"]
    # A change of controls that arrives while the panel is being edited waits (focus is in it), and is rendered as of
    # the time it was sent. When the panel is replaced the line says what the last push did, not what it had rendered.
    page.focus("#model-params input[name=kappa]")
    page.evaluate("""() => { const form = new FormData(document.querySelector('#model-params form'));
        form.set('cap', '5'); return fetch('/api/sim/params', {method: 'POST', body: form}); }""")
    page.wait_for_function("window.lobInformed.state().line === 'Informed traders: no market orders yet'")
    page.evaluate("stepSim(3000)")  # the line moves on, while the panel still holds the older render
    page.wait_for_function("window.lobInformed.state().text.includes('% of the model')")
    pushed = page.evaluate("window.lobInformed.state().text")
    assert "cap=3.0" in page.get_attribute("#model-params", "data-signature")  # not replaced yet
    page.evaluate("document.activeElement.blur()")
    page.wait_for_function("document.getElementById('model-params').dataset.signature.includes('cap=5.0')")
    page.wait_for_function("t => document.getElementById('informed-line').textContent === t", arg=pushed)
    # A reset starts the counts again.
    page.click("#sim-reset-btn")
    page.wait_for_function(NO_ORDERS_YET)
    page.wait_for_function(f"{INFORMED}.line === 'Informed traders: no market orders yet'")


def test_a_bar_pushed_for_another_version_of_the_chart_is_dropped_and_one_for_this_version_is_drawn(page, base_url):
    # A bar pushed before a reset or a change of interval can arrive after the reload that followed it (the answer
    # and the stream are different connections). It carries the version it was pushed for, and the chart, which knows
    # the version it loaded, leaves it out: applied, a bar of the last run would sit ahead of the new run's and the
    # series would refuse every bar that came after.
    open_dashboard(page, base_url)
    run_for(page, 1500, speed=1000)
    page.wait_for_function("candleSeries.data().length >= 5")
    state = page.evaluate("""() => { const d = candleSeries.data(), last = d[d.length - 1];
        return { version: loadedVersion, bars: d.length, time: last.time, close: last.close }; }""")
    assert state["version"] == page.evaluate("fetch('/api/candlesticks').then(r => r.json()).then(d => d.version)")
    stale = {"time": state["time"] + 7, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 5, "buy_volume": 5,
             "sell_volume": 0, "vwap": 1, "v": state["version"] - 1}  # 7: on no bar of the 2-unit interval either
    after = page.evaluate("""bar => { document.dispatchEvent(new CustomEvent('lob:candles', { detail: [bar] }));
        const d = candleSeries.data(), last = d[d.length - 1];
        return { bars: d.length, time: last.time, close: last.close, volumes: volumeSeries.data().length,
            hud: document.getElementById('hud-c').textContent }; }""", stale)
    assert (after["bars"], after["time"], after["close"]) == (state["bars"], state["time"], state["close"])
    assert after["volumes"] == state["bars"] and after["hud"] != "1"
    # The same bar for the version the chart holds is drawn, and one that names no version (a page's own) too.
    for bar, bars in (({**stale, "v": state["version"]}, state["bars"] + 1),
                      ({**{k: v for k, v in stale.items() if k != "v"}, "time": state["time"] + 9}, state["bars"] + 2)):
        got = page.evaluate("""bar => { document.dispatchEvent(new CustomEvent('lob:candles', { detail: [bar] }));
            const d = candleSeries.data(); return { bars: d.length, time: d[d.length - 1].time }; }""", bar)
        assert got == {"bars": bars, "time": bar["time"]}
    # A reset is another chart: the version moves on, the chart reloads with the new run's bars, and the stream's
    # bars for it are drawn (the run goes on from time 0, behind the bars just drawn, which are gone).
    page.click("#sim-reset-btn")
    page.wait_for_function("v => loadedVersion > v && candleSeries.data().length === 1", arg=state["version"])
    run_for(page, 1000, speed=1000)
    page.wait_for_function("candleSeries.data().length >= 3")
    times = page.evaluate("candleSeries.data().map(b => b.time)")
    assert times == sorted(times) and times[-1] < state["time"] and all(t % 2 == 0 for t in times)


def test_reveal_follows_the_interval_and_a_reset_and_the_value_restarts_with_the_book(page, base_url):
    open_dashboard(page, base_url)
    choose_informed(page)
    page.click("#reveal-toggle")
    page.wait_for_function(f"{INFORMED}.pressed")
    run_for(page, 4000, speed=1000)
    page.wait_for_function(f"{INFORMED}.points >= 15")
    reveal_matches_the_server(page)
    page.click("#btn-interval-15")
    page.wait_for_function("loadedInterval === 15")  # the chart has the new bars (currentInterval changes at the click)
    reveal_matches_the_server(page)
    page.wait_for_function(f"{INFORMED}.points >= 2")
    assert page.evaluate("lobInformed.data().every(p => p.time % 15 === 0)")
    page.click("#btn-interval-2")
    page.wait_for_function("loadedInterval === 2")
    reveal_matches_the_server(page)
    page.click("#sim-reset-btn")  # the value starts again at the mid with the book; the switch stays on
    page.wait_for_function(f"{INFORMED}.points <= 1")
    assert page.evaluate(INFORMED)["pressed"] and page.evaluate(INFORMED)["legend"]
    run_for(page, 2500, speed=1000)
    page.wait_for_function(f"{INFORMED}.points >= 8")
    reveal_matches_the_server(page)


def test_in_a_playback_the_switch_says_not_recorded_and_nothing_is_drawn_and_the_value_comes_back_live(page, base_url):
    open_dashboard(page, base_url)
    choose_informed(page)
    page.click("#reveal-toggle")
    page.wait_for_function(f"{INFORMED}.pressed")
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'true'")
    run_for(page, 2500, speed=1000)
    page.click("#record-btn")
    page.wait_for_function(f"{BADGE}.recording === 'false'")
    page.wait_for_function(f"{INFORMED}.points >= 5")
    page.click("#tab-analyze")
    page.wait_for_selector("#recordings-list [data-recording-id]")
    page.click("#recordings-list [data-recording-id]:first-child button[data-recording-action='play']")
    page.wait_for_function(f"{BADGE}.playback === 'true'")
    page.click("#tab-trade")
    page.wait_for_function(f"{INFORMED}.note === 'not recorded' && {INFORMED}.points === 0 && !{INFORMED}.legend")
    assert page.inner_text("#reveal-note") == "not recorded"
    assert page.inner_text("#informed-line").startswith("Informed traders: not recorded")
    assert page.get_attribute("#informed-line", "data-informed") == "yes"
    # (After a reload the stream sends only the newest bar, so a long step right away would skip the bars between.)
    page.evaluate("stepSim(1)")
    page.wait_for_function(f"{EVENTS} >= 1")
    page.evaluate("stepSim(10000)")  # to the end of the tape: its bars, none of the value
    page.wait_for_function(f"{BADGE}.pbEnded === 'true' && candleSeries.data().length > 3")
    assert page.evaluate(INFORMED)["points"] == 0
    assert page.evaluate("fetch('/api/candlesticks').then(r => r.json())"
                         ".then(d => d.reveal.active === false && d.candles.every(c => !('fair' in c)))")
    page.click("#playback-exit")
    page.wait_for_function(f"{BADGE}.playback === 'false'")
    page.wait_for_function(f"{INFORMED}.note === '' && {INFORMED}.legend && {INFORMED}.points >= 1")
    run_for(page, 2000, speed=1000)
    page.wait_for_function(f"{INFORMED}.points >= 5")
    reveal_matches_the_server(page)
    assert TRUTH.fullmatch(page.inner_text("#informed-line"))


def test_the_trade_view_stays_responsive_at_2000_events_a_second_with_the_informed_model_and_reveal_on(
        page, base_url, engine):
    if engine != "chrome":
        pytest.skip("long tasks are measured in Chromium")
    open_dashboard(page, base_url)
    choose_informed(page)
    page.click("#reveal-toggle")
    page.wait_for_function(f"{INFORMED}.pressed")
    page.evaluate("""() => { window.__long = []; new PerformanceObserver(list => {
        for (const e of list.getEntries()) window.__long.push(e.duration);
    }).observe({type: 'longtask', buffered: true}); }""")
    page.evaluate("setSpeedPreset(2000)")
    page.click("#sim-play-btn")
    page.wait_for_function(RUNNING)
    page.wait_for_function(f"{INFORMED}.points > 20")
    frames = page.evaluate("""() => new Promise(done => { const gaps = [], t0 = performance.now(); let last = t0;
        (function tick(now) { gaps.push(now - last); last = now;
            if (now - t0 < 3000) requestAnimationFrame(tick); else done(gaps); })(t0); })""")
    page.click("#sim-play-btn")
    page.wait_for_function(f"!({RUNNING})")
    frames.sort()
    long_tasks = page.evaluate("window.__long")
    assert len(frames) > 30
    assert frames[len(frames) // 2] < 50 and frames[-1] < 500, (frames[len(frames) // 2], frames[-1])
    assert not [d for d in long_tasks if d > 250], long_tasks
    assert page.evaluate(INFORMED)["error"] is None
    reveal_matches_the_server(page)
