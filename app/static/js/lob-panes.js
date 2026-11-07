// Charts on one time axis, with the controls the Pairs and Stat-arb views share: a horizontal zoom, a vertical scale
// for each pane, Center, Fit and Now, panes to show or hide, a choice of heights, and a crosshair across the panes.
// The toolbar comes from components/panes.html. A view says what lines there are and what each shows at a sample.
window.lobPanes = function ({ view, prefix, panes: PANES, palette, points, onCrosshair }) {
    const $ = (id) => document.getElementById(id);
    const RIGHT = 4;     // room to the right of the newest point, in points
    const FRAMED = 600;  // points shown after a history
    const SPAN_MIN = 30;    // what the zoom slider runs between
    const SPAN_MAX = 6000;
    const ZOOM_STEP = 1.25;  // a press of a zoom button
    let charts = null;
    let filling = false; // fewer points than the frame holds: they fill it from the left (see frame)
    let leader = null;   // the chart the user last acted on: the others show its range
    let pointerDown = false;
    let actedAt = -Infinity;
    let crosshair = null;     // the point under the crosshair, for the readout (else the newest)
    const each = (name, act) => view.querySelectorAll(`[data-${prefix}-${name}]`).forEach((button) => {
        act(button, button.getAttribute(`data-${prefix}-${name}`));
    });

    function saved(key, fallback) {
        try {
            const value = JSON.parse(localStorage.getItem(key));
            return value === null ? fallback : value;
        } catch (e) { return fallback; }
    }
    function save(key, value) {
        try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* not saved */ }
    }

    function fontSize() {
        const root = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
        return Math.max(8, Math.round(11 * root / 16));
    }
    const units = (t) => Math.round(Number(t)).toLocaleString('en-US');

    function look(P) {
        return {
            layout: { background: { type: 'solid', color: P.background }, textColor: P.text, fontSize: fontSize() },
            grid: { vertLines: { color: P.grid }, horzLines: { color: P.grid } },
            crosshair: { vertLine: { color: P.crosshair, labelBackgroundColor: P.label },
                         horzLine: { color: P.crosshair, labelBackgroundColor: P.label } },
            rightPriceScale: { borderColor: P.border },
            timeScale: { borderColor: P.border },
        };
    }

    function makeChart(el, P) {
        const chart = LightweightCharts.createChart(el, {
            width: el.clientWidth, height: el.clientHeight,
            localization: { timeFormatter: (t) => `time ${units(t)}` },
            rightPriceScale: { minimumWidth: 64, scaleMargins: { top: 0.08, bottom: 0.08 } },
            // Not fixLeftEdge: a chart pinned at its first point gains a point of its own with each new one,
            // and so shimmers while it fills. rightBarStaysOnScroll: a zoom keeps the right edge, and with it
            // the newest point in view.
            timeScale: { visible: false, rightOffset: RIGHT, minBarSpacing: 0.05, rightBarStaysOnScroll: true,
                         tickMarkFormatter: (t) => units(t) },
        });
        chart.applyOptions(look(P));
        new ResizeObserver(() => {
            if (el.clientWidth > 0) chart.applyOptions({ width: el.clientWidth, height: el.clientHeight });
        }).observe(el);
        return chart;
    }

    // Every line on the charts is one of these: which pane, how to color it and what it shows at a sample.
    function addLine(pane, key, color, value, { width = 2, style = 0, precision = 2, title = '', last = true } = {}) {
        const P = palette();
        const series = charts.panes[pane].chart.addLineSeries({
            color: color(P), lineWidth: width, lineStyle: style, title, priceLineVisible: false, lastValueVisible: last,
            crosshairMarkerVisible: width > 1,
            priceFormat: { type: 'price', precision, minMove: 1 / 10 ** precision },
            autoscaleInfoProvider: scaled(charts.panes[pane], key),
        });
        const entry = { pane, key, color, value, series };
        charts.lines.push(entry);
        return entry;
    }
    function removeLine(entry) {
        charts.panes[entry.pane].chart.removeSeries(entry.series);
        charts.lines.splice(charts.lines.indexOf(entry), 1);
        delete charts.panes[entry.pane].seen[entry.key];
    }
    const lineOf = (key) => charts.lines.find((entry) => entry.key === key);

    // A pane's vertical scale: it fits what is shown, stretched about its middle by the pane's zoom, so it stays
    // centered as the lines move, unlike a price axis stretched by hand. Each line reports its own range here;
    // all return the same one, from all the ranges reported.
    function scaled(pane, key) {
        return (original) => {
            const info = original();
            if (!info || !info.priceRange) {
                delete pane.seen[key];
                return info;
            }
            pane.seen[key] = info.priceRange;
            if (pane.zoom === 1) return info;
            let lo = Infinity;
            let hi = -Infinity;
            Object.values(pane.seen).forEach((r) => {
                lo = Math.min(lo, r.minValue);
                hi = Math.max(hi, r.maxValue);
            });
            const middle = (lo + hi) / 2;
            const half = Math.max((hi - lo) / 2, 1e-6) / pane.zoom;
            return { priceRange: { minValue: middle - half, maxValue: middle + half }, margins: info.margins };
        };
    }
    function rescale(pane) {  // applies a changed zoom, also to a price axis dragged out of auto-scale
        const p = charts.panes[pane];
        p.chart.priceScale('right').applyOptions({ autoScale: true });
        const first = charts.lines.find((entry) => entry.pane === pane);
        if (first) first.series.applyOptions({ autoscaleInfoProvider: scaled(p, first.key) });
    }
    function zoomPane(pane, by) {
        const p = charts.panes[pane];
        p.zoom = Math.min(16, Math.max(1 / 16, p.zoom * (by > 0 ? ZOOM_STEP : 1 / ZOOM_STEP)));
        rescale(pane);
    }
    function center(pane = null) {
        if (!charts) return;
        (pane ? [pane] : PANES).forEach((key) => {
            charts.panes[key].zoom = 1;
            rescale(key);
        });
    }

    function make() {
        if (charts) return;
        const P = palette();
        charts = { panes: {}, list: [], lines: [] };
        PANES.forEach((key) => {
            const chart = makeChart($(`${prefix}-${key}-chart`), P);
            charts.panes[key] = { chart, zoom: 1, seen: {}, wrap: view.querySelector(`[data-${prefix}-pane="${key}"]`) };
            charts.list.push(chart);
        });
        // One time axis: each chart has a point for every sample, so their logical ranges line up, and as they
        // all get the same points they scroll together. What the user does on one (dragging, zooming) goes to
        // the others from it alone: a range set on a chart takes effect only when it paints, so ranges passed
        // around among them, or set on the one under the user's hand, would undo the user's last move.
        PANES.forEach((key) => {
            const { chart } = charts.panes[key];
            const el = $(`${prefix}-${key}-chart`);
            const acting = () => {
                leader = chart;
                actedAt = performance.now();
            };
            // In the capture phase, before the chart's own listeners act on the same event.
            el.addEventListener('pointerdown', () => { acting(); pointerDown = true; }, true);
            el.addEventListener('wheel', acting, { capture: true, passive: true });
            chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
                if (range && chart === leader && userActing()) {
                    showOnOthers(range);
                    showZoom(range);
                }
            });
            chart.subscribeCrosshairMove((param) => crosshairMoved(chart, param));
        });
        ['pointerup', 'pointercancel'].forEach((name) => {
            window.addEventListener(name, () => {
                if (pointerDown) actedAt = performance.now();  // a touch drag glides on for a moment
                pointerDown = false;
            });
        });
        applyPanes();
    }

    const val = (t, v) => (v === null || v === undefined ? { time: t } : { time: t, value: v });
    function setAll() {
        if (!charts) return;
        // Emptied first. Given the same times again, a chart's only line leaves the library's list of points stale;
        // once another line has joined it and the oldest points have gone, its rows are out of order, and it throws
        // "Value is null" at every paint from then on (a P&L pane's zero line, when the first trader starts).
        charts.lines.forEach((entry) => entry.series.setData([]));
        charts.lines.forEach((entry) => entry.series.setData(points().map((p) => val(p.t, entry.value(p)))));
    }
    function setLine(entry) {
        entry.series.setData(points().map((p) => val(p.t, entry.value(p))));
    }
    function addPoint(p) {
        charts.lines.forEach((entry) => entry.series.update(val(p.t, entry.value(p))));
    }

    // ---- The time axis and the scales ----
    // The library follows the newest point by itself while that is in view, and leaves a range the user moved
    // back in time where it is, until the user brings the newest point into view again. It keeps the room
    // to the right of the newest point as points come, though, so a short history framed for FRAMED points
    // would be pushed off to the left: until the points reach the frame's right edge it holds still.
    const userActing = () => pointerDown || performance.now() - actedAt < 1000;
    const shownCharts = () => (charts ? PANES.filter((key) => !charts.panes[key].wrap.hidden).map((key) => charts.panes[key].chart) : []);
    const leadChart = () => (shownCharts().includes(leader) ? leader : shownCharts()[0]);
    const leadRange = () => (leadChart() ? leadChart().timeScale().getVisibleLogicalRange() : null);
    function showOnOthers(range) {
        shownCharts().forEach((chart) => {
            if (chart !== leader) { try { chart.timeScale().setVisibleLogicalRange(range); } catch (e) { /* no data yet */ } }
        });
    }
    function setRange(from, to) {
        shownCharts().forEach((chart) => {
            try { chart.timeScale().setVisibleLogicalRange({ from, to }); } catch (e) { /* no data yet */ }
        });
        showZoom({ from, to });
    }
    function setFilling(on) {
        filling = on;
        charts.list.forEach((chart) => chart.timeScale().applyOptions({ shiftVisibleRangeOnNewBar: !on }));
    }
    function frame() {  // after a history: the newest FRAMED points, or room for them
        if (!charts) return;
        setSpan(FRAMED, true);
    }
    // Show this many points: up to the newest if that is in view, filling from the left if there are fewer;
    // looking back in time, up to where the view ends now.
    function setSpan(span, follow = false) {
        if (!charts) return;
        const shown = Math.round(Math.max(SPAN_MIN, span));
        const last = points().length - 1;
        const range = leadRange();
        if (follow || !range || range.to >= last) {
            setFilling(last < shown);
            const from = Math.max(0, last - shown);
            setRange(from, from + shown + RIGHT);
        } else {
            setRange(range.to - shown - RIGHT, range.to);
        }
    }
    function fit() {  // every point (a frame's worth at least: a few would be drawn absurdly wide), and centered
        if (!charts) return;
        setSpan(Math.max(FRAMED, points().length - 1), true);
        center();
    }
    function zoomTime(by) {
        const range = leadRange();
        if (range) setSpan((range.to - range.from - RIGHT) * (by > 0 ? 1 / ZOOM_STEP : ZOOM_STEP));
    }
    function afterDropping() {  // the times shown stay as they are, unless some of them have gone
        const range = leadRange();
        if (!range || range.from >= 0) return;
        const last = points().length - 1;  // following: all there is; looking back: the oldest there is
        setRange(0, range.to >= last ? last + RIGHT : range.to - range.from);
    }
    function pointsAdded() {
        const range = leadRange();
        if (!range) return;
        const last = points().length - 1;
        if (filling && last >= range.to - RIGHT) {  // the frame is full: from here on it scrolls
            setFilling(false);
            setRange(last - (range.to - range.from - RIGHT), last + RIGHT);
        } else if (!userActing()) {  // a range passed on just as points came is behind them: catch it up
            const behind = shownCharts().some((chart) => {
                const r = chart.timeScale().getVisibleLogicalRange();
                return r && (Math.abs(r.from - range.from) > 0.01 || Math.abs(r.to - range.to) > 0.01);
            });
            if (behind) setRange(range.from, range.to);
        }
    }

    // The zoom controls show what the charts show, however it came about.
    const toSpan = (position) => Math.round(SPAN_MIN * Math.pow(SPAN_MAX / SPAN_MIN, Number(position) / 1000));
    const spanToSlider = (span) => 1000 * Math.log(Math.max(SPAN_MIN, span) / SPAN_MIN) / Math.log(SPAN_MAX / SPAN_MIN);
    function showZoom(range) {
        const span = Math.round(range.to - range.from - RIGHT);
        const chart = leadChart();
        const width = chart ? chart.timeScale().width() : 0;
        const px = width ? ` (${(width / (span + RIGHT)).toFixed(1)} px)` : '';
        $(`${prefix}-hzoom-badge`).textContent = `${span.toLocaleString('en-US')} points${px}`;
        const zoomSlider = $(`${prefix}-hzoom-slider`);
        if (!zoomSlider.matches(':active')) zoomSlider.value = spanToSlider(span);
        each('span', (button, value) => {
            const on = Math.abs(Number(value) - span) / span < 0.03;
            button.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    // Which panes show. Only the lowest has the time axis; a pane shown again takes up the others' range.
    function applyPanes() {
        const shown = saved(`lob-${prefix}-panes`, PANES);
        PANES.forEach((key) => {
            const on = shown.includes(key);
            charts.panes[key].wrap.hidden = !on;
            view.querySelector(`[data-${prefix}-toggle="${key}"]`).setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        const lowest = PANES.filter((key) => shown.includes(key)).pop();
        PANES.forEach((key) => charts.panes[key].chart.timeScale().applyOptions({ visible: key === lowest }));
    }
    function togglePane(pane) {
        const shown = saved(`lob-${prefix}-panes`, PANES);
        const next = shown.includes(pane) ? shown.filter((key) => key !== pane) : PANES.filter((key) => key === pane || shown.includes(key));
        if (!next.length) return;  // one always stays
        const range = leadRange();
        save(`lob-${prefix}-panes`, next);
        applyPanes();
        if (range) requestAnimationFrame(() => setRange(range.from, range.to));
    }
    function setHeight(factor) {
        $(`${prefix}-chart-card`).style.setProperty(`--${prefix}-h`, factor);
        save(`lob-${prefix}-height`, factor);
        each('height', (button, value) => {
            button.setAttribute('aria-pressed', Number(value) === factor ? 'true' : 'false');
        });
    }

    // The readout: the point under the crosshair, or the newest. The crosshair shows in every pane.
    const nearest = (t) => {
        const all = points();
        let lo = 0;
        let hi = all.length - 1;
        while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (all[mid].t < t) lo = mid + 1;
            else hi = mid;
        }
        return all[lo] || null;
    };
    function crosshairMoved(chart, param) {
        const p = param.time === undefined || param.time === null ? null : nearest(param.time);
        // Only the pane under the pointer leads: the others are told below, which reports back here. A pointer
        // leaving a pane comes with no event, and clearing the others reports back with no time.
        if ((p && !param.sourceEvent) || (!p && !crosshair)) return;
        crosshair = p;
        onCrosshair();
        PANES.forEach((key) => {
            const other = charts.panes[key].chart;
            if (other === chart || charts.panes[key].wrap.hidden) return;
            const first = charts.lines.find((entry) => entry.pane === key && entry.value(p || {}) !== null && entry.value(p || {}) !== undefined);
            if (!p || !first) other.clearCrosshairPosition();
            else other.setCrosshairPosition(first.value(p), p.t, first.series);
        });
    }

    function restyle() {
        if (!charts) return;
        const P = palette();
        charts.list.forEach((chart) => chart.applyOptions(look(P)));
        charts.lines.forEach((entry) => entry.series.applyOptions({ color: entry.color(P) }));
    }

    // The toolbar.
    $(`${prefix}-fit-btn`).addEventListener('click', fit);
    $(`${prefix}-center-btn`).addEventListener('click', () => center());
    $(`${prefix}-now-btn`).addEventListener('click', () => {
        const range = leadRange();
        if (range) setSpan(range.to - range.from - RIGHT, true);
    });
    $(`${prefix}-zoom-in`).addEventListener('click', () => zoomTime(1));
    $(`${prefix}-zoom-out`).addEventListener('click', () => zoomTime(-1));
    $(`${prefix}-hzoom-slider`).addEventListener('input', (e) => setSpan(toSpan(e.target.value)));
    each('span', (button, value) => button.addEventListener('click', () => setSpan(Number(value))));
    each('vzoom', (button, value) => {
        button.addEventListener('click', () => { if (charts) zoomPane(value, Number(button.dataset.by)); });
    });
    each('center', (button, value) => button.addEventListener('click', () => center(value)));
    each('toggle', (button, value) => button.addEventListener('click', () => { if (charts) togglePane(value); }));
    each('height', (button, value) => button.addEventListener('click', () => setHeight(Number(value))));

    return {
        make, ready: () => !!charts, addLine, removeLine, lineOf, setAll, setLine, addPoint, frame, fit, center, afterDropping,
        pointsAdded, restyle, crosshair: () => crosshair, restoreHeight: () => setHeight(saved(`lob-${prefix}-height`, 1)),
        // What the charts hold and show (for the browser tests).
        state() {
            const range = leadRange();
            const lead = leadChart();
            const scale = {};
            PANES.forEach((key) => {
                const pane = charts.panes[key];
                const first = charts.lines.find((entry) => entry.pane === key);
                const height = pane.chart.chartElement().clientHeight;
                scale[key] = { zoom: pane.zoom, auto: pane.chart.priceScale('right').options().autoScale, height,
                               top: first.series.coordinateToPrice(0), bottom: first.series.coordinateToPrice(height - 30) };
            });
            const all = points();
            return {
                points: all.length, first: all.length ? all[0].t : null, last: all.length ? all[all.length - 1].t : null,
                ranges: shownCharts().map((chart) => chart.timeScale().getVisibleLogicalRange()),
                times: lead ? lead.timeScale().getVisibleRange() : null, filling,
                following: !!range && range.to >= all.length - 1,
                span: range ? Math.round(range.to - range.from - RIGHT) : null, scale,
                axis: PANES.filter((key) => charts.panes[key].chart.timeScale().options().visible),
            };
        },
    };
};

// What a view with a market of its own does around its charts: its event stream, held in place of the page's while it
// is shown (index.html: a tab holds one stream at a time); its play, step and speed controls; its forms, with each
// field's message and the fields at fault marked; and its board, with its buttons. ``prefix`` names all of them.
window.lobMarketView = function ({ view, prefix, panels, on, onState, onShow, onClose, onReady }) {
    const $ = (id) => document.getElementById(id);
    const api = `/api/${prefix}`;
    const instance = $('lob-app').dataset.instance;
    const SPEED_MAX = Number(view.dataset.speedMax) || 500;
    const ICONS = { play: 'M8 5v14l11-7z', pause: 'M6 19h4V5H6v14zm8-14v14h4V5h-4z' };
    let source = null;

    // The settings in force, in the form: on connect, and when they were applied (in this tab or another).
    // While the form is being typed in they wait until it is left.
    const configForm = $(`${prefix}-config`);
    let shownSettings = null;
    let pendingSettings = null;
    function showSettings(settings) {
        const key = JSON.stringify(settings);
        if (key === shownSettings) return;
        if (configForm.contains(document.activeElement)) {
            pendingSettings = settings;
            return;
        }
        pendingSettings = null;
        shownSettings = key;
        Object.entries(settings).forEach(([name, value]) => {
            const input = configForm.elements[name];
            if (!input) return;
            if (input.type === 'checkbox') input.checked = !!value;
            else input.value = value;
        });
    }
    configForm.addEventListener('focusout', () => setTimeout(() => {
        if (pendingSettings) showSettings(pendingSettings);
    }, 0));

    function showState(state) {
        showRunning(state.running);
        if (!$(`${prefix}-speed-slider`).matches(':active')) showSpeed(state.speed);
        if (state.settings) showSettings(state.settings);
        if (onState) onState(state);
    }

    function replaceElement(target, html) {
        const template = document.createElement('template');
        template.innerHTML = html.trim();
        const node = template.content.firstElementChild;
        if (node) target.replaceWith(node);
    }

    // The board is pushed several times a second while the books run, and its rows move as their P&L
    // changes. While the pointer is on a row's buttons, or the keyboard is on one of them, its pushes
    // wait, so a click lands on the trader it was aimed at (as on the Trade view's leaderboard).
    const boardWrapper = $(`${prefix}-board-wrapper`);
    const onButtons = (el) => !!(el && el.closest && el.closest(`[data-${prefix}-actions]`));
    let pointerOn = false;
    let focusOn = false;
    let heldBoard = null;
    function hold() {
        if (pointerOn || focusOn || heldBoard === null) return;
        const html = heldBoard;
        heldBoard = null;
        replaceElement($(`${prefix}-board`), html);
    }
    boardWrapper.addEventListener('pointerover', (e) => { pointerOn = onButtons(e.target); hold(); });
    boardWrapper.addEventListener('pointerleave', () => { pointerOn = false; hold(); });
    boardWrapper.addEventListener('focusin', (e) => {
        let keyboard = true;
        try { keyboard = e.target.matches(':focus-visible'); } catch (err) { /* older browsers: count it */ }
        focusOn = onButtons(e.target) && keyboard;
        hold();
    });
    boardWrapper.addEventListener('focusout', (e) => { if (!onButtons(e.relatedTarget)) { focusOn = false; hold(); } });

    function onPanel(name, html) {
        if (name === `${prefix}_board` && (pointerOn || focusOn)) {
            heldBoard = html;
            return;
        }
        const target = $(panels[name]);
        if (target) replaceElement(target, html);
        if (name === `${prefix}_status`) showRunning($(`${prefix}-status`).dataset.running === 'true');
    }

    function open() {
        if (source) return;
        const opened = new EventSource(`/${prefix}/events`);
        source = opened;
        window.showConnection('connecting');
        opened.onopen = () => window.showConnection('live');
        opened.onerror = () => {
            window.showConnection('reconnecting');
            // The browser retries on its own unless the stream was refused outright.
            if (opened.readyState === EventSource.CLOSED) {
                if (source === opened) source = null;
                setTimeout(() => { if (window.currentView === prefix) open(); }, 3000);
            }
        };
        opened.addEventListener('hello', (e) => {
            if (JSON.parse(e.data).instance !== instance) window.location.reload();  // a different simulation
        });
        opened.addEventListener('notices', (e) => JSON.parse(e.data).forEach(window.showToast));
        opened.addEventListener(`${prefix}_state`, (e) => showState(JSON.parse(e.data)));
        Object.entries(on).forEach(([name, handle]) => opened.addEventListener(name, (e) => handle(e.data)));
        Object.keys(panels).forEach((name) => opened.addEventListener(name, (e) => onPanel(name, e.data)));
    }
    function close() {
        if (!source) return;
        source.close();
        source = null;
        onClose();  // the next stream starts with the history
    }

    // ---- Controls ----

    function showRunning(running) {
        const button = $(`${prefix}-play-btn`);
        button.setAttribute('aria-pressed', running ? 'true' : 'false');
        $(`${prefix}-play-icon`).setAttribute('d', running ? ICONS.pause : ICONS.play);
        $(`${prefix}-play-text`).textContent = running ? 'Pause' : 'Start';
    }

    async function control(action, fields = {}) {
        let res = null;
        try {
            res = await window.postForm(`${api}/control`, { action, ...fields });
        } catch (err) { /* no answer */ }
        if (!res || !res.ok) {
            if (window.showToast) window.showToast(`Not done: ${res ? `the server answered with status ${res.status}` : 'the server did not answer'}.`);
            return;
        }
        showRunning((await res.json()).running);
    }
    const toggle = () => control('toggle');
    const step = (n) => control('step', { units: n });

    $(`${prefix}-play-btn`).addEventListener('click', toggle);
    $(`${prefix}-reset-btn`).addEventListener('click', () => control('reset'));
    view.querySelectorAll(`[data-${prefix}-step]`).forEach((button) => {
        button.addEventListener('click', () => step(Number(button.getAttribute(`data-${prefix}-step`))));
    });

    // Speed, in units of time a second: the slider is logarithmic, from 1 to SPEED_MAX.
    const toSpeed = (position) => Math.max(1, Math.round(Math.pow(SPEED_MAX, Number(position) / 1000)));
    const toSlider = (speed) => Math.round(1000 * Math.log(Math.max(1, speed)) / Math.log(SPEED_MAX));
    function showSpeed(speed, { fromSlider = false } = {}) {
        $(`${prefix}-speed-badge`).textContent = `${speed} units/s`;
        if (!fromSlider) $(`${prefix}-speed-slider`).value = toSlider(speed);
        view.querySelectorAll(`[data-${prefix}-speed]`).forEach((button) => {
            button.setAttribute('aria-pressed', Number(button.getAttribute(`data-${prefix}-speed`)) === speed ? 'true' : 'false');
        });
    }
    async function setSpeed(speed) {
        showSpeed(speed);
        try {
            await window.postForm(`${api}/speed`, { speed });
        } catch (err) { /* the stream shows the speed in force */ }
    }
    const slider = $(`${prefix}-speed-slider`);
    slider.value = toSlider(Number(slider.dataset.speed));
    slider.addEventListener('input', () => showSpeed(toSpeed(slider.value), { fromSlider: true }));
    slider.addEventListener('change', () => setSpeed(toSpeed(slider.value)));
    view.querySelectorAll(`[data-${prefix}-speed]`).forEach((button) => {
        button.addEventListener('click', () => setSpeed(Number(button.getAttribute(`data-${prefix}-speed`))));
    });

    // The forms: each field's message under the form, and the fields at fault marked.
    async function submit(form, url, result) {
        form.querySelectorAll('[aria-invalid]').forEach((input) => input.removeAttribute('aria-invalid'));
        result.className = 'text-[0.625rem] leading-snug text-slate-400';
        result.textContent = '…';
        let res = null;
        let body = null;
        try {
            res = await fetch(url, { method: 'POST', body: new FormData(form) });
            body = await res.json();
        } catch (err) { /* said below */ }
        if (res && res.ok && body) {
            result.className = 'text-[0.625rem] leading-snug text-emerald-300';
            result.textContent = body.message;
            return body;
        }
        result.className = 'text-[0.625rem] leading-snug text-rose-300';
        if (body && body.errors) {
            Object.keys(body.errors).forEach((name) => {
                // The trader form has the same names in each kind's settings: the ones in force.
                const input = form.querySelector(`fieldset:not([disabled]) [name="${name}"]`) || form.querySelector(`[name="${name}"]`);
                if (input) input.setAttribute('aria-invalid', 'true');
            });
            result.textContent = Object.values(body.errors).join(' ');
        } else if (body && body.error) {
            result.textContent = body.error;
        } else {
            result.textContent = res ? `Not done: the server answered with status ${res.status}.` : 'Not done: the server did not answer.';
        }
        return null;
    }
    configForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (await submit(configForm, `${api}/config`, $(`${prefix}-config-result`))) {
            shownSettings = null;  // the stream's next state shows the values as applied
        }
    });
    const traderForm = $(`${prefix}-trader-form`);
    traderForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submit(traderForm, `${api}/traders/start`, $(`${prefix}-trader-result`));
    });

    // The board's buttons. They are marked unavailable (aria-disabled) rather than disabled or taken out,
    // as on the Trade view's leaderboard: WebKit may not report the pointer leaving a disabled element.
    const off = (button) => button.getAttribute('aria-disabled') === 'true';
    boardWrapper.addEventListener('click', async (e) => {
        const button = e.target.closest(`button[data-${prefix}-action]`);
        if (!button || off(button)) return;
        if (e.detail > 0) button.blur();  // a mouse click: the board need not wait for focus to move on
        const row = button.closest(`tr[data-${prefix}-id]`);
        const action = button.getAttribute(`data-${prefix}-action`);
        button.setAttribute('aria-disabled', 'true');
        let res = null;
        try {
            res = await window.postForm(`${api}/traders/${button.getAttribute(`data-${prefix}-target`)}/${action}`);
        } catch (err) { /* said below */ }
        button.removeAttribute('aria-disabled');
        if (!res || !res.ok) {
            const why = !res ? 'the server did not answer' : res.status === 404 ? 'that trader is no longer on the board'
                : `the server answered with status ${res.status}`;
            if (window.showToast) window.showToast(`Not done: ${why}.`);
            return;
        }
        if (action === 'remove') {
            row.classList.add('opacity-40', 'line-through');
            row.querySelectorAll(`button[data-${prefix}-action]`).forEach((b) => b.setAttribute('aria-disabled', 'true'));
            button.textContent = 'Removed';
        } else {
            const active = action === 'resume';
            row.dataset.active = String(active);
            button.setAttribute(`data-${prefix}-action`, active ? 'stop' : 'resume');
            button.textContent = active ? 'Stop' : 'Resume';
        }
    });

    // ---- The view shown or left ----

    function show() {
        onShow();
        open();
    }
    document.addEventListener('lob:view', (e) => {
        if (e.detail === prefix) show();
        else close();
    });
    // base.html's helpers (themeColor, postForm) come after the page's content: start once they are there.
    document.addEventListener('DOMContentLoaded', () => {
        showRunning($(`${prefix}-status`).dataset.running === 'true');
        if (onReady) onReady();
        if (window.currentView === prefix) show();  // the page opened on this view (before this script listened)
    });
    return { toggle, step, submit, showState };
};

// Fitting a trader over training runs, in a view's trader form (components/panes.html, fit_panel): a count of the
// combinations the ticked values make, Fit and Cancel, the progress, and the results the view's stream brings
// (components/fit_runs_result.html) with their Use and Start buttons. It also shows only the chosen kind's settings and
// its values to try (data-PREFIX-kind, data-PREFIX-fit-kind). ``submit`` is lobMarketView's. It gives the view
// ``showFitting`` for the state the stream brings and ``showFit`` for the results.
window.lobFitPanel = function ({ prefix, form, submit }) {
    const $ = (id) => document.getElementById(id);
    const api = `/api/${prefix}/traders`;
    const panel = $(`${prefix}-fit-panel`);
    const run = $(`${prefix}-fit-run`);
    const progress = $(`${prefix}-fit-progress`);
    const result = $(`${prefix}-fit-result`);
    const reply = $(`${prefix}-fit-reply`);
    let asked = false;
    let shown = null;  // the id of the fit whose results are on the page

    // Only the chosen kind's settings show and are sent (a disabled fieldset's are not).
    function chooseKind() {
        const kind = (form.querySelector('input[name="kind"]:checked') || {}).value;
        form.querySelectorAll(`fieldset[data-${prefix}-kind], fieldset[data-${prefix}-fit-kind]`).forEach((set) => {
            const on = (set.getAttribute(`data-${prefix}-kind`) || set.getAttribute(`data-${prefix}-fit-kind`)) === kind;
            set.hidden = !on;
            set.disabled = !on;
        });
        countCombinations();
    }
    form.addEventListener('change', (e) => { if (e.target.name === 'kind') chooseKind(); });

    // How many settings the ticked values make, before fitting: with more than a fit tries, a sample of them is.
    function countCombinations() {
        const set = form.querySelector(`fieldset[data-${prefix}-fit-kind]:not([disabled])`);
        const line = $(`${prefix}-fit-count`);
        if (!set) return;
        let combinations = 1;
        let ticked = 0;
        set.querySelectorAll('input[type="checkbox"][name^="fit_"]').forEach((box) => {
            if (!box.checked) return;
            ticked += 1;
            const text = set.querySelector(`[name="grid_${box.name.slice(4)}"]`).value;  // each kind has its own
            const values = text.split(/[,;]/).map((v) => v.trim()).filter(Boolean).map(Number);  // 2 and 2.0 are one
            combinations *= Math.max(1, new Set(values).size);
        });
        const most = Number(line.dataset.most);
        const s = combinations === 1 ? '' : 's';
        line.textContent = !ticked ? 'Tick at least one setting to fit.'
            : `${combinations.toLocaleString('en-US')} combination${s}, and your settings.`
              + (combinations > most ? ` At most ${most} are tried, fewer on long runs, drawn at random from the seed.` : '');
    }
    panel.addEventListener('input', countCombinations);
    panel.addEventListener('change', countCombinations);

    // A fit's progress comes by the stream, to every tab; only one runs at a time.
    function showFitting(fitting) {
        run.setAttribute('aria-disabled', fitting ? 'true' : 'false');
        $(`${prefix}-fit-cancel-wrap`).hidden = !fitting;
        progress.textContent = fitting ? `Fitting: ${fitting[0]} of ${fitting[1]} runs done…` : '';
    }
    // The results come by the stream too, before or after the answer. The tab that asked is taken to them: below
    // the chart, they can be out of sight, and above it they are brought clear of the sticky header. The same results
    // pushed again (the books' settings changed) are not the answer, and leave the reply to a Use or a Start as it is.
    function showFit(html) {
        result.innerHTML = html;
        const summary = $(`${prefix}-fit-summary`);
        const id = summary ? summary.dataset.fitId : null;
        if (id === shown) return;
        shown = id;
        reply.textContent = '';
        if (asked && html) {
            result.style.scrollMarginTop = `${document.querySelector('header').offsetHeight}px`;
            result.scrollIntoView({ block: 'nearest' });
        }
        asked = false;
    }
    async function fitTrader() {
        if (run.getAttribute('aria-disabled') === 'true') return;
        run.setAttribute('aria-disabled', 'true');  // until the stream says how it is
        asked = true;
        const body = await submit(form, `${api}/fit`, $(`${prefix}-fit-status`));
        if (!body || body.canceled) asked = false;
        run.setAttribute('aria-disabled', progress.textContent ? 'true' : 'false');
    }
    run.addEventListener('click', fitTrader);
    panel.addEventListener('keydown', (e) => {  // in this panel Enter fits, not starts
        if (e.key === 'Enter' && e.target.matches('input')) {
            e.preventDefault();
            fitTrader();
        }
    });
    $(`${prefix}-fit-cancel`).addEventListener('click', () => {
        window.postForm(`${api}/fit/cancel`).catch(() => {
            if (window.showToast) window.showToast('Not canceled: the server did not answer.');
        });
    });

    result.addEventListener('click', async (e) => {
        const use = e.target.closest(`[data-${prefix}-fit-use]`);
        const start = e.target.closest(`[data-${prefix}-fit-start]`);
        const summary = $(`${prefix}-fit-summary`);
        if (!summary || (!use && !start)) return;
        if (use) {
            const kind = summary.dataset.kind;
            form.querySelector(`input[name="kind"][value="${kind}"]`).checked = true;
            chooseKind();
            const set = form.querySelector(`fieldset[data-${prefix}-kind="${kind}"]`);
            Object.entries(JSON.parse(use.getAttribute(`data-${prefix}-fit-use`))).forEach(([name, value]) => {
                const input = set.querySelector(`[name="${name}"]`);
                if (!input) return;
                if (input.type === 'checkbox') input.checked = !!value;
                else input.value = value;
            });
            reply.className = 'mb-2 text-[0.625rem] leading-snug text-slate-300 empty:hidden';
            reply.textContent = 'These settings are in the form now, on the left, to change or start.';
            return;
        }
        let res = null;
        let body = null;
        try {
            res = await window.postForm(`${api}/start`, { fit_id: summary.dataset.fitId, rank: start.getAttribute(`data-${prefix}-fit-start`) });
            body = await res.json();
        } catch (err) { /* said below */ }
        const started = res && res.ok && body;
        reply.className = `mb-2 text-[0.625rem] leading-snug empty:hidden ${started ? 'text-emerald-300' : 'text-rose-300'}`;
        reply.textContent = body && (body.message || body.error) ? (body.message || body.error) : 'Not started: the server did not answer.';
    });

    // The browser restores a form after the scripts have run, and again on a step back or forward: so the choice is
    // read after that (a disabled setting is not restored).
    window.addEventListener('pageshow', chooseKind);
    if (document.readyState === 'complete') chooseKind();
    return { showFitting, showFit };
};
