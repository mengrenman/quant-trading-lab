// The Pairs view (components/pairs_view.html): two order books whose gap mean-reverts, and pairs
// traders in them (app/pairs.py). A market of its own, with its own clock and speed. Its event stream,
// /pairs/events, is open only while the view is shown, in place of the page's /events (index.html: a tab
// holds one stream at a time), and also brings the page's notices. Its charts are made the first time the
// view is shown. window.lobPairs gives base.html's keyboard shortcuts what they act on in this view.
(function () {
    const view = document.getElementById('view-pairs');
    if (!view || !window.LightweightCharts) return;
    const instance = document.getElementById('lob-app').dataset.instance;
    const SPEED_MAX = Number(view.dataset.speedMax) || 500;
    const SAMPLES = Number(view.dataset.samples) || 3000;
    const ICONS = { play: 'M8 5v14l11-7z', pause: 'M6 19h4V5H6v14zm8-14v14h4V5h-4z' };
    const PANELS = { pairs_status: 'pairs-status', pairs_books: 'pairs-books', pairs_board: 'pairs-board' };
    const $ = (id) => document.getElementById(id);

    let source = null;
    let version = null;
    let points = [];     // the chart's samples, oldest first (app/pairs.py, PairsSession._sample)
    let marks = [];      // [trader id, sample time, its new target], in the order they happened
    let traders = [];    // {id, name, color, active, kind, entry, exit}, in the order they started
    let shownChoice = null;  // whose levels, signal and hedge the chart shows: null the first trader's, '' no one's
    let charts = null;
    const PANES = ['price', 'gap', 'signal', 'hedge', 'pnl'];
    const RIGHT = 4;     // room to the right of the newest point, in points
    const FRAMED = 600;  // points shown after a history
    const SPAN_MIN = 30;    // what the zoom slider runs between
    const SPAN_MAX = 6000;
    const ZOOM_STEP = 1.25;  // a press of a zoom button
    let filling = false; // fewer points than the frame holds: they fill it from the left (see frame)
    let leader = null;   // the chart the user last acted on: the others show its range
    let pointerDown = false;
    let actedAt = -Infinity;
    let markersShown = null;  // which markers the gap chart has (they are redrawn only when they change)
    let crosshair = null;     // the point under the crosshair, for the readout (else the newest)

    function saved(key, fallback) {
        try {
            const value = JSON.parse(localStorage.getItem(key));
            return value === null ? fallback : value;
        } catch (e) { return fallback; }
    }
    function save(key, value) {
        try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* not saved */ }
    }

    // ---- Charts: prices, the gap, a signal, the hedge and each trader's P&L, on one time axis ----

    function palette() {
        const c = window.themeColor;
        return {
            background: c('darkBg'), text: c('slate-400'), grid: c('slate-800', 0.45), border: c('slate-800'),
            crosshair: c('sky-400', 0.4), label: c('slate-900'),
            a: c('sky-400'), b: c('violet-400'), fa: c('sky-400', 0.55), fb: c('violet-400', 0.55),
            gap: c('cyan-300'), fairGap: c('slate-400', 0.8), band: c('amber-400', 0.85), zero: c('slate-600'),
            hedge: c('lime-400'), beta: c('slate-400', 0.9),
            long: c('emerald-400'), short: c('rose-400'), flat: c('slate-300'),
        };
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

    function ensureCharts() {
        if (charts) return;
        const P = palette();
        const { Dotted, Dashed } = LightweightCharts.LineStyle;
        charts = { panes: {}, list: [], lines: [], traders: new Map(), levels: [] };
        PANES.forEach((key) => {
            const chart = makeChart($(`pairs-${key}-chart`), P);
            charts.panes[key] = { chart, zoom: 1, seen: {}, wrap: view.querySelector(`[data-pairs-pane="${key}"]`) };
            charts.list.push(chart);
        });
        const shown = () => { const id = shownId(); return id === null ? null : id; };
        const at = (field, i = null) => (p) => {
            const id = shown();
            const v = id === null || !p[field] ? null : p[field][id];
            return v === undefined || v === null ? null : (i === null ? v : v[i]);
        };
        addLine('price', 'fa', (Q) => Q.fa, (p) => p.fa, { width: 1, style: Dotted, precision: 1, last: false });
        addLine('price', 'fb', (Q) => Q.fb, (p) => p.fb, { width: 1, style: Dotted, precision: 1, last: false });
        addLine('price', 'a', (Q) => Q.a, (p) => p.a, { precision: 1, title: 'A' });
        addLine('price', 'b', (Q) => Q.b, (p) => p.b, { precision: 1, title: 'B' });
        addLine('gap', 'fairGap', (Q) => Q.fairGap, (p) => p.true, { width: 1, style: Dotted, last: false });
        addLine('gap', 'lo', (Q) => Q.band, at('bands', 0), { width: 1, style: Dashed, last: false });
        addLine('gap', 'mean', (Q) => Q.band, at('bands', 1), { width: 1, style: Dotted, last: false });
        addLine('gap', 'hi', (Q) => Q.band, at('bands', 2), { width: 1, style: Dashed, last: false });
        addLine('gap', 'gap', (Q) => Q.gap, (p) => p.gap, { title: 'gap' });
        addLine('signal', 'signalZero', (Q) => Q.zero, () => 0, { width: 1, style: Dotted, last: false });
        addLine('signal', 'signal0', (Q) => Q.gap, at('sig', 0), { title: '' });
        addLine('signal', 'signal1', (Q) => Q.b, at('sig', 1), { title: '' });
        addLine('hedge', 'beta', (Q) => Q.beta, (p) => p.beta, { width: 1, style: Dotted, precision: 3, title: 'beta' });
        addLine('hedge', 'hedge', (Q) => Q.hedge, at('hedge'), { precision: 3, title: 'hedge' });
        addLine('pnl', 'zero', (Q) => Q.zero, () => 0, { width: 1, style: Dotted, precision: 1, last: false });

        // One time axis: each chart has a point for every sample, so their logical ranges line up, and as they
        // all get the same points they scroll together. What the user does on one (dragging, zooming) goes to
        // the others from it alone: a range set on a chart takes effect only when it paints, so ranges passed
        // around among them, or set on the one under the user's hand, would undo the user's last move.
        PANES.forEach((key) => {
            const { chart } = charts.panes[key];
            const el = $(`pairs-${key}-chart`);
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
        syncTraders();
        setAll();
        frame();
    }

    const val = (t, v) => (v === null || v === undefined ? { time: t } : { time: t, value: v });
    function shownId() {
        if (shownChoice === '') return null;
        if (shownChoice !== null && traders.some((tr) => String(tr.id) === shownChoice)) return shownChoice;
        return traders.length ? String(traders[0].id) : null;
    }
    const shownTrader = () => traders.find((tr) => String(tr.id) === shownId()) || null;

    function setAll() {
        if (!charts) return;
        charts.lines.forEach((entry) => entry.series.setData(points.map((p) => val(p.t, entry.value(p)))));
        applyMarkers(true);
        showHud();
    }

    function addPoint(p) {
        charts.lines.forEach((entry) => entry.series.update(val(p.t, entry.value(p))));
    }

    // Where the chosen trader's position changed, on the gap: each mark on the first sample that shows it
    // (one not on the chart yet waits for its sample).
    function applyMarkers(redraw = false) {
        if (!charts) return;
        const id = shownId();
        const first = points.length ? points[0].t : 0;
        const last = points.length ? points[points.length - 1].t : -1;
        const shown = id === null ? [] : marks.filter((m) => String(m[0]) === id && m[1] >= first && m[1] <= last);
        const key = `${id} ${shown.length} ${shown.length ? `${shown[0][1]} ${shown[shown.length - 1][1]}` : ''}`;
        if (!redraw && key === markersShown) return;
        markersShown = key;
        const P = palette();
        lineOf('gap').series.setMarkers(shown.map(([, t, target]) => (
            target > 0 ? { time: t, position: 'belowBar', color: P.long, shape: 'arrowUp' }
                : target < 0 ? { time: t, position: 'aboveBar', color: P.short, shape: 'arrowDown' }
                    : { time: t, position: 'inBar', color: P.flat, shape: 'circle', size: 0.6 })));
    }

    // The levels of the trader shown, on the signal pane, and what its lines are.
    function showLevels() {
        if (!charts) return;
        const P = palette();
        const line0 = lineOf('signal0').series;
        charts.levels.forEach((level) => line0.removePriceLine(level));
        charts.levels = [];
        const tr = shownTrader();
        const legend = $('pairs-signal-legend');
        const swatch = (color, text) => {
            const item = document.createElement('span');
            item.className = 'flex items-center gap-1';
            const mark = document.createElement('span');
            mark.className = 'w-3 h-0.5 rounded';
            mark.style.backgroundColor = color;
            item.append(mark, document.createTextNode(text));
            return item;
        };
        if (!tr) {
            legend.textContent = 'of the trader shown, against its levels';
            return;
        }
        const { Dashed, Dotted } = LightweightCharts.LineStyle;
        const level = (price, style) => charts.levels.push(line0.createPriceLine({
            price, color: P.band, lineWidth: 1, lineStyle: style, axisLabelVisible: false, title: '' }));
        [tr.entry, -tr.entry].forEach((price) => level(price, Dashed));
        if (tr.exit > 0) [tr.exit, -tr.exit].forEach((price) => level(price, Dotted));
        legend.replaceChildren(...(tr.kind === 'copula'
            ? [swatch(P.gap, "A's flag"), swatch(P.b, "B's flag")]
            : [swatch(P.gap, tr.kind === 'kalman' ? "the filter's error, in its usual sizes" : "the gap's z-score")]));
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
        const last = points.length - 1;
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
        setSpan(Math.max(FRAMED, points.length - 1), true);
        center();
    }
    function zoomTime(by) {
        const range = leadRange();
        if (range) setSpan((range.to - range.from - RIGHT) * (by > 0 ? 1 / ZOOM_STEP : ZOOM_STEP));
    }
    function afterDropping() {  // the times shown stay as they are, unless some of them have gone
        const range = leadRange();
        if (!range || range.from >= 0) return;
        const last = points.length - 1;  // following: all there is; looking back: the oldest there is
        setRange(0, range.to >= last ? last + RIGHT : range.to - range.from);
    }
    function pointsAdded() {
        const range = leadRange();
        if (!range) return;
        const last = points.length - 1;
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
        $('pairs-hzoom-badge').textContent = `${span.toLocaleString('en-US')} points${px}`;
        const zoomSlider = $('pairs-hzoom-slider');
        if (!zoomSlider.matches(':active')) zoomSlider.value = spanToSlider(span);
        view.querySelectorAll('[data-pairs-span]').forEach((button) => {
            const on = Math.abs(Number(button.dataset.pairsSpan) - span) / span < 0.03;
            button.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    // Which panes show. Only the lowest has the time axis; a pane shown again takes up the others' range.
    function applyPanes() {
        const shown = saved('lob-pairs-panes', PANES);
        PANES.forEach((key) => {
            const on = shown.includes(key);
            charts.panes[key].wrap.hidden = !on;
            view.querySelector(`[data-pairs-toggle="${key}"]`).setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        const lowest = PANES.filter((key) => shown.includes(key)).pop();
        PANES.forEach((key) => charts.panes[key].chart.timeScale().applyOptions({ visible: key === lowest }));
    }
    function togglePane(pane) {
        const shown = saved('lob-pairs-panes', PANES);
        const next = shown.includes(pane) ? shown.filter((key) => key !== pane) : PANES.filter((key) => key === pane || shown.includes(key));
        if (!next.length) return;  // one always stays
        const range = leadRange();
        save('lob-pairs-panes', next);
        applyPanes();
        if (range) requestAnimationFrame(() => setRange(range.from, range.to));
    }
    function setHeight(factor) {
        $('pairs-chart-card').style.setProperty('--pairs-h', factor);
        save('lob-pairs-height', factor);
        view.querySelectorAll('[data-pairs-height]').forEach((button) => {
            button.setAttribute('aria-pressed', Number(button.dataset.pairsHeight) === factor ? 'true' : 'false');
        });
    }

    // The readout: the point under the crosshair, or the newest. The crosshair shows in every pane.
    const nearest = (t) => {
        let lo = 0;
        let hi = points.length - 1;
        while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (points[mid].t < t) lo = mid + 1;
            else hi = mid;
        }
        return points[lo] || null;
    };
    function crosshairMoved(chart, param) {
        const p = param.time === undefined || param.time === null ? null : nearest(param.time);
        // Only the pane under the pointer leads: the others are told below, which reports back here. A pointer
        // leaving a pane comes with no event, and clearing the others reports back with no time.
        if ((p && !param.sourceEvent) || (!p && !crosshair)) return;
        crosshair = p;
        showHud();
        PANES.forEach((key) => {
            const other = charts.panes[key].chart;
            if (other === chart || charts.panes[key].wrap.hidden) return;
            const first = charts.lines.find((entry) => entry.pane === key && entry.value(p || {}) !== null && entry.value(p || {}) !== undefined);
            if (!p || !first) other.clearCrosshairPosition();
            else other.setCrosshairPosition(first.value(p), p.t, first.series);
        });
    }
    function showHud() {
        const p = crosshair || points[points.length - 1];
        if (!p) return;
        const one = (v, d = 1) => (v === null || v === undefined ? '—' : Number(v).toFixed(d));
        const signed = (v, d = 2) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}`);
        const parts = [`time ${units(p.t)}`, `A ${one(p.a)}`, `B ${one(p.b)}`, `gap ${signed(p.gap, 1)}`, `fair ${signed(p.true, 1)}`, `beta ${one(p.beta, 3)}`];
        const id = shownId();
        const tr = shownTrader();
        if (id !== null && tr) {
            const sig = p.sig && p.sig[id];
            if (sig) parts.push(sig.length === 2 ? `flags ${signed(sig[0])} ${signed(sig[1])}` : `z ${signed(sig[0])}`);
            if (p.hedge && p.hedge[id] !== undefined) parts.push(`hedge ${one(p.hedge[id], 3)}`);
            if (p.pnl && p.pnl[id] !== undefined) parts.push(`P&L ${signed(p.pnl[id], 1)}`);
        }
        $('pairs-hud').textContent = parts.join('  ');
    }

    // The traders: a P&L line each, an entry in the P&L legend, and a choice in "Show".
    function syncTraders() {
        const select = $('pairs-bands-select');
        const chosen = select.value;
        select.replaceChildren(new Option('no one', ''), ...traders.map((tr) => new Option(tr.name, String(tr.id))));
        const id = shownId();
        select.value = id === null ? '' : id;
        if (chosen !== select.value && charts) setAll();
        const legend = $('pairs-pnl-legend');
        legend.replaceChildren(...traders.map((tr) => {
            const item = document.createElement('span');
            item.className = 'flex items-center gap-1' + (tr.active ? '' : ' opacity-60');
            const swatch = document.createElement('span');
            swatch.className = 'w-3 h-0.5 rounded';
            swatch.style.backgroundColor = window.themeColor(tr.color);
            item.append(swatch, document.createTextNode(tr.name + (tr.active ? '' : ' (stopped)')));
            return item;
        }));
        if (!charts) return;
        const ids = new Set(traders.map((tr) => String(tr.id)));
        charts.traders.forEach((entry, tid) => {
            if (!ids.has(tid)) { removeLine(entry); charts.traders.delete(tid); }
        });
        traders.forEach((tr) => {
            const key = String(tr.id);
            if (charts.traders.has(key)) return;
            const entry = addLine('pnl', `pnl${key}`, () => window.themeColor(tr.color), (p) => (p.pnl ? p.pnl[key] : null), { precision: 1, title: tr.name });
            entry.series.setData(points.map((p) => val(p.t, entry.value(p))));
            charts.traders.set(key, entry);
        });
        showLevels();
    }

    function restyle() {
        if (!charts) return;
        const P = palette();
        charts.list.forEach((chart) => chart.applyOptions(look(P)));
        charts.lines.forEach((entry) => entry.series.applyOptions({ color: entry.color(P) }));
        syncTraders();
        applyMarkers(true);
    }
    document.addEventListener('lob:theme', restyle);
    document.addEventListener('lob:text-size', restyle);

    // ---- What the stream brings ----

    function onHistory(h) {
        version = h.version;
        points = h.samples;
        marks = h.marks;
        traders = h.traders;
        onState(h);
        syncTraders();
        setAll();
        frame();
    }

    function onSamples(msg) {
        if (msg.version !== version) return;  // a history for the new market is on its way
        let added = 0;
        msg.samples.forEach((p) => {
            if (points.length && p.t <= points[points.length - 1].t) return;
            points.push(p);
            if (charts) addPoint(p);
            added += 1;
        });
        marks.push(...msg.marks);
        if (points.length > 2 * SAMPLES) {  // keep as many as the server does
            points = points.slice(-SAMPLES);
            marks = marks.filter((m) => m[1] >= points[0].t);
            setAll();
            afterDropping();
        } else if (added || msg.marks.length) {
            applyMarkers();
        }
        if (added) {
            pointsAdded();
            if (!crosshair) showHud();
        }
    }

    // The settings in force, in the form: on connect, and when they were applied (in this tab or another).
    // While the form is being typed in they wait until it is left.
    const configForm = $('pairs-config');
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

    function onState(state) {
        showRunning(state.running);
        if (!$('pairs-speed-slider').matches(':active')) showSpeed(state.speed);
        if (state.traders) {
            traders = state.traders;
            syncTraders();
        }
        if (state.settings) showSettings(state.settings);
        showFitting(state.fitting);
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
    const boardWrapper = $('pairs-board-wrapper');
    const onButtons = (el) => !!(el && el.closest && el.closest('[data-pairs-actions]'));
    let pointerOn = false;
    let focusOn = false;
    let heldBoard = null;
    function hold() {
        if (pointerOn || focusOn || heldBoard === null) return;
        const html = heldBoard;
        heldBoard = null;
        replaceElement($('pairs-board'), html);
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
        if (name === 'pairs_board' && (pointerOn || focusOn)) {
            heldBoard = html;
            return;
        }
        const target = $(PANELS[name]);
        if (target) replaceElement(target, html);
        if (name === 'pairs_status') showRunning($('pairs-status').dataset.running === 'true');
    }

    function open() {
        if (source) return;
        const opened = new EventSource('/pairs/events');
        source = opened;
        window.showConnection('connecting');
        opened.onopen = () => window.showConnection('live');
        opened.onerror = () => {
            window.showConnection('reconnecting');
            // The browser retries on its own unless the stream was refused outright.
            if (opened.readyState === EventSource.CLOSED) {
                if (source === opened) source = null;
                setTimeout(() => { if (window.currentView === 'pairs') open(); }, 3000);
            }
        };
        opened.addEventListener('hello', (e) => {
            if (JSON.parse(e.data).instance !== instance) window.location.reload();  // a different simulation
        });
        opened.addEventListener('notices', (e) => JSON.parse(e.data).forEach(window.showToast));
        opened.addEventListener('pairs_history', (e) => onHistory(JSON.parse(e.data)));
        opened.addEventListener('pairs_samples', (e) => onSamples(JSON.parse(e.data)));
        opened.addEventListener('pairs_state', (e) => onState(JSON.parse(e.data)));
        opened.addEventListener('pairs_fit', (e) => showFit(e.data));
        Object.keys(PANELS).forEach((name) => opened.addEventListener(name, (e) => onPanel(name, e.data)));
    }
    function close() {
        if (!source) return;
        source.close();
        source = null;
        version = null;  // the next stream starts with the history
    }

    // ---- Controls ----

    function showRunning(on) {
        const button = $('pairs-play-btn');
        button.setAttribute('aria-pressed', on ? 'true' : 'false');
        $('pairs-play-icon').setAttribute('d', on ? ICONS.pause : ICONS.play);
        $('pairs-play-text').textContent = on ? 'Pause' : 'Start';
    }

    async function control(action, fields = {}) {
        let res = null;
        try {
            res = await window.postForm('/api/pairs/control', { action, ...fields });
        } catch (err) { /* no answer */ }
        if (!res || !res.ok) {
            if (window.showToast) window.showToast(`Not done: ${res ? `the server answered with status ${res.status}` : 'the server did not answer'}.`);
            return;
        }
        showRunning((await res.json()).running);
    }
    const toggle = () => control('toggle');
    const step = (n) => control('step', { units: n });

    $('pairs-play-btn').addEventListener('click', toggle);
    $('pairs-reset-btn').addEventListener('click', () => control('reset'));
    view.querySelectorAll('[data-pairs-step]').forEach((button) => {
        button.addEventListener('click', () => step(Number(button.dataset.pairsStep)));
    });
    $('pairs-bands-select').addEventListener('change', (e) => {
        shownChoice = e.target.value;
        showLevels();
        setAll();
    });

    // The chart's toolbar.
    $('pairs-fit-btn').addEventListener('click', fit);
    $('pairs-center-btn').addEventListener('click', () => center());
    $('pairs-now-btn').addEventListener('click', () => {
        const range = leadRange();
        if (range) setSpan(range.to - range.from - RIGHT, true);
    });
    $('pairs-zoom-in').addEventListener('click', () => zoomTime(1));
    $('pairs-zoom-out').addEventListener('click', () => zoomTime(-1));
    $('pairs-hzoom-slider').addEventListener('input', (e) => setSpan(toSpan(e.target.value)));
    view.querySelectorAll('[data-pairs-span]').forEach((button) => {
        button.addEventListener('click', () => setSpan(Number(button.dataset.pairsSpan)));
    });
    view.querySelectorAll('[data-pairs-vzoom]').forEach((button) => {
        button.addEventListener('click', () => { if (charts) zoomPane(button.dataset.pairsVzoom, Number(button.dataset.by)); });
    });
    view.querySelectorAll('[data-pairs-center]').forEach((button) => {
        button.addEventListener('click', () => center(button.dataset.pairsCenter));
    });
    view.querySelectorAll('[data-pairs-toggle]').forEach((button) => {
        button.addEventListener('click', () => { if (charts) togglePane(button.dataset.pairsToggle); });
    });
    view.querySelectorAll('[data-pairs-height]').forEach((button) => {
        button.addEventListener('click', () => setHeight(Number(button.dataset.pairsHeight)));
    });

    // Speed, in units of time a second: the slider is logarithmic, from 1 to SPEED_MAX.
    const toSpeed = (position) => Math.max(1, Math.round(Math.pow(SPEED_MAX, Number(position) / 1000)));
    const toSlider = (speed) => Math.round(1000 * Math.log(Math.max(1, speed)) / Math.log(SPEED_MAX));
    function showSpeed(speed, { fromSlider = false } = {}) {
        $('pairs-speed-badge').textContent = `${speed} units/s`;
        if (!fromSlider) $('pairs-speed-slider').value = toSlider(speed);
        view.querySelectorAll('[data-pairs-speed]').forEach((button) => {
            button.setAttribute('aria-pressed', Number(button.dataset.pairsSpeed) === speed ? 'true' : 'false');
        });
    }
    async function setSpeed(speed) {
        showSpeed(speed);
        try {
            await window.postForm('/api/pairs/speed', { speed });
        } catch (err) { /* the stream shows the speed in force */ }
    }
    const slider = $('pairs-speed-slider');
    slider.value = toSlider(Number(slider.dataset.speed));
    slider.addEventListener('input', () => showSpeed(toSpeed(slider.value), { fromSlider: true }));
    slider.addEventListener('change', () => setSpeed(toSpeed(slider.value)));
    view.querySelectorAll('[data-pairs-speed]').forEach((button) => {
        button.addEventListener('click', () => setSpeed(Number(button.dataset.pairsSpeed)));
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
        if (await submit(configForm, '/api/pairs/config', $('pairs-config-result'))) {
            shownSettings = null;  // the stream's next state shows the values as applied
        }
    });
    const traderForm = $('pairs-trader-form');
    traderForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submit(traderForm, '/api/pairs/traders/start', $('pairs-trader-result'));
    });
    // Only the chosen kind's settings show and are sent (a disabled fieldset's are not).
    function chooseKind() {
        const kind = (traderForm.querySelector('input[name="kind"]:checked') || {}).value;
        traderForm.querySelectorAll('fieldset[data-pairs-kind], fieldset[data-pairs-fit-kind]').forEach((set) => {
            const on = (set.dataset.pairsKind || set.dataset.pairsFitKind) === kind;
            set.hidden = !on;
            set.disabled = !on;
        });
        countCombinations();
    }
    traderForm.addEventListener('change', (e) => { if (e.target.name === 'kind') chooseKind(); });

    // ---- Fitting a trader over training runs ----

    // How many settings the ticked values make, before fitting: with more than a fit tries, a sample of them is.
    function countCombinations() {
        const set = traderForm.querySelector('fieldset[data-pairs-fit-kind]:not([disabled])');
        const line = $('pairs-fit-count');
        if (!set) return;
        let combinations = 1;
        let ticked = 0;
        set.querySelectorAll('input[type="checkbox"][name^="fit_"]').forEach((box) => {
            if (!box.checked) return;
            ticked += 1;
            const text = set.querySelector(`[name="grid_${box.name.slice(4)}"]`).value;  // each kind has its own
            combinations *= Math.max(1, new Set(text.split(/[,;]/).map((v) => v.trim()).filter(Boolean)).size);
        });
        const most = Number(line.dataset.most);
        const s = combinations === 1 ? '' : 's';
        line.textContent = !ticked ? 'Tick at least one setting to fit.'
            : `${combinations.toLocaleString('en-US')} combination${s}, and your settings.`
              + (combinations > most ? ` At most ${most} are tried, fewer on long runs, drawn at random from the seed.` : '');
    }
    $('pairs-fit-panel').addEventListener('input', countCombinations);
    $('pairs-fit-panel').addEventListener('change', countCombinations);
    chooseKind();  // a reload can keep another choice

    // A fit's progress comes by the stream, to every tab; only one runs at a time.
    function showFitting(fitting) {
        $('pairs-fit-run').setAttribute('aria-disabled', fitting ? 'true' : 'false');
        $('pairs-fit-cancel-wrap').hidden = !fitting;
        $('pairs-fit-progress').textContent = fitting ? `Fitting: ${fitting[0]} of ${fitting[1]} runs done…` : '';
    }
    // The results come by the stream too, before or after the answer. The tab that asked is taken to them: below
    // the chart, they can be out of sight.
    let fitAsked = false;
    function showFit(html) {
        $('pairs-fit-result').innerHTML = html;
        $('pairs-fit-reply').textContent = '';
        if (fitAsked && html) $('pairs-fit-result').scrollIntoView({ block: 'nearest' });
        fitAsked = false;
    }
    async function fitTrader() {
        if ($('pairs-fit-run').getAttribute('aria-disabled') === 'true') return;
        $('pairs-fit-run').setAttribute('aria-disabled', 'true');  // until the stream says how it is
        fitAsked = true;
        const body = await submit(traderForm, '/api/pairs/traders/fit', $('pairs-fit-status'));
        if (!body || body.canceled) fitAsked = false;
        $('pairs-fit-run').setAttribute('aria-disabled', $('pairs-fit-progress').textContent ? 'true' : 'false');
    }
    $('pairs-fit-run').addEventListener('click', fitTrader);
    $('pairs-fit-panel').addEventListener('keydown', (e) => {  // in this panel Enter fits, not starts
        if (e.key === 'Enter' && e.target.matches('input')) {
            e.preventDefault();
            fitTrader();
        }
    });
    $('pairs-fit-cancel').addEventListener('click', () => {
        window.postForm('/api/pairs/traders/fit/cancel').catch(() => {
            if (window.showToast) window.showToast('Not canceled: the server did not answer.');
        });
    });
    $('pairs-fit-result').addEventListener('click', async (e) => {
        const reply = $('pairs-fit-reply');
        const use = e.target.closest('[data-pairs-fit-use]');
        const start = e.target.closest('[data-pairs-fit-start]');
        const summary = $('pairs-fit-summary');
        if (!summary || (!use && !start)) return;
        if (use) {
            const kind = summary.dataset.kind;
            traderForm.querySelector(`input[name="kind"][value="${kind}"]`).checked = true;
            chooseKind();
            const set = traderForm.querySelector(`fieldset[data-pairs-kind="${kind}"]`);
            Object.entries(JSON.parse(use.dataset.pairsFitUse)).forEach(([name, value]) => {
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
            res = await window.postForm('/api/pairs/traders/start', { fit_id: summary.dataset.fitId, rank: start.dataset.pairsFitStart });
            body = await res.json();
        } catch (err) { /* said below */ }
        const started = res && res.ok && body;
        reply.className = `mb-2 text-[0.625rem] leading-snug empty:hidden ${started ? 'text-emerald-300' : 'text-rose-300'}`;
        reply.textContent = body && (body.message || body.error) ? (body.message || body.error) : 'Not started: the server did not answer.';
    });

    // The board's buttons. They are marked unavailable (aria-disabled) rather than disabled or taken out,
    // as on the Trade view's leaderboard: WebKit may not report the pointer leaving a disabled element.
    const off = (button) => button.getAttribute('aria-disabled') === 'true';
    boardWrapper.addEventListener('click', async (e) => {
        const button = e.target.closest('button[data-pairs-action]');
        if (!button || off(button)) return;
        if (e.detail > 0) button.blur();  // a mouse click: the board need not wait for focus to move on
        const row = button.closest('tr[data-pairs-id]');
        const action = button.dataset.pairsAction;
        button.setAttribute('aria-disabled', 'true');
        let res = null;
        try {
            res = await window.postForm(`/api/pairs/traders/${button.dataset.pairsTarget}/${action}`);
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
            row.querySelectorAll('button[data-pairs-action]').forEach((b) => b.setAttribute('aria-disabled', 'true'));
            button.textContent = 'Removed';
        } else {
            const active = action === 'resume';
            row.dataset.active = String(active);
            button.dataset.pairsAction = active ? 'stop' : 'resume';
            button.textContent = active ? 'Stop' : 'Resume';
        }
    });

    // ---- The view shown or left ----

    function show() {
        ensureCharts();
        open();
    }
    document.addEventListener('lob:view', (e) => {
        if (e.detail === 'pairs') show();
        else close();
    });
    // What the charts hold and show (for the browser tests).
    function chartState() {
        if (!charts) return null;
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
        return {
            points: points.length, first: points.length ? points[0].t : null,
            last: points.length ? points[points.length - 1].t : null,
            ranges: shownCharts().map((chart) => chart.timeScale().getVisibleLogicalRange()),
            times: lead ? lead.timeScale().getVisibleRange() : null, filling,
            following: !!range && range.to >= points.length - 1,
            span: range ? Math.round(range.to - range.from - RIGHT) : null, scale,
            axis: PANES.filter((key) => charts.panes[key].chart.timeScale().options().visible),
            traders: [...charts.traders.keys()], bands: shownId(), levels: charts.levels.length,
            markers: marks.filter((m) => String(m[0]) === shownId()).length,
        };
    }
    window.lobPairs = { toggle, step, fit, center, chartState };
    // base.html's helpers (themeColor, postForm) come after the page's content: start once they are there.
    document.addEventListener('DOMContentLoaded', () => {
        showRunning($('pairs-status').dataset.running === 'true');
        setHeight(saved('lob-pairs-height', 1));
        if (window.currentView === 'pairs') show();  // the page opened on this view (before this script listened)
    });
})();
