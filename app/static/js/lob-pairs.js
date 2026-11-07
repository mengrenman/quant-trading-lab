// The Pairs view (components/pairs_view.html): two order books whose gap mean-reverts, and pairs
// traders in them (app/pairs.py). A market of its own, with its own clock and speed. Its event stream,
// /pairs/events, is open only while the view is shown, in place of the page's /events (index.html: a tab
// holds one stream at a time), and also brings the page's notices. Its charts are made the first time the
// view is shown. window.lobPairs gives base.html's keyboard shortcuts what they act on in this view.
(function () {
    const view = document.getElementById('view-pairs');
    if (!view || !window.LightweightCharts) return;
    const SAMPLES = Number(view.dataset.samples) || 3000;
    const PANELS = { pairs_status: 'pairs-status', pairs_books: 'pairs-books', pairs_board: 'pairs-board' };
    const $ = (id) => document.getElementById(id);

    let version = null;
    let points = [];     // the chart's samples, oldest first (app/pairs.py, PairsSession._sample)
    let marks = [];      // [trader id, sample time, its new target], in the order they happened
    let traders = [];    // {id, name, color, active, kind, entry, exit}, in the order they started
    let shownChoice = null;  // whose levels, signal and hedge the chart shows: null the first trader's, '' no one's
    let markersShown = null;  // which markers the gap chart has (they are redrawn only when they change)
    const traderLines = new Map();  // each trader's P&L line
    let levelLines = [];            // the levels of the trader shown, on the signal pane

    // ---- Charts: prices, the gap, a signal, the hedge and each trader's P&L, on one time axis (lob-panes.js) ----

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
    const units = (t) => Math.round(Number(t)).toLocaleString('en-US');
    const panes = window.lobPanes({ view, prefix: 'pairs', panes: ['price', 'gap', 'signal', 'hedge', 'pnl'], palette,
                                    points: () => points, onCrosshair: () => showHud() });
    const { addLine, removeLine, lineOf } = panes;

    function ensureCharts() {
        if (panes.ready()) return;
        panes.make();
        const { Dotted, Dashed } = LightweightCharts.LineStyle;
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

        syncTraders();
        setAll();
        panes.frame();
    }

    function shownId() {
        if (shownChoice === '') return null;
        if (shownChoice !== null && traders.some((tr) => String(tr.id) === shownChoice)) return shownChoice;
        return traders.length ? String(traders[0].id) : null;
    }
    const shownTrader = () => traders.find((tr) => String(tr.id) === shownId()) || null;

    function setAll() {
        if (!panes.ready()) return;
        panes.setAll();
        applyMarkers(true);
        showHud();
    }

    // Where the chosen trader's position changed, on the gap: each mark on the first sample that shows it
    // (one not on the chart yet waits for its sample).
    function applyMarkers(redraw = false) {
        if (!panes.ready()) return;
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
        if (!panes.ready()) return;
        const P = palette();
        const line0 = lineOf('signal0').series;
        levelLines.forEach((level) => line0.removePriceLine(level));
        levelLines = [];
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
        const level = (price, style) => levelLines.push(line0.createPriceLine({
            price, color: P.band, lineWidth: 1, lineStyle: style, axisLabelVisible: false, title: '' }));
        [tr.entry, -tr.entry].forEach((price) => level(price, Dashed));
        if (tr.exit > 0) [tr.exit, -tr.exit].forEach((price) => level(price, Dotted));
        legend.replaceChildren(...(tr.kind === 'copula'
            ? [swatch(P.gap, "A's flag"), swatch(P.b, "B's flag")]
            : [swatch(P.gap, tr.kind === 'kalman' ? "the filter's error, in its usual sizes" : "the gap's z-score")]));
    }

    function showHud() {
        const p = panes.crosshair() || points[points.length - 1];
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
        if (chosen !== select.value && panes.ready()) setAll();
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
        if (!panes.ready()) return;
        const ids = new Set(traders.map((tr) => String(tr.id)));
        traderLines.forEach((entry, tid) => {
            if (!ids.has(tid)) { removeLine(entry); traderLines.delete(tid); }
        });
        traders.forEach((tr) => {
            const key = String(tr.id);
            if (traderLines.has(key)) return;
            const entry = addLine('pnl', `pnl${key}`, () => window.themeColor(tr.color), (p) => (p.pnl ? p.pnl[key] : null), { precision: 1, title: tr.name });
            panes.setLine(entry);
            traderLines.set(key, entry);
        });
        showLevels();
    }

    function restyle() {
        if (!panes.ready()) return;
        panes.restyle();
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
        market.showState(h);
        syncTraders();
        setAll();
        panes.frame();
    }

    function onSamples(msg) {
        if (msg.version !== version) return;  // a history for the new market is on its way
        let added = 0;
        msg.samples.forEach((p) => {
            if (points.length && p.t <= points[points.length - 1].t) return;
            points.push(p);
            if (panes.ready()) panes.addPoint(p);
            added += 1;
        });
        marks.push(...msg.marks);
        if (points.length > 2 * SAMPLES) {  // keep as many as the server does
            points = points.slice(-SAMPLES);
            marks = marks.filter((m) => m[1] >= points[0].t);
            setAll();
            panes.afterDropping();
        } else if (added || msg.marks.length) {
            applyMarkers();
        }
        if (added) {
            panes.pointsAdded();
            if (!panes.crosshair()) showHud();
        }
    }

    function onState(state) {
        if (state.traders) {
            traders = state.traders;
            syncTraders();
        }
        fit.showFitting(state.fitting);
    }

    // The fit panel comes first, as the stream's callbacks use it; its submit is the market's, once there is one.
    const fit = window.lobFitPanel({ prefix: 'pairs', form: $('pairs-trader-form'), submit: (...args) => market.submit(...args) });
    const market = window.lobMarketView({
        view, prefix: 'pairs', panels: PANELS, onState,
        on: { pairs_history: (data) => onHistory(JSON.parse(data)), pairs_samples: (data) => onSamples(JSON.parse(data)),
              pairs_fit: (data) => fit.showFit(data) },
        onShow: ensureCharts,
        onClose: () => { version = null; },
        onReady: panes.restoreHeight,
    });

    $('pairs-bands-select').addEventListener('change', (e) => {
        shownChoice = e.target.value;
        showLevels();
        setAll();
    });

    // What the charts hold and show (for the browser tests).
    function chartState() {
        if (!panes.ready()) return null;
        return { ...panes.state(), traders: [...traderLines.keys()], bands: shownId(), levels: levelLines.length,
                 markers: marks.filter((m) => String(m[0]) === shownId()).length };
    }
    window.lobPairs = { toggle: market.toggle, step: market.step, fit: panes.fit, center: panes.center, chartState };
})();
