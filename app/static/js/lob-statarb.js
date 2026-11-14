// The Stat-arb view (components/statarb_view.html): many order books with several common factors, and traders of many
// relations (app/statarb.py). The charts' controls and the view's stream, controls and forms are lob-panes.js's.
(function () {
    const view = document.getElementById('view-statarb');
    if (!view || !window.LightweightCharts) return;
    const PANELS = { statarb_status: 'statarb-status', statarb_books: 'statarb-books', statarb_board: 'statarb-board' };
    const MAX_SIGNALS = Number(view.dataset.signals) || 17;  // slots for a trader's lines: the server keeps 12 of them, or every book with an index
    const SECTOR_COLORS = ['sky-400', 'amber-300', 'lime-400', 'violet-300'];  // theme colors, by sector
    const INDEX_COLOR = 'slate-100';  // the index book's own, none of a sector's
    const $ = (id) => document.getElementById(id);

    let version = null;
    let points = [];     // the chart's samples, oldest first (app/statarb.py, StatArbSession._sample)
    let traders = [];    // {id, name, color, active, kind, entry, exit}, in the order they started
    let names = [];      // each book's name
    let groups = [];     // each book's sector, for its color
    let indexBook = null;  // the place of the index book (the last), or null: drawn over the number of the other books
    let shownChoice = null;  // whose signals and holdings the chart shows: null the first trader's, '' no one's
    const bookLines = [];
    const traderLines = new Map();
    let levelLines = [];

    function palette() {
        const c = window.themeColor;
        return {
            background: c('darkBg'), text: c('slate-400'), grid: c('slate-800', 0.45), border: c('slate-800'),
            crosshair: c('sky-400', 0.4), label: c('slate-900'), band: c('amber-400', 0.85), zero: c('slate-600'),
            signal: c('cyan-300', 0.85), gross: c('lime-400'), exposure: c('rose-400'),
        };
    }
    const panes = window.lobPanes({ view, prefix: 'statarb', panes: ['books', 'signal', 'held', 'pnl'], palette,
                                    points: () => points, onCrosshair: () => showHud() });
    const { addLine, removeLine, lineOf } = panes;

    function shownId() {
        if (shownChoice === '') return null;
        if (shownChoice !== null && traders.some((tr) => String(tr.id) === shownChoice)) return shownChoice;
        return traders.length ? String(traders[0].id) : null;
    }
    const shownTrader = () => traders.find((tr) => String(tr.id) === shownId()) || null;

    function ensureCharts() {
        if (panes.ready()) return;
        panes.make();
        const { Dotted } = LightweightCharts.LineStyle;
        const of = (field) => (p) => {
            const id = shownId();
            const v = id === null || !p[field] ? null : p[field][id];
            return v === undefined ? null : v;
        };
        for (let i = MAX_SIGNALS - 1; i >= 0; i -= 1) {
            addLine('signal', `signal${i}`, (Q) => Q.signal, (p) => {
                const values = of('sig')(p);
                const v = values ? values[i] : null;
                return v === undefined ? null : v;
            }, { width: 1, last: false, breaks: true });  // a deal trader's lines have gaps, where a deal ends
        }
        addLine('signal', 'signalZero', (Q) => Q.zero, () => 0, { width: 1, style: Dotted, last: false });
        addLine('held', 'gross', (Q) => Q.gross, of('gross'), { precision: 0, title: 'lots' });
        addLine('held', 'exposure', (Q) => Q.exposure, of('exposure'), { precision: 1, title: 'factors' });
        addLine('pnl', 'zero', (Q) => Q.zero, () => 0, { width: 1, style: Dotted, precision: 1, last: false });
        syncBooks();
        syncTraders();
        setAll();
        panes.frame();
    }

    function setAll() {
        if (!panes.ready()) return;
        panes.setAll();
        showHud();
    }

    // A line a book, colored by its sector. The index book's (if any) is over the number of the others, so that it sits
    // among them, and has a color of its own.
    function syncBooks() {
        if (!panes.ready()) return;
        while (bookLines.length > names.length) removeLine(bookLines.pop());
        while (bookLines.length < names.length) {
            const i = bookLines.length;
            const entry = addLine('books', `book${i}`, () => bookColor(i), (p) => bookValue(p, i), { width: 1, precision: 1, last: false });
            bookLines.push(entry);
        }
        const P = palette();
        bookLines.forEach((entry, i) => entry.series.applyOptions({ color: entry.color(P), lineWidth: i === indexBook ? 2 : 1 }));
        const legend = $('statarb-books-legend');
        const sectors = [...new Set(groups.filter((_, i) => i !== indexBook))];
        const swatched = (color, label) => {
            const item = document.createElement('span');
            item.className = 'flex items-center gap-1';
            const swatch = document.createElement('span');
            swatch.className = 'w-3 h-0.5 rounded';
            swatch.style.backgroundColor = window.themeColor(color);
            item.append(swatch, document.createTextNode(label));
            return item;
        };
        const items = sectors.map((group) => {
            const first = names[groups.indexOf(group)];
            return swatched(SECTOR_COLORS[group % SECTOR_COLORS.length], sectors.length > 1 ? `sector ${first[0]}` : 'every book');
        });
        if (indexBook !== null) items.push(swatched(INDEX_COLOR, `${names[indexBook]}, the index over ${indexBook}`));
        legend.replaceChildren(...items);
    }
    function bookColor(i) {
        return window.themeColor(i === indexBook ? INDEX_COLOR : SECTOR_COLORS[groups[i] % SECTOR_COLORS.length]);
    }
    function bookValue(p, i) {
        if (!p.mids) return null;
        return i === indexBook ? p.mids[i] / indexBook : p.mids[i];
    }

    // The levels of the trader shown, on the signal pane.
    function showLevels() {
        if (!panes.ready()) return;
        const P = palette();
        const line = lineOf('signalZero').series;
        levelLines.forEach((level) => line.removePriceLine(level));
        levelLines = [];
        const tr = shownTrader();
        const legend = $('statarb-signal-legend');
        if (!tr) {
            legend.textContent = 'of the trader shown, one a relation, against its levels';
            return;
        }
        const { Dashed, Dotted } = LightweightCharts.LineStyle;
        const level = (price, style) => levelLines.push(line.createPriceLine({
            price, color: P.band, lineWidth: 1, lineStyle: style, axisLabelVisible: false, title: '' }));
        if (tr.kind === 'deals') {  // gaps in ticks, entered at or above one level: 0 is the line that is always there
            if (tr.entry > 0) level(tr.entry, Dashed);
        } else if (tr.entry || tr.exit) {  // a trader that scores every book has none
            [tr.entry, -tr.entry].forEach((price) => level(price, Dashed));
            if (tr.exit > 0) [tr.exit, -tr.exit].forEach((price) => level(price, Dotted));
        }
        const BEFORE = 'before it is made neutral';
        const says = {
            pairs: 'each pair\'s z-score', baskets: 'each basket\'s z-score', residuals: 'each book\'s s-score',
            copula: 'half the difference of each pair\'s flags',
            reversal: `each book's score, minus its residual move over the look back, ${BEFORE}`,
            momentum: `each book's score, its residual move over the look back, ${BEFORE}`,
            leadlag: `each book's score, what the other books' last moves say of its next, ${BEFORE}`,
            flow: `each book's score, its bought share of trades or its queue imbalance, ${BEFORE}`,
            index: 'the z-score of the index\'s spread over the books it is the sum of',
            deals: `the gap of each deal on, in ticks (the offer's worth less its target's mid), entered at ${tr.entry} or more`,
        };
        legend.textContent = `${tr.name}: ${says[tr.kind] || 'the signal of each relation'}`;
    }

    function showHud() {
        const p = panes.crosshair() || points[points.length - 1];
        if (!p) return;
        const parts = [`time ${Math.round(Number(p.t)).toLocaleString('en-US')}`];
        const id = shownId();
        if (id !== null) {
            const signed = (v) => (v === null || v === undefined ? '—' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(1)}`);
            if (p.pnl && p.pnl[id] !== undefined) parts.push(`P&L ${signed(p.pnl[id])}`);
            if (p.gross && p.gross[id] !== undefined) parts.push(`held ${p.gross[id]}`);
            if (p.exposure && p.exposure[id] !== undefined) parts.push(`factors ${Number(p.exposure[id]).toFixed(1)}`);
        }
        $('statarb-hud').textContent = parts.join('  ');
    }

    // The traders: a P&L line each, an entry in the P&L legend, and a choice in "Show".
    function syncTraders() {
        const select = $('statarb-shown-select');
        const chosen = select.value;
        select.replaceChildren(new Option('no one', ''), ...traders.map((tr) => new Option(tr.name, String(tr.id))));
        const id = shownId();
        select.value = id === null ? '' : id;
        if (chosen !== select.value && panes.ready()) setAll();
        $('statarb-pnl-legend').replaceChildren(...traders.map((tr) => {
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
        syncBooks();
        syncTraders();
    }
    document.addEventListener('lob:theme', restyle);
    document.addEventListener('lob:text-size', restyle);

    function onHistory(h) {
        version = h.version;
        points = h.samples;
        traders = h.traders;
        names = h.names;
        groups = h.groups;
        indexBook = h.index === undefined ? null : h.index;
        market.showState(h);
        syncBooks();
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
        if (points.length > 6000) {  // keep as many as the server does
            points = points.slice(-3000);
            setAll();
            panes.afterDropping();
        }
        if (added) {
            panes.pointsAdded();
            if (!panes.crosshair()) showHud();
        }
    }

    // The fit panel comes first, as the stream's callbacks use it; its submit is the market's, once there is one.
    const fit = window.lobFitPanel({ prefix: 'statarb', form: $('statarb-trader-form'), submit: (...args) => market.submit(...args) });
    const market = window.lobMarketView({
        view, prefix: 'statarb', panels: PANELS,
        onState: (state) => {
            if (state.traders) {
                traders = state.traders;
                syncTraders();
            }
            if (state.settings) chooseShape();
            fit.showFitting(state.fitting);
        },
        on: { statarb_history: (data) => onHistory(JSON.parse(data)), statarb_samples: (data) => onSamples(JSON.parse(data)),
              statarb_fit: (data) => fit.showFit(data) },
        onShow: ensureCharts,
        onClose: () => { version = null; },
        onReady: panes.restoreHeight,
    });

    $('statarb-shown-select').addEventListener('change', (e) => {
        shownChoice = e.target.value;
        showLevels();
        setAll();
    });

    // Only the chosen shape's settings show and are sent (a disabled fieldset's are not).
    const configForm = $('statarb-config');
    function chooseShape() {
        const shape = (configForm.querySelector('input[name="shape"]:checked') || {}).value;
        configForm.querySelectorAll('fieldset[data-statarb-shape]').forEach((set) => {
            const on = set.dataset.statarbShape === shape;
            set.hidden = !on;
            set.disabled = !on;
        });
    }
    configForm.addEventListener('change', (e) => { if (e.target.name === 'shape') chooseShape(); });
    chooseShape();

    // What the tests see: the answer comes on the status panel, by the stream.
    $('statarb-test-btn').addEventListener('click', async () => {
        const button = $('statarb-test-btn');
        if (button.getAttribute('aria-disabled') === 'true') return;
        button.setAttribute('aria-disabled', 'true');
        try {
            await window.postForm('/api/statarb/test');
        } catch (err) {
            if (window.showToast) window.showToast('Not done: the server did not answer.');
        }
        button.removeAttribute('aria-disabled');
    });

    // What the charts hold and show (for the browser tests).
    function chartState() {
        if (!panes.ready()) return null;
        const slots = Array.from({ length: MAX_SIGNALS }, (_, i) => i);
        const drawn = (i) => lineOf(`signal${i}`).series.data().some((p) => p.value !== undefined);
        // Where a line of the trader shown stops, with a value before a sample without one, and how many points the lines
        // hold made clear, so that nothing is drawn across the gap (a deal's line: lob-panes.js ``breaks``).
        const id = shownId();
        const fed = (p, i) => { const v = id !== null && p.sig && p.sig[id] ? p.sig[id][i] : null; return v === undefined ? null : v; };
        const gaps = slots.reduce((n, i) => n + points.filter((p, k) => k > 0 && fed(p, i) === null && fed(points[k - 1], i) !== null).length, 0);
        const cleared = slots.reduce((n, i) => n + lineOf(`signal${i}`).series.data().filter((p) => p.color === panes.clear).length, 0);
        const lastValue = (entry) => { const data = entry.series.data(); return data.length ? data[data.length - 1].value : null; };
        return { ...panes.state(), traders: [...traderLines.keys()], shown: shownId(), levels: levelLines.length, books: bookLines.length,
                 signals: slots.filter(drawn).length, gaps, cleared,
                 index: indexBook, bookLines: bookLines.map((entry) => ({ color: entry.series.options().color, width: entry.series.options().lineWidth,
                                                                         last: lastValue(entry) })) };
    }
    window.lobStatarb = { toggle: market.toggle, step: market.step, fit: panes.fit, center: panes.center, chartState };
})();
