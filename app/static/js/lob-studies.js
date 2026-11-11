// The microstructure studies band (components/studies.html): a tile for each study with its value, what goes with
// it, and a sparkline of its once-a-unit samples against the mid.
//
// The markup is static and never replaced by a push; everything here updates it in place, and only while the Trade
// view is shown (it catches up when the view comes back). The data are the stream's "studies" events (the latest
// values, and the samples since the last push; app/studies.py) and GET /api/studies, which gives all of them. The
// page loads them when it opens, and again whenever what it holds may be out of date: a push with another version
// (a reset, a playback's start or seek, a collection's end, a change of settings), a push that leaves a gap in the
// units, and the stream connecting again (after a drop, or coming back from Pairs or Stat-arb, where it is closed).
// Pushes that arrive while it loads wait, and are merged after. It never fills a gap: the sparkline shows the units
// it has.
//
// Which tiles show and the sparklines' window are this browser's (localStorage "lob-studies"); the settings of a
// study are the session's, posted to /api/studies/config and shared by every tab.
(function () {
    const band = document.getElementById('studies-band');
    if (!band) return;
    const tilesBox = document.getElementById('studies-tiles');
    const status = document.getElementById('studies-status');
    const panel = document.getElementById('studies-panel');
    const form = document.getElementById('studies-form');
    const windowSelect = document.getElementById('studies-window');
    const MAX_ROWS = 600;  // units kept (the server's)
    const STORE = 'lob-studies';
    const NS = { TOP: 2.5, BOTTOM: 21.5, WIDTH: 100 };  // the sparkline's frame, in its viewBox units

    const tiles = {};
    tilesBox.querySelectorAll('[data-tile]').forEach((el) => {
        tiles[el.dataset.tile] = {
            el, key: el.dataset.tile, value: el.querySelector('[data-value]'), lines: el.querySelector('[data-lines]'),
            spark: el.querySelector('[data-spark]'), zero: el.querySelector('[data-zero]'),
            mid: el.querySelector('[data-mid]'), line: el.querySelector('[data-line]'),
            shown: { value: '', tone: '', lines: '', mid: '', line: '', zero: null, stroke: '' },
        };
    });

    let rows = [];  // [unit, mid, one value for each key]
    let keys = [];
    let version = null;
    let unit = null;  // the last unit of simulation time sampled in what is held (rows is empty before the first)
    let latest = {};  // the latest values by study
    let settings = {};
    let enabled = {};
    let failed = null;
    let suspended = false;
    let loaded = false;
    let loading = false;
    let queued = [];  // pushes that arrived while the page was loading
    let drawPending = false;
    let prefs = { hidden: [], window: 120 };
    let open = null;  // the study whose panel is open

    try {
        const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
        if (saved && Array.isArray(saved.hidden)) prefs.hidden = saved.hidden.filter((k) => k in tiles);
        if (saved && Number.isFinite(saved.window) && saved.window >= 5 && saved.window <= MAX_ROWS) prefs.window = saved.window;
    } catch (e) { /* the defaults */ }
    function savePrefs() {
        try { localStorage.setItem(STORE, JSON.stringify(prefs)); } catch (e) { /* not saved */ }
    }

    const isShown = () => !document.getElementById('view-trade').hidden;

    // ---- Loading and merging ----

    async function load() {
        if (loading) return;
        loading = true;
        let data = null;
        try {
            data = await (await fetch('/api/studies')).json();
        } catch (err) {
            console.error('Failed to load the studies:', err);
        }
        loading = false;  // before merging what arrived meanwhile, which may itself ask for a reload
        const pending = queued;
        queued = [];
        if (data) {
            keys = data.keys;
            rows = data.samples.slice(-MAX_ROWS);
            version = data.version;
            unit = data.unit;
            take(data);
            loaded = true;
            pending.forEach(merge);
        }
        schedule();
    }

    function take(push) {
        latest = push.values || {};
        settings = push.settings || {};
        enabled = push.enabled || {};
        failed = push.failed;
        suspended = !!push.suspended;
        if (push.keys && push.keys.length) keys = push.keys;
    }

    function merge(push) {
        if (!loaded) return;  // a reload is under way or due
        if (push.version !== version) {  // started over, or the settings changed: what is held is out of date
            loaded = false;
            load();
            return;
        }
        const fresh = push.samples;
        if (fresh.length) {
            if (unit !== null && fresh[0][0] > unit + 1) {  // units missed: never bridge them
                loaded = false;
                load();
                return;
            }
            fresh.forEach((row) => {
                const end = rows.length ? rows[rows.length - 1][0] : null;
                if (end === null || row[0] > end) rows.push(row);
            });
            if (rows.length > MAX_ROWS) rows.splice(0, rows.length - MAX_ROWS);
        }
        if (push.unit !== null) unit = push.unit;
        take(push);
        schedule();
    }

    document.addEventListener('lob:studies', (e) => {
        if (loading) queued.push(e.detail);
        else if (loaded) merge(e.detail);  // before the first load the pushes are not needed
    });
    let connectedBefore = false;
    document.addEventListener('lob:connection', (e) => {
        if (e.detail !== 'live') return;
        // Pushes were missed while the stream was down or closed (Pairs, Stat-arb): load again. The first connection
        // needs nothing (a page opened on those views loaded before it had any): a new stream's first push has every
        // sample, which a gap in the units turns into a reload, and a reset, a playback's start or seek and a
        // collection's end change the version, which every push carries. The chart's own reloads (an interval, the
        // footprint) are not the band's business.
        if (connectedBefore) {
            loaded = false;
            load();
        }
        connectedBefore = true;
    });
    document.addEventListener('lob:view', (e) => {
        if (e.detail === 'trade') schedule();
    });
    document.addEventListener('lob:text-size', schedule);
    document.addEventListener('DOMContentLoaded', load);
    if (document.readyState !== 'loading') load();

    // ---- Formatting ----

    const g = (x) => String(Number(Number(x).toFixed(3)));  // 5.0 -> "5", 0.25 -> "0.25"
    const signed = (x, digits = 0) => `${x > 0 ? '+' : x < 0 ? '-' : ''}${Math.abs(x).toLocaleString('en-US', {
        minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
    const pct = (x) => (x == null ? '—' : `${(100 * x).toFixed(1)}%`);
    const fixed = (x, digits) => (x == null ? '—' : x.toFixed(digits));
    const units = (x, digits) => (x == null ? '—' : `${x.toFixed(digits)}u`);
    const SIDES = { true: 'true side', tick: 'tick rule', quote: 'quote rule', mid: 'against the mid', bvc: 'bulk volume classification' };

    // What a tile shows from the study's latest values and settings: the main text, which way it leans
    // (+1 buying or above, -1 selling or below, 0 neutral, 2 a warning), and the lines that go with it.
    const FORMATS = {
        ofi: (v, s) => ({
            text: v.ofi == null ? 'warming up' : `${signed(v.ofi)} lots`, lean: v.ofi == null ? 0 : Math.sign(v.ofi),
            lines: [`over ${g(s.window)} units`] }),
        cvd: (v) => ({ text: `${signed(v.cvd)} lots`, lean: Math.sign(v.cvd), lines: ['since the start'] }),
        delta: (v, s) => {
            const lines = [`${SIDES[s.side]}, ${g(s.window)} units`];
            if (s.side !== 'true') lines.push(`coverage ${pct(v.coverage)} · accuracy ${pct(v.accuracy)}`);
            if (s.side === 'quote') lines.push(s.quote_lag === 0 ? 'lag 0: a check, not a result' : `quote lag ${s.quote_lag} units`);
            return { text: v.delta == null ? 'warming up' : `${signed(v.delta)} lots`, lean: v.delta == null ? 0 : Math.sign(v.delta), lines };
        },
        vpin: (v, s) => {
            const lines = [s.side === 'mid' ? `${SIDES[s.side]}, lag ${s.quote_lag}` : SIDES[s.side],
                `buckets ${Math.min(v.buckets, s.buckets)}/${s.buckets} of ${v.volume == null ? '…' : `${g(v.volume)} lots`}`];
            if (s.side === 'bvc') lines.push('near 0.5 by construction');
            return { text: v.vpin == null ? 'warming up' : v.vpin.toFixed(3), lean: 0, lines };
        },
        otr: (v, s) => ({
            text: v.otr == null ? (v.messages == null ? 'warming up' : 'no trades') : `${v.otr.toFixed(2)} msgs/trade`, lean: 0,
            lines: [`TTO ${fixed(v.tto, 3)} lots/lot`,
                `${v.messages == null ? '—' : v.messages.toLocaleString('en-US')} msgs · ${v.trades == null ? '—' : v.trades.toLocaleString('en-US')} trades, ${g(s.window)} units`,
                s.scope === 'model' ? "the model's orders" : "everyone's orders"] }),
        imbalance: (v, s) => ({
            text: v.imbalance == null ? '—' : signed(v.imbalance, 3), lean: v.imbalance == null ? 0 : Math.sign(v.imbalance),
            lines: [`${s.levels} level${s.levels === 1 ? '' : 's'}, ${s.weighting === 'touch' ? 'touch-weighted' : 'equal weight'}`
                + (v.scale == null ? '' : ` (S ${v.scale.toFixed(1)})`)] }),
        spread: (v) => ({
            text: v.spread == null ? '—' : `${v.spread} tick${v.spread === 1 ? '' : 's'}`, lean: v.shock ? 2 : 0,
            lines: [v.z == null ? 'z —' : `z ${signed(v.z, 2)}${v.shock ? ' · SHOCK' : ''}`] }),
        resilience: (v) => {
            if (v.baseline_bid == null && v.baseline_ask == null) return { text: 'warming up', lean: 0, lines: ['needs 20 units of samples'] };  // no baseline to measure against
            const now = [v.depleted_bid != null && `bid for ${v.depleted_bid.toFixed(1)}u`, v.depleted_ask != null && `ask for ${v.depleted_ask.toFixed(1)}u`]
                .filter(Boolean).join(', ');
            const lines = [now ? `depleted: ${now}` : v.started ? 'none depleted now' : 'no depletions'];
            if (v.started) {
                lines.push(`recovery ${units(v.median_recovery, 2)} · abandoned ${pct(v.abandoned_share)}`);
            }
            if (v.quote_moves) lines.push(`quote restored ${pct(v.restored_share)} in ${units(v.median_restoration, 2)}`);
            // The score is undefined while a side is depleted and no recovery has been timed yet: the tile then says
            // which side is depleted, instead of a dash.
            const sides = [v.depleted_bid != null && 'bid', v.depleted_ask != null && 'ask'].filter(Boolean);
            const text = v.score != null ? `score ${v.score.toFixed(2)}` : sides.length === 2 ? 'both depleted' : sides.length ? `${sides[0]} depleted` : '—';
            return { text, lean: 0, lines };
        },
    };
    // Studies whose value is signed, so that their sparkline shows the zero line.
    const SIGNED = new Set(['ofi', 'cvd', 'delta', 'imbalance']);
    const TONES = { 1: 'text-emerald-400', '-1': 'text-rose-400', 0: 'text-slate-100', 2: 'text-amber-400' };
    const STROKES = { 1: 'stroke-emerald-400', '-1': 'stroke-rose-400', 0: 'stroke-cyan-400', 2: 'stroke-amber-400' };

    // ---- Drawing ----

    function schedule() {
        if (drawPending || !isShown()) return;
        drawPending = true;
        requestAnimationFrame(() => {
            drawPending = false;
            draw();
        });
    }

    // The points of one sparkline: unit and value pairs, each series on its own y scale, x by the unit's place in
    // the window (the last unit at the right edge), a break wherever a value is missing.
    function points(pairs, first, span, zero) {
        const usable = pairs.filter(([, y]) => y != null);
        if (!usable.length) return { text: '', y0: null };
        let lo = Math.min(...usable.map(([, y]) => y));
        let hi = Math.max(...usable.map(([, y]) => y));
        if (zero) {
            lo = Math.min(lo, 0);
            hi = Math.max(hi, 0);
        }
        const scale = hi > lo ? (NS.BOTTOM - NS.TOP) / (hi - lo) : 0;
        const y = (value) => (hi > lo ? NS.BOTTOM - (value - lo) * scale : (NS.TOP + NS.BOTTOM) / 2);
        const text = usable.map(([unit, value]) => `${(NS.WIDTH * (unit - first) / span).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
        return { text, y0: zero ? y(0) : null };
    }

    function setText(node, text) {
        if (node.textContent !== text) node.textContent = text;
    }

    function drawTile(tile, inWindow, first, span) {
        const key = tile.key;
        const column = 2 + keys.indexOf(key);
        const on = enabled[key] !== false;
        const values = latest[key];
        let state = 'ok';
        let shown = { text: '…', lean: 0, lines: [] };
        if (!loaded || !(key in settings)) state = 'loading';
        else if (!on) {
            state = 'off';
            shown = { text: 'off', lean: 0, lines: ['switched off in its settings'] };
        } else if (suspended || !values) {
            state = 'paused';
            shown = { text: '—', lean: 0, lines: [suspended ? 'off while the playback seeks' : 'no values yet'] };
        } else {
            shown = FORMATS[key](values, settings[key]);
            if (shown.text === 'warming up') state = 'warming';
        }
        if (tile.el.dataset.state !== state) tile.el.dataset.state = state;
        setText(tile.value, shown.text);
        const tone = state === 'ok' || state === 'warming' ? (state === 'warming' ? 'text-slate-500' : TONES[shown.lean]) : 'text-slate-500';
        if (tile.shown.tone !== tone) {
            tile.value.classList.remove('text-emerald-400', 'text-rose-400', 'text-slate-100', 'text-amber-400', 'text-slate-500');
            tile.value.classList.add(tone);
            tile.shown.tone = tone;
        }
        const linesText = shown.lines.join('\n');
        if (tile.shown.lines !== linesText) {
            tile.shown.lines = linesText;
            while (tile.lines.children.length > shown.lines.length) tile.lines.lastElementChild.remove();
            while (tile.lines.children.length < shown.lines.length) tile.lines.appendChild(document.createElement('div'));
            shown.lines.forEach((line, i) => setText(tile.lines.children[i], line));
        }
        // The sparkline: the units of the window, the mid and the study's samples.
        const study = points(inWindow.map((row) => [row[0], row[column]]), first, span, SIGNED.has(key));
        const mid = points(inWindow.map((row) => [row[0], row[1]]), first, span, false);
        if (tile.shown.line !== study.text) {
            tile.shown.line = study.text;
            tile.line.setAttribute('points', study.text);
        }
        if (tile.shown.mid !== mid.text) {
            tile.shown.mid = mid.text;
            tile.mid.setAttribute('points', mid.text);
        }
        const zero = study.y0 === null ? null : study.y0.toFixed(2);
        if (tile.shown.zero !== zero) {
            tile.shown.zero = zero;
            tile.zero.setAttribute('visibility', zero === null ? 'hidden' : 'visible');
            if (zero !== null) {
                tile.zero.setAttribute('y1', zero);
                tile.zero.setAttribute('y2', zero);
            }
        }
        const stroke = STROKES[state === 'ok' ? shown.lean : 0];
        if (tile.shown.stroke !== stroke) {
            tile.line.classList.remove('stroke-emerald-400', 'stroke-rose-400', 'stroke-cyan-400', 'stroke-amber-400');
            tile.line.classList.add(stroke);
            tile.shown.stroke = stroke;
        }
    }

    function draw() {
        const last = rows.length ? rows[rows.length - 1][0] : null;
        const span = prefs.window;
        const first = last === null ? 0 : last - span;
        const inWindow = last === null ? [] : rows.filter((row) => row[0] > first);
        Object.values(tiles).forEach((tile) => {
            if (!tile.el.hidden) drawTile(tile, inWindow, first, span);
        });
        let text = 'loading…';
        let tone = 'text-slate-500';
        if (loaded) {
            if (failed) {
                text = `switched off after an error (${failed}); the simulation goes on`;
                tone = 'text-rose-400';
            } else if (suspended) text = 'off while the playback seeks: they start again at the target';
            else text = last === null ? 'sampled once a unit of simulation time' : `sampled once a unit of simulation time · unit ${last}`;
        }
        setText(status, text);
        status.classList.toggle('text-rose-400', tone === 'text-rose-400');
        status.classList.toggle('text-slate-500', tone !== 'text-rose-400');
        if (open && !panel.hidden) fillForm(open, false);
        band.dataset.loaded = loaded ? 'true' : 'false';
    }

    // ---- Which tiles show, and the sparklines' window ----

    function applyPrefs() {
        Object.values(tiles).forEach((tile) => { tile.el.hidden = prefs.hidden.includes(tile.key); });
        band.querySelectorAll('[data-chip]').forEach((chip) => {
            chip.setAttribute('aria-pressed', prefs.hidden.includes(chip.dataset.chip) ? 'false' : 'true');
        });
        windowSelect.value = String(prefs.window);
        if (windowSelect.value !== String(prefs.window)) {  // a window the menu does not offer
            const option = new Option(String(prefs.window), String(prefs.window));
            windowSelect.add(option);
            windowSelect.value = String(prefs.window);
        }
        if (open && prefs.hidden.includes(open)) closePanel();
        schedule();
    }
    band.querySelectorAll('[data-chip]').forEach((chip) => {
        chip.addEventListener('click', () => {
            const key = chip.dataset.chip;
            prefs.hidden = prefs.hidden.includes(key) ? prefs.hidden.filter((k) => k !== key) : [...prefs.hidden, key];
            savePrefs();
            applyPrefs();
        });
    });
    windowSelect.addEventListener('change', () => {
        prefs.window = Number(windowSelect.value);
        savePrefs();
        schedule();
    });
    applyPrefs();

    // ---- The panel: what a study measures, and its settings ----

    // Whether the form already shows the study's settings (the form's own Apply, once the page has them back).
    function showsSettings(set, key) {
        const same = [...set.querySelectorAll('[name]')].every((input) => {
            const value = settings[key][input.name];
            return input.type === 'number' ? Number(input.value) === Number(value) : input.value === String(value);
        });
        return same && document.getElementById('studies-enabled').checked === (enabled[key] !== false);
    }

    // Put the study's settings in the form when they are not what it was last given (another tab changed them), and
    // never while the form is being typed in, so a push cannot undo what is typed; once focus leaves it, it follows.
    // A change this form made itself (it shows the new settings already) is taken as given, so that text typed after
    // the Apply, and not yet applied, is not replaced when focus leaves.
    let given = '';
    function fillForm(key, force) {
        const set = form.querySelector(`fieldset[data-fields="${key}"]`);
        if (!set || !(key in settings)) return;
        const now = JSON.stringify([key, settings[key], enabled[key]]);
        if (!force && now === given) return;
        if (!force && form.contains(document.activeElement)) {
            if (showsSettings(set, key)) given = now;
            return;
        }
        given = now;
        set.querySelectorAll('[name]').forEach((input) => {
            const value = String(settings[key][input.name]);
            if (input.value !== value) input.value = value;
        });
        document.getElementById('studies-enabled').checked = enabled[key] !== false;
    }

    function openPanel(key, mode) {
        if (open === key && panel.dataset.mode === mode && !panel.hidden) {
            closePanel();
            return;
        }
        open = key;
        panel.hidden = false;
        panel.dataset.mode = mode;
        document.getElementById('studies-panel-title').textContent = tiles[key].el.dataset.label;
        panel.querySelectorAll('[data-help]').forEach((block) => { block.hidden = block.dataset.help !== key; });
        form.hidden = mode !== 'settings';
        form.querySelectorAll('fieldset[data-fields]').forEach((set) => {
            const on = set.dataset.fields === key;
            set.hidden = !on;
            set.disabled = !on;
        });
        form.elements.study.value = key;
        document.getElementById('studies-result').textContent = '';
        fillForm(key, true);
        Object.values(tiles).forEach((tile) => { tile.el.dataset.selected = tile.key === key ? 'true' : 'false'; });
    }

    function closePanel() {
        open = null;
        panel.hidden = true;
        Object.values(tiles).forEach((tile) => { tile.el.dataset.selected = 'false'; });
    }

    tilesBox.addEventListener('click', (e) => {
        const button = e.target.closest('[data-open]');
        if (!button) return;
        openPanel(button.closest('[data-tile]').dataset.tile, button.dataset.open);
    });
    document.getElementById('studies-panel-close').addEventListener('click', closePanel);
    document.addEventListener('keydown', (e) => {
        const where = document.activeElement;
        if (e.key === 'Escape' && open && (where === document.body || band.contains(where))) closePanel();
    });

    // The enable switch is sent as on or off (an unchecked box sends nothing).
    form.addEventListener('htmx:configRequest', (e) => {
        e.detail.parameters.enabled = document.getElementById('studies-enabled').checked ? 'on' : 'off';
    });

    window.lobStudies = {  // for the browser tests
        state: () => ({
            loaded, version, unit, keys: keys.slice(), rows: rows.length, last: rows.length ? rows[rows.length - 1][0] : null,
            values: latest, settings, enabled, failed, suspended, window: prefs.window, hidden: prefs.hidden.slice(),
        }),
        rows: () => rows,
        reload: () => { loaded = false; return load(); },
    };
})();
