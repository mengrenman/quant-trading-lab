// The tape of fills (components/trades.html): the latest fills newest first, with filters, a pause, the trade rate,
// and an optional mark on the fills the tick rule gets wrong.
//
// The markup is static and about 50 rows are built once and refilled where they are, so a typed filter, a scroll
// position or a hovered row survives the pushes. The fills are the stream's "tape" events and GET /api/tape: a fill
// is [number, simulation time, price, size, side] (side 1 when the buyer was the aggressor, -1 the seller), the
// numbers are never reused, and the page keeps the last 2,000. A push says whether its fills replace what the page
// holds ("reload": it is a new version, or the page had fallen behind the server's ring); otherwise they are the
// fills after the last one held, and a gap in the numbers makes the page load them all again. Pushes that arrive
// while it loads wait. It draws only while the Trade view is shown, and catches up when it is shown again.
//
// The list holds still, showing the fills up to the moment it stopped, while it is paused or scrolled away from the
// top, and the button says how many have come since; "Jump to newest" goes back. The rate is the fills in the last
// N units of simulation time (N is a setting) over N, from the simulation times stored with the fills. The tick
// rule is taken in the order of the fills the page holds (above the last fill's price a buy, below a sell, at it
// the same as before; unknown before the second fill), as lob/studies.py's delta of trades takes it.
(function () {
    const box = document.getElementById('tape-rows');
    if (!box) return;
    const list = document.getElementById('tape-list');
    const empty = document.getElementById('tape-empty');
    const rateText = document.getElementById('tape-rate');
    const shareText = document.getElementById('tape-share');
    const countText = document.getElementById('tape-count');
    const minInput = document.getElementById('tape-min');
    const sideSelect = document.getElementById('tape-side');
    const windowInput = document.getElementById('tape-window');
    const tickBox = document.getElementById('tape-tick');
    const pauseButton = document.getElementById('tape-pause');
    const newestButton = document.getElementById('tape-newest');
    const MAX_FILLS = 2000;
    const MAX_ROWS = 50;
    const AT_TOP = 4;  // px of scroll that still count as the top
    const STORE = 'lob-tape';
    const BADGES = {
        1: 'px-1.5 py-0.5 rounded text-[0.625rem] font-sans font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
        '-1': 'px-1.5 py-0.5 rounded text-[0.625rem] font-sans font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30',
    };
    const PRICES = { 1: 'text-right font-bold text-emerald-400', '-1': 'text-right font-bold text-rose-400' };

    let fills = [];  // [number, time, price, size, side, tick rule's side or null], oldest first
    let version = null;
    let seq = 0;  // the number of the last fill held
    let lastPrice = null;  // the tick rule's state after the last fill held
    let lastSign = null;
    let loaded = false;
    let loading = false;
    let queued = [];
    let drawPending = false;
    let paused = false;
    let shownSeq = 0;  // the newest fill the list shows (or would show, with filters): the list holds at this
    let prefs = { min: 1, side: 'all', window: 10, mark: false };
    let rate = null;
    let share = null;

    try {
        const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
        if (saved) {
            if (Number.isInteger(saved.min) && saved.min >= 1) prefs.min = saved.min;
            if (['all', 'buy', 'sell'].includes(saved.side)) prefs.side = saved.side;
            if (Number.isInteger(saved.window) && saved.window >= 1 && saved.window <= 100) prefs.window = saved.window;
            prefs.mark = saved.mark === true;
        }
    } catch (e) { /* the defaults */ }
    function savePrefs() {
        try { localStorage.setItem(STORE, JSON.stringify(prefs)); } catch (e) { /* not saved */ }
    }
    minInput.value = String(prefs.min);
    sideSelect.value = prefs.side;
    windowInput.value = String(prefs.window);
    tickBox.checked = prefs.mark;

    const isShown = () => !document.getElementById('view-trade').hidden;
    const frozen = () => paused || list.scrollTop > AT_TOP;

    // The rows: built once.
    const rows = [];
    for (let i = 0; i < MAX_ROWS; i++) {
        const el = document.createElement('div');
        el.className = 'grid grid-cols-4 items-center px-3 py-0.5 rounded hover:bg-slate-800/60 transition-colors border-b border-slate-900/50';
        el.style.display = 'none';
        el.innerHTML = '<div class="flex items-center gap-1"><span data-side></span>'
            + '<span data-tick hidden class="text-[0.625rem] font-sans font-bold text-amber-300">×</span></div>'
            + '<div data-price></div><div data-size class="text-right text-slate-200"></div>'
            + '<div data-time class="text-right text-slate-500 text-[0.6875rem]"></div>';
        box.appendChild(el);
        rows.push({ el, badge: el.querySelector('[data-side]'), tick: el.querySelector('[data-tick]'),
                    price: el.querySelector('[data-price]'), size: el.querySelector('[data-size]'),
                    time: el.querySelector('[data-time]'), side: 0, shown: null });
    }

    // ---- The fills ----

    function append(row) {
        const price = row[2];
        let sign = null;
        if (lastPrice !== null) sign = price > lastPrice ? 1 : price < lastPrice ? -1 : lastSign;
        lastPrice = price;
        lastSign = sign;
        fills.push([row[0], row[1], price, row[3], row[4], sign]);
        seq = row[0];
    }

    function replace(push) {
        fills = [];
        lastPrice = lastSign = null;
        push.fills.forEach(append);
        if (fills.length > MAX_FILLS) fills.splice(0, fills.length - MAX_FILLS);
        version = push.version;
        seq = Math.max(seq, push.seq);
        loaded = true;
        // A new market (or one that has moved on past what was shown) leaves nothing to hold still on.
        if (!fills.length || fills[0][0] > shownSeq) showPaused(false);
    }

    async function load() {
        if (loading) return;
        loading = true;
        let data = null;
        try {
            data = await (await fetch('/api/tape')).json();
        } catch (err) {
            console.error('Failed to load the tape:', err);
        }
        loading = false;  // before merging what arrived meanwhile, which may itself ask for a reload
        const pending = queued;
        queued = [];
        if (data) {
            replace(data);
            pending.forEach(merge);
        }
        schedule();
    }

    function merge(push) {
        if (!loaded) return;  // a reload is under way or due
        if (push.reload) {
            replace(push);
        } else if (push.version !== version) {
            loaded = false;
            load();
            return;
        } else {
            const fresh = push.fills.filter((fill) => fill[0] > seq);
            if (fresh.length && fresh[0][0] > seq + 1) {  // fills missed: load them all again
                loaded = false;
                load();
                return;
            }
            fresh.forEach(append);
            if (fills.length > MAX_FILLS) fills.splice(0, fills.length - MAX_FILLS);
        }
        schedule();
    }

    document.addEventListener('lob:tape', (e) => {
        if (loading) queued.push(e.detail);
        else if (loaded) merge(e.detail);  // before the first load the pushes are not needed
    });
    let connectedBefore = false;
    document.addEventListener('lob:connection', (e) => {
        if (e.detail !== 'live') return;
        // Pushes were missed while the stream was down or closed (Pairs, Stat-arb): load them all again. The first
        // connection needs nothing (a page opened on those views loaded before it had any): a new stream's first push
        // is the whole tape, and a new market is a new version, which is in every push. The chart's own reloads (an
        // interval, the footprint) are not the tape's business.
        if (connectedBefore) {
            loaded = false;
            load();
        }
        connectedBefore = true;
    });
    document.addEventListener('lob:view', (e) => {
        if (e.detail === 'trade') schedule();
    });
    document.addEventListener('DOMContentLoaded', load);
    if (document.readyState !== 'loading') load();

    // ---- Drawing ----

    function schedule() {
        if (drawPending || !isShown()) return;
        drawPending = true;
        requestAnimationFrame(() => {
            drawPending = false;
            draw(false);
        });
    }

    const passes = (fill) => fill[3] >= prefs.min && (prefs.side === 'all' || fill[4] === (prefs.side === 'buy' ? 1 : -1));

    function fillRow(r, fill) {
        const side = fill[4];
        if (r.side !== side) {
            r.side = side;
            r.badge.className = BADGES[side];
            r.badge.textContent = side === 1 ? 'BUY' : 'SELL';
            r.badge.title = side === 1 ? 'Buyer-initiated: a buy order hit the ask' : 'Seller-initiated: a sell order hit the bid';
            r.price.className = PRICES[side];
        }
        const text = [String(fill[2]), String(fill[3]), `${fill[1].toFixed(2)}s`];
        if (r.shown === null || r.shown[0] !== text[0]) r.price.textContent = text[0];
        if (r.shown === null || r.shown[1] !== text[1]) r.size.textContent = text[1];
        if (r.shown === null || r.shown[2] !== text[2]) r.time.textContent = text[2];
        r.shown = text;
        const wrong = prefs.mark && fill[5] !== null && fill[5] !== side;
        if (r.tick.hidden === wrong) r.tick.hidden = !wrong;
        if (wrong) r.tick.title = `The tick rule calls this a ${fill[5] === 1 ? 'buy' : 'sell'}; the ${side === 1 ? 'buyer' : 'seller'} was the aggressor`;
        r.el.dataset.seq = String(fill[0]);
        if (r.el.style.display !== '') r.el.style.display = '';
    }

    // The newest fills at or before ``limit`` that pass the filters, newest first.
    function pick(limit) {
        const out = [];
        for (let i = fills.length - 1; i >= 0 && out.length < MAX_ROWS; i--) {
            const fill = fills[i];
            if (fill[0] <= limit && passes(fill)) out.push(fill);
        }
        return out;
    }

    function updateRate() {
        if (!fills.length) {
            rate = share = null;
        } else {
            const last = fills[fills.length - 1][1];
            const cutoff = last - prefs.window;
            let n = 0;
            let classified = 0;
            let wrong = 0;
            for (let i = fills.length - 1; i >= 0 && fills[i][1] > cutoff; i--) {
                n++;
                if (fills[i][5] !== null) {
                    classified++;
                    if (fills[i][5] !== fills[i][4]) wrong++;
                }
            }
            const span = Math.min(prefs.window, last - fills[0][1]);  // what the page holds may cover less than the window
            rate = span >= 1 ? n / span : null;
            share = classified ? { wrong, classified, share: wrong / classified } : null;
        }
        const text = rate === null ? '—' : `${rate.toFixed(1)} fills/unit`;
        if (rateText.textContent !== text) rateText.textContent = text;
        rateText.dataset.rate = rate === null ? '' : String(rate);
        const shareLine = prefs.mark && share ? `tick rule wrong on ${(100 * share.share).toFixed(1)}% of ${share.classified} fills (${prefs.window} units)` : '';
        if (shareText.textContent !== shareLine) shareText.textContent = shareLine;
    }

    function draw(force) {
        updateRate();
        const hold = frozen();
        if (!hold) shownSeq = fills.length ? fills[fills.length - 1][0] : shownSeq;
        if (hold && !force) {
            showNewest();
            return;
        }
        const picked = pick(shownSeq);
        picked.forEach((fill, i) => fillRow(rows[i], fill));
        for (let i = picked.length; i < MAX_ROWS; i++) {
            if (rows[i].el.style.display !== 'none') rows[i].el.style.display = 'none';
        }
        const message = picked.length ? '' : fills.length ? 'No fills match the filters' : 'No executions yet';
        if (empty.textContent !== message) empty.textContent = message;
        empty.style.display = message ? '' : 'none';
        const count = `${picked.length} shown of ${fills.length}`;
        if (countText.textContent !== count) countText.textContent = count;
        showNewest();
    }

    // The button that returns to the newest fills, with how many have come since the list stopped.
    function showNewest() {
        const hold = frozen();
        let later = 0;
        if (hold) {
            for (let i = fills.length - 1; i >= 0 && fills[i][0] > shownSeq; i--) {
                if (passes(fills[i])) later++;
            }
        }
        newestButton.hidden = !hold;
        const label = later ? `Jump to newest (+${later > 99 ? '99+' : later})` : 'Jump to newest';
        if (newestButton.textContent !== label) newestButton.textContent = label;
    }

    // ---- Controls ----

    // The flag and the button that shows it, always together.
    function showPaused(on) {
        paused = on;
        pauseButton.setAttribute('aria-pressed', on ? 'true' : 'false');
        pauseButton.textContent = on ? 'Resume' : 'Pause';
    }
    function setPaused(on) {
        showPaused(on);
        draw(false);
    }
    pauseButton.addEventListener('click', () => setPaused(!paused));
    newestButton.addEventListener('click', () => {
        list.scrollTop = 0;
        setPaused(false);
        draw(true);
    });
    list.addEventListener('scroll', () => {
        if (list.scrollTop <= AT_TOP && !paused) draw(true);  // back at the top: the newest again
        else showNewest();
    });
    minInput.addEventListener('input', () => {
        const value = parseInt(minInput.value, 10);
        prefs.min = Number.isFinite(value) && value >= 1 ? value : 1;
        savePrefs();
        draw(true);
    });
    sideSelect.addEventListener('change', () => {
        prefs.side = sideSelect.value;
        savePrefs();
        draw(true);
    });
    windowInput.addEventListener('input', () => {
        const value = parseInt(windowInput.value, 10);
        if (!Number.isFinite(value) || value < 1 || value > 100) return;  // while typing
        prefs.window = value;
        savePrefs();
        draw(false);
    });
    tickBox.addEventListener('change', () => {
        prefs.mark = tickBox.checked;
        savePrefs();
        draw(true);
    });
    setPaused(false);

    window.lobTape = {  // for the browser tests
        state: () => ({
            loaded, version, seq, held: fills.length, paused, frozen: frozen(), shownSeq, rate, share, prefs: { ...prefs },
            shown: rows.filter((r) => r.el.style.display !== 'none').map((r) => ({
                seq: Number(r.el.dataset.seq), side: r.side, price: r.price.textContent, size: r.size.textContent,
                time: r.time.textContent, tick: !r.tick.hidden })),
        }),
        fills: () => fills,
    };
})();
