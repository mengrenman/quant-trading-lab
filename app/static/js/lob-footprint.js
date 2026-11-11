// The footprint on the candle chart (the toggle on the chart's toolbar, components/chart.html): in each bar, at each
// price, the lots sold into the bid (the left half of the cell) and bought from the ask (the right half), the
// color deeper the more volume, the numbers when there is room, the busiest price (the point of control) outlined,
// and the diagonal imbalances marked: buying at a price at least `ratio` times the selling one price below it, and
// selling at a price at least `ratio` times the buying one price above it, each of at least `minLots` lots.
//
// It is a Lightweight Charts series primitive attached to the candle series, drawn in the library's own canvas, so
// it follows every scroll, zoom and autoscale and lets the pointer through. While it is on the candle bodies are
// transparent (their outlines and wicks stay), not hidden: a hidden series stops the price scale autoscaling.
// A finished bar's column is drawn once on a canvas of its own and copied while nothing changes (its width, row
// height, the theme and the settings); the newest bar is drawn afresh at each paint, and only the bars in view are
// drawn.
//
// The data are the candles' ``cells``: [price, lots sold into the bid, lots bought from the ask], by rising price,
// which the server sends for the session when the footprint is on (POST /api/chart/footprint, the session's flag:
// every tab follows it, through the reload that /api/candlesticks gives). The page checks them against the bars
// they belong to: the buy cells sum to the bar's buy volume and all the cells to its volume. The result is in
// the chart container's data-footprint-check ("ok", "bad" or "none" while no bar has cells), never a console error.
// Which tiles to draw, the numbers' size and the like are this browser's: only the imbalance settings are kept
// (localStorage "lob-footprint").
(function () {
    const container = document.getElementById('price-chart-container');
    const toggle = document.getElementById('footprint-toggle');
    if (!container || !toggle) return;
    const more = document.getElementById('footprint-more');
    const hint = document.getElementById('footprint-hint');
    const ratioInput = document.getElementById('footprint-ratio');
    const minInput = document.getElementById('footprint-min');
    const wideButton = document.getElementById('footprint-wide');

    const CLEAR = 'rgba(0, 0, 0, 0)';
    const NUMBERS_ROW_PX = 9;  // the numbers need rows this tall and bars this far apart (CSS px)
    const NUMBERS_BAR_PX = 40;
    const IMBALANCE_BAR_PX = 12;  // a column narrower than this has no imbalance outlines
    const WIDE_SPACING = 48;  // what "Widen bars" sets
    const LEGIBLE_INTERVAL = 15;
    const STEPS = 16;  // shades of a cell's color
    const MAX_ROWS = 400;  // price rows a bar may span and still be drawn
    const MAX_CACHED = 80;  // finished bars that keep their column
    const MAX_BARS = 2000;  // bars kept (the oldest let go): a run of hours at a small interval makes many more than are ever fetched
    const NARROW_SPACING = 24;  // switched on at a bar spacing under this, the chart is widened once (see widenIfNarrow)

    let on = false;  // the chart carries cells: the session's flag, as /api/candlesticks last said
    let interval = 2;
    let attachedTo = null;  // the series the primitive is on
    let chart = null;
    let series = null;
    let request = () => {};  // asks the chart for a repaint (the primitive's requestUpdate)
    const entries = new Map();  // bar time -> {time, cells, volume, buy, version, derived, cache}
    let times = [];  // the bars' times, ascending
    let bad = new Set();  // the times of bars whose cells disagree with their volumes
    let withCells = 0;  // the bars held that have cells, kept as bars come and go
    let widenNext = false;  // the user has just switched it on: the next load widens the chart, if it is narrow
    let themeVersion = 0;
    let palette = null;
    let settings = { ratio: 3, minLots: 10 };
    let paints = 0;
    let columns = 0;  // columns drawn to a canvas of their own
    let copies = 0;  // columns copied from there
    let last = null;  // the last paint, for the tests
    let error = null;
    const cached = new Set();

    try {
        const saved = JSON.parse(localStorage.getItem('lob-footprint') || 'null');
        if (saved && saved.ratio >= 1 && saved.ratio <= 50) settings.ratio = saved.ratio;
        if (saved && Number.isInteger(saved.minLots) && saved.minLots >= 1) settings.minLots = saved.minLots;
    } catch (e) { /* the defaults */ }
    ratioInput.value = String(settings.ratio);
    minInput.value = String(settings.minLots);

    // ---- Data ----

    function check(entry) {
        if (!entry.cells) {
            bad.delete(entry.time);
            return;
        }
        let all = 0;
        let bought = 0;
        entry.cells.forEach((cell) => {
            all += cell[1] + cell[2];
            bought += cell[2];
        });
        if (all === entry.volume && bought === entry.buy) bad.delete(entry.time);
        else bad.add(entry.time);
    }

    function setEntry(c) {
        const previous = entries.get(c.time);
        const entry = { time: c.time, cells: c.cells || null, volume: c.volume, buy: c.buy_volume,
                        version: previous ? previous.version + 1 : 0, derived: null, cache: null };
        if (previous && previous.cache) cached.delete(previous);
        if (previous && previous.cells) withCells--;
        if (entry.cells) withCells++;
        entries.set(c.time, entry);
        if (!previous) {
            const n = times.length;
            if (!n || c.time > times[n - 1]) times.push(c.time);
            else {  // an older bar: insert in order
                let lo = 0;
                let hi = n;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (times[mid] < c.time) lo = mid + 1;
                    else hi = mid;
                }
                times.splice(lo, 0, c.time);
            }
        }
        check(entry);
        while (times.length > MAX_BARS) {  // the oldest go
            const time = times.shift();
            const old = entries.get(time);
            if (old.cells) withCells--;
            if (old.cache) cached.delete(old);
            entries.delete(time);
            bad.delete(time);
        }
    }

    function showStatus() {
        container.dataset.footprint = on ? 'on' : 'off';
        container.dataset.footprintBars = String(withCells);
        container.dataset.footprintBad = String(bad.size);
        container.dataset.footprintCheck = !on || !withCells ? 'none' : bad.size ? 'bad' : 'ok';
        toggle.setAttribute('aria-pressed', on ? 'true' : 'false');
        more.hidden = !on;
        const short = on && interval < LEGIBLE_INTERVAL;
        hint.classList.toggle('text-amber-400', short);
        hint.classList.toggle('text-slate-500', !short);
        const slider = document.getElementById('hzoom-slider');
        if (slider) slider.max = on ? '72' : '32';
    }

    // A fresh set of bars (the chart reloaded): ``flag`` is whether the session carries cells.
    function load(flag, candles, newInterval) {
        on = !!flag;
        interval = newInterval;
        entries.clear();
        cached.clear();
        times = [];
        bad = new Set();
        withCells = 0;
        if (on) candles.forEach(setEntry);
        applyMode();
        if (widenNext && on) {  // (a reload that was under way before the switch says off: the next one is the one)
            widenNext = false;
            // After the rest of this reload, which sets the chart's bars and spacing.
            if (interval >= LEGIBLE_INTERVAL) setTimeout(widenIfNarrow, 0);
        }
    }

    // One bar pushed (or the same bar again, changed).
    function bar(c) {
        if (!on) return;
        setEntry(c);
        showStatus();
    }

    // ---- What a bar's column shows ----

    function derive(entry) {
        const key = `${settings.ratio}|${settings.minLots}|${entry.version}`;
        if (entry.derived && entry.derived.key === key) return entry.derived;
        const cells = entry.cells;
        const sold = new Map();
        const bought = new Map();
        let top = -Infinity;
        let bottom = Infinity;
        let busiest = 0;
        let poc = null;
        let peak = 0;
        let chars = 1;
        cells.forEach(([price, sell, buy]) => {
            chars = Math.max(chars, compact(sell).length, compact(buy).length);
            sold.set(price, sell);
            bought.set(price, buy);
            top = Math.max(top, price);
            bottom = Math.min(bottom, price);
            if (sell + buy > busiest) {
                busiest = sell + buy;
                poc = price;
            }
            peak = Math.max(peak, sell, buy);
        });
        const buyImbalance = new Set();
        const sellImbalance = new Set();
        cells.forEach(([price, sell, buy]) => {
            const below = sold.get(price - 1) || 0;  // the selling one price below the buying
            if (buy > 0 && buy >= settings.minLots && buy >= settings.ratio * below) buyImbalance.add(price);
            const above = bought.get(price + 1) || 0;  // the buying one price above the selling
            if (sell > 0 && sell >= settings.minLots && sell >= settings.ratio * above) sellImbalance.add(price);
        });
        entry.derived = { key, top, bottom, poc, peak, chars, buyImbalance, sellImbalance, rows: top - bottom + 1 };
        return entry.derived;
    }

    function buildPalette() {
        const shades = (name) => {
            const [r, g, b] = window.themeTriplet(name);
            return Array.from({ length: STEPS }, (_, i) => `rgba(${r}, ${g}, ${b}, ${(0.12 + 0.76 * (i / (STEPS - 1))).toFixed(3)})`);
        };
        palette = {
            buy: shades('emerald-500'), sell: shades('rose-500'),
            buyEdge: window.themeColor('emerald-300'), sellEdge: window.themeColor('rose-300'),
            poc: window.themeColor('amber-400'), text: window.themeColor('slate-100'),
            faint: window.themeColor('slate-800', 0.35),
        };
    }

    const level = (lots, peak) => Math.min(STEPS - 1, Math.max(0, Math.round(Math.sqrt(lots / peak) * (STEPS - 1))));
    const compact = (lots) => (lots >= 10000 ? `${Math.round(lots / 1000)}k` : String(lots));

    // Draw one bar's column into ``ctx`` with its top-left corner at (x0, y0), in bitmap pixels: ``wide`` pixels
    // across (the left half sold, the right bought), ``tall`` a price row, ``digits`` the numbers' font size or 0.
    function drawColumn(ctx, entry, d, x0, y0, wide, tall, digits, line, edges) {
        const half = Math.floor(wide / 2);
        const gap = tall >= 4 ? 1 : 0;
        if (digits) {
            ctx.font = `${digits}px "JetBrains Mono", ui-monospace, monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
        }
        entry.cells.forEach(([price, sell, buy]) => {
            const k = d.top - price;
            const ya = Math.round(y0 + k * tall);
            const h = Math.max(1, Math.round(y0 + (k + 1) * tall) - ya - gap);
            if (sell > 0) {
                ctx.fillStyle = palette.sell[level(sell, d.peak)];
                ctx.fillRect(x0, ya, half, h);
            }
            if (buy > 0) {
                ctx.fillStyle = palette.buy[level(buy, d.peak)];
                ctx.fillRect(x0 + half, ya, wide - half, h);
            }
            if (digits) {
                ctx.fillStyle = palette.text;
                const mid = ya + h / 2 + 0.5;
                if (sell > 0) ctx.fillText(compact(sell), x0 + half / 2, mid);
                if (buy > 0) ctx.fillText(compact(buy), x0 + half + (wide - half) / 2, mid);
            }
            ctx.lineWidth = line;
            if (edges && d.sellImbalance.has(price)) {
                ctx.strokeStyle = palette.sellEdge;
                ctx.strokeRect(x0 + line / 2, ya + line / 2, half - line, h - line);
            }
            if (edges && d.buyImbalance.has(price)) {
                ctx.strokeStyle = palette.buyEdge;
                ctx.strokeRect(x0 + half + line / 2, ya + line / 2, wide - half - line, h - line);
            }
            if (price === d.poc) {
                ctx.strokeStyle = palette.poc;
                ctx.strokeRect(x0 + line / 2, ya + line / 2, wide - line, h - line);
            }
        });
    }

    // ---- The primitive ----

    function paint(scope) {
        const ctx = scope.context;
        const hpr = scope.horizontalPixelRatio;
        const vpr = scope.verticalPixelRatio;
        paints++;
        const frame = { paint: paints, bars: [], spacing: null, rowHeight: null, numbers: false, copied: 0, drawn: 0 };
        last = frame;
        if (!on || !chart || !series || !times.length) return;
        if (!palette) buildPalette();
        const scale = chart.timeScale();
        const range = scale.getVisibleRange();
        const spacing = scale.options().barSpacing;
        if (!range || !(spacing > 0)) return;
        frame.spacing = spacing;
        let lo = 0;
        let hi = times.length;
        while (lo < hi) {  // the first bar that can be in view
            const mid = (lo + hi) >> 1;
            if (times[mid] < range.from - interval) lo = mid + 1;
            else hi = mid;
        }
        const wide = Math.max(2, Math.round(Math.max(1, spacing - Math.max(1, spacing * 0.08)) * hpr));
        const line = Math.max(1, Math.round(1.25 * Math.min(hpr, vpr)));
        const edges = wide / hpr >= IMBALANCE_BAR_PX;  // narrower, the outlines would hide the colors: only the point of control
        const newest = times[times.length - 1];
        const height = scope.mediaSize.height;
        for (let i = lo; i < times.length && times[i] <= range.to + interval; i++) {
            const entry = entries.get(times[i]);
            if (!entry || !entry.cells || !entry.cells.length) continue;
            const x = scale.timeToCoordinate(entry.time);
            if (x === null) continue;
            const d = derive(entry);
            if (d.rows > MAX_ROWS) continue;
            const yTop = series.priceToCoordinate(d.top);
            const yBelow = series.priceToCoordinate(d.top - 1);
            const yBottom = series.priceToCoordinate(d.bottom);
            if (yTop === null || yBelow === null || yBottom === null) continue;
            const rowHeight = Math.abs(yBelow - yTop);
            if (rowHeight <= 0 || yBottom + rowHeight < 0 || yTop - rowHeight > height) continue;  // out of sight
            const numbers = rowHeight >= NUMBERS_ROW_PX && spacing >= NUMBERS_BAR_PX;
            const left = Math.round((x - wide / hpr / 2) * hpr);
            const top = (yTop - rowHeight / 2) * vpr;
            const tall = rowHeight * vpr;
            // A font the busiest number fits in (a digit is about 0.62 of the font's size across), or none.
            const fits = Math.floor((wide / 2 - 2) / (0.62 * d.chars));
            const digits = numbers ? Math.min(Math.round(11 * Math.min(hpr, vpr)), Math.floor(tall - 2), fits) : 0;
            frame.rowHeight = rowHeight;
            frame.numbers = numbers;
            if (entry.time === newest || d.rows * tall > 4000) {  // the newest changes at every push: drawn afresh
                drawColumn(ctx, entry, d, left, top, wide, tall, digits >= 6 ? digits : 0, line, edges);
                frame.drawn++;
            } else {
                const key = `${wide}|${tall.toFixed(2)}|${digits >= 6 ? digits : 0}|${themeVersion}|${d.key}`;
                if (!entry.cache || entry.cache.key !== key) {
                    const canvas = entry.cache ? entry.cache.canvas : document.createElement('canvas');
                    canvas.width = wide;
                    canvas.height = Math.ceil(d.rows * tall) + 2;
                    drawColumn(canvas.getContext('2d'), entry, d, 0, 0, wide, tall, digits >= 6 ? digits : 0, line, edges);
                    entry.cache = { key, canvas };
                    cached.add(entry);
                    columns++;
                } else {
                    copies++;
                    frame.copied++;
                }
                ctx.drawImage(entry.cache.canvas, left, Math.round(top));
            }
            frame.bars.push({ time: entry.time, x, top: yTop, bottom: yBottom, rows: d.rows, poc: d.poc,
                              buyImbalances: d.buyImbalance.size, sellImbalances: d.sellImbalance.size });
        }
        if (cached.size > MAX_CACHED) {  // the columns of bars out of view are let go
            const visible = new Set(frame.bars.map((b) => b.time));
            cached.forEach((entry) => {
                if (!visible.has(entry.time)) {
                    entry.cache = null;
                    cached.delete(entry);
                }
            });
        }
    }

    const renderer = {
        draw(target) {
            target.useBitmapCoordinateSpace((scope) => {
                try {
                    paint(scope);
                } catch (err) {  // a drawing fault must not stop the chart's own painting
                    if (!error) console.warn('The footprint could not be drawn:', err);
                    error = String(err);
                }
            });
        },
    };
    const paneView = { zOrder: () => 'top', renderer: () => renderer };
    const primitive = {
        attached(params) {
            chart = params.chart;
            series = params.series;
            request = params.requestUpdate;
        },
        detached() {
            chart = series = null;
            request = () => {};
        },
        updateAllViews() {},
        paneViews: () => [paneView],
    };

    // ---- On and off ----

    function theSeries() {
        return typeof candleSeries !== 'undefined' ? candleSeries : null;
    }

    // The candle bodies: transparent while the footprint is on (their outlines and wicks stay).
    function setBodies() {
        const s = theSeries();
        if (!s || typeof chartPalette !== 'function') return;
        const P = chartPalette();
        s.applyOptions({
            upColor: on ? CLEAR : P.up, downColor: on ? CLEAR : P.down,
            borderUpColor: P.up, borderDownColor: P.down, wickUpColor: P.up, wickDownColor: P.down,
        });
    }

    function applyMode() {
        const s = theSeries();
        if (s && on && attachedTo !== s) {  // the chart may have been made again: attach to the series it has now
            if (attachedTo) {
                try { attachedTo.detachPrimitive(primitive); } catch (e) { /* that series is gone */ }
            }
            s.attachPrimitive(primitive);
            attachedTo = s;
        } else if (s && !on && attachedTo) {
            try { attachedTo.detachPrimitive(primitive); } catch (e) { /* that series is gone */ }
            attachedTo = null;
        }
        setBodies();
        showStatus();
        request();
    }

    toggle.addEventListener('click', async () => {
        const want = toggle.getAttribute('aria-pressed') !== 'true';
        widenNext = want;  // before the post: the stream's chart reset may reach the page before the answer does
        try {
            await window.postForm('/api/chart/footprint', { on: want });
        } catch (err) {
            widenNext = false;
            console.error('Failed to switch the footprint:', err);
            return;
        }
        if (typeof loadChartData === 'function') loadChartData();  // the cells come with the bars
    });

    function setting() {
        const ratio = parseFloat(ratioInput.value);
        const minLots = parseInt(minInput.value, 10);
        if (ratio >= 1 && ratio <= 50) settings.ratio = ratio;
        if (Number.isFinite(minLots) && minLots >= 1) settings.minLots = minLots;
        try { localStorage.setItem('lob-footprint', JSON.stringify(settings)); } catch (e) { /* not saved */ }
        request();
    }
    ratioInput.addEventListener('input', setting);
    minInput.addEventListener('input', setting);
    // Bars WIDE_SPACING apart ending at the newest (a range is set rather than the spacing: with few bars the chart
    // holds the spacing but not the position, which would leave the newest bars out of sight). The volume chart
    // follows the price chart's range, and each chart corrects a range it cannot show (the two panes are not as wide
    // until their price axes are, and the one's correction can reach the other), so the range is looked at again
    // a moment later and set again if the left edge is not where it was asked to be.
    function widen(attempt) {
        const s = theSeries();
        if (!s || typeof priceChart === 'undefined' || !priceChart) return;
        const n = s.data().length;
        if (!n) return;
        const scale = priceChart.timeScale();
        const bars = scale.width() / WIDE_SPACING;  // bars across
        const to = Math.max(n - 1 + (scale.options().rightOffset || 0), bars - 0.5);  // with few bars, they start at the left
        const from = to - bars;
        scale.setVisibleLogicalRange({ from, to });
        setTimeout(() => {
            const range = scale.getVisibleLogicalRange();
            if (attempt < 4 && range && range.from > Math.max(from, 0) + 0.5) widen(attempt + 1);
            else if (typeof syncZoomControlsFromChart === 'function') syncZoomControlsFromChart();
        }, 120);
    }
    wideButton.addEventListener('click', () => widen(0));

    // Switched on at the usual few pixels a bar, the footprint is slivers without numbers even at an interval where
    // it reads: widen the chart once, as "Widen bars" does (a chart without bars yet is given the spacing, for the
    // bars to come). A spacing the user has already widened to is left alone, and so is every later reload.
    function widenIfNarrow() {
        if (!on || typeof priceChart === 'undefined' || !priceChart) return;
        if (priceChart.timeScale().options().barSpacing >= NARROW_SPACING) return;
        const s = theSeries();
        if (s && s.data().length) {
            widen(0);
            return;
        }
        [priceChart, typeof volumeChart === 'undefined' ? null : volumeChart].forEach((c) => {
            if (c) c.timeScale().applyOptions({ barSpacing: WIDE_SPACING });
        });
        if (typeof syncZoomControlsFromChart === 'function') syncZoomControlsFromChart();
    }

    // The theme (and the color-blind palette) changed: colors again, and the candle bodies again (the chart's own
    // theme step has just set them opaque).
    document.addEventListener('lob:chart-theme', () => {
        themeVersion++;
        palette = null;
        if (on) setBodies();
        request();
    });
    showStatus();

    window.lobFootprint = {
        load, bar, on: () => on,
        state: () => {  // what was drawn, for the browser tests
            let counted = 0;  // counted afresh: the data attribute's figure is kept incrementally
            entries.forEach((entry) => { if (entry.cells) counted++; });
            return { on, interval, attached: !!attachedTo, bars: entries.size, withCells: counted, bad: [...bad], paints, columns, copies,
                     settings: { ...settings }, error, last, times: times.slice() };
        },
        cells: (time) => (entries.get(time) || {}).cells || null,
        derived: (time) => {
            const entry = entries.get(time);
            if (!entry || !entry.cells) return null;
            const d = derive(entry);
            return { top: d.top, bottom: d.bottom, poc: d.poc, buyImbalance: [...d.buyImbalance], sellImbalance: [...d.sellImbalance] };
        },
        repaint: () => request(),
    };
})();
