// The order-book heatmap (components/heatmap.html): the visible depth at each price over simulation
// time, one column per unit of time, with the trades and the mid-price.
//
// The history comes from /api/heatmap the first time the Analyze view is shown, and again when the
// heatmap starts over (a new version: reset, or after a collection) or the stream reconnects. The
// stream's heatmap events then bring the columns from the last one pushed on: the latest column's
// trades grow until the next column starts, so it comes again until then.
(function () {
    const canvas = document.getElementById('heatmap-canvas');
    if (!canvas) return;
    const tip = document.getElementById('heatmap-tip');
    const span = document.getElementById('heatmap-span');
    const MAX_COLUMNS = 600;
    const MIN_COLUMN_PX = 2;  // the narrowest a column is drawn; older ones scroll off the left
    const MAX_COLUMN_PX = 16;  // the widest (a short history is not stretched across)
    const SHADES = 40;  // steps of the color scale
    const T_MIN = 0.05;  // the scale runs over depth / median depth from T_MIN to T_MAX (log steps)
    const T_MAX = 50;

    let columns = [];  // sorted by t
    let version = null;
    let loaded = false;
    let loading = false;
    let queued = [];  // pushes that arrived while the history was loading
    let showTrades = true;
    let showHidden = false;
    let drawPending = false;
    let geometry = null;  // the last drawing's layout, for the tooltip

    window.heatmapColumns = () => columns;  // for the browser tests

    const isShown = () => !document.getElementById('view-analyze').hidden;

    async function load() {
        if (loading) return;
        loading = true;
        try {
            const data = await (await fetch('/api/heatmap')).json();
            columns = data.columns;
            version = data.version;
            loaded = true;
            const pending = queued;
            queued = [];
            pending.forEach(merge);
        } catch (err) {
            console.error('Failed to load the heatmap:', err);
        } finally {
            loading = false;
        }
        schedule();
    }

    function merge(push) {
        if (push.version !== version) {  // the heatmap started over
            loaded = false;
            load();
            return;
        }
        push.columns.forEach((col) => {
            const last = columns[columns.length - 1];
            if (!last || col.t > last.t) columns.push(col);
            else if (col.t === last.t) columns[columns.length - 1] = col;
        });
        if (columns.length > MAX_COLUMNS) columns.splice(0, columns.length - MAX_COLUMNS);
        schedule();
    }

    document.addEventListener('lob:heatmap', (e) => {
        if (loading) queued.push(e.detail);
        else if (loaded) merge(e.detail);  // before the first load the pushes are not needed
    });
    document.addEventListener('lob:view', (e) => {
        if (e.detail !== 'analyze') return;
        if (!loaded) load();
        else schedule();
    });
    let connectedBefore = false;
    document.addEventListener('lob:connection', (e) => {
        if (e.detail !== 'live') return;
        if (connectedBefore && loaded) load();  // pushes were missed while the stream was down
        connectedBefore = true;
    });
    document.addEventListener('lob:theme', schedule);
    document.addEventListener('lob:text-size', schedule);
    new ResizeObserver(schedule).observe(canvas);

    function toggle(button, on) {
        button.setAttribute('aria-pressed', on ? 'true' : 'false');
        schedule();
        return on;
    }
    const tradesButton = document.getElementById('heatmap-trades');
    const hiddenButton = document.getElementById('heatmap-hidden');
    tradesButton.addEventListener('click', () => { showTrades = toggle(tradesButton, !showTrades); });
    hiddenButton.addEventListener('click', () => { showHidden = toggle(hiddenButton, !showHidden); });

    function schedule() {
        if (drawPending || !isShown()) return;
        drawPending = true;
        requestAnimationFrame(() => {
            drawPending = false;
            draw();
        });
    }

    // A theme color as [r, g, b] (themes are CSS variables of "R G B": app/static/css/themes.css).
    function triplet(name) {
        return getComputedStyle(document.documentElement).getPropertyValue(`--c-${name}`).trim().split(/\s+/).map(Number);
    }

    function rgb(name) {
        return triplet(name).join(', ');
    }

    // The color scale of one side, by depth relative to the median depth in view (t): up to twice
    // the median, the side's color grows from faint to full; beyond, it moves away from the
    // background (toward white on a dark theme, black on a light one), so a wall of ten or twenty
    // times the usual depth stands out.
    function shades(name) {
        const [r, g, b] = triplet(name);
        const [br, bg, bb] = triplet('darkBg');
        const light = 0.2126 * br + 0.7152 * bg + 0.0722 * bb > 128;
        const target = light ? 0 : 255;
        const reach = light ? 0.6 : 0.85;
        return Array.from({ length: SHADES }, (_, i) => {
            const t = Math.exp(Math.log(T_MIN) + (i / (SHADES - 1)) * (Math.log(T_MAX) - Math.log(T_MIN)));
            if (t <= 2) return `rgba(${r}, ${g}, ${b}, ${(0.12 + 0.88 * Math.pow(t / 2, 0.7)).toFixed(3)})`;
            const w = reach * Math.min(1, Math.log(t / 2) / Math.log(10));
            const mix = (c) => Math.round(c + (target - c) * w);
            return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
        });
    }

    function shadeIndex(t) {
        const i = Math.round((Math.log(t) - Math.log(T_MIN)) / (Math.log(T_MAX) - Math.log(T_MIN)) * (SHADES - 1));
        return Math.max(0, Math.min(SHADES - 1, i));
    }

    function draw() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.fillStyle = `rgb(${rgb('darkBg')})`;
        ctx.fillRect(0, 0, width, height);

        const rootPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
        const fontPx = Math.max(8, Math.round(10 * rootPx / 16));
        const axis = Math.round(3.2 * fontPx);  // room for the price labels on the right
        const plot = width - axis;
        const n = Math.min(columns.length, Math.floor(plot / MIN_COLUMN_PX));
        if (!n) {
            span.textContent = loaded ? 'No columns yet' : 'Loading…';
            geometry = null;
            return;
        }
        const shown = columns.slice(columns.length - n);
        const colW = Math.min(MAX_COLUMN_PX, plot / n);
        const x0 = plot - colW * n;  // newest at the right

        // The prices in view: where there are orders, trades or the mid, and a tick or two more.
        let lo = Infinity;
        let hi = -Infinity;
        const sizes = [];
        let maxTrade = 0;
        shown.forEach((c) => {
            c.depth.forEach((d, k) => {
                if (!d) return;
                sizes.push(Math.abs(d));
                lo = Math.min(lo, c.lo + k);
                hi = Math.max(hi, c.lo + k);
            });
            c.traded.forEach(([price, lots]) => {
                maxTrade = Math.max(maxTrade, lots);
                lo = Math.min(lo, price);
                hi = Math.max(hi, price);
            });
            if (c.mid != null) {
                lo = Math.min(lo, Math.floor(c.mid));
                hi = Math.max(hi, Math.ceil(c.mid));
            }
        });
        if (lo > hi) {  // an empty book
            lo = shown[shown.length - 1].lo;
            hi = lo + shown[shown.length - 1].depth.length - 1;
        }
        lo -= 2;
        hi += 2;
        const rowH = height / (hi - lo + 1);
        const top = (price) => height - (price - lo + 1) * rowH;  // the top edge of a price's row

        sizes.sort((a, b) => a - b);
        const median = sizes.length ? sizes[Math.floor(sizes.length / 2)] : 1;
        const bidShades = shades('emerald-500');
        const askShades = shades('rose-500');
        const cellW = Math.ceil(colW);
        const cellH = Math.max(1, Math.ceil(rowH));
        shown.forEach((c, i) => {
            const x = x0 + i * colW;
            c.depth.forEach((d, k) => {
                if (!d) return;
                ctx.fillStyle = (d > 0 ? bidShades : askShades)[shadeIndex(Math.abs(d) / median)];
                ctx.fillRect(x, top(c.lo + k), cellW, cellH);
            });
        });

        if (showHidden) {  // iceberg reserves, which the visible depth leaves out
            ctx.fillStyle = `rgba(${rgb('cyan-400')}, 0.45)`;
            ctx.strokeStyle = `rgb(${rgb('cyan-300')})`;
            ctx.lineWidth = 1;
            shown.forEach((c, i) => {
                c.hidden.forEach(([price]) => {
                    ctx.fillRect(x0 + i * colW, top(price), cellW, cellH);
                    ctx.strokeRect(x0 + i * colW + 0.5, top(price) + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
                });
            });
        }

        if (showTrades && maxTrade > 0) {
            ctx.fillStyle = `rgba(${rgb('amber-300')}, 0.75)`;
            const most = Math.max(1.5, Math.min(colW * 0.6, rowH * 0.6, 5));
            shown.forEach((c, i) => {
                const x = x0 + (i + 0.5) * colW;
                c.traded.forEach(([price, lots]) => {
                    ctx.beginPath();
                    ctx.arc(x, top(price) + rowH / 2, 1 + (most - 1) * Math.sqrt(lots / maxTrade), 0, 2 * Math.PI);
                    ctx.fill();
                });
            });
        }

        // The mid-price.
        ctx.strokeStyle = `rgb(${rgb('cyan-300')})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let drawing = false;
        shown.forEach((c, i) => {
            if (c.mid == null) {
                drawing = false;
                return;
            }
            const x = x0 + (i + 0.5) * colW;
            const y = top(c.mid) + rowH / 2;
            if (drawing) ctx.lineTo(x, y);
            else ctx.moveTo(x, y);
            drawing = true;
        });
        ctx.stroke();

        // Price labels on the right, far enough apart to read.
        ctx.fillStyle = `rgb(${rgb('slate-400')})`;
        ctx.font = `${fontPx}px 'JetBrains Mono', monospace`;
        ctx.textBaseline = 'middle';
        const step = [1, 2, 5, 10, 20, 50, 100].find((s) => s * rowH >= fontPx * 1.6) || 200;
        for (let price = Math.ceil(lo / step) * step; price <= hi; price += step) {
            ctx.fillText(String(price), plot + 4, top(price) + rowH / 2);
        }

        geometry = { shown, x0, colW, lo, rowH, height, plot };
        const first = shown[0].t;
        const last = shown[shown.length - 1].t;
        span.textContent = `Time ${first} to ${last}: ${n} column${n === 1 ? '' : 's'}`;
    }

    // What is under the pointer.
    canvas.addEventListener('mousemove', (e) => {
        const g = geometry;
        const box = canvas.getBoundingClientRect();
        const x = e.clientX - box.left;
        const y = e.clientY - box.top;
        const i = g ? Math.floor((x - g.x0) / g.colW) : -1;
        if (!g || x >= g.plot || i < 0 || i >= g.shown.length) {
            tip.hidden = true;
            return;
        }
        const c = g.shown[i];
        const price = g.lo + Math.floor((g.height - y) / g.rowH);
        const d = c.depth[price - c.lo] || 0;
        const find = (pairs) => (pairs.find(([p]) => p === price) || [price, 0])[1];
        const parts = [`Time ${c.t}, price ${price}: ${d > 0 ? `${d} lots bid` : d < 0 ? `${-d} lots offered` : 'no orders'}`];
        const hidden = find(c.hidden);
        const traded = find(c.traded);
        if (hidden) parts.push(`${hidden} hidden`);
        if (traded) parts.push(`${traded} traded`);
        tip.textContent = parts.join(' · ');
        tip.hidden = false;
        const left = x + 12 + tip.offsetWidth > box.width ? x - tip.offsetWidth - 12 : x + 12;
        tip.style.left = `${Math.max(0, left)}px`;
        tip.style.top = `${Math.max(0, y - 28)}px`;
    });
    canvas.addEventListener('mouseleave', () => { tip.hidden = true; });

    if (isShown()) load();  // the page opened on the Analyze view
})();
