// The order book tick by tick (components/ticks_ladder.html): a row for every price near the mid.
//
// In the levels view the rows are the best levels with orders, so they shift as levels empty and
// fill. Here they are fixed prices, updated where they are, so a price stays put while you aim at
// it, and the empty prices are rows too: one-click trading and dragging your orders can use any
// price (static/js/lob-ladder.js does both, in either view). The rows are built around an anchor
// price and built again around the mid by Center, or by themselves once the mid has moved far, but
// never while the pointer is on the ladder or a level is pressed. The data come from the stream's
// ticks events: the book 60 ticks either side of the mid (ticks_view in app/main.py); 40 show.
(function () {
    const wrapper = document.getElementById('ladder-wrapper');
    const ladder = document.getElementById('ticks-ladder');
    const rowsBox = document.getElementById('ticks-rows');
    const scroll = document.getElementById('ticks-scroll');
    if (!wrapper || !ladder || !rowsBox || !scroll) return;

    const HALF = 40;  // rows either side of the anchor
    const RECENTER_AT = 20;  // ticks the mid may move from the anchor before the rows follow it
    const CELL = 'relative h-full flex items-center gap-1 px-2 cursor-pointer hover:bg-slate-800/70';
    const BADGE = 'text-[0.5625rem] font-sans px-1 rounded border bg-cyan-500/15 text-cyan-300 border-cyan-500/50 hover:bg-cyan-500/30';
    const BOT_BADGE = 'text-[0.5625rem] font-sans px-1 rounded border bg-amber-500/15 text-amber-300 border-amber-500/50';

    let anchor = null;
    let data = null;
    let rows = new Map();  // price -> its row's parts

    const inTicks = () => wrapper.dataset.mode === 'ticks';
    const busy = () => ladder.matches(':hover') || window.ladderHeld;

    window.setLadderMode = function (mode) {
        wrapper.dataset.mode = mode === 'ticks' ? 'ticks' : 'levels';
        try { localStorage.setItem('lob-ladder-mode', wrapper.dataset.mode); } catch (e) { /* not saved */ }
        if (inTicks()) show();
        else if (window.centerDomAtMid) requestAnimationFrame(window.centerDomAtMid);
        document.dispatchEvent(new CustomEvent('lob:ladder-mode', { detail: wrapper.dataset.mode }));
    };

    // The price to build around: the mid, rounded (or the one side's best price).
    function middle() {
        if (data.mid != null) return Math.round(data.mid);
        return data.best_bid ?? data.best_ask ?? data.lo + Math.floor(data.bids.length / 2);
    }

    function cellParts(cell) {
        return { bar: cell.querySelector('[data-bar]'), depth: cell.querySelector('[data-depth]'),
                 own: cell.querySelector('[data-own-slot]'), ownKey: '', bot: cell.querySelector('[data-bot-slot]'), botKey: '' };
    }

    function build(at) {
        anchor = at;
        rows = new Map();
        const rowsHtml = [];
        for (let price = anchor + HALF; price >= Math.max(1, anchor - HALF); price--) {
            rowsHtml.push(`<div class="grid grid-cols-[1fr_4rem_1fr] h-[1.3125rem] items-stretch" data-price="${price}">
                <div class="${CELL} justify-end" data-side="bid" data-price="${price}" title="${price}: click to use it as the limit price (with 1-click on, to buy there)">
                    <span class="absolute right-0 inset-y-0.5 rounded-l bg-emerald-500/20 pointer-events-none" data-bar></span>
                    <span class="z-10" data-bot-slot></span><span class="z-10" data-own-slot></span><span class="z-10 text-slate-200" data-depth></span>
                </div>
                <div class="flex items-center justify-center font-bold text-slate-500" data-price-cell>${price}</div>
                <div class="${CELL} justify-start" data-side="ask" data-price="${price}" title="${price}: click to use it as the limit price (with 1-click on, to sell there)">
                    <span class="absolute left-0 inset-y-0.5 rounded-r bg-rose-500/20 pointer-events-none" data-bar></span>
                    <span class="z-10 text-slate-200" data-depth></span><span class="z-10" data-own-slot></span><span class="z-10" data-bot-slot></span>
                </div>
            </div>`);
        }
        rowsBox.innerHTML = rowsHtml.join('');
        [...rowsBox.children].forEach((row) => {
            const [bid, priceCell, ask] = row.children;
            rows.set(Number(row.dataset.price), { row, priceCell, tone: '', bid: cellParts(bid), ask: cellParts(ask) });
        });
    }

    function fillCell(part, depth, max, own, side, price, bot) {
        const text = depth == null ? '·' : depth ? depth.toLocaleString('en-US') : '';
        if (part.depth.textContent !== text) part.depth.textContent = text;
        part.bar.style.width = depth ? `${Math.min(100, (100 * depth) / max)}%` : '0';
        const botKey = bot ? String(bot) : '';
        if (part.botKey !== botKey) {
            part.botKey = botKey;
            part.bot.innerHTML = bot ? `<span class="${BOT_BADGE}" title="Your market maker's quote here: ${bot} lots">bot ${bot}</span>` : '';
        }
        const key = own ? `${own[0]}:${own[1]}` : '';
        if (part.ownKey === key) return;
        part.ownKey = key;
        part.own.innerHTML = own
            ? `<span data-own data-lots="${own[0] + own[1]}" class="${BADGE}" title="Your resting ${side === 'bid' ? 'buy' : 'sell'} orders at ${price}: ${own[0]} lots showing${own[1] ? `, ${own[1]} hidden in icebergs` : ''}. Drag to another price to move them there.">you ${own[0]}${own[1] ? ` +${own[1]}` : ''}</span>`
            : '';
    }

    function fill() {
        if (!data || anchor === null) return;
        const { lo, bids, asks, best_bid: bestBid, best_ask: bestAsk } = data;
        let max = 1;
        rows.forEach((_, price) => {
            const k = price - lo;
            if (k >= 0 && k < bids.length) max = Math.max(max, bids[k], asks[k]);
        });
        const own = new Map(data.own.map(([side, price, size, hidden]) => [`${side}:${price}`, [size, hidden]]));
        const bots = new Map((data.bot || []).map(([side, price, size]) => [`${side}:${price}`, size]));
        rows.forEach((r, price) => {
            const k = price - lo;
            const known = k >= 0 && k < bids.length;  // outside the pushed window the depth is not known
            fillCell(r.bid, known ? bids[k] : null, max, own.get(`bid:${price}`), 'bid', price, bots.get(`bid:${price}`));
            fillCell(r.ask, known ? asks[k] : null, max, own.get(`ask:${price}`), 'ask', price, bots.get(`ask:${price}`));
            // Bid prices green, ask prices red, the spread between them gray; the best quotes stand out.
            const tone = bestBid != null && price <= bestBid ? 'text-emerald-400'
                : bestAsk != null && price >= bestAsk ? 'text-rose-400' : 'text-slate-500';
            if (r.tone !== tone) {
                r.priceCell.classList.remove('text-emerald-400', 'text-rose-400', 'text-slate-500');
                r.priceCell.classList.add(tone);
                r.tone = tone;
            }
            r.row.classList.toggle('bg-slate-800/60', price === bestBid || price === bestAsk);
            r.row.classList.toggle('bg-slate-900/80', bestBid != null && bestAsk != null && price > bestBid && price < bestAsk);
        });
    }

    // Scroll so that the anchor (the mid when the rows were built) is in the middle.
    function centerView() {
        const r = rows.get(anchor);
        if (!r) return;
        scroll.scrollTop = r.row.offsetTop - rowsBox.offsetTop - (scroll.clientHeight - r.row.offsetHeight) / 2;
    }

    function rebuild() {
        build(middle());
        fill();
        centerView();
        document.dispatchEvent(new CustomEvent('lob:ticks-filled'));
    }

    function show() {
        if (!data) return;
        if (anchor === null || Math.abs(middle() - anchor) > RECENTER_AT) rebuild();
        else {
            fill();
            centerView();
        }
    }

    document.addEventListener('lob:ticks', (e) => {
        data = e.detail;
        if (!inTicks()) return;
        if (anchor === null || (Math.abs(middle() - anchor) > RECENTER_AT && !busy())) rebuild();
        else fill();
    });
    document.getElementById('ticks-center').addEventListener('click', () => { if (data) rebuild(); });
    document.addEventListener('lob:text-size', () => { if (inTicks()) centerView(); });
    document.addEventListener('lob:view', (e) => { if (e.detail === 'trade' && inTicks()) requestAnimationFrame(centerView); });

    window.ticksLadder = { rows: () => rows, anchor: () => anchor };  // for the browser tests
})();
