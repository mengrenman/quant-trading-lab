// Trading from the order book (components/ladder.html).
//
// A click on a price level copies its price into the limit order form or, with one-click trading
// on, places a limit order there for the size in the form: a buy on a bid level, a sell on an ask
// level. Dragging the "you" badge of your orders onto another level moves them there
// (POST /api/order/move: they are canceled and entered again, so they go to the back of the queue).
//
// The ladder is pushed up to five times a second and lists only levels with orders, so its rows
// shift as levels empty and fill. The handlers therefore sit on its wrapper, which stays, and read
// the level from the row under the pointer, and the book holds still (window.ladderHeld, see
// swapPanel in index.html) while a level is pressed or dragged, and while you aim at the levels
// with one-click on, refreshing every two seconds: the level you click is the level you see.
(function () {
    const wrapper = document.getElementById('ladder-wrapper');
    const hint = document.getElementById('ladder-hint');
    if (!wrapper || !hint) return;

    const DRAG_START = 4;     // pixels the pointer moves before a press on your badge becomes a drag
    const CLICK_SLOP = 8;     // a press that wanders further than this is no click
    const MAX_PRESS_MS = 15000;  // a press holds the book's pushes for at most this long
    const AIM_REFRESH_MS = 2000;  // while aiming with one-click on, the book refreshes this often
    const AFTER_ACTION_MS = 700;  // after an order or a move the book runs freely a moment, to show it

    let armed = false;
    let press = null;  // {price, side, own, lots, x, y, dragging}
    let dropRow = null;
    let pressTimer = null;
    let aimTimer = null;
    let aiming = false;
    let pointer = null;  // the last position over the wrapper, to redo the hint after a refresh
    let aimPausedUntil = 0;

    function syncToggle() {
        wrapper.dataset.armed = armed ? 'true' : 'false';
        const toggle = document.getElementById('one-click-toggle');
        if (!toggle) return;
        toggle.setAttribute('aria-pressed', armed ? 'true' : 'false');
        const state = toggle.querySelector('[data-one-click-state]');
        if (state) state.textContent = armed ? 'on' : 'off';
    }

    window.toggleOneClick = function () {
        armed = !armed;
        syncToggle();
        hideHint();
        if (!armed) aim(false);
    };
    document.addEventListener('lob:panel', (e) => { if (e.detail === 'ladder') syncToggle(); });

    function limitSize() {
        const input = document.getElementById('limit-order-size');
        const size = input ? Number(input.value) : NaN;
        return Number.isInteger(size) && size >= 1 ? size : null;
    }

    // The level under a point: its row, price and side (null off the levels).
    function levelAt(x, y) {
        const el = document.elementFromPoint(x, y);
        const row = el && el.closest('#ladder-container [data-price]');
        return row ? { row, price: Number(row.dataset.price), side: row.dataset.side } : null;
    }

    function showHint(text, x, y) {
        const box = wrapper.getBoundingClientRect();
        hint.textContent = text;
        hint.hidden = false;
        let left = x - box.left + 14;
        if (left + hint.offsetWidth > box.width) left = x - box.left - hint.offsetWidth - 14;
        hint.style.left = `${Math.max(0, left)}px`;
        hint.style.top = `${Math.max(0, y - box.top - 30)}px`;
    }

    function hideHint() {
        hint.hidden = true;
    }

    function markDrop(row) {
        if (dropRow === row) return;
        if (dropRow) dropRow.classList.remove('drop-target');
        dropRow = row;
        if (dropRow) dropRow.classList.add('drop-target');
    }

    // The book is held while either a press or aiming wants it; the last push waits meanwhile.
    function updateHold() {
        const held = press !== null || aiming;
        if (window.ladderHeld && !held && window.releaseLadder) {
            window.ladderHeld = false;
            window.releaseLadder();
        }
        window.ladderHeld = held;
    }

    function aim(on) {
        if (aiming === on) return;
        aiming = on;
        clearInterval(aimTimer);
        if (on) {
            aimTimer = setInterval(() => {  // show the latest book, then hold it again
                if (press || !window.releaseLadder) return;
                window.releaseLadder();
                if (pointer) hover(pointer.x, pointer.y);
            }, AIM_REFRESH_MS);
        }
        updateHold();
    }

    function finish() {
        press = null;
        clearTimeout(pressTimer);
        markDrop(null);
        wrapper.classList.remove('dragging');
        hideHint();
        updateHold();
    }

    function notify(text) {
        if (window.showToast) window.showToast(text);
    }

    // Let the pushes that show an order or a move through before holding the book again.
    function afterAction() {
        aimPausedUntil = Date.now() + AFTER_ACTION_MS;
        aim(false);
        setTimeout(() => { if (pointer && !press) hover(pointer.x, pointer.y); }, AFTER_ACTION_MS);
    }

    // Whether a buy (or sell) at this price would trade on arrival: at or across the best opposite quote.
    function crosses(side, price) {
        const quotes = window.bookQuotes ? window.bookQuotes() : { bid: null, ask: null };
        return side === 'bid' ? quotes.ask !== null && price >= quotes.ask : quotes.bid !== null && price <= quotes.bid;
    }

    async function placeOrder(side, price) {
        const size = limitSize();
        if (size === null) {
            notify('No order sent: enter a size of at least 1 lot in the limit order form.');
            return;
        }
        try {
            const res = await window.postForm('/api/order/limit', { price, size, side });
            if (!res.ok) throw new Error(`the server answered with status ${res.status}`);
            notify(window.htmlText(await res.text()));
        } catch (err) {
            notify(`One-click order failed: ${err.message}.`);
        }
        afterAction();
    }

    async function moveOrders(side, fromPrice, toPrice) {
        try {
            const res = await window.postForm('/api/order/move', { side, from_price: fromPrice, to_price: toPrice });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || `the server answered with status ${res.status}`);
            notify(data.text);
        } catch (err) {
            notify(`Move failed: ${err.message}.`);
        }
        afterAction();
    }

    // The pointer over the book, not pressed: with one-click on, hold the book and say what a click does.
    function hover(x, y) {
        const level = armed ? levelAt(x, y) : null;
        aim(level !== null && Date.now() >= aimPausedUntil);
        if (!level) {
            hideHint();
            return;
        }
        const size = limitSize();
        if (size === null) showHint('Enter a size in the limit order form', x, y);
        else showHint(`${level.side === 'bid' ? 'Buy' : 'Sell'} ${size} at ${level.price}`
                      + (crosses(level.side, level.price) ? ': crosses the spread and trades now' : ''), x, y);
    }

    wrapper.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        const level = levelAt(e.clientX, e.clientY);
        if (!level) return;
        const badge = e.target.closest('[data-own]');
        press = { price: level.price, side: level.side, own: !!badge, lots: badge ? Number(badge.dataset.lots) : 0,
                  x: e.clientX, y: e.clientY, dragging: false };
        wrapper.setPointerCapture(e.pointerId);
        clearTimeout(pressTimer);
        pressTimer = setTimeout(finish, MAX_PRESS_MS);
        updateHold();
        if (badge) e.preventDefault();  // no text selection or native drag
    });

    wrapper.addEventListener('pointermove', (e) => {
        pointer = { x: e.clientX, y: e.clientY };
        if (!press) {
            hover(e.clientX, e.clientY);
            return;
        }
        const moved = Math.hypot(e.clientX - press.x, e.clientY - press.y);
        if (press.own && !press.dragging && moved > DRAG_START) {
            press.dragging = true;
            wrapper.classList.add('dragging');
        }
        if (!press.dragging) return;
        const level = levelAt(e.clientX, e.clientY);
        const target = level && level.price !== press.price ? level : null;
        markDrop(target ? target.row : null);
        const kind = press.side === 'bid' ? 'buy' : 'sell';
        const trades = target && crosses(press.side, target.price) ? ': crosses the spread and trades now' : '';
        showHint(target ? `Move your ${kind} orders (${press.lots} lots) to ${target.price}${trades}`
                        : 'Drop on another level to move your orders there', e.clientX, e.clientY);
    });

    wrapper.addEventListener('pointerup', (e) => {
        if (!press) return;
        const p = press;
        const level = levelAt(e.clientX, e.clientY);
        finish();
        if (p.dragging) {
            if (level && level.price !== p.price) moveOrders(p.side, p.price, level.price);
        } else if (Math.hypot(e.clientX - p.x, e.clientY - p.y) <= CLICK_SLOP) {
            if (armed) placeOrder(p.side, p.price);
            else if (window.setOrderPrice) window.setOrderPrice(p.price);
        }
    });

    wrapper.addEventListener('pointercancel', finish);
    wrapper.addEventListener('pointerleave', () => {
        pointer = null;
        if (press) return;
        hideHint();
        aim(false);
    });

    syncToggle();
})();
