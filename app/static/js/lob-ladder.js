// Trading from the order book, in both of its views: by levels (components/ladder.html) and tick by
// tick (components/ticks_ladder.html, static/js/lob-ticks.js).
//
// A click on a price copies it into the limit order form or, with one-click trading on, places a
// limit order there for the size in the form: a buy on a bid level or in the bid column, a sell on
// an ask level or in the ask column (the price column of the tick view only copies). Dragging the
// "you" badge of your orders onto another price moves them there (POST /api/order/move: they are
// canceled and entered again, so they go to the back of the queue).
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

    function syncToggle() {  // one toggle in each view's header
        wrapper.dataset.armed = armed ? 'true' : 'false';
        wrapper.querySelectorAll('.one-click-toggle').forEach((toggle) => {
            toggle.setAttribute('aria-pressed', armed ? 'true' : 'false');
            const state = toggle.querySelector('[data-one-click-state]');
            if (state) state.textContent = armed ? 'on' : 'off';
        });
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

    // The price under a point: the element that carries it (a levels row, or a cell of the tick view),
    // the price, and the side if it has one (null off the prices).
    function levelAt(x, y) {
        const el = document.elementFromPoint(x, y);
        const row = el && el.closest('#ladder-wrapper [data-price]');
        return row ? { row, price: Number(row.dataset.price), side: row.dataset.side || null } : null;
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
        if (!level.side) showHint(`Click the bid or the ask column to trade at ${level.price}`, x, y);
        else if (size === null) showHint('Enter a size in the limit order form', x, y);
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
            if (armed && p.side) placeOrder(p.side, p.price);
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

    // ---- Without a mouse ----
    // Tab reaches the book (its wrapper, which stays while the levels view is replaced). The arrow
    // keys move a cursor over its prices: up and down, and in the tick view left and right between
    // bids and asks. Enter does what a click would there. M picks up your orders at the cursor; the
    // arrows then choose a price, Enter puts them there, and Escape (or M again) leaves them where
    // they were and takes the cursor back to them. The cursor follows its price when the levels view is replaced, and screen readers
    // follow it (aria-activedescendant) and hear the hint, a live region.
    let cursor = null;  // {price, side}
    let carrying = null;  // {price, side, lots, from} while moving orders from the keyboard
    const KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter', 'Escape', 'm', 'M'];
    const inTicks = () => wrapper.dataset.mode === 'ticks';
    const priceOf = (el) => Number(el.dataset.price);

    // What the cursor can stand on, top to bottom: the levels view's rows, or one column of the tick view.
    function targets(side) {
        return inTicks() ? [...wrapper.querySelectorAll(`#ticks-rows [data-side="${side}"]`)]
            : [...wrapper.querySelectorAll('#ladder-container [data-price][data-side]')];
    }

    function targetOf(c) {
        const scope = inTicks() ? '#ticks-rows' : '#ladder-container';
        return c ? wrapper.querySelector(`${scope} [data-side="${c.side}"][data-price="${c.price}"]`) : null;
    }

    // The nearest target to a price (on the cursor's side in the levels view, where rows come and go).
    function nearest(c) {
        const list = targets(c.side).filter((el) => inTicks() || el.dataset.side === c.side);
        const pool = list.length ? list : targets(c.side);
        return pool.reduce((best, el) => (!best || Math.abs(priceOf(el) - c.price) < Math.abs(priceOf(best) - c.price)
            ? el : best), null);
    }

    function startCursor() {  // at the best bid
        const bid = window.bookQuotes ? window.bookQuotes().bid : null;
        const el = nearest({ price: bid ?? 0, side: 'bid' });
        return el ? { price: priceOf(el), side: el.dataset.side } : null;
    }

    function step(dir) {  // -1 up, +1 down
        const here = targetOf(cursor) || nearest(cursor);
        const list = targets(cursor.side);
        const next = list[Math.max(0, Math.min(list.length - 1, list.indexOf(here) + dir))];
        if (next) cursor = { price: priceOf(next), side: next.dataset.side };
    }

    function ownLots(el) {
        const badge = el && el.querySelector('[data-own]');
        return badge ? Number(badge.dataset.lots) : 0;
    }

    function describe(el) {
        const price = priceOf(el);
        const kind = (side) => (side === 'bid' ? 'buy' : 'sell');
        if (carrying) {
            if (price === carrying.price) {
                return `Your ${kind(carrying.side)} orders (${carrying.lots} lots) at ${price}: arrows to choose a price, Enter to put them there, Escape to keep them`;
            }
            const trades = crosses(carrying.side, price) ? ': crosses the spread and trades now' : '';
            return `Move your ${kind(carrying.side)} orders (${carrying.lots} lots) to ${price}${trades}: Enter, or Escape to keep them`;
        }
        const depthCell = el.querySelector('[data-depth]');
        const depth = el.dataset.depth ?? (depthCell ? depthCell.textContent : '');
        const side = el.dataset.side;
        let text = `${price}: ${depth || 'no'} lots ${side === 'bid' ? 'bid' : 'offered'}. Enter: `;
        const size = limitSize();
        if (armed && size !== null) text += `${kind(side)} ${size} at ${price}${crosses(side, price) ? ', which trades now' : ''}`;
        else text += 'copy the price into the order form';
        if (ownLots(el)) text += '; M: move your orders';
        return text;
    }

    // Show the cursor; ``scroll`` brings it into view (after a key, not after every push).
    function render(scroll = false) {
        wrapper.querySelectorAll('.kbd-cursor').forEach((el) => el.classList.remove('kbd-cursor'));
        if (document.activeElement !== wrapper || !cursor) return;
        let el = targetOf(cursor);
        if (!el) {  // its level left the book: the nearest one on its side
            el = nearest(cursor);
            if (!el) return;
            cursor = { price: priceOf(el), side: el.dataset.side };
        }
        el.classList.add('kbd-cursor');
        if (!el.id) el.id = `book-${inTicks() ? 'tick' : 'level'}-${cursor.side}-${cursor.price}`;
        wrapper.setAttribute('aria-activedescendant', el.id);
        if (scroll) el.scrollIntoView({ block: 'nearest' });
        if (carrying) markDrop(cursor.price !== carrying.price ? el : null);
        const box = el.getBoundingClientRect();
        showHint(describe(el), box.left + box.width * 0.55, box.bottom + 26);
    }

    wrapper.addEventListener('keydown', (e) => {
        if (e.target !== wrapper || !KEYS.includes(e.key) || e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();  // keeps the page's own shortcuts (the right arrow steps the simulation) out of it
        if (!cursor) cursor = startCursor();
        if (!cursor) return;
        if (e.key === 'ArrowUp') step(-1);
        else if (e.key === 'ArrowDown') step(1);
        else if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && inTicks()) {
            cursor = { price: cursor.price, side: e.key === 'ArrowLeft' ? 'bid' : 'ask' };
        } else if (e.key === 'Escape' || ((e.key === 'm' || e.key === 'M') && carrying)) {
            if (carrying) cursor = carrying.from;  // back to the orders
            carrying = null;
            markDrop(null);
        } else if (e.key === 'm' || e.key === 'M') {
            const el = targetOf(cursor);
            const lots = ownLots(el);
            const badge = el && el.querySelector('[data-own]');
            const side = badge ? (badge.closest('[data-side]') || el).dataset.side : null;
            if (lots && side) carrying = { price: cursor.price, side, lots, from: { ...cursor } };
        } else if (e.key === 'Enter') {
            if (carrying) {
                const from = carrying;
                carrying = null;
                markDrop(null);
                if (cursor.price !== from.price) moveOrders(from.side, from.price, cursor.price);
            } else if (armed && cursor.side) {
                placeOrder(cursor.side, cursor.price);
            } else if (window.setOrderPrice) {
                window.setOrderPrice(cursor.price);
            }
        }
        render(true);
    });
    wrapper.addEventListener('focus', () => {
        if (!cursor || !targetOf(cursor)) cursor = startCursor();
        render(true);
    });
    wrapper.addEventListener('blur', () => {
        carrying = null;
        markDrop(null);
        wrapper.querySelectorAll('.kbd-cursor').forEach((el) => el.classList.remove('kbd-cursor'));
        wrapper.removeAttribute('aria-activedescendant');
        hideHint();
    });
    document.addEventListener('lob:panel', (e) => { if (e.detail === 'ladder') render(); });
    document.addEventListener('lob:ticks-filled', () => render());
    document.addEventListener('lob:ladder-mode', () => {
        cursor = null;
        carrying = null;
        if (document.activeElement === wrapper) {
            cursor = startCursor();
            render();
        }
    });
    window.ladderCursor = () => cursor;  // for the browser tests

    syncToggle();
})();
