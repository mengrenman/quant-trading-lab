// Guided lessons (components/lessons.html lists them; the lesson dock in index.html runs one).
//
// Each lesson is a few steps. A step explains something (its text may read the book, so numbers are
// current), can do it for you ("Do it": the same requests the page's own controls send), can point
// at the part of the page it is about ("Show me": switches view and outlines it), and may check that
// it was done. The dock stays in the corner across views; the lesson, the step and what the steps
// noted are kept in this browser, so a reload resumes.
(function () {
    const dock = document.getElementById('lesson-dock');
    const catalog = document.getElementById('lesson-catalog');
    const spotlight = document.getElementById('lesson-spotlight');
    if (!dock || !catalog || !spotlight) return;

    const $ = (id) => document.getElementById(id);

    // ---- What the steps read and do ----

    const quotes = () => (window.bookQuotes ? window.bookQuotes() : { bid: null, ask: null, mid: null });

    // The order book's levels on one side, best first: price, depth and the running total from the best.
    function levels(side) {
        const rows = [...document.querySelectorAll(`#dom-${side}s-section [data-price]`)].map((row) => ({
            price: Number(row.dataset.price), depth: Number(row.dataset.depth), cum: Number(row.dataset.cum),
        }));
        return side === 'ask' ? rows.reverse() : rows;  // the asks are listed with the best at the bottom
    }

    function ownLevels() {
        try {
            return JSON.parse($('active-orders-container').dataset.levels || '[]');
        } catch (e) {
            return [];
        }
    }

    const textOf = (html) => {
        const template = document.createElement('template');
        template.innerHTML = html;
        return template.content.textContent.replace(/\s+/g, ' ').trim();
    };

    async function post(url, fields = {}) {
        const body = new FormData();
        Object.entries(fields).forEach(([key, value]) => body.append(key, value));
        const res = await fetch(url, { method: 'POST', body });
        if (!res.ok) {
            const detail = await res.json().then((d) => d.detail).catch(() => null);
            throw new Error(typeof detail === 'string' ? detail : `the server answered with status ${res.status}`);
        }
        return res;
    }

    // An order through the order endpoints; its result also shows under the order form.
    async function order(url, fields) {
        const html = await (await post(url, fields)).text();
        const result = $('order-result');
        if (result) result.innerHTML = html;
        return textOf(html);
    }

    async function play(speed) {
        if (speed && window.updateSimSpeed) window.updateSimSpeed(speed);
        if (!window.simIsRunning) window.startSim();
        return `The simulation is running${speed ? ` at ${speed} events a second` : ''}.`;
    }

    async function pause() {
        if (window.simIsRunning) window.pauseSim();
        return 'Paused.';
    }

    async function cancelSide(side) {
        const data = await (await post('/api/order/cancel_all', { side })).json();
        return `Canceled ${data.canceled} order${data.canceled === 1 ? '' : 's'}.`;
    }

    const paused = () => !window.simIsRunning;
    const resultSays = (pattern) => () => pattern.test(($('order-result') || {}).textContent || '');
    const PAUSE_STEP = (text) => ({ text, do: ['Pause', pause], show: ['trade', 'sim-play-btn'], check: paused });

    // ---- The lessons ----

    const LESSONS = [
        {
            id: 'impact',
            title: 'Price impact: walking the book',
            summary: 'Send a market order larger than the best level, and watch it take the next levels and move the price.',
            steps: [
                PAUSE_STEP('A market order trades with the best price on the other side first. If it is larger than the depth there, it goes on to the next price, and the next: it <b>walks the book</b>. Pause the simulation so the book holds still while you look.'),
                {
                    text: 'The asks in the order book are the sell orders waiting, the best (lowest) at the bottom, next to the mid. Each row is a price with the lots resting there; Accumulated adds them up from the best ask outward.',
                    show: ['trade', 'ladder-wrapper'],
                },
                {
                    text: () => {
                        const asks = levels('ask');
                        const n = Math.min(3, asks.length);
                        if (!n) return 'There are no asks to buy from: run the simulation for a moment first.';
                        return `Buy everything on the ${n} best ask levels: ${asks[n - 1].cum} lots, at prices from ${asks[0].price} to ${asks[n - 1].price}.`;
                    },
                    do: ['Send the market buy', async (ctx) => {
                        const asks = levels('ask');
                        const n = Math.min(3, asks.length);
                        if (!n) throw new Error('there are no asks in the book');
                        ctx.bestAsk = asks[0].price;
                        return order('/api/order/market', {
                            size: asks[n - 1].cum, side: 'bid', protection: Math.max(10, asks[n - 1].price - asks[0].price),
                        });
                    }],
                    show: ['trade', 'order-form-container'],
                    check: resultSays(/Bought/),
                },
                {
                    text: (ctx) => `The order paid ${ctx.bestAsk ?? 'the best ask'} for the first lots and more for the rest, so its average is above the best ask it started from. The levels it emptied are gone: the best ask, and the mid with it, moved up. That move is the order's <b>price impact</b>. The arrow on the chart marks your fill.`,
                    show: ['trade', 'metrics-wrapper'],
                },
                {
                    text: 'Now start the simulation and watch the asks: new sell orders arrive and the spread closes again. How quickly a book refills after a large order is its <b>resilience</b>.',
                    do: ['Start the simulation', () => play(100)],
                    show: ['trade', 'ladder-wrapper'],
                },
                {
                    text: 'Impact grows with the size of an order, but in real markets less than in proportion to it: it is concave. The Analyze view measures this for the current model (Concave price impact, under Stylized facts). Try a larger order, with a protection limit in the market order form so that it stops walking.',
                    show: ['analyze', 'facts-wrapper'],
                },
            ],
        },
        {
            id: 'walls',
            title: 'Walls: large resting orders',
            summary: 'Place a buy order far larger than the levels around it, and see how it holds the price up.',
            steps: [
                PAUSE_STEP('A <b>wall</b> is a resting order far larger than the levels around it. Pause the simulation first.'),
                {
                    text: () => {
                        const q = quotes();
                        if (q.bid === null) return 'There are no bids: run the simulation for a moment first.';
                        return `Place a wall: a buy of 300 lots at ${q.bid - 3}, three ticks below the best bid.`;
                    },
                    do: ['Place the wall', async (ctx) => {
                        const q = quotes();
                        if (q.bid === null) throw new Error('there are no bids in the book');
                        ctx.wall = q.bid - 3;
                        return order('/api/order/limit', { price: ctx.wall, size: 300, side: 'bid' });
                    }],
                    show: ['trade', 'ladder-wrapper'],
                    check: () => ownLevels().some((l) => l.side === 'bid' && l.size >= 300),
                },
                {
                    text: 'In the depth chart under the order book, the wall is a cliff in the bid curve, and the triangle on its price axis marks your order. On the price chart, the dashed line is your wall.',
                    show: ['trade', 'depth-chart'],
                },
                {
                    text: 'Start the simulation. While the wall rests, the best bid cannot fall below it: sellers have to trade all 300 lots before the price can go lower, so it tends to bounce off the wall.',
                    do: ['Start the simulation', () => play(200)],
                    show: ['trade', 'chart-panel'],
                },
                {
                    text: 'In the Analyze view, the order-book heatmap shows the wall as a bright band, with the mid-price line running above it.',
                    show: ['analyze', 'heatmap-container'],
                },
                {
                    text: 'A wall can be pulled at any moment, and some are placed only to be canceled (spoofing, which exchanges forbid), so traders read them with care. Cancel yours.',
                    do: ['Cancel your bids', () => cancelSide('bid')],
                    show: ['trade', 'active-orders-wrapper'],
                    check: () => !ownLevels().some((l) => l.side === 'bid'),
                },
            ],
        },
        {
            id: 'icebergs',
            title: 'Hidden liquidity: icebergs',
            summary: 'An iceberg shows only part of its size. Rest one at the best ask and watch it trade more than the book shows.',
            steps: [
                PAUSE_STEP('An <b>iceberg</b> order shows only part of its size, its peak. When the peak has traded, the next one comes out of the hidden reserve, at the back of the queue at its price. Pause the simulation first.'),
                {
                    text: () => {
                        const q = quotes();
                        if (q.ask === null) return 'There are no asks: run the simulation for a moment first.';
                        return `Place an iceberg: a sell of 200 lots at the best ask, ${q.ask}, showing 10 at a time. The book shows 10 more lots there; your badge says how many are hidden.`;
                    },
                    do: ['Place the iceberg', async (ctx) => {
                        const q = quotes();
                        if (q.ask === null) throw new Error('there are no asks in the book');
                        ctx.iceberg = q.ask;
                        return order('/api/order/limit', { price: q.ask, size: 200, side: 'ask', display: 10 });
                    }],
                    show: ['trade', 'ladder-wrapper'],
                    check: () => ownLevels().some((l) => l.side === 'ask' && l.hidden > 0),
                },
                {
                    text: 'Start the simulation slowly and watch the best ask. Each time buyers trade with your iceberg, a notice says how many lots it has left, while the book never shows more than its peak of 10.',
                    do: ['Start the simulation', () => play(20)],
                    show: ['trade', 'ladder-wrapper'],
                    check: () => /Your sell order #\d+ filled/.test(($('active-orders-container') || {}).textContent || ''),
                },
                {
                    text: 'In the Analyze view, the heatmap shows the iceberg as a price that keeps trading while its visible depth stays small. Reveal hidden outlines the reserve, which the other traders cannot see.',
                    do: ['Reveal hidden', async () => {
                        const button = $('heatmap-hidden');
                        if (button && button.getAttribute('aria-pressed') !== 'true') button.click();
                        if (window.showView) window.showView('analyze');
                        return 'Hidden reserves are outlined on the heatmap.';
                    }],
                    show: ['analyze', 'heatmap-container'],
                },
                {
                    text: 'Exchanges let traders hide size like this so that a large order does not scare the price away. Others look for icebergs in trades that add up to more than the book showed. Cancel your iceberg.',
                    do: ['Cancel your asks', () => cancelSide('ask')],
                    show: ['trade', 'active-orders-wrapper'],
                    check: () => !ownLevels().some((l) => l.side === 'ask'),
                },
            ],
        },
        {
            id: 'ofi',
            title: 'Order flow imbalance',
            summary: 'Measure how the orders arriving at the best quotes move the price (Cont, Kukanov and Stoikov, 2014).',
            steps: [
                {
                    text: '<b>Order flow imbalance</b> (OFI) adds up, over an interval, the lots added to the best bid or taken from the best ask, less the lots taken from the bid or added to the ask. Buying pressure makes it positive, selling pressure negative.',
                    show: ['analyze', 'ofi-wrapper'],
                },
                {
                    text: 'Measuring it takes a sample. Collect one: 400,000 events, in a few seconds.',
                    do: ['Collect', async () => {
                        await post('/api/sim/collect', { n: 400000 });
                        return 'Collecting…';
                    }],
                    show: ['analyze', 'facts-wrapper'],
                    check: () => document.querySelectorAll('#ofi-container circle').length > 50,
                },
                {
                    text: 'Each point is one interval of 100 events: its OFI against the change in the mid-price. The slope of the line is how far the mid moves per lot of imbalance, and R² how much of the price changes OFI explains.',
                    show: ['analyze', 'ofi-wrapper'],
                },
                {
                    text: 'Cont et al. found the slope smaller where the book is deeper: the same imbalance moves a deep book less. Try another model: switch to Mike and Farmer (2008), collect again, and compare the slope and the depth at the best quotes.',
                    do: ['Switch and collect', async () => {
                        await post('/api/sim/config', { model: 'mike_farmer_2008' });
                        await post('/api/sim/collect', { n: 400000 });
                        return 'Collecting with Mike and Farmer (2008)…';
                    }],
                    show: ['analyze', 'ofi-wrapper'],
                },
                {
                    text: 'Unlike the depth imbalance in the metrics strip, which one large resting order can dominate, OFI counts only what changes at the best quotes. Notebook 07 compares it across models and depths.',
                    show: ['trade', 'metrics-wrapper'],
                },
            ],
        },
        {
            id: 'split',
            title: 'Splitting a large order',
            summary: 'Buy 300 lots as 10 orders over time, and compare with one market order for all of it.',
            steps: [
                {
                    text: 'A large market order walks deep into the book and pays more at each level it reaches. Splitting it into smaller orders spread over time lets the book refill in between: a <b>TWAP</b> schedule (time-weighted average price).',
                },
                {
                    text: 'Send a buy of 300 lots as 10 market orders, one every 5 units of simulation time. The first goes out now.',
                    do: ['Send the split order', () => order('/api/order/market', {
                        size: 300, side: 'bid', slices: 10, interval: 5, protection: 10,
                    })],
                    show: ['trade', 'active-orders-wrapper'],
                    check: () => /Buying 300 lots|Split buy/.test(($('active-orders-container') || {}).textContent || ''),
                },
                {
                    text: 'The other slices go out as simulation time passes, so start the simulation. The panel under My Active Orders shows how far it has got.',
                    do: ['Start the simulation', () => play(200)],
                    show: ['trade', 'active-orders-wrapper'],
                    check: () => /Split buy (done|canceled)/.test(($('active-orders-container') || {}).textContent || ''),
                },
                {
                    text: 'When the last slice is out, the summary compares the split\'s average price with what one market order for all 300 lots would have paid at the start, and with the market\'s VWAP over the same time.',
                    show: ['trade', 'active-orders-wrapper'],
                },
                {
                    text: 'Splitting usually pays less than sweeping the book at once, but while you wait the price can move against you, and others may notice the pattern. Execution algorithms weigh these costs against each other.',
                },
            ],
        },
    ];
    window.LOB_LESSONS = LESSONS;  // for the browser tests

    // ---- Where you are ----

    const STATE_KEY = 'lob-lesson';
    const DONE_KEY = 'lob-lessons-done';
    const read = (key, fallback) => {
        try {
            return JSON.parse(localStorage.getItem(key)) ?? fallback;
        } catch (e) {
            return fallback;
        }
    };
    const write = (key, value) => {
        try {
            if (value === null) localStorage.removeItem(key);
            else localStorage.setItem(key, JSON.stringify(value));
        } catch (e) { /* not saved */ }
    };

    let state = read(STATE_KEY, null);  // {id, step, ctx, open}
    let done = new Set(read(DONE_KEY, []));
    const lessonOf = (id) => LESSONS.find((l) => l.id === id);
    if (state && !lessonOf(state.id)) state = null;

    const save = () => write(STATE_KEY, state);

    // ---- The catalog ----

    function renderCatalog() {
        catalog.innerHTML = '';
        LESSONS.forEach((lesson) => {
            const card = document.createElement('div');
            card.className = 'flex flex-col gap-1.5 p-3 rounded-lg border border-slate-800 bg-slate-900/60';
            card.dataset.lesson = lesson.id;
            const going = state && state.id === lesson.id;
            const label = going ? 'Resume' : done.has(lesson.id) ? 'Again' : 'Start';
            card.innerHTML = `
                <div class="flex items-start justify-between gap-2">
                    <div class="font-semibold text-slate-100">${lesson.title}</div>
                    ${done.has(lesson.id) ? '<span class="flex-shrink-0 text-[0.625rem] font-semibold text-emerald-300" title="You finished this lesson">✓ done</span>' : ''}
                </div>
                <div class="text-[0.6875rem] text-slate-400 leading-snug flex-1">${lesson.summary}</div>
                <div class="flex items-center justify-between gap-2 pt-1">
                    <span class="text-[0.625rem] text-slate-500">${lesson.steps.length} steps${going ? `, at step ${state.step + 1}` : ''}</span>
                    <button type="button" class="px-2.5 py-1 rounded-md font-semibold bg-amber-500/15 hover:bg-amber-500/25 text-amber-200 border border-amber-500/40 transition cursor-pointer">${label}</button>
                </div>`;
            card.querySelector('button').addEventListener('click', () => start(lesson.id, !going));
            catalog.appendChild(card);
        });
        const count = LESSONS.filter((l) => done.has(l.id)).length;
        $('lessons-done').textContent = count ? `${count} of ${LESSONS.length} done` : '';
    }

    // ---- The dock ----

    let lastText = '';

    function start(id, fresh) {
        if (fresh || !state || state.id !== id) state = { id, step: 0, ctx: {} };
        state.open = true;
        save();
        renderDock(true);
        renderCatalog();
        const s = step();
        if (s.show && window.showView) window.showView(s.show[0]);  // to the view the lesson starts in
    }

    function close() {
        if (state) {
            state.open = false;
            save();
        }
        dock.hidden = true;
        hideSpotlight();
        renderCatalog();
    }

    function step() {
        return state ? lessonOf(state.id).steps[state.step] : null;
    }

    function textOfStep(s) {
        return typeof s.text === 'function' ? s.text(state.ctx) : s.text;
    }

    function renderDock(fresh) {
        if (!state || !state.open) {
            dock.hidden = true;
            return;
        }
        const lesson = lessonOf(state.id);
        const s = step();
        dock.hidden = false;
        $('lesson-title').textContent = lesson.title;
        $('lesson-progress').textContent = `Step ${state.step + 1} of ${lesson.steps.length}`;
        lastText = textOfStep(s);
        $('lesson-text').innerHTML = lastText;
        if (fresh) $('lesson-result').textContent = '';
        const doButton = $('lesson-do');
        doButton.hidden = !s.do;
        doButton.disabled = false;
        if (s.do) doButton.textContent = s.do[0];
        $('lesson-show').hidden = !s.show;
        $('lesson-back').disabled = state.step === 0;
        const last = state.step === lesson.steps.length - 1;
        $('lesson-next').textContent = last ? 'Finish' : 'Next';
        updateCheck();
    }

    function updateCheck() {
        const s = step();
        const next = $('lesson-next');
        const ok = !!(s && s.check && s.check(state.ctx));
        next.dataset.done = ok ? 'true' : 'false';
        if (ok) next.textContent = `✓ ${state.step === lessonOf(state.id).steps.length - 1 ? 'Finish' : 'Next'}`;
        else next.textContent = state.step === lessonOf(state.id).steps.length - 1 ? 'Finish' : 'Next';
    }

    function go(delta) {
        const lesson = lessonOf(state.id);
        const target = state.step + delta;
        if (target >= lesson.steps.length) {  // finished
            done.add(state.id);
            write(DONE_KEY, [...done]);
            state = null;
            save();
            dock.hidden = true;
            hideSpotlight();
            renderCatalog();
            return;
        }
        state.step = Math.max(0, target);
        save();
        hideSpotlight();
        renderDock(true);
    }

    async function doIt() {
        const s = step();
        if (!s || !s.do) return;
        const button = $('lesson-do');
        const result = $('lesson-result');
        button.disabled = true;
        result.textContent = 'Working…';
        try {
            const message = await s.do[1](state.ctx);
            save();  // the step may have noted something for later steps
            result.textContent = message || 'Done.';
        } catch (err) {
            result.textContent = `That did not work: ${err.message}.`;
        } finally {
            button.disabled = false;
            updateCheck();
        }
    }

    // ---- Show me: switch view, scroll to the element and outline it for a few seconds ----

    let spotTimer = null;
    let spotFrame = null;

    function hideSpotlight() {
        clearTimeout(spotTimer);
        cancelAnimationFrame(spotFrame);
        spotlight.hidden = true;
    }

    function showMe() {
        const s = step();
        if (!s || !s.show) return;
        const [view, id] = s.show;
        if (window.showView) window.showView(view);
        hideSpotlight();
        requestAnimationFrame(() => {
            const el = $(id);
            if (!el) return;
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            spotlight.hidden = false;
            const follow = () => {  // the element can move (scrolling) or be replaced (pushes)
                const target = $(id);
                if (target) {
                    const r = target.getBoundingClientRect();
                    Object.assign(spotlight.style, {
                        left: `${r.left - 4}px`, top: `${r.top - 4}px`, width: `${r.width + 8}px`, height: `${r.height + 8}px`,
                    });
                }
                spotFrame = requestAnimationFrame(follow);
            };
            follow();
            spotTimer = setTimeout(hideSpotlight, 4000);
        });
    }

    $('lesson-do').addEventListener('click', doIt);
    $('lesson-show').addEventListener('click', showMe);
    $('lesson-back').addEventListener('click', () => go(-1));
    $('lesson-next').addEventListener('click', () => go(1));
    $('lesson-close').addEventListener('click', close);

    // Keep the check and any numbers in the text current.
    setInterval(() => {
        if (!state || !state.open || dock.hidden) return;
        updateCheck();
        const text = textOfStep(step());
        if (text !== lastText) {
            lastText = text;
            $('lesson-text').innerHTML = text;
        }
    }, 500);

    window.startLesson = (id) => start(id, true);  // for links and the browser tests

    renderCatalog();
    renderDock(false);
})();
