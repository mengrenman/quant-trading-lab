// The triggers card (components/triggers_card.html): the rule editor, and the list of rules with what each has seen.
//
// The card is static and never replaced by a push. The list is filled in place from the stream's "triggers" events
// (app/triggers.py, LOBManager.triggers_status: every rule's definition and status, whenever it changed) and from
// data-initial, the rules when the page was made. While the pointer or the keyboard is on a row's buttons the list
// waits (the last status is kept and shown when it ends), so a click lands on the rule it was aimed at even when a
// firing changes the rule's button; what a click did shows in its row at once.
//
// Fired rules also mark the candle chart: the chart asks lobTriggers.markers(bars, view) for them when it sets its markers
// (components/chart.html), one for each bar with a firing, and this says when they change (lob:trigger-marks).
(function () {
    const card = document.getElementById('triggers-card');
    if (!card) return;
    const form = document.getElementById('trigger-form');
    const conditionsBox = document.getElementById('trigger-conditions');
    const addButton = document.getElementById('trigger-add-condition');
    const board = document.getElementById('triggers-board');
    const table = document.getElementById('triggers-table');
    const rowsBox = document.getElementById('triggers-rows');
    const emptyNote = document.getElementById('triggers-empty');
    const status = document.getElementById('triggers-status');
    const result = document.getElementById('triggers-result');
    const combineBox = document.getElementById('trigger-combine');
    const oneShot = document.getElementById('trigger-one-shot');
    const submit = document.getElementById('trigger-submit');
    const cancelEdit = document.getElementById('trigger-cancel-edit');
    const editingTag = document.getElementById('trigger-editing');
    const maxConditions = Number(card.dataset.maxConditions);
    const EDGES = JSON.parse(card.dataset.edges || '{}');

    let latest = { rules: [], marks: [] };
    try { latest = JSON.parse(card.dataset.initial || 'null') || latest; } catch (e) { /* none yet */ }
    let held = false;
    let queued = null;  // the status that arrived while the list was held
    let editing = null;  // the id of the rule the editor is changing
    let oneShotTouched = false;
    let marksKey = '';
    let marksWaiting = false;  // a firing is later than the newest bar the chart holds
    const rows = new Map();  // rule id -> { tr, refs, shown }

    // ---- Numbers and words ----

    const g = (x) => {
        if (x == null) return '—';
        if (x === 0) return '0';
        const a = Math.abs(x);
        let text = a >= 100 ? x.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
            : a >= 1 ? x.toFixed(3) : String(Number(x.toPrecision(5)));
        if (text.includes('.')) text = text.replace(/0+$/, '').replace(/\.$/, '');
        return text;
    };
    const ACTION_WORDS = { notice: 'notice', pause: 'pause', cancel: 'cancel my orders', flatten: 'flatten' };
    const TONES = {
        armed: ['bg-emerald-400', 'text-emerald-400'], 'warming up': ['bg-amber-400', 'text-amber-300'],
        'study off': ['bg-rose-400', 'text-rose-300'], disarmed: ['bg-slate-600', 'text-slate-400'],
    };
    const DOT_CLASSES = ['bg-emerald-400', 'bg-amber-400', 'bg-rose-400', 'bg-slate-600'];
    const STATE_CLASSES = ['text-emerald-400', 'text-amber-300', 'text-rose-300', 'text-slate-400'];
    const setText = (node, text) => { if (node.textContent !== text) node.textContent = text; };

    // ---- The list ----

    function makeRow(rule) {
        const tr = document.getElementById('trigger-row-template').content.firstElementChild.cloneNode(true);
        tr.dataset.ruleId = String(rule.id);
        const refs = {
            dot: tr.querySelector('[data-dot]'), name: tr.querySelector('[data-name]'), state: tr.querySelector('[data-state]'),
            text: tr.querySelector('[data-text]'), how: tr.querySelector('[data-how]'), values: tr.querySelector('[data-values]'),
            fired: tr.querySelector('[data-fired]'), toggle: tr.querySelector('[data-action="arm"]'),
        };
        refs.toggle.removeAttribute('data-action');  // the first update gives it its label, look and action
        return { tr, refs, shown: { state: null } };
    }

    function updateRow(row, rule) {
        const { tr, refs } = row;
        tr.dataset.armed = String(rule.armed);
        tr.dataset.state = rule.state;
        tr.dataset.count = String(rule.count);
        if (tr.dataset.removed) return;  // deleted here: it leaves with the next push after the hold
        setText(refs.name, rule.name);
        if (refs.name.title !== rule.name) refs.name.title = rule.name;  // the cell cuts a long name short
        setText(refs.state, rule.state === 'armed' && rule.one_shot ? 'armed, one-shot' : rule.state);
        if (row.shown.state !== rule.state) {
            row.shown.state = rule.state;
            const [dot, text] = TONES[rule.state] || TONES.armed;
            refs.dot.classList.remove(...DOT_CLASSES);
            refs.dot.classList.add(dot);
            refs.state.classList.remove(...STATE_CLASSES);
            refs.state.classList.add(text);
            refs.dot.classList.toggle('animate-pulse', rule.state === 'armed');
        }
        setText(refs.text, `${rule.conditions.length > 1 ? (rule.combine === 'all' ? 'all of: ' : 'any of: ') : ''}${rule.text}`);
        setText(refs.how, `${rule.actions.map((a) => ACTION_WORDS[a]).join(', ')} · cooldown ${g(rule.cooldown)}${rule.one_shot ? ' · one-shot' : ''}`);
        // What each condition reads at the last check: its value, marked when it holds.
        const values = refs.values;
        while (values.children.length > rule.conditions.length) values.lastElementChild.remove();
        while (values.children.length < rule.conditions.length) values.appendChild(document.createElement('div'));
        rule.conditions.forEach((c, i) => {
            const node = values.children[i];
            const text = c.off ? 'off' : c.now == null ? '—' : `${g(c.now)}${c.unit ? ' ' + c.unit : ''}`;
            setText(node, `${c.label} ${text}`);
            node.title = c.now == null && c.none ? `${c.label}: ${c.none}` : c.text;
            node.classList.toggle('text-emerald-400', c.met === true);
            node.classList.toggle('text-slate-400', c.met !== true);
        });
        setText(refs.fired, rule.fired_at == null ? (rule.count ? `${rule.count}×` : '—') : `t ${rule.fired_at.toFixed(1)} · ${rule.count}×`);
        const toggle = refs.toggle;
        const action = rule.armed ? 'disarm' : 'arm';
        if (toggle.dataset.action !== action) {
            toggle.dataset.action = action;
            toggle.textContent = rule.armed ? 'Disarm' : 'Arm';
            toggle.title = rule.armed ? 'Stop checking this rule' : 'Check this rule again, from now';
            toggle.className = `px-2 py-0.5 text-[0.625rem] font-medium rounded border transition cursor-pointer aria-disabled:opacity-50 aria-disabled:cursor-default ${
                rule.armed ? 'text-amber-200 bg-amber-500/10 hover:bg-amber-500/25 border-amber-500/40'
                    : 'text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/25 border-emerald-500/40'}`;
        }
    }

    function statusLine(s) {
        const notes = [];
        if (s.needs_model) notes.push('needs a running model: the manual model takes no steps, so a rule is checked only after an order of yours');
        if (s.playback) notes.push('playback: only the notice and the pause act, cancel and flatten are off; counts start from the playback\'s start');
        if (s.seeking) notes.push('seeking: nothing acts, rules start again at the target');
        if (s.studies_off && !s.seeking) notes.push('the studies are off: rules wait');
        if (!notes.length && s.rules.length) {
            const armed = s.rules.filter((r) => r.armed).length;
            notes.push(`${s.rules.length} rule${s.rules.length === 1 ? '' : 's'}, ${armed} armed`);
        }
        return { text: notes.join(' · '), warn: !!(s.needs_model || s.playback || s.seeking || s.studies_off) };
    }

    function render(s) {
        const ids = new Set(s.rules.map((r) => r.id));
        rows.forEach((row, id) => {
            if (!ids.has(id)) {
                row.tr.remove();
                rows.delete(id);
            }
        });
        s.rules.forEach((rule, index) => {
            let row = rows.get(rule.id);
            if (!row) {
                row = makeRow(rule);
                rows.set(rule.id, row);
            }
            const at = rowsBox.children[index] || null;
            if (at !== row.tr) rowsBox.insertBefore(row.tr, at);
            updateRow(row, rule);
        });
        table.hidden = s.rules.length === 0;
        emptyNote.hidden = s.rules.length > 0;
        const line = statusLine(s);
        setText(status, line.text);
        status.classList.toggle('text-amber-300', line.warn);
        status.classList.toggle('text-slate-500', !line.warn);
        const key = JSON.stringify(s.marks || []);
        if (key !== marksKey) {
            marksKey = key;
            document.dispatchEvent(new CustomEvent('lob:trigger-marks'));
        }
        if (editing !== null && !ids.has(editing)) endEdit(true);  // it was deleted elsewhere
    }

    function receive(s) {
        latest = s;
        if (held) queued = s;
        else render(s);
    }
    document.addEventListener('lob:triggers', (e) => receive(e.detail));
    render(latest);

    // ---- Holding the list while a click is on its way ----

    const onButtons = (el) => !!(el && el.closest && el.closest('[data-trigger-actions]'));
    let pointerOn = false;
    let focusOn = false;
    function hold() {
        held = pointerOn || focusOn;
        if (!held && queued) {
            const s = queued;
            queued = null;
            render(s);
        }
    }
    board.addEventListener('pointerover', (e) => { pointerOn = onButtons(e.target); hold(); });
    board.addEventListener('pointerleave', () => { pointerOn = false; hold(); });
    board.addEventListener('focusin', (e) => {
        let keyboard = true;
        try { keyboard = e.target.matches(':focus-visible'); } catch (err) { /* older browsers: count it */ }
        focusOn = onButtons(e.target) && keyboard;
        hold();
    });
    board.addEventListener('focusout', (e) => { if (!onButtons(e.relatedTarget)) { focusOn = false; hold(); } });

    // Buttons are marked unavailable (aria-disabled) rather than disabled or taken out: an element under the pointer that
    // is disabled or removed can keep WebKit from telling the board the pointer has left it, which would hold it for good.
    const off = (button) => button.getAttribute('aria-disabled') === 'true';
    board.addEventListener('click', async (e) => {
        const button = e.target.closest('button[data-action]');
        if (!button || off(button)) return;
        const tr = button.closest('tr[data-rule-id]');
        const id = Number(tr.dataset.ruleId);
        const action = button.dataset.action;
        if (e.detail > 0) button.blur();  // a mouse click: the list need not wait for focus to move on
        if (action === 'edit') {
            const rule = latest.rules.find((r) => r.id === id);
            if (rule) startEdit(rule);
            return;
        }
        button.setAttribute('aria-disabled', 'true');
        let res = null;
        try {
            res = await window.postForm(`/api/triggers/${id}/${action}`);
        } catch (err) { /* no answer: said below */ }
        button.removeAttribute('aria-disabled');
        if (!res || !res.ok) {
            const why = !res ? 'the server did not answer' : res.status === 404 ? 'that rule is no longer there'
                : `the server answered with status ${res.status}`;
            if (window.showToast) window.showToast(`Not done: ${why}.`);
            return;
        }
        const row = rows.get(id);
        if (action === 'delete') {
            tr.classList.add('opacity-40', 'line-through');
            tr.dataset.removed = 'true';
            tr.querySelectorAll('button[data-action]').forEach((b) => b.setAttribute('aria-disabled', 'true'));
            button.textContent = 'Deleted';
            if (editing === id) endEdit(true);
        } else if (row) {  // arm or disarm: its row says so now, the rest when the list catches up
            const armed = action === 'arm';
            tr.dataset.armed = String(armed);
            tr.dataset.state = armed ? 'armed' : 'disarmed';
            button.dataset.action = armed ? 'disarm' : 'arm';
            button.textContent = armed ? 'Disarm' : 'Arm';
            setText(row.refs.state, armed ? 'armed' : 'disarmed');
        }
    });

    // ---- The editor ----

    const conditionRows = () => [...conditionsBox.querySelectorAll('[data-condition]')];

    function refreshConditions() {
        const list = conditionRows();
        list.forEach((row, i) => {
            const remove = row.querySelector('[data-remove-condition]');
            if (list.length === 1) remove.setAttribute('aria-disabled', 'true');
            else remove.removeAttribute('aria-disabled');
            setText(row.querySelector('[data-edge]'), EDGES[row.querySelector('[name="op"]').value] || '');
        });
        if (list.length >= maxConditions) addButton.setAttribute('aria-disabled', 'true');
        else addButton.removeAttribute('aria-disabled');
        combineBox.hidden = list.length < 2;
        const crossings = list.filter((row) => row.querySelector('[name="op"]').value.startsWith('crosses')).length;
        const allOf = form.elements.combine.value === 'all';
        if (list.length > 1 && allOf && crossings > 1) {
            result.innerHTML = '<span class="text-amber-300" data-hint="crossings">A crossing is true only on the event it happens, and two studies do not cross on one event: an all-of rule can hold only one crossing. Choose "any of them does", or make the others above or below.</span>';
        } else if (result.querySelector('[data-hint]')) {
            result.innerHTML = '';
        }
    }

    function addCondition() {
        if (conditionRows().length >= maxConditions) return null;
        const row = document.getElementById('trigger-condition-template').content.firstElementChild.cloneNode(true);
        conditionsBox.appendChild(row);
        refreshConditions();
        return row;
    }
    addButton.addEventListener('click', () => { if (!off(addButton)) addCondition(); });
    conditionsBox.addEventListener('click', (e) => {
        const remove = e.target.closest('[data-remove-condition]');
        if (!remove || off(remove)) return;
        remove.closest('[data-condition]').remove();
        refreshConditions();
    });
    conditionsBox.addEventListener('change', refreshConditions);
    form.addEventListener('change', (e) => {
        if (e.target.name === 'combine') refreshConditions();
        if (e.target.name === 'action' && e.target.value === 'pause' && !oneShotTouched) oneShot.checked = e.target.checked;
    });
    oneShot.addEventListener('change', () => { oneShotTouched = true; });

    function setCondition(row, value, op, threshold) {
        row.querySelector('[name="value"]').value = value;
        row.querySelector('[name="op"]').value = op;
        row.querySelector('[name="threshold"]').value = String(threshold);
    }

    function resetForm() {
        conditionRows().slice(1).forEach((row) => row.remove());
        const first = conditionRows()[0];
        first.querySelector('[name="value"]').selectedIndex = 0;
        first.querySelector('[name="op"]').selectedIndex = 0;
        first.querySelector('[name="threshold"]').value = '0';
        form.elements.name.value = '';
        form.elements.cooldown.value = form.elements.cooldown.defaultValue;
        form.querySelectorAll('input[name="combine"]').forEach((r) => { r.checked = r.value === 'all'; });
        form.querySelectorAll('input[name="action"]').forEach((box) => { box.checked = box.value === 'notice'; });
        oneShot.checked = false;
        oneShotTouched = false;
        refreshConditions();
    }

    function startEdit(rule) {
        resetForm();
        rule.conditions.forEach((c, i) => {
            const row = i === 0 ? conditionRows()[0] : addCondition();
            if (row) setCondition(row, c.value, c.op, c.threshold);
        });
        form.elements.name.value = rule.name;
        form.elements.cooldown.value = String(rule.cooldown);
        form.querySelectorAll('input[name="combine"]').forEach((r) => { r.checked = r.value === rule.combine; });
        form.querySelectorAll('input[name="action"]').forEach((box) => { box.checked = rule.actions.includes(box.value); });
        oneShot.checked = rule.one_shot;
        oneShotTouched = true;
        editing = rule.id;
        setText(editingTag, `editing #${rule.id}`);
        editingTag.hidden = false;
        setText(submit, 'Save changes');
        cancelEdit.hidden = false;
        result.innerHTML = '';
        refreshConditions();
        form.elements.name.focus();
    }

    function endEdit(reset) {
        editing = null;
        editingTag.hidden = true;
        setText(submit, 'Add the rule');
        cancelEdit.hidden = true;
        if (reset) resetForm();
    }
    cancelEdit.addEventListener('click', () => { endEdit(true); result.innerHTML = ''; });

    // The checkbox sends nothing when it is off, so it is sent as on or off; a rule being edited is posted to its own path.
    form.addEventListener('htmx:configRequest', (e) => {
        e.detail.parameters.one_shot = oneShot.checked ? 'on' : 'off';
        if (editing !== null) e.detail.path = `/api/triggers/${editing}`;
    });
    // A rule that was made or saved is in the list: the editor starts clean, so that a second press of Add cannot make a
    // duplicate and the next rule does not inherit this one's conditions (the message of what was done stays).
    form.addEventListener('htmx:afterRequest', (e) => {
        if (e.detail.elt !== form || !e.detail.xhr || e.detail.xhr.status !== 200) return;
        if (editing !== null) endEdit(true);
        else resetForm();
    });
    refreshConditions();

    // ---- The chart's markers ----

    // The marks for the chart, one for each bar with a firing: the bars as the series holds them (ascending, a fixed step
    // apart). A firing later than the newest bar waits for it (waiting() says so). Only the newest firing of each rule
    // has its name: a rule that fires every few bars would print its name over its neighbors' until none could be read.
    // The names of rules whose newest firings are closer than their labels are wide share the later one ("A +1"), so no
    // two labels overlap; ``view`` says how wide a bar is (``spacing``, in pixels) and in what font the chart writes.
    const measure = (() => {
        const context = document.createElement('canvas').getContext('2d');
        return (text, font) => {
            if (!context) return 7.2 * text.length;
            context.font = font;
            return context.measureText(text).width;
        };
    })();
    const NAME_CHARS = 20;
    const LABEL_PAD = 8;  // pixels of air around a label, which the chart centers on its bar
    const labelText = ({ names, count }) => `${names[0].slice(0, NAME_CHARS)}${names.length > 1 ? ` +${names.length - 1}`
        : count > 1 ? ` ×${count}` : ''}`;

    function markers(bars, view) {
        marksWaiting = false;
        const marks = latest.marks || [];
        if (!bars.length || !marks.length) return [];
        const first = bars[0].time;
        const last = bars[bars.length - 1].time;
        const step = bars.length > 1 ? bars[1].time - bars[0].time
            : (typeof currentInterval === 'number' ? currentInterval : 2);
        const color = window.themeColor ? themeColor('violet-400') : '#a78bfa';
        const times = new Set();
        const newest = new Map();  // rule id -> { time, name, count: its firings in that bar }
        const counts = new Map();  // `${bar}:${rule id}` -> firings
        marks.forEach(([t, id, name]) => {
            if (t < first) return;
            if (t >= last + step) {
                marksWaiting = true;
                return;
            }
            const time = Math.floor(t / step) * step;
            times.add(time);
            const key = `${time}:${id}`;
            counts.set(key, (counts.get(key) || 0) + 1);
            const at = newest.get(id);
            if (!at || time >= at.time) newest.set(id, { time, name });
        });
        // One label for each bar that holds a rule's newest firing, then labels too close to read are folded together.
        const labels = new Map();
        newest.forEach(({ time, name }, id) => {
            const label = labels.get(time) || { time, names: [], count: counts.get(`${time}:${id}`) };
            label.names.push(name);
            labels.set(time, label);
        });
        const kept = [];
        const width = (label) => measure(labelText(label), view && view.font || '12px monospace') + LABEL_PAD;
        [...labels.values()].sort((a, b) => b.time - a.time).forEach((label) => {
            const near = kept[kept.length - 1];
            if (view && view.spacing && near
                && Math.abs(near.time - label.time) / step * view.spacing < (width(near) + width(label)) / 2) {
                near.names.push(...label.names);  // the later label names them both
            } else {
                kept.push(label);
            }
        });
        const text = new Map(kept.map((label) => [label.time, labelText(label)]));
        return [...times].sort((a, b) => a - b).map((time) => ({
            time, position: 'aboveBar', shape: 'circle', color, text: text.get(time) || '',
        }));
    }

    window.lobTriggers = {  // for the chart, and for the browser tests
        markers,
        waiting: () => marksWaiting,
        state: () => ({
            rules: latest.rules.map((r) => ({ ...r, conditions: r.conditions.map((c) => ({ ...c })) })), marks: (latest.marks || []).slice(),
            held, queued: queued !== null, editing, shown: [...rowsBox.children].map((tr) => tr.dataset.ruleId),
            status: status.textContent, needsModel: !!latest.needs_model, playback: !!latest.playback,
        }),
    };
})();
