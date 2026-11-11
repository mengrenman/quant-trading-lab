// The informed-traders model on the page (the "ZI with informed traders" model, lob/models/informed.py).
//
// Its ground truth: a line in the model-parameters bar (#informed-line, components/model_params.html) says what share
// of the model's market-order lots the informed traders sent and what makers lose to them, counted live from the
// model's own orders; with another model it says there are none, and in a playback that it is not recorded. The
// text comes from the server (the stream's `informed` event, sent when it changed, and the same text when the panel is
// rendered), so this only puts it in place: and again after the page replaces the panel on a change of controls from
// the stream (index.html's applyControls, which can wait while the panel is being edited), which would otherwise
// bring back the text it was rendered with.
//
// Reveal: a switch on the chart's toolbar (#reveal-toggle) for the session (every tab follows it). On, the candles
// carry `fair`, the hidden value at the last unit of the bar sampled, and the value is drawn as a line on the candle
// chart in violet with a legend entry "hidden value (revealed)". It is disabled with its reason for a model without a
// hidden value, and says "not recorded" in a playback, which draws nothing.
(function () {
    const toggle = document.getElementById('reveal-toggle');
    const note = document.getElementById('reveal-note');
    const legend = document.getElementById('fair-legend');
    const container = document.getElementById('price-chart-container');
    if (!toggle || !note || !legend || !container) return;

    const TITLE = toggle.title;
    let status = null;  // what the server last said (the stream's event, or the answer that carried the candles)
    let series = null;  // the line on the candle chart
    let seriesChart = null;  // the chart it was made on (the page may make the charts again)
    let error = null;

    function lineColor() {
        return typeof themeColor === 'function' ? themeColor('violet-400') : '#a78bfa';
    }

    function ensureSeries() {
        if (typeof priceChart === 'undefined' || !priceChart) return null;
        if (series && seriesChart !== priceChart) series = null;  // the chart it was on is gone
        if (!series) {
            series = priceChart.addLineSeries({
                color: lineColor(),
                lineWidth: 2,
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: false,
                title: 'hidden value',
                priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
            });
            seriesChart = priceChart;
        }
        return series;
    }

    function drawnPoints() {
        return series && seriesChart === priceChart ? series.data().length : 0;
    }

    // ---- The switch ----

    function setStatus(reveal) {
        if (!reveal) return;
        status = reveal;
        toggle.setAttribute('aria-pressed', reveal.on ? 'true' : 'false');
        toggle.disabled = !!reveal.disabled;
        toggle.title = reveal.note ? `${TITLE} (${reveal.note})` : TITLE;
        if (note.textContent !== (reveal.note || '')) note.textContent = reveal.note || '';
        legend.hidden = !reveal.active;
        container.dataset.reveal = reveal.active ? 'on' : 'off';
    }

    toggle.addEventListener('click', async () => {
        if (toggle.disabled) return;
        const want = toggle.getAttribute('aria-pressed') !== 'true';
        try {
            await window.postForm('/api/chart/reveal', { on: want });
        } catch (err) {
            console.error('Failed to switch Reveal:', err);
            return;
        }
        if (typeof loadChartData === 'function') loadChartData();  // the value comes with the bars
    });

    // ---- The line ----

    // A fresh set of bars (the chart reloaded): the line is those that have a value, or nothing.
    function load(reveal, candles) {
        setStatus(reveal);
        const points = (candles || []).filter((c) => c.fair != null).map((c) => ({ time: c.time, value: c.fair }));
        try {
            if (!points.length) {
                if (series && seriesChart === priceChart) series.setData([]);
                return;
            }
            const s = ensureSeries();
            if (!s) return;
            s.setData([]);  // as the other lines are: setting the same times again to a line that is not empty can corrupt the chart's point map
            s.setData(points);
        } catch (err) {
            error = String(err);
            console.error('Failed to draw the hidden value:', err);
        }
    }

    // One bar pushed (or the same bar again, changed).
    function bar(c) {
        if (c.fair == null) return;
        try {
            const s = ensureSeries();
            if (s) s.update({ time: c.time, value: c.fair });
        } catch (err) {
            error = String(err);  // an older bar than the line's last (a push from before a reload): the reload brings it
        }
    }

    document.addEventListener('lob:chart-theme', () => {
        if (series && seriesChart === priceChart) series.applyOptions({ color: lineColor() });
    });

    // ---- The ground truth line ----

    let text = null;  // the line as last sent

    function showLine() {
        const el = document.getElementById('informed-line');
        if (el && text != null && el.textContent !== text) el.textContent = text;
    }

    document.addEventListener('lob:informed', (e) => {
        const d = e.detail || {};
        text = d.line;
        const el = document.getElementById('informed-line');
        if (el) el.dataset.informed = d.model ? 'yes' : 'no';
        showLine();
        setStatus(d.reveal);
    });

    // The panel was replaced by a change of controls from the stream (which waits while the panel is being edited, so
    // it can be as old as the push that rendered it): the text of the last push goes back in.
    document.addEventListener('lob:model-params', showLine);

    window.lobInformed = {
        load, bar,
        state: () => ({  // what is shown, for the browser tests
            status, text, error,
            points: drawnPoints(),
            color: series && seriesChart === priceChart ? series.options().color : null,
            legend: !legend.hidden,
            pressed: toggle.getAttribute('aria-pressed') === 'true',
            disabled: toggle.disabled,
            note: note.textContent,
            line: (document.getElementById('informed-line') || {}).textContent || null,
        }),
        data: () => (series && seriesChart === priceChart ? series.data().map((p) => ({ time: p.time, value: p.value })) : []),
    };
})();
