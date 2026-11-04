// Recording and playback (app/recording.py; the endpoints are in app/main.py).
//
// The status badge (components/sim_badge.html, pushed with the panels) says whether the session is
// recording or playing back, and where. This keeps the Record button, the Reset button (Rewind in a
// playback) and the playback bar (components/sim_controls.html) in step with it, moves the playback
// when its timeline is dragged, and runs the Recordings card (components/recordings_card.html).
(function () {
    const $ = (id) => document.getElementById(id);
    const count = (n) => Number(n).toLocaleString('en-US');
    const clock = (t) => Number(t).toFixed(2);
    let dragging = false;  // while the timeline is held, pushes leave it alone

    function toast(text) {
        if (window.showToast) window.showToast(text);
    }

    async function reason(res) {
        if (!res) return 'the server did not answer';
        const detail = await res.json().then((d) => d.detail).catch(() => null);
        return typeof detail === 'string' ? detail.replace(/\.$/, '') : `the server answered with status ${res.status}`;
    }

    async function post(url, fields) {
        try {
            return await window.postForm(url, fields);
        } catch (err) {
            return null;
        }
    }

    function sync() {
        const badge = $('sim-stats-badge');
        if (!badge) return;
        const d = badge.dataset;

        const recording = d.recording === 'true';
        const record = $('record-btn');
        if (record) {
            record.setAttribute('aria-pressed', String(recording));
            $('record-text').textContent = recording ? `Stop recording (${count(d.recorded)} events)` : 'Record';
            $('record-dot').className = recording ? 'w-2 h-2 rounded-sm bg-rose-400 animate-pulse' : 'w-2 h-2 rounded-full bg-rose-500';
            record.title = recording ? 'Stop recording and keep it (Analyze, Recordings)'
                : 'Record from now on, to play back later (Analyze, Recordings)';
        }

        const playing = d.playback === 'true';
        const bar = $('playback-bar');
        if (bar) bar.hidden = !playing;
        const reset = $('sim-reset-btn');
        if (reset) {
            $('sim-reset-text').textContent = playing ? 'Rewind' : 'Reset Book';
            reset.title = playing ? 'Back to the start of the recording' : 'Restart from the seed with a fresh book';
        }
        if (!playing || !bar) return;

        const start = Number(d.pbStart);
        const end = Number(d.pbEnd);
        const now = Math.min(Number(d.pbTime), end);
        $('playback-name').textContent = d.pbName;
        const timeline = $('playback-timeline');
        if (!dragging) {
            timeline.min = String(start);
            timeline.max = String(Math.max(end, start + 1e-6));
            timeline.value = String(now);
            $('playback-clock').textContent = `${clock(now)} of ${clock(start)} to ${clock(end)}`;
        }
        let note;
        if (d.pbSeeking) {
            const [done, to] = d.pbSeeking.split('/');
            note = `Moving there: event ${count(done)} of ${count(to)}…`;
        } else if (d.pbEnded === 'true') {
            note = 'The end of the recording: Rewind plays it again.';
        } else {
            note = `Event ${count(d.pbCursor)} of ${count(d.pbEvents)}, recorded with ${d.pbRecordedWith}`;
        }
        $('playback-note').textContent = note;
        const exit = $('playback-exit');
        exit.textContent = `Exit playback: ${d.pbTakesOver} carries on live`;
        exit.title = `End the playback here: ${d.pbTakesOver}, the model selected, takes over the book as it is`;
    }

    document.addEventListener('lob:panel', (e) => {
        if (e.detail === 'sim_status') sync();
    });
    sync();

    window.toggleRecording = async function () {
        const recording = (($('sim-stats-badge') || {}).dataset || {}).recording === 'true';
        const res = await post(recording ? '/api/recording/stop' : '/api/recording/start');
        if (!res || !res.ok) toast(`${await reason(res)}.`);
    };

    window.exitPlayback = async function () {
        const res = await post('/api/playback/exit');
        if (!res || !res.ok) toast(`${await reason(res)}.`);
    };

    // The timeline: the time it points at shows while it is dragged; letting go moves the playback there.
    const timeline = $('playback-timeline');
    if (timeline) {
        timeline.addEventListener('pointerdown', () => { dragging = true; });
        timeline.addEventListener('pointerup', () => { dragging = false; });
        timeline.addEventListener('pointercancel', () => { dragging = false; });
        timeline.addEventListener('input', () => {
            const d = $('sim-stats-badge').dataset;
            $('playback-clock').textContent = `${clock(timeline.value)} of ${clock(d.pbStart)} to ${clock(d.pbEnd)}`;
        });
        timeline.addEventListener('change', async () => {
            dragging = false;  // from here the pushes show it moving there
            const res = await post('/api/playback/seek', { time: timeline.value });
            if (!res || !res.ok) toast(`${await reason(res)}.`);
        });
    }

    // The Recordings card: its list is pushed when it changes, so its buttons are handled here.
    const list = $('recordings-list-wrapper');
    if (list) {
        list.addEventListener('click', async (e) => {
            const button = e.target.closest('button[data-recording-action]');
            if (!button || button.disabled) return;
            const id = button.closest('[data-recording-id]').dataset.recordingId;
            const action = button.dataset.recordingAction;
            button.disabled = true;
            const res = action === 'play' ? await post('/api/playback/start', { recording_id: id })
                : await post(`/api/recordings/${id}/delete`);
            button.disabled = false;
            if (!res || !res.ok) toast(`${await reason(res)}.`);
        });
    }

    const upload = $('recording-upload');
    if (upload) {
        upload.addEventListener('submit', async (e) => {
            e.preventDefault();
            const result = $('recording-upload-result');
            const file = upload.querySelector('input[type="file"]').files[0];
            if (!file) return;
            result.className = 'basis-full text-[0.6875rem] leading-snug min-h-[1rem] text-slate-400';
            result.textContent = 'Reading it…';
            const body = new FormData();
            body.append('file', file);
            let res = null;
            try {
                res = await fetch('/api/recordings/upload', { method: 'POST', body });
            } catch (err) { /* said below */ }
            if (res && res.ok) {
                const data = await res.json();
                result.className = 'basis-full text-[0.6875rem] leading-snug min-h-[1rem] text-emerald-400';
                result.textContent = `Added ${data.name}: ${count(data.events)} events. Play it back from the list.`;
                upload.reset();
            } else {
                result.className = 'basis-full text-[0.6875rem] leading-snug min-h-[1rem] text-rose-400';
                result.textContent = `${await reason(res)}.`;
            }
        });
    }
})();
