// ============================================================
// NLB Tracking Demo — MC_RTT velocity decoding
//
// Renders REAL held-out reaches exported by export_nlb_demo.py
// (web-demo/nlb_results.js → window.NLB_DEMO_RESULTS):
//   • true cursor path  (integrated true finger velocity)
//   • per-model decoded path (integrated decoded velocity)
//   • real 98-channel spike-count neural data
// Reaches play back to back; the trail never crosses a reach
// boundary (each reach is integrated from its own origin).
// Nothing here is synthetic — see export_nlb_demo.py.
// ============================================================

(function () {
    const R = window.NLB_DEMO_RESULTS;
    if (!R || !R.actual) {
        console.error('NLB_DEMO_RESULTS missing — run export_nlb_demo.py to generate web-demo/nlb_results.js');
        return;
    }

    const N = R.meta.n_frames;
    const NUM_CHANNELS = R.meta.n_channels;

    const actual = { x: Float32Array.from(R.actual.x), y: Float32Array.from(R.actual.y) };
    const ecogData = R.neural.map(row => Float32Array.from(row));

    const MODEL_ORDER = ['transformer', 'lstm', 'cnn2d', 'mlp'];
    const models = {};
    for (const key of MODEL_ORDER) {
        const m = R.models[key];
        if (!m) continue;
        models[key] = {
            name: m.name,
            velR2Mean: m.vel_r2_mean,
            velR2Std: m.vel_r2_std,
            seedR2: m.seed_r2,
            xPred: Float32Array.from(m.x),
            yPred: Float32Array.from(m.y),
        };
    }

    // Per-frame reach index, and per-reach framing (data-space center + half-span)
    // so each reach fills its panel regardless of where it drifts.
    const segOfFrame = new Int16Array(N);
    const segs = R.segments.map((s, i) => {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (let f = s.start; f < s.start + s.length; f++) {
            segOfFrame[f] = i;
            for (const arr of [actual, ...Object.values(models)]) {
                const xs = arr.x || arr.xPred, ys = arr.y || arr.yPred;
                minX = Math.min(minX, xs[f]); maxX = Math.max(maxX, xs[f]);
                minY = Math.min(minY, ys[f]); maxY = Math.max(maxY, ys[f]);
            }
        }
        const half = Math.max(maxX - minX, maxY - minY) / 2 || 1;
        return { start: s.start, length: s.length,
                 cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, half };
    });

    // Display scaling for the neural panels (from the real value range).
    let neuralAbsMax = 1e-6, neuralMin = Infinity, neuralMax = -Infinity;
    for (let ch = 0; ch < NUM_CHANNELS; ch++) {
        for (let f = 0; f < N; f++) {
            const v = ecogData[ch][f];
            if (Math.abs(v) > neuralAbsMax) neuralAbsMax = Math.abs(v);
            if (v < neuralMin) neuralMin = v;
            if (v > neuralMax) neuralMax = v;
        }
    }
    const neuralRange = (neuralMax - neuralMin) || 1;

    // --- Canvas setup ---
    const neuralCanvas = document.getElementById('nlb-neural-canvas');
    const heatmapCanvas = document.getElementById('nlb-heatmap-canvas');
    const decodeCanvas = document.getElementById('nlb-decode-canvas');
    const neuralCtx = neuralCanvas.getContext('2d');
    const heatmapCtx = heatmapCanvas.getContext('2d');
    const decodeCtx = decodeCanvas.getContext('2d');

    function resizeCanvases() {
        const dpr = window.devicePixelRatio || 1;
        [neuralCanvas, heatmapCanvas, decodeCanvas].forEach(canvas => {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            canvas.style.width = rect.width + 'px';
            canvas.style.height = rect.height + 'px';
            canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        });
    }

    // --- State ---
    let currentModel = 'transformer';
    let currentFrame = 0;
    let playhead = 0;
    let lastTs = null;
    let isPlaying = false;
    let animId = null;
    let speed = 2;
    // 20 ms bins → 50 bins/s is real time; speed 2 ≈ real time, slider 1–10 scales it.
    const BINS_PER_SEC_PER_SPEED = 25;
    const WINDOW = 80; // neural frames visible at a time

    // --- Spike-count traces (subset of channels) ---
    function drawNeural() {
        const w = neuralCanvas.width / (window.devicePixelRatio || 1);
        const h = neuralCanvas.height / (window.devicePixelRatio || 1);
        neuralCtx.clearRect(0, 0, w, h);

        const seg = segs[segOfFrame[currentFrame]];
        const startFrame = Math.max(seg.start, currentFrame - WINDOW);
        const channelsToShow = Math.min(NUM_CHANNELS, 24);
        const top = 25;
        const rowH = (h - top - 10) / channelsToShow;
        const colW = (w - 10) / WINDOW;
        const amp = (rowH * 0.45) / neuralAbsMax;

        neuralCtx.strokeStyle = 'rgba(26, 26, 26, 0.7)';
        neuralCtx.lineWidth = 1;
        for (let ch = 0; ch < channelsToShow; ch++) {
            const midY = top + ch * rowH + rowH / 2;
            neuralCtx.beginPath();
            let started = false;
            for (let f = startFrame; f < currentFrame; f++) {
                const x = 5 + (f - startFrame) * colW;
                const y = midY - ecogData[ch][f] * amp;
                if (!started) { neuralCtx.moveTo(x, y); started = true; }
                else neuralCtx.lineTo(x, y);
            }
            neuralCtx.stroke();
        }
    }

    // --- Spike-count heatmap (all channels × time) ---
    function drawHeatmap() {
        const w = heatmapCanvas.width / (window.devicePixelRatio || 1);
        const h = heatmapCanvas.height / (window.devicePixelRatio || 1);
        heatmapCtx.clearRect(0, 0, w, h);

        const seg = segs[segOfFrame[currentFrame]];
        const startFrame = Math.max(seg.start, currentFrame - WINDOW);
        const top = 20;
        const rowH = (h - top - 6) / NUM_CHANNELS;
        const colW = (w - 10) / WINDOW;

        for (let ch = 0; ch < NUM_CHANNELS; ch++) {
            for (let f = startFrame; f < currentFrame; f++) {
                const t = Math.max(0, Math.min(1, (ecogData[ch][f] - neuralMin) / neuralRange));
                const r = Math.round(245 - t * 215);
                const g = Math.round(245 - t * 187);
                const b = Math.round(243 - t * 105);
                const x = 5 + (f - startFrame) * colW;
                const y = top + ch * rowH;
                heatmapCtx.fillStyle = `rgb(${r},${g},${b})`;
                heatmapCtx.fillRect(x, y, Math.max(1, colW), Math.max(1, rowH));
            }
        }
    }

    // --- Decoded cursor path (true vs decoded, framed per reach) ---
    function drawDecode() {
        const w = decodeCanvas.width / (window.devicePixelRatio || 1);
        const h = decodeCanvas.height / (window.devicePixelRatio || 1);
        decodeCtx.clearRect(0, 0, w, h);

        const model = models[currentModel];
        const seg = segs[segOfFrame[currentFrame]];
        const midX = w / 2, midY = h / 2;
        const scale = (Math.min(w, h) * 0.42) / seg.half;
        const sx = (x) => midX + (x - seg.cx) * scale;
        const sy = (y) => midY - (y - seg.cy) * scale;

        // Full reach paths (faint) so the target shape is visible.
        const drawPath = (xs, ys, color) => {
            decodeCtx.beginPath();
            decodeCtx.strokeStyle = color;
            decodeCtx.lineWidth = 1.5;
            for (let f = seg.start; f < seg.start + seg.length; f++) {
                const px = sx(xs[f]), py = sy(ys[f]);
                if (f === seg.start) decodeCtx.moveTo(px, py);
                else decodeCtx.lineTo(px, py);
            }
            decodeCtx.stroke();
        };
        drawPath(actual.x, actual.y, 'rgba(59, 130, 246, 0.18)');
        drawPath(model.xPred, model.yPred, 'rgba(239, 68, 68, 0.18)');

        // Traversed portion (bright), from reach start to current frame.
        const drawTrail = (xs, ys, color) => {
            decodeCtx.beginPath();
            decodeCtx.strokeStyle = color;
            decodeCtx.lineWidth = 2.5;
            for (let f = seg.start; f <= currentFrame; f++) {
                const px = sx(xs[f]), py = sy(ys[f]);
                if (f === seg.start) decodeCtx.moveTo(px, py);
                else decodeCtx.lineTo(px, py);
            }
            decodeCtx.stroke();
        };
        drawTrail(actual.x, actual.y, 'rgba(59, 130, 246, 0.85)');
        drawTrail(model.xPred, model.yPred, 'rgba(239, 68, 68, 0.85)');

        // Current cursor positions.
        const dot = (xs, ys, color) => {
            decodeCtx.beginPath();
            decodeCtx.arc(sx(xs[currentFrame]), sy(ys[currentFrame]), 6, 0, Math.PI * 2);
            decodeCtx.fillStyle = color;
            decodeCtx.fill();
        };
        dot(actual.x, actual.y, '#3B82F6');
        dot(model.xPred, model.yPred, '#EF4444');

        // Legend
        decodeCtx.font = '10px "JetBrains Mono"';
        decodeCtx.textAlign = 'left';
        decodeCtx.fillStyle = '#3B82F6';
        decodeCtx.fillRect(10, h - 18, 8, 8);
        decodeCtx.fillStyle = '#6B6B6B';
        decodeCtx.fillText('True', 22, h - 10);
        decodeCtx.fillStyle = '#EF4444';
        decodeCtx.fillRect(60, h - 18, 8, 8);
        decodeCtx.fillStyle = '#6B6B6B';
        decodeCtx.fillText('Decoded', 72, h - 10);
    }

    function updateReachLabel() {
        const idx = segOfFrame[currentFrame];
        document.getElementById('nlb-reach-label').textContent =
            `— reach ${idx + 1}/${segs.length}`;
    }

    function updateReadout() {
        const m = models[currentModel];
        document.getElementById('nlb-live-r2-value').textContent =
            m.velR2Mean != null ? m.velR2Mean.toFixed(3) : '—';
    }

    function drawAll() {
        drawNeural();
        drawHeatmap();
        drawDecode();
        updateReachLabel();
    }

    // --- Animation loop (time-based) ---
    function animate(ts) {
        if (!isPlaying) return;
        if (lastTs === null) lastTs = ts;
        const dt = Math.min((ts - lastTs) / 1000, 0.1);
        lastTs = ts;

        playhead += speed * BINS_PER_SEC_PER_SPEED * dt;
        if (playhead >= N - 1) playhead = 0;
        currentFrame = Math.floor(playhead);

        drawAll();
        animId = requestAnimationFrame(animate);
    }

    function startPlaying() {
        isPlaying = true;
        lastTs = null;
        document.getElementById('nlb-icon-play').style.display = 'none';
        document.getElementById('nlb-icon-pause').style.display = 'block';
        animId = requestAnimationFrame(animate);
    }

    function stopPlaying() {
        isPlaying = false;
        document.getElementById('nlb-icon-play').style.display = 'block';
        document.getElementById('nlb-icon-pause').style.display = 'none';
        if (animId) cancelAnimationFrame(animId);
    }

    // --- Result cards ---
    function fillResultCards() {
        document.querySelectorAll('.nlb-result-card[data-model]').forEach(card => {
            const m = models[card.dataset.model];
            if (!m) return;
            card.querySelector('.result-corr').textContent =
                m.velR2Mean != null ? m.velR2Mean.toFixed(3) : '—';
            const label = card.querySelector('.result-label');
            if (label && m.velR2Std != null) label.textContent = `± ${m.velR2Std.toFixed(3)} vel R² (5 seeds)`;
            const span = card.querySelector('.result-detail span');
            if (span && m.seedR2 != null) span.textContent = `replay seed: ${m.seedR2.toFixed(3)}`;
        });
    }

    // --- Event Listeners ---
    document.getElementById('nlb-btn-play').addEventListener('click', () => {
        if (isPlaying) stopPlaying(); else startPlaying();
    });
    document.getElementById('nlb-btn-reset').addEventListener('click', () => {
        stopPlaying();
        currentFrame = 0;
        playhead = 0;
        drawAll();
    });
    document.getElementById('nlb-speed-slider').addEventListener('input', (e) => {
        speed = parseInt(e.target.value, 10);
    });
    document.querySelectorAll('.nlb-model-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const key = tab.dataset.model;
            if (!models[key]) return;
            document.querySelector('.nlb-model-tab.active').classList.remove('active');
            tab.classList.add('active');
            currentModel = key;
            updateReadout();
            drawAll();
        });
    });
    window.addEventListener('resize', () => {
        resizeCanvases();
        drawAll();
    });

    // --- Init ---
    document.fonts.ready.then(() => {
        resizeCanvases();
        fillResultCards();
        updateReadout();
        drawAll();
        startPlaying();
    });
})();
