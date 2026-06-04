// ============================================================
// Neural Decoding Demo — Animated BCI Visualization
//
// Renders REAL held-out test-set data exported by the training
// scripts (web-demo/results.js → window.DEMO_RESULTS):
//   • real hand trajectory (actual kinematics)
//   • real per-model decoded trajectories
//   • real 95-channel spike-count neural data
// Nothing here is synthetic — see train_cnn.py / train_all.py.
// ============================================================

(function () {
    const R = window.DEMO_RESULTS;
    if (!R || !R.base) {
        console.error('DEMO_RESULTS missing — run train_cnn.py then train_all.py to generate web-demo/results.js');
        return;
    }

    const NUM_POINTS = R.base.n_points;
    const NUM_CHANNELS = R.base.n_channels;

    // --- Real hand trajectory (held-out test set) ---
    const traj = {
        xActual: Float32Array.from(R.base.actual.x_pos),
        yActual: Float32Array.from(R.base.actual.y_pos),
    };

    // --- Real neural data: channels × time spike counts (z-scored) ---
    const ecogData = R.base.neural.map(row => Float32Array.from(row));

    // Display scaling derived from the real data ranges.
    let neuralAbsMax = 1e-6, neuralMin = Infinity, neuralMax = -Infinity;
    for (let ch = 0; ch < NUM_CHANNELS; ch++) {
        for (let f = 0; f < NUM_POINTS; f++) {
            const v = ecogData[ch][f];
            if (Math.abs(v) > neuralAbsMax) neuralAbsMax = Math.abs(v);
            if (v < neuralMin) neuralMin = v;
            if (v > neuralMax) neuralMax = v;
        }
    }
    const neuralRange = (neuralMax - neuralMin) || 1;

    // --- Real per-model decoded trajectories + correlations ---
    // Display order for tabs / chart / cards.
    const MODEL_ORDER = ['lstm', 'mlp', 'cnn2d', 'transformer'];
    const models = {};
    for (const key of MODEL_ORDER) {
        const m = R.models[key];
        if (!m) continue;
        models[key] = {
            name: m.name,
            corr: m.corr,
            xPred: Float32Array.from(m.pred.x_pos),
            yPred: Float32Array.from(m.pred.y_pos),
        };
    }

    // --- Decode-view geometry ---
    // The hand trajectory is a center-out reaching pattern: it sits near its
    // mean most of the time (std ~28) with occasional reaches to ±100. Center
    // the view on the actual-trajectory mean and scale by a robust spread (a
    // high percentile of the centered extent across actual + all predictions),
    // NOT the global max — otherwise the rare reaches squish typical motion
    // into a blob at the origin.
    let viewCx = 0, viewCy = 0;
    for (let i = 0; i < NUM_POINTS; i++) { viewCx += traj.xActual[i]; viewCy += traj.yActual[i]; }
    viewCx /= NUM_POINTS; viewCy /= NUM_POINTS;

    const radii = [];
    const pushRadius = (x, y) => radii.push(Math.max(Math.abs(x - viewCx), Math.abs(y - viewCy)));
    for (let i = 0; i < NUM_POINTS; i++) pushRadius(traj.xActual[i], traj.yActual[i]);
    for (const key of MODEL_ORDER) {
        const m = models[key];
        if (!m) continue;
        for (let i = 0; i < NUM_POINTS; i++) pushRadius(m.xPred[i], m.yPred[i]);
    }
    radii.sort((a, b) => a - b);
    // 92nd percentile: typical motion fills the panel; the biggest reaches may
    // extend slightly past the edge (acceptable).
    const viewRadius = Math.max(radii[Math.floor(radii.length * 0.92)] || 1, 1e-6);

    // --- Canvas setup ---
    const neuralCanvas = document.getElementById('neural-canvas');
    const heatmapCanvas = document.getElementById('heatmap-canvas');
    const decodeCanvas = document.getElementById('decode-canvas');
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
    let currentModel = 'lstm';
    let currentFrame = 0;
    let isPlaying = false;
    let animId = null;
    let speed = 2;
    const WINDOW = 80; // frames visible at a time

    // --- Draw spike-count traces (subset of channels) ---
    function drawNeural() {
        const w = neuralCanvas.width / (window.devicePixelRatio || 1);
        const h = neuralCanvas.height / (window.devicePixelRatio || 1);
        neuralCtx.clearRect(0, 0, w, h);

        const startFrame = Math.max(0, currentFrame - WINDOW);
        const endFrame = currentFrame;
        const channelsToShow = Math.min(NUM_CHANNELS, 24); // show subset for clarity
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
            for (let f = startFrame; f < endFrame; f++) {
                const x = 5 + (f - startFrame) * colW;
                const y = midY - ecogData[ch][f] * amp;
                if (!started) { neuralCtx.moveTo(x, y); started = true; }
                else neuralCtx.lineTo(x, y);
            }
            neuralCtx.stroke();
        }

        const timeStr = ((currentFrame * 50) / 1000).toFixed(1) + 's';
        neuralCtx.font = '10px "JetBrains Mono"';
        neuralCtx.fillStyle = '#6B6B6B';
        neuralCtx.textAlign = 'right';
        neuralCtx.fillText(timeStr, w - 8, h - 5);
    }

    // --- Draw spike-count heatmap (all 95 channels × time) ---
    function drawHeatmap() {
        const w = heatmapCanvas.width / (window.devicePixelRatio || 1);
        const h = heatmapCanvas.height / (window.devicePixelRatio || 1);
        heatmapCtx.clearRect(0, 0, w, h);

        const startFrame = Math.max(0, currentFrame - WINDOW);
        const endFrame = currentFrame;
        const top = 20;
        const rowH = (h - top - 6) / NUM_CHANNELS;
        const colW = (w - 10) / WINDOW;

        // Sequential light→blue colormap over the real value range.
        for (let ch = 0; ch < NUM_CHANNELS; ch++) {
            for (let f = startFrame; f < endFrame; f++) {
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

    // --- Draw decoded position ---
    function drawDecode() {
        const w = decodeCanvas.width / (window.devicePixelRatio || 1);
        const h = decodeCanvas.height / (window.devicePixelRatio || 1);
        decodeCtx.clearRect(0, 0, w, h);

        const model = models[currentModel];
        const midX = w / 2;
        const midY = h / 2;
        // Map the robust spread (centered on the trajectory mean) to ~45% of the
        // smaller canvas dimension so typical motion fills the panel.
        const scale = (Math.min(w, h) * 0.45) / viewRadius;
        const sx = (x) => midX + (x - viewCx) * scale;
        const sy = (y) => midY - (y - viewCy) * scale;

        // Grid/crosshair
        decodeCtx.strokeStyle = '#E5E4E0';
        decodeCtx.lineWidth = 1;
        decodeCtx.beginPath();
        decodeCtx.moveTo(midX, 25);
        decodeCtx.lineTo(midX, h - 10);
        decodeCtx.moveTo(10, midY);
        decodeCtx.lineTo(w - 10, midY);
        decodeCtx.stroke();

        // Trail
        const trailLen = Math.min(32, currentFrame);
        const startF = currentFrame - trailLen;

        if (trailLen > 1) {
            decodeCtx.beginPath();
            decodeCtx.strokeStyle = 'rgba(59, 130, 246, 0.35)';
            decodeCtx.lineWidth = 2;
            for (let f = startF; f < currentFrame; f++) {
                const px = sx(traj.xActual[f]);
                const py = sy(traj.yActual[f]);
                if (f === startF) decodeCtx.moveTo(px, py);
                else decodeCtx.lineTo(px, py);
            }
            decodeCtx.stroke();

            decodeCtx.beginPath();
            decodeCtx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
            decodeCtx.lineWidth = 2;
            for (let f = startF; f < currentFrame; f++) {
                const px = sx(model.xPred[f]);
                const py = sy(model.yPred[f]);
                if (f === startF) decodeCtx.moveTo(px, py);
                else decodeCtx.lineTo(px, py);
            }
            decodeCtx.stroke();
        }

        if (currentFrame > 0) {
            const ax = sx(traj.xActual[currentFrame]);
            const ay = sy(traj.yActual[currentFrame]);
            decodeCtx.beginPath();
            decodeCtx.arc(ax, ay, 6, 0, Math.PI * 2);
            decodeCtx.fillStyle = '#3B82F6';
            decodeCtx.fill();

            const px = sx(model.xPred[currentFrame]);
            const py = sy(model.yPred[currentFrame]);
            decodeCtx.beginPath();
            decodeCtx.arc(px, py, 6, 0, Math.PI * 2);
            decodeCtx.fillStyle = '#EF4444';
            decodeCtx.fill();
        }

        // Legend
        decodeCtx.font = '10px "JetBrains Mono"';
        decodeCtx.textAlign = 'left';
        decodeCtx.fillStyle = '#3B82F6';
        decodeCtx.fillRect(10, h - 18, 8, 8);
        decodeCtx.fillStyle = '#6B6B6B';
        decodeCtx.fillText('Actual', 22, h - 10);

        decodeCtx.fillStyle = '#EF4444';
        decodeCtx.fillRect(75, h - 18, 8, 8);
        decodeCtx.fillStyle = '#6B6B6B';
        decodeCtx.fillText('Predicted', 87, h - 10);
    }

    // --- Update correlation display ---
    function updateCorrelation() {
        const model = models[currentModel];
        document.getElementById('live-corr-value').textContent = model.corr.average.toFixed(3);
    }

    // --- Animation loop ---
    function animate() {
        if (!isPlaying) return;
        currentFrame += speed;
        if (currentFrame >= NUM_POINTS - 1) currentFrame = 0;
        drawNeural();
        drawHeatmap();
        drawDecode();
        animId = requestAnimationFrame(animate);
    }

    function startPlaying() {
        isPlaying = true;
        document.getElementById('icon-play').style.display = 'none';
        document.getElementById('icon-pause').style.display = 'block';
        animate();
    }

    function stopPlaying() {
        isPlaying = false;
        document.getElementById('icon-play').style.display = 'block';
        document.getElementById('icon-pause').style.display = 'none';
        if (animId) cancelAnimationFrame(animId);
    }

    // --- Bar Chart (real correlations) ---
    const METRIC_KEYS = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'average'];
    const METRIC_LABELS = ['X Position', 'Y Position', 'X Velocity', 'Y Velocity', 'Average'];
    const BAR_COLORS = { mlp: '#6B6B6B', cnn2d: '#3B82F6', lstm: '#1A1A1A', transformer: '#A0A0A0' };

    function drawBarChart() {
        const canvas = document.getElementById('bar-chart');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = (rect.height - 48) * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = (rect.height - 48) + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const w = rect.width;
        const h = rect.height - 48;
        const leftPad = 80, rightPad = 20, topPad = 20, bottomPad = 40;
        const chartW = w - leftPad - rightPad;
        const chartH = h - topPad - bottomPad;
        const groupW = chartW / METRIC_KEYS.length;
        const barW = groupW * 0.17;
        const barGap = 2;
        const barModels = MODEL_ORDER.filter(k => models[k]);

        // Y axis gridlines
        ctx.strokeStyle = '#E5E4E0';
        ctx.lineWidth = 1;
        for (let v = 0; v <= 1; v += 0.25) {
            const y = topPad + chartH * (1 - v);
            ctx.beginPath();
            ctx.moveTo(leftPad, y);
            ctx.lineTo(w - rightPad, y);
            ctx.stroke();
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillStyle = '#6B6B6B';
            ctx.textAlign = 'right';
            ctx.fillText(v.toFixed(2), leftPad - 8, y + 3);
        }

        const zeroY = topPad + chartH;
        ctx.strokeStyle = '#1A1A1A';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leftPad, zeroY);
        ctx.lineTo(w - rightPad, zeroY);
        ctx.stroke();

        METRIC_KEYS.forEach((metric, i) => {
            const groupX = leftPad + i * groupW + groupW / 2;
            const totalW = barW * barModels.length + barGap * (barModels.length - 1);
            const startX = groupX - totalW / 2;
            barModels.forEach((key, j) => {
                const val = Math.max(0, models[key].corr[metric]);
                const barH = val * chartH;
                ctx.fillStyle = BAR_COLORS[key];
                ctx.fillRect(startX + j * (barW + barGap), zeroY - barH, barW, barH);
            });
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillStyle = '#6B6B6B';
            ctx.textAlign = 'center';
            ctx.fillText(METRIC_LABELS[i], groupX, zeroY + 16);
        });

        // Legend
        const legendX = leftPad + 10;
        const legendY = topPad + 5;
        barModels.forEach((key, i) => {
            ctx.fillStyle = BAR_COLORS[key];
            ctx.fillRect(legendX + i * 90, legendY, 10, 10);
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillStyle = '#6B6B6B';
            ctx.textAlign = 'left';
            ctx.fillText(models[key].name, legendX + i * 90 + 14, legendY + 9);
        });
    }

    // --- Populate the result cards from real correlations ---
    function fillResultCards() {
        document.querySelectorAll('.result-card[data-model]').forEach(card => {
            const m = models[card.dataset.model];
            if (!m) return;
            card.querySelector('.result-corr').textContent = m.corr.average.toFixed(3);
            const spans = card.querySelectorAll('.result-detail span');
            const vals = [m.corr.x_pos, m.corr.y_pos, m.corr.x_vel, m.corr.y_vel];
            const lbls = ['X pos', 'Y pos', 'X vel', 'Y vel'];
            spans.forEach((s, i) => { if (i < 4) s.textContent = `${lbls[i]}: ${vals[i].toFixed(3)}`; });
        });
    }

    // --- Event Listeners ---
    document.getElementById('btn-play').addEventListener('click', () => {
        if (isPlaying) stopPlaying();
        else startPlaying();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        stopPlaying();
        currentFrame = 0;
        drawNeural();
        drawHeatmap();
        drawDecode();
    });

    document.getElementById('speed-slider').addEventListener('input', (e) => {
        speed = parseInt(e.target.value, 10);
    });

    document.querySelectorAll('.model-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const key = tab.dataset.model;
            if (!models[key]) return;
            document.querySelector('.model-tab.active').classList.remove('active');
            tab.classList.add('active');
            currentModel = key;
            updateCorrelation();
            drawNeural();
            drawHeatmap();
            drawDecode();
        });
    });

    window.addEventListener('resize', () => {
        resizeCanvases();
        drawNeural();
        drawHeatmap();
        drawDecode();
        drawBarChart();
    });

    // --- Init ---
    document.fonts.ready.then(() => {
        resizeCanvases();
        fillResultCards();
        updateCorrelation();
        drawNeural();
        drawHeatmap();
        drawDecode();
        drawBarChart();
        startPlaying();
    });
})();
