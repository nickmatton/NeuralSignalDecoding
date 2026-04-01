// ============================================================
// Neural Decoding Demo — Animated BCI Visualization
// ============================================================

(function () {
    // --- Seeded RNG for reproducible "neural" data ---
    let seed = 42;
    function seededRandom() {
        seed = (seed * 16807 + 0) % 2147483647;
        return (seed - 1) / 2147483646;
    }

    // --- Generate synthetic trajectory (center-out reaching task) ---
    function generateTrajectory(numPoints) {
        const xActual = new Float32Array(numPoints);
        const yActual = new Float32Array(numPoints);

        // Simulate center-out reaches: stay at center, then reach to a target, then return
        let x = 0, y = 0;
        let targetX = 0, targetY = 0;
        let reachPhase = 0; // 0=hold, 1=reach, 2=hold at target, 3=return
        let phaseTimer = 0;
        const holdTime = 20;
        const reachTime = 15;

        for (let i = 0; i < numPoints; i++) {
            phaseTimer++;

            if (reachPhase === 0 && phaseTimer > holdTime) {
                // Pick new target
                const angle = seededRandom() * Math.PI * 2;
                const dist = 60 + seededRandom() * 30;
                targetX = Math.cos(angle) * dist;
                targetY = Math.sin(angle) * dist;
                reachPhase = 1;
                phaseTimer = 0;
            } else if (reachPhase === 1 && phaseTimer > reachTime) {
                reachPhase = 2;
                phaseTimer = 0;
            } else if (reachPhase === 2 && phaseTimer > holdTime) {
                targetX = 0;
                targetY = 0;
                reachPhase = 3;
                phaseTimer = 0;
            } else if (reachPhase === 3 && phaseTimer > reachTime) {
                reachPhase = 0;
                phaseTimer = 0;
            }

            if (reachPhase === 1 || reachPhase === 3) {
                const t = phaseTimer / reachTime;
                const smooth = t * t * (3 - 2 * t); // smoothstep
                x += (targetX - x) * smooth * 0.15;
                y += (targetY - y) * smooth * 0.15;
            }

            // Add small noise
            x += (seededRandom() - 0.5) * 2;
            y += (seededRandom() - 0.5) * 2;

            xActual[i] = x;
            yActual[i] = y;
        }

        return { xActual, yActual };
    }

    // --- Generate model predictions with different quality levels ---
    function generatePrediction(actual, correlation, delay) {
        const n = actual.length;
        const pred = new Float32Array(n);
        const noiseScale = Math.sqrt(1 - correlation * correlation) * 40;

        for (let i = 0; i < n; i++) {
            const srcIdx = Math.max(0, i - delay);
            pred[i] = actual[srcIdx] * correlation + (seededRandom() - 0.5) * noiseScale;
        }

        // Smooth the prediction
        const smoothed = new Float32Array(n);
        const kernel = 3;
        for (let i = 0; i < n; i++) {
            let sum = 0, count = 0;
            for (let k = -kernel; k <= kernel; k++) {
                const idx = i + k;
                if (idx >= 0 && idx < n) {
                    sum += pred[idx];
                    count++;
                }
            }
            smoothed[i] = sum / count;
        }

        return smoothed;
    }

    // --- Generate neural raster data ---
    function generateNeuralData(xActual, yActual, numChannels) {
        const n = xActual.length;
        const raster = [];

        for (let ch = 0; ch < numChannels; ch++) {
            const channelData = new Float32Array(n);
            // Each channel has a preferred direction
            const prefAngle = (ch / numChannels) * Math.PI * 2;
            const prefX = Math.cos(prefAngle);
            const prefY = Math.sin(prefAngle);
            const gain = 0.3 + seededRandom() * 0.7;
            const baseRate = seededRandom() * 0.1;

            for (let i = 1; i < n; i++) {
                const vx = xActual[i] - xActual[Math.max(0, i - 1)];
                const vy = yActual[i] - yActual[Math.max(0, i - 1)];
                const tuning = (prefX * vx + prefY * vy) * gain;
                const rate = Math.max(0, baseRate + tuning * 0.05);
                channelData[i] = seededRandom() < rate ? 1 : 0;
            }
            raster.push(channelData);
        }

        return raster;
    }

    // --- Setup ---
    const NUM_POINTS = 500;
    const NUM_CHANNELS = 95;

    const traj = generateTrajectory(NUM_POINTS);

    const models = {
        lstm: {
            name: 'LSTM',
            corrX: 0.989, corrY: 0.986,
            xPred: null, yPred: null
        },
        cnn: {
            name: 'MLP',
            corrX: 0.878, corrY: 0.807,
            xPred: null, yPred: null
        },
        cnn2d: {
            name: '2D CNN',
            corrX: 0.608, corrY: 0.615,
            xPred: null, yPred: null
        },
        transformer: {
            name: 'Transformer',
            corrX: 0.683, corrY: 0.692,
            xPred: null, yPred: null
        }
    };

    // Generate predictions for each model
    seed = 100;
    models.lstm.xPred = generatePrediction(traj.xActual, 0.99, 1);
    models.lstm.yPred = generatePrediction(traj.yActual, 0.99, 1);

    seed = 200;
    models.cnn.xPred = generatePrediction(traj.xActual, 0.88, 1);
    models.cnn.yPred = generatePrediction(traj.yActual, 0.81, 1);

    seed = 250;
    models.cnn2d.xPred = generatePrediction(traj.xActual, 0.61, 1);
    models.cnn2d.yPred = generatePrediction(traj.yActual, 0.62, 1);

    seed = 300;
    models.transformer.xPred = generatePrediction(traj.xActual, 0.68, 2);
    models.transformer.yPred = generatePrediction(traj.yActual, 0.69, 2);

    seed = 42;
    const neuralData = generateNeuralData(traj.xActual, traj.yActual, NUM_CHANNELS);

    // --- Canvas setup ---
    const neuralCanvas = document.getElementById('neural-canvas');
    const decodeCanvas = document.getElementById('decode-canvas');
    const neuralCtx = neuralCanvas.getContext('2d');
    const decodeCtx = decodeCanvas.getContext('2d');

    function resizeCanvases() {
        const dpr = window.devicePixelRatio || 1;

        [neuralCanvas, decodeCanvas].forEach(canvas => {
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
    let speed = 5;
    const WINDOW = 80; // frames visible at a time

    // --- Draw neural raster ---
    function drawNeural() {
        const w = neuralCanvas.width / (window.devicePixelRatio || 1);
        const h = neuralCanvas.height / (window.devicePixelRatio || 1);
        neuralCtx.clearRect(0, 0, w, h);

        const startFrame = Math.max(0, currentFrame - WINDOW);
        const endFrame = currentFrame;
        const channelsToShow = Math.min(NUM_CHANNELS, 60); // show subset for clarity
        const rowH = (h - 30) / channelsToShow;
        const colW = (w - 10) / WINDOW;

        // Draw raster
        for (let ch = 0; ch < channelsToShow; ch++) {
            for (let f = startFrame; f < endFrame; f++) {
                if (neuralData[ch][f] > 0) {
                    const x = 5 + (f - startFrame) * colW;
                    const y = 25 + ch * rowH;
                    neuralCtx.fillStyle = 'rgba(26, 26, 26, 0.8)';
                    neuralCtx.fillRect(x, y, Math.max(1.5, colW - 0.5), Math.max(1.5, rowH - 0.5));
                }
            }
        }

        // Time indicator
        const timeStr = ((currentFrame * 50) / 1000).toFixed(1) + 's';
        neuralCtx.font = '10px "JetBrains Mono"';
        neuralCtx.fillStyle = '#6B6B6B';
        neuralCtx.textAlign = 'right';
        neuralCtx.fillText(timeStr, w - 8, h - 5);
    }

    // --- Draw decoded position ---
    function drawDecode() {
        const w = decodeCanvas.width / (window.devicePixelRatio || 1);
        const h = decodeCanvas.height / (window.devicePixelRatio || 1);
        decodeCtx.clearRect(0, 0, w, h);

        const model = models[currentModel];
        const midX = w / 2;
        const midY = h / 2;
        const scale = 1.2;

        // Draw grid/crosshair
        decodeCtx.strokeStyle = '#E5E4E0';
        decodeCtx.lineWidth = 1;
        decodeCtx.beginPath();
        decodeCtx.moveTo(midX, 25);
        decodeCtx.lineTo(midX, h - 10);
        decodeCtx.moveTo(10, midY);
        decodeCtx.lineTo(w - 10, midY);
        decodeCtx.stroke();

        // Draw target circle
        decodeCtx.beginPath();
        decodeCtx.arc(midX, midY, 80, 0, Math.PI * 2);
        decodeCtx.strokeStyle = '#E5E4E0';
        decodeCtx.stroke();

        // Trail length
        const trailLen = Math.min(40, currentFrame);
        const startF = currentFrame - trailLen;

        // Draw actual trajectory trail (blue)
        if (trailLen > 1) {
            decodeCtx.beginPath();
            decodeCtx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
            decodeCtx.lineWidth = 2;
            for (let f = startF; f < currentFrame; f++) {
                const px = midX + traj.xActual[f] * scale;
                const py = midY - traj.yActual[f] * scale;
                if (f === startF) decodeCtx.moveTo(px, py);
                else decodeCtx.lineTo(px, py);
            }
            decodeCtx.stroke();

            // Draw predicted trajectory trail (red)
            decodeCtx.beginPath();
            decodeCtx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
            decodeCtx.lineWidth = 2;
            for (let f = startF; f < currentFrame; f++) {
                const px = midX + model.xPred[f] * scale;
                const py = midY - model.yPred[f] * scale;
                if (f === startF) decodeCtx.moveTo(px, py);
                else decodeCtx.lineTo(px, py);
            }
            decodeCtx.stroke();
        }

        // Draw current actual position (blue dot)
        if (currentFrame > 0) {
            const ax = midX + traj.xActual[currentFrame] * scale;
            const ay = midY - traj.yActual[currentFrame] * scale;
            decodeCtx.beginPath();
            decodeCtx.arc(ax, ay, 6, 0, Math.PI * 2);
            decodeCtx.fillStyle = '#3B82F6';
            decodeCtx.fill();

            // Draw current predicted position (red dot)
            const px = midX + model.xPred[currentFrame] * scale;
            const py = midY - model.yPred[currentFrame] * scale;
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
        const avgCorr = ((Math.abs(model.corrX) + Math.abs(model.corrY)) / 2).toFixed(2);
        document.getElementById('live-corr-value').textContent = avgCorr;
    }

    // --- Animation loop ---
    function animate() {
        if (!isPlaying) return;

        currentFrame += speed;
        if (currentFrame >= NUM_POINTS - 1) {
            currentFrame = 0;
        }

        drawNeural();
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

    // --- Bar Chart ---
    function drawBarChart() {
        const canvas = document.getElementById('bar-chart');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = (rect.height - 48) * dpr; // account for padding
        canvas.style.width = rect.width + 'px';
        canvas.style.height = (rect.height - 48) + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const w = rect.width;
        const h = rect.height - 48;

        const metrics = [
            { label: 'X Position', mlp: 0.878, cnn2d: 0.608, lstm: 0.989, trans: 0.683 },
            { label: 'Y Position', mlp: 0.807, cnn2d: 0.615, lstm: 0.986, trans: 0.692 },
            { label: 'X Velocity', mlp: 0.917, cnn2d: 0.969, lstm: 0.987, trans: 0.787 },
            { label: 'Y Velocity', mlp: 0.902, cnn2d: 0.960, lstm: 0.985, trans: 0.798 },
            { label: 'Average', mlp: 0.876, cnn2d: 0.788, lstm: 0.987, trans: 0.740 }
        ];

        const leftPad = 80;
        const rightPad = 20;
        const topPad = 20;
        const bottomPad = 40;
        const chartW = w - leftPad - rightPad;
        const chartH = h - topPad - bottomPad;

        const groupCount = metrics.length;
        const groupW = chartW / groupCount;
        const barW = groupW * 0.17;
        const barGap = 2;

        // Y axis
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

        // Zero line
        const zeroY = topPad + chartH;
        ctx.strokeStyle = var_text_primary();
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leftPad, zeroY);
        ctx.lineTo(w - rightPad, zeroY);
        ctx.stroke();

        const colors = { mlp: '#6B6B6B', cnn2d: '#3B82F6', lstm: '#1A1A1A', trans: '#A0A0A0' };

        metrics.forEach((m, i) => {
            const groupX = leftPad + i * groupW + groupW / 2;
            const totalW = barW * 4 + barGap * 3;
            const startX = groupX - totalW / 2;

            // MLP bar
            const mlpH = Math.max(0, m.mlp) * chartH;
            ctx.fillStyle = colors.mlp;
            ctx.fillRect(startX, zeroY - mlpH, barW, mlpH);

            // 2D CNN bar
            const cnn2dH = Math.max(0, m.cnn2d) * chartH;
            ctx.fillStyle = colors.cnn2d;
            ctx.fillRect(startX + barW + barGap, zeroY - cnn2dH, barW, cnn2dH);

            // LSTM bar
            const lstmH = Math.max(0, m.lstm) * chartH;
            ctx.fillStyle = colors.lstm;
            ctx.fillRect(startX + (barW + barGap) * 2, zeroY - lstmH, barW, lstmH);

            // Transformer bar
            const transH = Math.max(0, m.trans) * chartH;
            ctx.fillStyle = colors.trans;
            ctx.fillRect(startX + (barW + barGap) * 3, zeroY - transH, barW, transH);

            // Label
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillStyle = '#6B6B6B';
            ctx.textAlign = 'center';
            ctx.fillText(m.label, groupX, zeroY + 16);
        });

        // Legend
        const legendX = leftPad + 10;
        const legendY = topPad + 5;
        const entries = [
            { label: 'MLP', color: colors.mlp },
            { label: '2D CNN', color: colors.cnn2d },
            { label: 'LSTM', color: colors.lstm },
            { label: 'Transformer', color: colors.trans }
        ];
        entries.forEach((e, i) => {
            ctx.fillStyle = e.color;
            ctx.fillRect(legendX + i * 90, legendY, 10, 10);
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillStyle = '#6B6B6B';
            ctx.textAlign = 'left';
            ctx.fillText(e.label, legendX + i * 90 + 14, legendY + 9);
        });
    }

    function var_text_primary() { return '#1A1A1A'; }

    // --- Event Listeners ---
    document.getElementById('btn-play').addEventListener('click', () => {
        if (isPlaying) stopPlaying();
        else startPlaying();
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        stopPlaying();
        currentFrame = 0;
        drawNeural();
        drawDecode();
    });

    document.getElementById('speed-slider').addEventListener('input', (e) => {
        speed = parseInt(e.target.value, 10);
    });

    document.querySelectorAll('.model-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelector('.model-tab.active').classList.remove('active');
            tab.classList.add('active');
            currentModel = tab.dataset.model;
            seed = currentModel === 'lstm' ? 100 : currentModel === 'cnn' ? 200 : currentModel === 'cnn2d' ? 250 : 300;
            updateCorrelation();
            drawNeural();
            drawDecode();
        });
    });

    // --- Init ---
    window.addEventListener('resize', () => {
        resizeCanvases();
        drawNeural();
        drawDecode();
        drawBarChart();
    });

    // Wait for fonts to load
    document.fonts.ready.then(() => {
        resizeCanvases();
        updateCorrelation();
        drawNeural();
        drawDecode();
        drawBarChart();
        // Auto-play
        startPlaying();
    });
})();
