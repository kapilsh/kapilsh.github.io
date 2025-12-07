// Hopper GEMM Pipeline Timeline Visualization
// ============================================================================

const commonPipelineStyles = `
    <style>
        .pipeline-container * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .pipeline-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #111827;
            color: #ffffff;
            padding: 2rem;
            border-radius: 0.5rem;
        }

        .pipeline-title {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #c084fc, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .pipeline-subtitle {
            text-align: center;
            color: #9ca3af;
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }

        .pipeline-controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .pipeline-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.2s;
            color: white;
        }

        .pipeline-btn-primary {
            background-color: #9333ea;
        }

        .pipeline-btn-primary:hover {
            background-color: #7e22ce;
        }

        .pipeline-btn-secondary {
            background-color: #374151;
        }

        .pipeline-btn-secondary:hover {
            background-color: #4b5563;
        }

        .pipeline-time-display {
            text-align: center;
            margin-bottom: 2rem;
        }

        .pipeline-time-value {
            font-size: 1.5rem;
            font-family: monospace;
        }

        .pipeline-time-value .current {
            color: #a78bfa;
        }

        .pipeline-time-info {
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        .pipeline-active-ops {
            font-size: 0.875rem;
            color: #22d3ee;
            margin-top: 0.5rem;
        }

        .pipeline-card {
            background-color: #1f2937;
            border-radius: 0.5rem;
            padding: 1.5rem;
            border: 2px solid #374151;
            margin-bottom: 1.5rem;
        }

        .pipeline-card-title {
            font-size: 1.125rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .pipeline-legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .pipeline-legend-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .pipeline-legend-icon {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .pipeline-legend-text {
            flex: 1;
        }

        .pipeline-legend-name {
            font-weight: bold;
        }

        .pipeline-legend-duration {
            font-size: 0.75rem;
            color: #9ca3af;
        }

        .pipeline-timeline-container {
            background-color: #1f2937;
            border-radius: 0.5rem;
            padding: 1.5rem;
            border: 2px solid #374151;
            overflow-x: auto;
            margin-bottom: 2rem;
        }

        .pipeline-timeline-wrapper {
            min-width: 880px;
        }

        .pipeline-time-axis {
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
            margin-left: 12rem;
        }

        .pipeline-time-label {
            font-size: 0.875rem;
            color: #9ca3af;
            font-weight: bold;
            margin-right: 1rem;
        }

        .pipeline-time-track {
            position: relative;
            flex: 1;
            height: 2rem;
            background-color: #111827;
            border-radius: 0.25rem;
            border: 1px solid #4b5563;
        }

        .pipeline-time-marker {
            position: absolute;
            top: -2rem;
            font-size: 0.75rem;
            color: #6b7280;
            transform: translateX(-0.5rem);
        }

        .pipeline-time-line {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 1px;
            background-color: #4b5563;
        }

        .pipeline-current-time-marker {
            position: absolute;
            top: 0;
            bottom: 0;
            width: 4px;
            background-color: #fbbf24;
            transition: left 0.04s linear;
            z-index: 10;
        }

        .pipeline-current-time-arrow {
            position: absolute;
            top: -2rem;
            left: 50%;
            transform: translateX(-50%);
            color: #fbbf24;
            font-size: 1.5rem;
        }

        .pipeline-stream-lane {
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }

        .pipeline-stream-label {
            width: 11rem;
            padding-right: 1rem;
        }

        .pipeline-stream-name {
            font-size: 0.875rem;
            font-weight: bold;
        }

        .pipeline-stream-type {
            font-size: 0.75rem;
            color: #6b7280;
        }

        .pipeline-producer .pipeline-stream-name {
            color: #22d3ee;
        }

        .pipeline-consumer .pipeline-stream-name {
            color: #a78bfa;
        }

        .pipeline-lane-track {
            position: relative;
            flex: 1;
            height: 5rem;
            background-color: #111827;
            border-radius: 0.25rem;
            border: 1px solid #374151;
        }

        .pipeline-operation {
            position: absolute;
            top: 0.25rem;
            bottom: 0.25rem;
            border-radius: 0.5rem;
            border: 2px solid #4b5563;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: opacity 0.3s, border-color 0.3s, box-shadow 0.3s;
            overflow: hidden;
        }

        .pipeline-operation.active {
            border-color: #ffffff;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
            z-index: 5;
        }

        .pipeline-operation-progress {
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.2);
            transition: width 0.04s linear;
        }

        .pipeline-operation-content {
            position: relative;
            z-index: 2;
            text-align: center;
            padding: 0 0.5rem;
        }

        .pipeline-operation-icon {
            font-size: 1.5rem;
        }

        .pipeline-operation-label {
            font-size: 0.875rem;
            font-weight: bold;
            margin-top: 0.25rem;
        }

        .pipeline-operation-pulse {
            position: absolute;
            inset: 0;
            background-color: rgba(255, 255, 255, 0.1);
            animation: pipeline-pulse 1s infinite;
        }

        @keyframes pipeline-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .pipeline-bg-blue-500 { background-color: #3b82f6; }
        .pipeline-bg-cyan-500 { background-color: #06b6d4; }
        .pipeline-bg-purple-500 { background-color: #a855f7; }
        .pipeline-bg-orange-500 { background-color: #f97316; }
        .pipeline-bg-green-500 { background-color: #22c55e; }

        .pipeline-info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }

        .pipeline-info-card {
            background-color: #1f2937;
            border-radius: 0.5rem;
            padding: 1.5rem;
            border: 2px solid #374151;
        }

        .pipeline-info-card.cyan { border-color: #22d3ee; }
        .pipeline-info-card.purple { border-color: #a78bfa; }

        .pipeline-info-title {
            font-size: 1.25rem;
            font-weight: bold;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .pipeline-info-title.cyan { color: #22d3ee; }
        .pipeline-info-title.purple { color: #a78bfa; }

        .pipeline-info-content {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            font-size: 0.875rem;
        }

        .pipeline-info-item {
            color: #d1d5db;
        }

        .pipeline-info-label {
            font-weight: bold;
        }

        .pipeline-insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            font-size: 0.875rem;
            margin-top: 1rem;
        }

        .pipeline-insight-item {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .pipeline-insight-title {
            font-weight: bold;
        }

        .pipeline-insight-title.yellow { color: #fbbf24; }
        .pipeline-insight-title.green { color: #22c55e; }
        .pipeline-insight-title.pink { color: #ec4899; }

        .pipeline-insight-text {
            color: #9ca3af;
        }
    </style>
`;

class HopperGEMMPipelineViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        // Configuration
        this.config = {
            maxTime: 400,
            timeScale: 2.2,
            numTiles: 8,
            animationInterval: 40,
        };

        this.opTypes = {
            TMA_A: { name: 'TMA Load A', color: 'pipeline-bg-blue-500', duration: 25, icon: '📥' },
            TMA_B: { name: 'TMA Load B', color: 'pipeline-bg-cyan-500', duration: 25, icon: '📥' },
            WGMMA: { name: 'WGMMA', color: 'pipeline-bg-purple-500', duration: 40, icon: '⚡' },
            EPILOGUE: { name: 'Epilogue', color: 'pipeline-bg-orange-500', duration: 15, icon: '🔧' },
            GMEM_WRITE: { name: 'Write GMEM', color: 'pipeline-bg-green-500', duration: 20, icon: '📤' },
        };

        this.streams = [
            { id: 'producer_0', name: 'Producer Warp 0', type: 'producer' },
            { id: 'producer_1', name: 'Producer Warp 1', type: 'producer' },
            { id: 'consumer_0', name: 'Consumer Warp 0', type: 'consumer' },
            { id: 'consumer_1', name: 'Consumer Warp 1', type: 'consumer' },
            { id: 'consumer_2', name: 'Consumer Warp 2', type: 'consumer' },
            { id: 'consumer_3', name: 'Consumer Warp 3', type: 'consumer' },
        ];

        // State
        this.state = {
            isPlaying: false,
            currentTime: 0,
            operations: [],
            animationTimer: null,
        };

        this.init();
    }

    init() {
        this.container.innerHTML = commonPipelineStyles + `
            <div class="pipeline-container">
                <h1 class="pipeline-title">Hopper GEMM Pipeline Timeline</h1>
                <p class="pipeline-subtitle">TMA Async Loads → WGMMA Compute → Epilogue → GMEM Write</p>

                <!-- Controls -->
                <div class="pipeline-controls">
                    <button id="pipelinePlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="pipelinePlayIcon">▶</span>
                        <span id="pipelinePlayText">Play</span>
                    </button>
                    <button id="pipelineResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <!-- Time Display -->
                <div class="pipeline-time-display">
                    <div class="pipeline-time-value">
                        Time: <span class="current" id="pipelineCurrentTime">0</span> / <span id="pipelineMaxTime">${this.config.maxTime}</span>
                    </div>
                    <div class="pipeline-time-info">(1 unit ≈ 10 GPU cycles)</div>
                    <div class="pipeline-active-ops">Active Operations: <span id="pipelineActiveOps">0</span></div>
                </div>

                <!-- Legend -->
                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">Operation Types</h3>
                    <div class="pipeline-legend-grid" id="pipelineLegendGrid"></div>
                </div>

                <!-- Timeline -->
                <div class="pipeline-timeline-container">
                    <div class="pipeline-timeline-wrapper" id="pipelineTimelineWrapper">
                        <div class="pipeline-time-axis">
                            <div class="pipeline-time-label">Time →</div>
                            <div class="pipeline-time-track" id="pipelineTimeTrack"></div>
                        </div>
                        <div id="pipelineStreamLanes"></div>
                    </div>
                </div>

                <!-- Info Cards -->
                <div class="pipeline-info-grid">
                    <div class="pipeline-info-card cyan">
                        <h3 class="pipeline-info-title cyan">
                            <span>💾</span>
                            Producer Stage (TMA)
                        </h3>
                        <div class="pipeline-info-content">
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Warp Specialization:</span> Dedicated producer warps issue TMA instructions</p>
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Async Copy:</span> GMEM → SMEM without register usage</p>
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Overlap:</span> Multiple tiles can load simultaneously</p>
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Latency Hiding:</span> ~150-200 cycles hidden by pipelining</p>
                        </div>
                    </div>

                    <div class="pipeline-info-card purple">
                        <h3 class="pipeline-info-title purple">
                            <span>🖥️</span>
                            Consumer Stage (WGMMA)
                        </h3>
                        <div class="pipeline-info-content">
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Tensor Cores:</span> 4th gen tensor cores with warpgroup MMA</p>
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Dependencies:</span> Waits for TMA completion via async barriers</p>
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Throughput:</span> 989 TFLOPS FP16 on H100 (80GB)</p>
                            <p class="pipeline-info-item"><span class="pipeline-info-label">Occupancy:</span> Multiple consumer warps maximize utilization</p>
                        </div>
                    </div>
                </div>

                <!-- Key Insights -->
                <div class="pipeline-card" style="margin-top: 1.5rem;">
                    <h3 class="pipeline-card-title">🎯 Key Pipeline Insights</h3>
                    <div class="pipeline-insights-grid">
                        <div class="pipeline-insight-item">
                            <div class="pipeline-insight-title yellow">Pipelining</div>
                            <p class="pipeline-insight-text">While consumer warps compute tile N, producer warps asynchronously load tiles N+1 and N+2, achieving near-perfect overlap.</p>
                        </div>
                        <div class="pipeline-insight-item">
                            <div class="pipeline-insight-title green">Zero Overhead</div>
                            <p class="pipeline-insight-text">TMA doesn't use registers, allowing 100% of register file for computation. Async barriers coordinate without wasting cycles.</p>
                        </div>
                        <div class="pipeline-insight-item">
                            <div class="pipeline-insight-title pink">Software Pipelining</div>
                            <p class="pipeline-insight-text">CUTLASS 3.x uses CollectiveBuilder to automatically generate optimal pipeline schedules with configurable stage counts.</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.state.operations = this.initializeOperations();
        this.initializeLegend();
        this.initializeTimeTrack();
        this.initializeStreamLanes();
        this.updateVisualization();

        // Event listeners
        document.getElementById('pipelinePlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('pipelineResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    initializeOperations() {
        const ops = [];
        let opId = 0;

        for (let tile = 0; tile < this.config.numTiles; tile++) {
            const baseTime = tile * 45;

            // TMA Load A
            ops.push({
                id: opId++,
                streamId: 'producer_0',
                type: 'TMA_A',
                startTime: baseTime,
                endTime: baseTime + this.opTypes.TMA_A.duration,
                tile: tile,
                label: `A${tile}`,
            });

            // TMA Load B
            ops.push({
                id: opId++,
                streamId: 'producer_1',
                type: 'TMA_B',
                startTime: baseTime + 2,
                endTime: baseTime + 2 + this.opTypes.TMA_B.duration,
                tile: tile,
                label: `B${tile}`,
            });

            // WGMMA
            const wgmmaStart = baseTime + this.opTypes.TMA_A.duration + 3;
            const consumerWarp = tile % 4;
            ops.push({
                id: opId++,
                streamId: `consumer_${consumerWarp}`,
                type: 'WGMMA',
                startTime: wgmmaStart,
                endTime: wgmmaStart + this.opTypes.WGMMA.duration,
                tile: tile,
                label: `C${tile}`,
            });

            // Epilogue
            const epilogueStart = wgmmaStart + this.opTypes.WGMMA.duration + 1;
            ops.push({
                id: opId++,
                streamId: `consumer_${consumerWarp}`,
                type: 'EPILOGUE',
                startTime: epilogueStart,
                endTime: epilogueStart + this.opTypes.EPILOGUE.duration,
                tile: tile,
                label: `E${tile}`,
            });

            // GMEM Write
            const writeStart = epilogueStart + this.opTypes.EPILOGUE.duration + 1;
            ops.push({
                id: opId++,
                streamId: `consumer_${consumerWarp}`,
                type: 'GMEM_WRITE',
                startTime: writeStart,
                endTime: writeStart + this.opTypes.GMEM_WRITE.duration,
                tile: tile,
                label: `W${tile}`,
            });
        }

        return ops;
    }

    initializeLegend() {
        const legendGrid = document.getElementById('pipelineLegendGrid');
        legendGrid.innerHTML = '';

        Object.entries(this.opTypes).forEach(([, type]) => {
            const item = document.createElement('div');
            item.className = 'pipeline-legend-item';
            item.innerHTML = `
                <div class="pipeline-legend-icon ${type.color}">${type.icon}</div>
                <div class="pipeline-legend-text">
                    <div class="pipeline-legend-name">${type.name}</div>
                    <div class="pipeline-legend-duration">${type.duration} units</div>
                </div>
            `;
            legendGrid.appendChild(item);
        });
    }

    initializeTimeTrack() {
        const timeTrack = document.getElementById('pipelineTimeTrack');
        const numMarkers = Math.floor(this.config.maxTime / 40) + 1;

        timeTrack.innerHTML = '';

        // Time markers
        for (let i = 0; i < numMarkers; i++) {
            const marker = document.createElement('div');
            marker.className = 'pipeline-time-marker';
            marker.style.left = `${i * 40 * this.config.timeScale}px`;
            marker.textContent = i * 40;
            timeTrack.appendChild(marker);

            const line = document.createElement('div');
            line.className = 'pipeline-time-line';
            line.style.left = `${i * 40 * this.config.timeScale}px`;
            timeTrack.appendChild(line);
        }

        // Current time marker
        const currentMarker = document.createElement('div');
        currentMarker.id = 'pipelineCurrentTimeMarker';
        currentMarker.className = 'pipeline-current-time-marker';
        currentMarker.innerHTML = '<div class="pipeline-current-time-arrow">▼</div>';
        timeTrack.appendChild(currentMarker);

        // Set width
        timeTrack.style.minWidth = `${this.config.maxTime * this.config.timeScale}px`;
    }

    initializeStreamLanes() {
        const streamLanes = document.getElementById('pipelineStreamLanes');
        streamLanes.innerHTML = '';

        this.streams.forEach(stream => {
            const lane = document.createElement('div');
            lane.className = 'pipeline-stream-lane';
            lane.innerHTML = `
                <div class="pipeline-stream-label pipeline-${stream.type}">
                    <div class="pipeline-stream-name">${stream.name}</div>
                    <div class="pipeline-stream-type">${stream.type === 'producer' ? 'TMA Loads' : 'Compute'}</div>
                </div>
                <div class="pipeline-lane-track" id="pipeline-lane-${stream.id}" style="min-width: ${this.config.maxTime * this.config.timeScale}px"></div>
            `;
            streamLanes.appendChild(lane);
        });

        // Create operation elements
        this.state.operations.forEach(op => {
            const opType = this.opTypes[op.type];
            const lane = document.getElementById(`pipeline-lane-${op.streamId}`);

            const opElement = document.createElement('div');
            opElement.id = `pipeline-op-${op.id}`;
            opElement.className = `pipeline-operation ${opType.color}`;
            opElement.style.left = `${op.startTime * this.config.timeScale}px`;
            opElement.style.width = `${(op.endTime - op.startTime) * this.config.timeScale}px`;
            opElement.style.display = 'none'; // Initially hidden
            opElement.innerHTML = `
                <div class="pipeline-operation-progress" id="pipeline-progress-${op.id}" style="width: 0%"></div>
                <div class="pipeline-operation-content">
                    <div class="pipeline-operation-icon">${opType.icon}</div>
                    <div class="pipeline-operation-label">${op.label}</div>
                </div>
            `;

            lane.appendChild(opElement);
        });
    }

    updateVisualization() {
        const currentTimeEl = document.getElementById('pipelineCurrentTime');
        const currentMarker = document.getElementById('pipelineCurrentTimeMarker');

        currentTimeEl.textContent = this.state.currentTime;
        if (currentMarker) {
            currentMarker.style.left = `${this.state.currentTime * this.config.timeScale}px`;
        }

        let activeCount = 0;

        this.state.operations.forEach(op => {
            const isActive = this.state.currentTime >= op.startTime && this.state.currentTime <= op.endTime;
            const isVisible = this.state.currentTime >= op.startTime;

            if (isActive) activeCount++;

            const opElement = document.getElementById(`pipeline-op-${op.id}`);
            const progressBar = document.getElementById(`pipeline-progress-${op.id}`);

            if (opElement) {
                // Show/hide bubble based on visibility
                if (isVisible) {
                    opElement.style.display = 'flex';
                    opElement.style.opacity = isActive ? '1' : '0.6';
                } else {
                    opElement.style.display = 'none';
                }

                if (isVisible && isActive) {
                    opElement.classList.add('active');
                    if (!opElement.querySelector('.pipeline-operation-pulse')) {
                        const pulse = document.createElement('div');
                        pulse.className = 'pipeline-operation-pulse';
                        opElement.appendChild(pulse);
                    }
                } else {
                    opElement.classList.remove('active');
                    const pulse = opElement.querySelector('.pipeline-operation-pulse');
                    if (pulse) pulse.remove();
                }

                if (progressBar && isActive) {
                    const progress = (this.state.currentTime - op.startTime) / (op.endTime - op.startTime);
                    progressBar.style.width = `${progress * 100}%`;
                } else if (progressBar && this.state.currentTime > op.endTime) {
                    progressBar.style.width = '100%';
                } else if (progressBar) {
                    progressBar.style.width = '0%';
                }
            }
        });

        document.getElementById('pipelineActiveOps').textContent = activeCount;
    }

    animate() {
        if (this.state.isPlaying) {
            this.state.currentTime++;

            if (this.state.currentTime >= this.config.maxTime) {
                this.state.currentTime = this.config.maxTime;
                this.stopAnimation();
            }

            this.updateVisualization();
        }
    }

    startAnimation() {
        this.state.isPlaying = true;
        this.state.animationTimer = setInterval(() => this.animate(), this.config.animationInterval);

        document.getElementById('pipelinePlayIcon').textContent = '⏸';
        document.getElementById('pipelinePlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('pipelinePlayIcon').textContent = '▶';
        document.getElementById('pipelinePlayText').textContent = 'Play';
    }

    resetAnimation() {
        this.stopAnimation();
        this.state.currentTime = 0;
        this.updateVisualization();
    }

    togglePlay() {
        if (!this.state.isPlaying && this.state.currentTime >= this.config.maxTime) {
            this.resetAnimation();
            setTimeout(() => this.startAnimation(), 100);
        } else if (this.state.isPlaying) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
}

// Auto-initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('hopper-gemm-pipeline-viz');
    if (container) {
        new HopperGEMMPipelineViz('hopper-gemm-pipeline-viz');
    }
});

// ============================================================================
// CUTLASS 3.x Conceptual GEMM Hierarchy Visualization
// ============================================================================

const cutlassHierarchyStyles = `
    <style>
        .cutlass-hierarchy-container * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .cutlass-hierarchy-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 2.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        }

        .cutlass-hierarchy-title {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #10b981, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .cutlass-hierarchy-subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 2.5rem;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .cutlass-hierarchy-diagram {
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            padding: 2rem 0;
        }

        .cutlass-level {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            position: relative;
        }

        .cutlass-level-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }

        .cutlass-level-number {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 50%;
            font-weight: bold;
            font-size: 1.1rem;
            flex-shrink: 0;
        }

        .cutlass-level-title {
            font-size: 1.4rem;
            font-weight: bold;
            flex: 1;
        }

        .cutlass-level-card {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 0.75rem;
            padding: 1.5rem;
            border: 2px solid;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }

        .cutlass-level-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }

        /* Level-specific colors */
        .cutlass-level-1 .cutlass-level-number {
            background: linear-gradient(135deg, #10b981, #059669);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
        }

        .cutlass-level-1 .cutlass-level-title {
            color: #10b981;
        }

        .cutlass-level-1 .cutlass-level-card {
            border-color: #10b981;
        }

        .cutlass-level-2 .cutlass-level-number {
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
        }

        .cutlass-level-2 .cutlass-level-title {
            color: #3b82f6;
        }

        .cutlass-level-2 .cutlass-level-card {
            border-color: #3b82f6;
        }

        .cutlass-level-3 .cutlass-level-number {
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
        }

        .cutlass-level-3 .cutlass-level-title {
            color: #8b5cf6;
        }

        .cutlass-level-3 .cutlass-level-card {
            border-color: #8b5cf6;
        }

        .cutlass-level-4 .cutlass-level-number {
            background: linear-gradient(135deg, #ec4899, #db2777);
            box-shadow: 0 0 20px rgba(236, 72, 153, 0.4);
        }

        .cutlass-level-4 .cutlass-level-title {
            color: #ec4899;
        }

        .cutlass-level-4 .cutlass-level-card {
            border-color: #ec4899;
        }

        .cutlass-level-5 .cutlass-level-number {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
        }

        .cutlass-level-5 .cutlass-level-title {
            color: #f59e0b;
        }

        .cutlass-level-5 .cutlass-level-card {
            border-color: #f59e0b;
        }

        .cutlass-card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .cutlass-card-icon {
            font-size: 2rem;
        }

        .cutlass-card-name {
            font-size: 1.1rem;
            font-weight: bold;
            color: #f1f5f9;
        }

        .cutlass-card-type {
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        .cutlass-card-description {
            color: #cbd5e1;
            line-height: 1.6;
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }

        .cutlass-card-components {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .cutlass-component-tag {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.4);
            padding: 0.4rem 0.8rem;
            border-radius: 0.375rem;
            font-size: 0.85rem;
            font-family: 'Courier New', monospace;
            color: #93c5fd;
            transition: all 0.2s;
        }

        .cutlass-component-tag:hover {
            background: rgba(59, 130, 246, 0.3);
            border-color: rgba(59, 130, 246, 0.6);
        }

        .cutlass-multi-column {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }

        .cutlass-connector {
            position: absolute;
            left: 1.25rem;
            width: 2px;
            background: linear-gradient(180deg,
                rgba(16, 185, 129, 0.3) 0%,
                rgba(59, 130, 246, 0.3) 25%,
                rgba(139, 92, 246, 0.3) 50%,
                rgba(236, 72, 153, 0.3) 75%,
                rgba(245, 158, 11, 0.3) 100%
            );
            top: 3.5rem;
            bottom: 2rem;
            z-index: 0;
        }

        .cutlass-key-points {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 2px solid rgba(148, 163, 184, 0.2);
        }

        .cutlass-key-points-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #f1f5f9;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .cutlass-key-points-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .cutlass-key-point {
            display: flex;
            align-items: start;
            gap: 0.75rem;
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .cutlass-key-point-icon {
            color: #3b82f6;
            font-weight: bold;
            flex-shrink: 0;
        }

        @media (max-width: 768px) {
            .cutlass-hierarchy-container {
                padding: 1.5rem;
            }

            .cutlass-hierarchy-title {
                font-size: 1.5rem;
            }

            .cutlass-multi-column {
                grid-template-columns: 1fr;
            }
        }
    </style>
`;

class CutlassHierarchyViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        this.levels = [
            {
                number: 1,
                title: "Atom Layer",
                icon: "⚛️",
                description: "Architecture-specific instructions and associated meta-information. These are the foundational hardware primitives that directly map to GPU instructions.",
                components: [
                    { name: "MMA Atom", type: "cute::Mma_Atom<>", desc: "Warp-level matrix multiply-accumulate primitives (e.g., mma.sync.m16n8k16.f16)" },
                    { name: "Copy Atom", type: "cute::Copy_Atom<>", desc: "Thread-level memory operations (e.g., ldmatrix, TMA load/store)" }
                ],
                tags: ["Hardware Instructions", "SM Architecture", "Register-Level Ops"]
            },
            {
                number: 2,
                title: "Tiled MMA/Copy",
                icon: "🔲",
                description: "Spatial micro-kernels that allow for arbitrary interleaving and tiling of architecture-specific atoms. Enables flexible data layouts and efficient computation patterns.",
                components: [
                    { name: "Tiled MMA", type: "cute::TiledMma<>", desc: "Thread block-level tiling of MMA atoms with configurable layouts" },
                    { name: "Tiled Copy", type: "cute::TiledCopy<>", desc: "Thread block-level tiling of copy operations with swizzling support" }
                ],
                tags: ["Thread Block Level", "Spatial Tiling", "Layout Composition", "CuTe Layouts"]
            },
            {
                number: 3,
                title: "Collective Layer",
                icon: "🔄",
                description: "Temporal micro-kernels that use architecture-specific synchronization to orchestrate the execution of one or more spatial micro-kernels. The workhorse of CUTLASS 3.x.",
                components: [
                    { name: "Collective Mainloop", type: "cutlass::gemm::collective::CollectiveMma<>", desc: "Orchestrates data movement (TMA) and computation (WGMMA) with software pipelining" },
                    { name: "Collective Epilogue", type: "cutlass::epilogue::collective::CollectiveEpilogue<>", desc: "Post-processing operations (accumulator transformations, reductions, output writes)" }
                ],
                tags: ["Software Pipelining", "Async Barriers", "Producer-Consumer", "Hopper TMA", "WGMMA"]
            },
            {
                number: 4,
                title: "Kernel Layer",
                icon: "🖥️",
                description: "Device code for executing a kernel over a grid of threadblocks/clusters. Composes mainloop and epilogue abstractions and manages tile scheduling strategies.",
                components: [
                    { name: "GEMM Universal", type: "cutlass::gemm::kernel::GemmUniversal<>", desc: "Universal GEMM kernel supporting multiple problem modes and scheduling" }
                ],
                tags: ["Device Kernel", "Grid Management", "Tile Scheduling", "Stream-K", "Persistent Kernels"]
            },
            {
                number: 5,
                title: "Device Layer",
                icon: "🌐",
                description: "Host-side setup and interface providing reusable, stateful kernel management and launch capabilities. User-facing API for CUTLASS.",
                components: [
                    { name: "Device Adapter", type: "cutlass::gemm::device::GemmUniversalAdapter<>", desc: "Host-side interface for kernel configuration, workspace management, and launch" }
                ],
                tags: ["Host API", "Kernel Launch", "Workspace Management", "Parameter Configuration"]
            }
        ];

        this.init();
    }

    init() {
        this.container.innerHTML = cutlassHierarchyStyles + `
            <div class="cutlass-hierarchy-container">
                <h1 class="cutlass-hierarchy-title">CUTLASS 3.x GEMM Hierarchy</h1>
                <p class="cutlass-hierarchy-subtitle">
                    Five orthogonal, composable abstraction layers from hardware instructions to host API.<br>
                    Each layer builds upon the previous, enabling maximum code reuse and architectural portability.
                </p>

                <div class="cutlass-hierarchy-diagram">
                    <div class="cutlass-connector"></div>
                    ${this.renderLevels()}
                </div>

                ${this.renderKeyPoints()}
            </div>
        `;
    }

    renderLevels() {
        return this.levels.map(level => `
            <div class="cutlass-level cutlass-level-${level.number}">
                <div class="cutlass-level-header">
                    <div class="cutlass-level-number">${level.number}</div>
                    <div class="cutlass-level-title">${level.title}</div>
                </div>

                <div class="cutlass-level-card">
                    <div class="cutlass-card-header">
                        <div class="cutlass-card-icon">${level.icon}</div>
                        <div>
                            <div class="cutlass-card-name">${level.title}</div>
                        </div>
                    </div>

                    <div class="cutlass-card-description">${level.description}</div>

                    <div class="cutlass-multi-column">
                        ${level.components.map(comp => `
                            <div>
                                <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 0.25rem;">
                                    ${comp.name}
                                </div>
                                <div class="cutlass-card-type">${comp.type}</div>
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem; line-height: 1.5;">
                                    ${comp.desc}
                                </div>
                            </div>
                        `).join('')}
                    </div>

                    <div class="cutlass-card-components">
                        ${level.tags.map(tag => `
                            <div class="cutlass-component-tag">${tag}</div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `).join('');
    }

    renderKeyPoints() {
        return `
            <div class="cutlass-key-points">
                <div class="cutlass-key-points-title">
                    <span>💡</span>
                    Design Philosophy
                </div>
                <div class="cutlass-key-points-list">
                    <div class="cutlass-key-point">
                        <span class="cutlass-key-point-icon">▸</span>
                        <span><strong>Orthogonality:</strong> Different MMA operations can combine with different copy operations independently, maximizing code reuse across architectures.</span>
                    </div>
                    <div class="cutlass-key-point">
                        <span class="cutlass-key-point-icon">▸</span>
                        <span><strong>Composability:</strong> Each layer is a composition point—higher layers build upon lower layers through template parameters, not inheritance.</span>
                    </div>
                    <div class="cutlass-key-point">
                        <span class="cutlass-key-point-icon">▸</span>
                        <span><strong>Specialization:</strong> Dispatch policies (e.g., MainloopSm90TmaGmmaWarpSpecialized) specialize implementations for specific architectures while maintaining a common interface.</span>
                    </div>
                    <div class="cutlass-key-point">
                        <span class="cutlass-key-point-icon">▸</span>
                        <span><strong>Zero Overhead:</strong> Template metaprogramming and compile-time layout reasoning (via CuTe) ensure zero runtime overhead for abstraction layers.</span>
                    </div>
                    <div class="cutlass-key-point">
                        <span class="cutlass-key-point-icon">▸</span>
                        <span><strong>Future-Proof:</strong> New hardware features (e.g., Hopper's TMA, WGMMA) slot into existing layers without changing the overall hierarchy structure.</span>
                    </div>
                </div>
            </div>
        `;
    }
}

// Auto-initialize CUTLASS hierarchy viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('cutlass-3x-hierarchy-viz');
    if (container) {
        new CutlassHierarchyViz('cutlass-3x-hierarchy-viz');
    }
});
