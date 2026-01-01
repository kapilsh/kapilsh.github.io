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
            min-width: 600px;
        }

        .pipeline-time-axis {
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-top: 2rem;
        }

        .pipeline-time-label {
            width: 11rem;
            min-width: 11rem;
            font-size: 0.875rem;
            color: #9ca3af;
            font-weight: bold;
            padding-right: 1rem;
            flex-shrink: 0;
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
            min-width: 11rem;
            padding-right: 1rem;
            flex-shrink: 0;
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
            height: 3rem;
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
            font-size: 1rem;
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
        .pipeline-bg-purple-400 { background-color: #c084fc; }
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
            maxTime: 450,
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
                <h1 class="pipeline-title">Example Hopper Warp Specialized GEMM Pipeline</h1>
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
            </div>
        `;

        this.state.operations = this.initializeOperations();
        this.initializeLegend();
        this.initializeTimeTrack();
        this.initializeStreamLanes();
        // Don't call updateVisualization() here - operations should be hidden until play is pressed

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
            const consumerWarp = tile % 2;
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
                </div>
            `;

            lane.appendChild(opElement);
        });
    }

    updateVisualization() {
        const currentMarker = document.getElementById('pipelineCurrentTimeMarker');

        if (currentMarker) {
            currentMarker.style.left = `${this.state.currentTime * this.config.timeScale}px`;
        }

        this.state.operations.forEach(op => {
            const isActive = this.state.currentTime >= op.startTime && this.state.currentTime <= op.endTime;
            const isVisible = this.state.currentTime >= op.startTime;

            // Cache DOM elements on first access
            if (!op._cachedElements) {
                op._cachedElements = {
                    opElement: document.getElementById(`pipeline-op-${op.id}`),
                    progressBar: document.getElementById(`pipeline-progress-${op.id}`)
                };
            }

            const { opElement, progressBar } = op._cachedElements;

            if (opElement) {
                // Track previous state to avoid unnecessary updates
                const prevActive = op._prevActive;
                const prevVisible = op._prevVisible;

                // Only update if state changed
                if (isVisible !== prevVisible) {
                    opElement.style.display = isVisible ? 'flex' : 'none';
                    op._prevVisible = isVisible;
                }

                if (isVisible && isActive !== prevActive) {
                    opElement.style.opacity = isActive ? '1' : '0.6';

                    if (isActive) {
                        opElement.classList.add('active');
                        if (!op._pulseElement) {
                            op._pulseElement = document.createElement('div');
                            op._pulseElement.className = 'pipeline-operation-pulse';
                            opElement.appendChild(op._pulseElement);
                        }
                    } else {
                        opElement.classList.remove('active');
                        if (op._pulseElement) {
                            op._pulseElement.remove();
                            op._pulseElement = null;
                        }
                    }
                    op._prevActive = isActive;
                }

                if (progressBar && isActive) {
                    const progress = (this.state.currentTime - op.startTime) / (op.endTime - op.startTime);
                    progressBar.style.width = `${progress * 100}%`;
                } else if (progressBar && this.state.currentTime > op.endTime && op._prevProgressComplete !== true) {
                    progressBar.style.width = '100%';
                    op._prevProgressComplete = true;
                } else if (progressBar && !isVisible && op._prevProgressComplete !== false) {
                    progressBar.style.width = '0%';
                    op._prevProgressComplete = false;
                }
            }
        });
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

        .cutlass-code-block {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 1rem;
            overflow-x: auto;
            font-family: 'Courier New', Consolas, monospace;
            font-size: 0.85rem;
            line-height: 1.6;
        }

        .cutlass-code-block code {
            color: #e2e8f0;
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
                title: "Device Layer",
                icon: "🌐",
                description: "Host-side setup and interface providing a stateless wrapper around a kernel type",
                components: [
                    { name: "Device Adapter", type: "cutlass::gemm::device::GemmUniversalAdapter<>", desc: "Host-side interface for kernel configuration, workspace management, and launch" }
                ],
                codeExample: `using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

Gemm gemm_op;
cutlass::Status status = gemm_op.can_implement(arguments);
status = gemm_op.initialize(arguments, workspace.get());
status = gemm_op.run();`
            },
            {
                number: 2,
                title: "Kernel Layer",
                icon: "🖥️",
                description: "GEMM as a composition of a collective mainloop and a collective epilogue where each new kernel schedule is a specialization with some schedule tags",
                components: [
                    { name: "GEMM Universal", type: "cutlass::gemm::kernel::GemmUniversal<>", desc: "Universal GEMM kernel supporting multiple problem modes and scheduling" }
                ],
                specializations: [
                    { name: "KernelTmaWarpSpecialized", desc: "Standard TMA-based warp specialization" },
                    { name: "KernelTmaWarpSpecializedPingpong", desc: "Persistent thread blocks with dual-buffering" },
                    { name: "KernelTmaWarpSpecializedCooperative", desc: "Cooperative scheduling with Stream-K decomposition" }
                ],
                schedulers: [
                    { name: "PersistentScheduler", desc: "Persistent kernel work distribution" },
                    { name: "StreamKScheduler", desc: "Stream-K problem decomposition" }
                ],
                codeExample: `template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecialized
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};`
            },
            {
                number: 3,
                title: "Collective Layer",
                icon: "🔄",
                description: "Temporal micro-kernels that use architecture-specific synchronization to orchestrate the execution of one or more spatial micro-kernels. This is the main GEMM configuration API.",
                components: [
                    { name: "Collective Mainloop", type: "cutlass::gemm::collective::CollectiveMma<>", desc: "Orchestrates data movement (TMA) and computation (WGMMA) with software pipelining" },
                    { name: "Collective Epilogue", type: "cutlass::epilogue::collective::CollectiveEpilogue<>", desc: "Post-processing operations (accumulator transformations, reductions, output writes)" }
                ],
                autoOptions: [
                    { name: "KernelScheduleAuto", desc: "Framework selects optimal kernel schedule" },
                    { name: "StageCountAuto", desc: "Automatic pipeline stage determination" },
                    { name: "EpilogueScheduleAuto", desc: "Automatic epilogue scheduling" }
                ],
                codeExample: `using CollectiveOp = typename cutlass::gemm::collective::CollectiveBuilder<
  arch::Sm90, arch::OpClassTensorOp,
  half_t, LayoutA, 8,
  half_t, LayoutB, 8,
  float,
  Shape<_128,_128,_64>, Shape<_1,_2,_1>,
  gemm::collective::StageCountAuto,
  gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Epilogue with fusion
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
  Shape<_128,_128,_64>, Shape<_1,_1,_1>,
  cutlass::epilogue::collective::EpilogueTileAuto,
  ElementAccumulator, ElementCompute,
  ElementC, LayoutC, AlignmentC,
  ElementD, LayoutD, AlignmentD,
  EpilogueScheduleAuto
>::CollectiveOp;`
            },
            {
                number: 4,
                title: "Tiled MMA/Copy",
                icon: "🔲",
                description: "Spatial micro-kernels that allow for arbitrary interleaving and tiling of architecture-specific atoms. Enables varied data layouts and computation patterns.",
                components: [
                    { name: "Tiled MMA", type: "cute::TiledMma<>", desc: "Thread block-level tiling of MMA atoms with configurable layouts" },
                    { name: "Tiled Copy", type: "cute::TiledCopy<>", desc: "Thread block-level tiling of copy operations with swizzling support" }
                ],
                codeExample: `// Tiled MMA with warp group shape
using TiledMma = decltype(cute::make_tiled_mma(
  cute::GMMA::ss_op_selector<
    Element, Element, ElementAccumulator,
    cute::Shape<_64, _128, _16>
  >(),
  cute::Layout<Shape<_2,_1,_1>>{}  // 2x1x1 warp group arrangement
));

// Tiled Copy with TMA
using GmemTiledCopyA = decltype(cute::make_tiled_copy(
  cute::Copy_Atom<cute::SM90_TMA_LOAD, Element>{},
  cute::Layout<Shape<_1,_1,_1>>{},
  cute::make_layout(cute::make_shape(Int<32>{}, Int<4>{}))
));`
            },
            {
                number: 5,
                title: "Atom Layer",
                icon: "⚛️",
                description: "Architecture-specific instructions and associated meta-information. These are the foundational hardware primitives that directly map to GPU instructions.",
                components: [
                    { name: "MMA Atom", type: "cute::Mma_Atom<>", desc: "Warp-level matrix multiply-accumulate primitives (e.g., mma.sync.m16n8k16.f16)" },
                    { name: "Copy Atom", type: "cute::Copy_Atom<>", desc: "Thread-level memory operations (e.g., ldmatrix, TMA load/store)" }
                ],
                examples: [
                    { name: "WGMMA (Hopper)", code: "SM90_64x128x16_F16F16F16_SS" },
                    { name: "MMA (Ampere)", code: "SM80_16x8x16_F16F16F16F16_TN" },
                    { name: "TMA Load", code: "SM90_TMA_LOAD" },
                    { name: "TMA Store", code: "SM90_TMA_STORE" },
                    { name: "Shared Memory Load", code: "SM75_U32x4_LDSM_N" }
                ],
                codeExample: `// MMA Atom - WGMMA for Hopper
using MmaAtom = cute::MMA_Atom<
  cute::SM90_64x128x16_F16F16F16_SS  // 64x128x16 WGMMA operation
>;

// Copy Atom - TMA for async loading
using CopyAtom = cute::Copy_Atom<
  cute::SM90_TMA_LOAD,               // TMA load instruction
  Element
>;

// Copy Atom - Shared memory to register
using SmemCopyAtom = cute::Copy_Atom<
  cute::SM75_U32x4_LDSM_N,           // ldmatrix instruction
  Element
>;`
            }
        ];

        this.init();
    }

    init() {
        this.container.innerHTML = cutlassHierarchyStyles + `
            <div class="cutlass-hierarchy-container">
                <h1 class="cutlass-hierarchy-title">CUTLASS 3.x GEMM Hierarchy</h1>

                <div class="cutlass-hierarchy-diagram">
                    <div class="cutlass-connector"></div>
                    ${this.renderLevels()}
                </div>
            </div>
            
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
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

                    ${level.specializations ? `
                        <div style="margin-top: 1rem;">
                            <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 0.5rem; font-size: 0.9rem;">
                                Kernel Schedules:
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 0.75rem;">
                                ${level.specializations.map(spec => `
                                    <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 0.375rem; padding: 0.75rem;">
                                        <div style="font-family: 'Courier New', monospace; color: #60a5fa; font-size: 0.8rem; font-weight: bold;">
                                            ${spec.name}
                                        </div>
                                        <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.25rem;">
                                            ${spec.desc}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}

                    ${level.schedulers ? `
                        <div style="margin-top: 1rem;">
                            <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 0.5rem; font-size: 0.9rem;">
                                Tile Schedulers:
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
                                ${level.schedulers.map(sched => `
                                    <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 0.375rem; padding: 0.75rem;">
                                        <div style="font-family: 'Courier New', monospace; color: #a78bfa; font-size: 0.8rem; font-weight: bold;">
                                            ${sched.name}
                                        </div>
                                        <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.25rem;">
                                            ${sched.desc}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}

                    ${level.autoOptions ? `
                        <div style="margin-top: 1rem;">
                            <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 0.5rem; font-size: 0.9rem;">
                                Auto Configuration Options:
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
                                ${level.autoOptions.map(opt => `
                                    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 0.375rem; padding: 0.75rem;">
                                        <div style="font-family: 'Courier New', monospace; color: #34d399; font-size: 0.8rem; font-weight: bold;">
                                            ${opt.name}
                                        </div>
                                        <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.25rem;">
                                            ${opt.desc}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}

                    ${level.examples ? `
                        <div style="margin-top: 1rem;">
                            <div style="font-weight: bold; color: #f1f5f9; margin-bottom: 0.5rem; font-size: 0.9rem;">
                                Hardware Instruction Examples:
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem;">
                                ${level.examples.map(ex => `
                                    <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 0.375rem; padding: 0.75rem;">
                                        <div style="color: #fbbf24; font-size: 0.75rem; font-weight: bold; margin-bottom: 0.25rem;">
                                            ${ex.name}
                                        </div>
                                        <div style="font-family: 'Courier New', monospace; color: #fcd34d; font-size: 0.75rem;">
                                            ${ex.code}
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}

                    ${level.codeExample ? `
                        <pre class="cutlass-code-block"><code>${this.escapeHtml(level.codeExample)}</code></pre>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }
}

// Auto-initialize CUTLASS hierarchy viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('cutlass-3x-hierarchy-viz');
    if (container) {
        new CutlassHierarchyViz('cutlass-3x-hierarchy-viz');
    }
});

// ============================================================================
// Warp Specialization Visualization
// ============================================================================

class WarpSpecializationViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        // Configuration
        this.config = {
            maxTime: 220,
            timeScale: 2.5,
            numTiles: 4,
            animationInterval: 40,
        };

        this.opTypes = {
            TMA_LOAD: { name: 'TMA Load', color: 'pipeline-bg-cyan-500', duration: 20, icon: '📥' },
            BARRIER_ARRIVE: { name: 'Barrier Arrive', color: 'pipeline-bg-blue-500', duration: 15, icon: '🚪' },
            BARRIER_WAIT: { name: 'Barrier Wait', color: 'pipeline-bg-orange-500', duration: 15, icon: '⏳' },
            WGMMA: { name: 'WGMMA Compute', color: 'pipeline-bg-purple-500', duration: 30, icon: '⚡' },
        };

        this.streams = [
            { id: 'producer', name: 'Producer Warp', type: 'producer' },
            { id: 'consumer', name: 'Consumer Warp', type: 'consumer' },
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
                <h1 class="pipeline-title">Warp Specialization: Producer-Consumer Pattern</h1>
                <p class="pipeline-subtitle">Simple example with 1 producer warp and 1 consumer warp coordinated via async barriers</p>

                <!-- Controls -->
                <div class="pipeline-controls">
                    <button id="warpSpecPlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="warpSpecPlayIcon">▶</span>
                        <span id="warpSpecPlayText">Play</span>
                    </button>
                    <button id="warpSpecResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <!-- Legend -->
                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">Operation Types</h3>
                    <div class="pipeline-legend-grid" id="warpSpecLegendGrid"></div>
                </div>

                <!-- Timeline -->
                <div class="pipeline-timeline-container">
                    <div class="pipeline-timeline-wrapper" id="warpSpecTimelineWrapper">
                        <div class="pipeline-time-axis">
                            <div class="pipeline-time-label">Time →</div>
                            <div class="pipeline-time-track" id="warpSpecTimeTrack"></div>
                        </div>
                        <div id="warpSpecStreamLanes"></div>
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
        document.getElementById('warpSpecPlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('warpSpecResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    initializeOperations() {
        const ops = [];
        let opId = 0;

        for (let tile = 0; tile < this.config.numTiles; tile++) {
            const baseTime = tile * 50;

            // Producer: TMA Load
            ops.push({
                id: opId++,
                streamId: 'producer',
                type: 'TMA_LOAD',
                startTime: baseTime,
                endTime: baseTime + this.opTypes.TMA_LOAD.duration,
                tile: tile,
                label: `Load ${tile}`,
            });

            // Producer: Barrier Arrive (signals load complete)
            const arriveTime = baseTime + this.opTypes.TMA_LOAD.duration;
            ops.push({
                id: opId++,
                streamId: 'producer',
                type: 'BARRIER_ARRIVE',
                startTime: arriveTime,
                endTime: arriveTime + this.opTypes.BARRIER_ARRIVE.duration,
                tile: tile,
                label: `Arrive ${tile}`,
            });

            // Consumer: Barrier Wait (waits for data)
            // Wait starts at the same time as barrier arrive to show synchronization
            const waitStart = arriveTime;
            const waitEnd = waitStart + this.opTypes.BARRIER_WAIT.duration;
            ops.push({
                id: opId++,
                streamId: 'consumer',
                type: 'BARRIER_WAIT',
                startTime: waitStart,
                endTime: waitEnd,
                tile: tile,
                label: `Wait ${tile}`,
            });

            // Consumer: WGMMA Compute (starts right after wait completes)
            const computeStart = waitEnd;
            ops.push({
                id: opId++,
                streamId: 'consumer',
                type: 'WGMMA',
                startTime: computeStart,
                endTime: computeStart + this.opTypes.WGMMA.duration,
                tile: tile,
                label: `Compute ${tile}`,
            });
        }

        return ops;
    }

    initializeLegend() {
        const legendGrid = document.getElementById('warpSpecLegendGrid');
        legendGrid.innerHTML = '';

        Object.entries(this.opTypes).forEach(([, type]) => {
            const item = document.createElement('div');
            item.className = 'pipeline-legend-item';
            item.innerHTML = `
                <div class="pipeline-legend-icon ${type.color}">${type.icon}</div>
                <div class="pipeline-legend-text">
                    <div class="pipeline-legend-name">${type.name}</div>
                </div>
            `;
            legendGrid.appendChild(item);
        });
    }

    initializeTimeTrack() {
        const timeTrack = document.getElementById('warpSpecTimeTrack');
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
        currentMarker.id = 'warpSpecCurrentTimeMarker';
        currentMarker.className = 'pipeline-current-time-marker';
        currentMarker.innerHTML = '<div class="pipeline-current-time-arrow">▼</div>';
        timeTrack.appendChild(currentMarker);

        // Set width
        timeTrack.style.minWidth = `${this.config.maxTime * this.config.timeScale}px`;
    }

    initializeStreamLanes() {
        const streamLanes = document.getElementById('warpSpecStreamLanes');
        streamLanes.innerHTML = '';

        this.streams.forEach(stream => {
            const lane = document.createElement('div');
            lane.className = 'pipeline-stream-lane';
            lane.innerHTML = `
                <div class="pipeline-stream-label pipeline-${stream.type}">
                    <div class="pipeline-stream-name">${stream.name}</div>
                    <div class="pipeline-stream-type">${stream.type === 'producer' ? 'Memory Ops' : 'Compute Ops'}</div>
                </div>
                <div class="pipeline-lane-track" id="warpSpec-lane-${stream.id}" style="min-width: ${this.config.maxTime * this.config.timeScale}px"></div>
            `;
            streamLanes.appendChild(lane);
        });

        // Create operation elements
        this.state.operations.forEach(op => {
            const opType = this.opTypes[op.type];
            const lane = document.getElementById(`warpSpec-lane-${op.streamId}`);

            const opElement = document.createElement('div');
            opElement.id = `warpSpec-op-${op.id}`;
            opElement.className = `pipeline-operation ${opType.color}`;
            opElement.style.left = `${op.startTime * this.config.timeScale}px`;
            opElement.style.width = `${(op.endTime - op.startTime) * this.config.timeScale}px`;
            opElement.style.display = 'none'; // Initially hidden
            opElement.innerHTML = `
                <div class="pipeline-operation-progress" id="warpSpec-progress-${op.id}" style="width: 0%"></div>
                <div class="pipeline-operation-content">
                    <div class="pipeline-operation-icon">${opType.icon}</div>
                </div>
            `;

            lane.appendChild(opElement);
        });
    }

    updateVisualization() {
        const currentMarker = document.getElementById('warpSpecCurrentTimeMarker');

        if (currentMarker) {
            currentMarker.style.left = `${this.state.currentTime * this.config.timeScale}px`;
        }

        this.state.operations.forEach(op => {
            const isActive = this.state.currentTime >= op.startTime && this.state.currentTime <= op.endTime;
            const isVisible = this.state.currentTime > op.startTime || (this.state.currentTime === op.startTime && this.state.isPlaying);

            const opElement = document.getElementById(`warpSpec-op-${op.id}`);
            const progressBar = document.getElementById(`warpSpec-progress-${op.id}`);

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

        document.getElementById('warpSpecPlayIcon').textContent = '⏸';
        document.getElementById('warpSpecPlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('warpSpecPlayIcon').textContent = '▶';
        document.getElementById('warpSpecPlayText').textContent = 'Play';
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

// Auto-initialize warp specialization viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('warp-specialization-viz');
    if (container) {
        new WarpSpecializationViz('warp-specialization-viz');
    }
});

// ============================================================================
// Persistent Cooperative Kernel Visualization
// ============================================================================

class PersistentCooperativeViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        // Configuration
        this.config = {
            maxTime: 450,
            timeScale: 2.2,
            numTiles: 6,
            numThreadBlocks: 2, // 2 persistent thread blocks processing multiple tiles
            animationInterval: 40,
        };

        this.opTypes = {
            TMA_A: { name: 'TMA Load A', color: 'pipeline-bg-blue-500', duration: 25, icon: '📥' },
            TMA_B: { name: 'TMA Load B', color: 'pipeline-bg-cyan-500', duration: 25, icon: '📥' },
            WGMMA: { name: 'WGMMA', color: 'pipeline-bg-purple-500', duration: 40, icon: '⚡' },
            EPILOGUE: { name: 'Epilogue', color: 'pipeline-bg-orange-500', duration: 15, icon: '🔧' },
            GMEM_WRITE: { name: 'Write GMEM', color: 'pipeline-bg-green-500', duration: 20, icon: '📤' },
        };

        // 2 Producer warps (shared) + 4 Consumer warps (2 for top half, 2 for bottom half)
        this.streams = [
            { id: 'producer_0', name: 'Producer 0 (Load A)', type: 'producer' },
            { id: 'producer_1', name: 'Producer 1 (Load B)', type: 'producer' },
            { id: 'consumer_0', name: 'Consumer 0 (Top Half)', type: 'consumer' },
            { id: 'consumer_1', name: 'Consumer 1 (Top Half)', type: 'consumer' },
            { id: 'consumer_2', name: 'Consumer 2 (Bottom Half)', type: 'consumer' },
            { id: 'consumer_3', name: 'Consumer 3 (Bottom Half)', type: 'consumer' }
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
        const totalTime = this.getTotalTime();

        this.container.innerHTML = commonPipelineStyles + `
            <div class="pipeline-container">
                <h1 class="pipeline-title">Persistent Cooperative Kernel Timeline</h1>
                <p class="pipeline-subtitle">2 Persistent Thread Blocks × Multiple Tiles → Cooperative Consumers Split Each Tile</p>

                <!-- Controls -->
                <div class="pipeline-controls">
                    <button id="persistentCoopPlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="persistentCoopPlayIcon">▶</span>
                        <span id="persistentCoopPlayText">Play</span>
                    </button>
                    <button id="persistentCoopResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <!-- Legend -->
                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">Operation Types</h3>
                    <div class="pipeline-legend-grid" id="persistentCoopLegendGrid"></div>
                </div>

                <!-- Timeline -->
                <div class="pipeline-timeline-container">
                    <div class="pipeline-timeline-wrapper">
                        <div class="pipeline-time-axis">
                            <div class="pipeline-time-label">Time →</div>
                            <div class="pipeline-time-track" id="persistentCoopTimeTrack"></div>
                        </div>
                        <div id="persistentCoopStreamLanes"></div>
                    </div>
                </div>
            </div>
        `;

        this.state.operations = this.initializeOperations();
        this.initializeLegend();
        this.initializeTimeTrack();
        this.initializeStreamLanes();
        // Don't call updateVisualization() here - operations should be hidden until play is pressed

        // Event listeners
        document.getElementById('persistentCoopPlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('persistentCoopResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    getTotalTime() {
        // Calculate based on all tiles completing
        const halfWGMMADuration = this.opTypes.WGMMA.duration * 0.6;
        const lastTileStart = (this.config.numTiles - 1) * 35;
        const lastTileEnd = lastTileStart + this.opTypes.TMA_A.duration + 3 +
            halfWGMMADuration + 1 +
            this.opTypes.EPILOGUE.duration + 1 +
            this.opTypes.GMEM_WRITE.duration + 20;
        return lastTileEnd;
    }

    initializeOperations() {
        const ops = [];
        let opId = 0;

        for (let tile = 0; tile < this.config.numTiles; tile++) {
            const baseTime = tile * 35; // Reduced spacing for better overlap

            // Producer 0: TMA Load A (parallel with Producer 1)
            ops.push({
                id: opId++,
                streamId: 'producer_0',
                type: 'TMA_A',
                startTime: baseTime,
                endTime: baseTime + this.opTypes.TMA_A.duration,
                tile: tile,
                label: `A${tile}`,
            });

            // Producer 1: TMA Load B (parallel with Producer 0)
            ops.push({
                id: opId++,
                streamId: 'producer_1',
                type: 'TMA_B',
                startTime: baseTime,
                endTime: baseTime + this.opTypes.TMA_B.duration,
                tile: tile,
                label: `B${tile}`,
            });

            // WGMMA starts after loads complete
            const wgmmaStart = baseTime + this.opTypes.TMA_A.duration + 3;
            const halfWGMMADuration = this.opTypes.WGMMA.duration * 0.6; // Each does ~half the work

            // Consumer 0 & 1: WGMMA on Top Half (parallel)
            const consumer01 = tile % 2; // Alternate between consumer 0 and 1
            ops.push({
                id: opId++,
                streamId: `consumer_${consumer01}`,
                type: 'WGMMA',
                startTime: wgmmaStart,
                endTime: wgmmaStart + halfWGMMADuration,
                tile: tile,
                label: `C${tile}`,
            });

            // Consumer 2 & 3: WGMMA on Bottom Half (parallel with top)
            const consumer23 = 2 + (tile % 2); // Alternate between consumer 2 and 3
            ops.push({
                id: opId++,
                streamId: `consumer_${consumer23}`,
                type: 'WGMMA',
                startTime: wgmmaStart,
                endTime: wgmmaStart + halfWGMMADuration,
                tile: tile,
                label: `C${tile}`,
            });

            // Epilogue - runs on one consumer after WGMMA
            const epilogueStart = wgmmaStart + halfWGMMADuration + 1;
            ops.push({
                id: opId++,
                streamId: `consumer_${consumer01}`,
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
                streamId: `consumer_${consumer01}`,
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
        const legendGrid = document.getElementById('persistentCoopLegendGrid');
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
        const timeTrack = document.getElementById('persistentCoopTimeTrack');
        const totalTime = this.getTotalTime();
        const numMarkers = Math.floor(totalTime / 40) + 1;

        timeTrack.innerHTML = '';

        // Time markers
        for (let i = 0; i < numMarkers; i++) {
            const marker = document.createElement('div');
            marker.className = 'pipeline-time-marker';
            marker.style.left = `${i * 40 * this.config.timeScale}px`;
            marker.textContent = i * 40;
            timeTrack.appendChild(marker);
        }

        // Current time marker
        const currentMarker = document.createElement('div');
        currentMarker.id = 'persistentCoopCurrentTimeMarker';
        currentMarker.className = 'pipeline-current-time-marker';
        currentMarker.style.left = '0px';
        timeTrack.appendChild(currentMarker);

        // Set width
        timeTrack.style.minWidth = `${totalTime * this.config.timeScale}px`;
    }

    initializeStreamLanes() {
        const streamLanes = document.getElementById('persistentCoopStreamLanes');
        streamLanes.innerHTML = '';

        this.streams.forEach(stream => {
            const lane = document.createElement('div');
            lane.className = 'pipeline-stream-lane';
            const totalTime = this.getTotalTime();
            lane.innerHTML = `
                <div class="pipeline-stream-label pipeline-${stream.type}">
                    <div class="pipeline-stream-name">${stream.name}</div>
                    <div class="pipeline-stream-type">${stream.type === 'producer' ? 'TMA Loads' : 'Compute'}</div>
                </div>
                <div class="pipeline-lane-track" id="persistentCoop-lane-${stream.id}" style="min-width: ${totalTime * this.config.timeScale}px"></div>
            `;
            streamLanes.appendChild(lane);
        });

        // Create operation elements
        this.state.operations.forEach(op => {
            const opType = this.opTypes[op.type];
            const lane = document.getElementById(`persistentCoop-lane-${op.streamId}`);

            const opElement = document.createElement('div');
            opElement.id = `persistentCoop-op-${op.id}`;
            opElement.className = `pipeline-operation ${opType.color}`;
            opElement.style.left = `${op.startTime * this.config.timeScale}px`;
            opElement.style.width = `${(op.endTime - op.startTime) * this.config.timeScale}px`;
            opElement.style.display = 'none'; // Initially hidden
            opElement.innerHTML = `
                <div class="pipeline-operation-progress" id="persistentCoop-progress-${op.id}" style="width: 0%"></div>
                <div class="pipeline-operation-content">
                    <div class="pipeline-operation-icon">${opType.icon}</div>
                </div>
            `;

            lane.appendChild(opElement);
        });
    }

    updateVisualization() {
        const currentMarker = document.getElementById('persistentCoopCurrentTimeMarker');

        if (currentMarker) {
            currentMarker.style.left = `${this.state.currentTime * this.config.timeScale}px`;
        }

        this.state.operations.forEach(op => {
            const isActive = this.state.currentTime >= op.startTime && this.state.currentTime <= op.endTime;
            const isVisible = this.state.currentTime >= op.startTime;

            // Cache DOM elements on first access
            if (!op._cachedElements) {
                op._cachedElements = {
                    opElement: document.getElementById(`persistentCoop-op-${op.id}`),
                    progressBar: document.getElementById(`persistentCoop-progress-${op.id}`)
                };
            }

            const { opElement, progressBar } = op._cachedElements;

            if (opElement) {
                // Track previous state to avoid unnecessary updates
                const prevActive = op._prevActive;
                const prevVisible = op._prevVisible;

                // Only update if state changed
                if (isVisible !== prevVisible) {
                    opElement.style.display = isVisible ? 'flex' : 'none';
                    op._prevVisible = isVisible;
                }

                if (isVisible && isActive !== prevActive) {
                    opElement.style.opacity = isActive ? '1' : '0.6';

                    if (isActive) {
                        opElement.classList.add('active');
                        if (!op._pulseElement) {
                            op._pulseElement = document.createElement('div');
                            op._pulseElement.className = 'pipeline-operation-pulse';
                            opElement.appendChild(op._pulseElement);
                        }
                    } else {
                        opElement.classList.remove('active');
                        if (op._pulseElement) {
                            op._pulseElement.remove();
                            op._pulseElement = null;
                        }
                    }
                    op._prevActive = isActive;
                }

                if (progressBar && isActive) {
                    const progress = (this.state.currentTime - op.startTime) / (op.endTime - op.startTime);
                    progressBar.style.width = `${progress * 100}%`;
                } else if (progressBar && this.state.currentTime > op.endTime && op._prevProgressComplete !== true) {
                    progressBar.style.width = '100%';
                    op._prevProgressComplete = true;
                } else if (progressBar && !isVisible && op._prevProgressComplete !== false) {
                    progressBar.style.width = '0%';
                    op._prevProgressComplete = false;
                }
            }
        });
    }

    animate() {
        if (this.state.isPlaying) {
            this.state.currentTime++;

            const totalTime = this.getTotalTime();
            if (this.state.currentTime >= totalTime) {
                this.state.currentTime = totalTime;
                this.stopAnimation();
            }

            this.updateVisualization();
        }
    }

    startAnimation() {
        this.state.isPlaying = true;
        this.state.animationTimer = setInterval(() => this.animate(), this.config.animationInterval);

        document.getElementById('persistentCoopPlayIcon').textContent = '⏸';
        document.getElementById('persistentCoopPlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('persistentCoopPlayIcon').textContent = '▶';
        document.getElementById('persistentCoopPlayText').textContent = 'Play';
    }

    resetAnimation() {
        this.stopAnimation();
        this.state.currentTime = 0;
        this.updateVisualization();
    }

    togglePlay() {
        const totalTime = this.getTotalTime();
        if (!this.state.isPlaying && this.state.currentTime >= totalTime) {
            this.resetAnimation();
            setTimeout(() => this.startAnimation(), 100);
        } else if (this.state.isPlaying) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
}

// Auto-initialize persistent cooperative viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('persistent-cooperative-viz');
    if (container) {
        new PersistentCooperativeViz('persistent-cooperative-viz');
    }
});

// ============================================================================
// Wave Quantization Visualization
// ============================================================================

class WaveQuantizationViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        this.config = {
            numSMs: 4,
            numTiles: 9,
            numWaves: 3,
            tileTime: 40,
            animationInterval: 50,
        };

        this.state = {
            isPlaying: false,
            currentTime: 0,
            animationTimer: null,
        };

        this.init();
    }

    init() {
        const totalTime = this.config.tileTime * this.config.numWaves;

        this.container.innerHTML = commonPipelineStyles + `
            <div class="pipeline-container">
                <h1 class="pipeline-title">Wave Quantization Problem</h1>
                <p class="pipeline-subtitle">Simple Example: 4 SMs computing 9 tiles (3 waves)</p>

                <div class="pipeline-controls">
                    <button id="waveQuantPlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="waveQuantPlayIcon">▶</span>
                        <span id="waveQuantPlayText">Play</span>
                    </button>
                    <button id="waveQuantResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <div class="pipeline-time-display">
                    <div class="pipeline-time-value">
                        Wave: <span class="current" id="waveQuantCurrentWave">1</span> / 3
                    </div>
                </div>

                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">SM Utilization Timeline</h3>
                    <div id="waveQuantSMGrid"></div>
                </div>
            </div>
        `;

        this.initializeSMGrid();

        document.getElementById('waveQuantPlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('waveQuantResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    initializeSMGrid() {
        const grid = document.getElementById('waveQuantSMGrid');
        grid.innerHTML = '';

        // Create a visual grid showing SMs across waves
        const gridContainer = document.createElement('div');
        gridContainer.style.cssText = 'display: flex; flex-direction: column; gap: 1.5rem; margin-top: 1rem;';

        for (let wave = 0; wave < this.config.numWaves; wave++) {
            const waveDiv = document.createElement('div');

            // Main row container with SM boxes and info
            const waveRow = document.createElement('div');
            waveRow.style.cssText = 'display: flex; gap: 1.5rem; align-items: stretch;';

            // Left side: SM boxes
            const smContainer = document.createElement('div');
            smContainer.style.cssText = 'flex: 2; display: flex; flex-direction: column; gap: 0.5rem;';

            const waveLabel = document.createElement('div');
            waveLabel.style.cssText = 'color: #ffffff; font-size: 0.85rem; font-weight: bold;';
            waveLabel.textContent = `Wave ${wave + 1}`;
            smContainer.appendChild(waveLabel);

            const smRow = document.createElement('div');
            smRow.style.cssText = 'display: flex; gap: 1rem; align-items: center;';

            for (let sm = 0; sm < this.config.numSMs; sm++) {
                const tileNum = wave * this.config.numSMs + sm;
                const smBox = document.createElement('div');
                smBox.id = `waveQuant-w${wave}-sm${sm}`;
                smBox.style.cssText = `
                    flex: 1;
                    height: 80px;
                    background: #1f2937;
                    border: 2px solid #374151;
                    border-radius: 0.5rem;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s;
                    position: relative;
                `;

                const smLabel = document.createElement('div');
                smLabel.style.cssText = 'font-size: 0.75rem; color: #6b7280; font-weight: bold;';
                smLabel.textContent = `SM ${sm}`;

                const tileLabel = document.createElement('div');
                tileLabel.id = `waveQuant-w${wave}-sm${sm}-tile`;
                tileLabel.style.cssText = 'font-size: 1.25rem; color: #9ca3af; font-weight: bold; margin-top: 0.25rem;';
                tileLabel.textContent = tileNum < this.config.numTiles ? `Tile ${tileNum}` : '—';

                smBox.appendChild(smLabel);
                smBox.appendChild(tileLabel);
                smRow.appendChild(smBox);
            }

            smContainer.appendChild(smRow);
            waveRow.appendChild(smContainer);

            // Right side: Wave info card
            const infoCard = document.createElement('div');
            infoCard.style.cssText = `
                flex: 1;
                background: #1f2937;
                border: 2px solid ${wave === 2 ? '#ef4444' : '#22c55e'};
                border-radius: 0.5rem;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 0.5rem;
            `;

            const numActiveSMs = Math.min(this.config.numSMs, this.config.numTiles - wave * this.config.numSMs);
            const utilization = (numActiveSMs / this.config.numSMs * 100).toFixed(0);
            const isPartial = wave === 2;

            const tilesInWave = [];
            for (let sm = 0; sm < this.config.numSMs; sm++) {
                const tileNum = wave * this.config.numSMs + sm;
                if (tileNum < this.config.numTiles) {
                    tilesInWave.push(tileNum);
                }
            }

            infoCard.innerHTML = `
                <div style="font-size: 0.9rem; font-weight: bold; color: ${isPartial ? '#ef4444' : '#22c55e'};">
                    ${isPartial ? '⚠️ Partial Wave' : '✓ Full Wave'}
                </div>
                <div style="font-size: 0.8rem; color: #d1d5db;">
                    <strong>SMs:</strong> ${numActiveSMs} / ${this.config.numSMs} (${utilization}%)
                </div>
                <div style="font-size: 0.8rem; color: #d1d5db;">
                    <strong>Tiles:</strong> ${tilesInWave.join(', ')}
                </div>
                ${isPartial ? `<div style="font-size: 0.75rem; color: #fca5a5; margin-top: 0.25rem;"><strong>${this.config.numSMs - numActiveSMs} SMs idle!</strong></div>` : ''}
            `;

            waveRow.appendChild(infoCard);
            waveDiv.appendChild(waveRow);
            gridContainer.appendChild(waveDiv);
        }

        grid.appendChild(gridContainer);
    }

    updateVisualization() {
        const currentWave = Math.floor(this.state.currentTime / this.config.tileTime);
        const timeInWave = this.state.currentTime % this.config.tileTime;
        const waveProgress = timeInWave / this.config.tileTime;

        document.getElementById('waveQuantCurrentWave').textContent = Math.min(currentWave + 1, this.config.numWaves);

        for (let wave = 0; wave < this.config.numWaves; wave++) {
            for (let sm = 0; sm < this.config.numSMs; sm++) {
                const tileNum = wave * this.config.numSMs + sm;
                const smBox = document.getElementById(`waveQuant-w${wave}-sm${sm}`);
                const tileLabel = document.getElementById(`waveQuant-w${wave}-sm${sm}-tile`);

                if (!smBox || !tileLabel) continue;

                let bgColor, borderColor, labelColor;

                if (wave < currentWave || (wave === currentWave && waveProgress >= 0.99)) {
                    // Completed
                    if (tileNum < this.config.numTiles) {
                        bgColor = '#22c55e';
                        borderColor = '#16a34a';
                        labelColor = '#ffffff';
                    } else {
                        bgColor = '#1f2937';
                        borderColor = '#374151';
                        labelColor = '#4b5563';
                    }
                } else if (wave === currentWave) {
                    // Active wave
                    if (tileNum < this.config.numTiles) {
                        bgColor = '#8b5cf6';
                        borderColor = '#7c3aed';
                        labelColor = '#ffffff';
                        smBox.style.transform = 'scale(1.05)';
                        smBox.style.boxShadow = '0 0 20px rgba(139, 92, 246, 0.5)';
                    } else {
                        bgColor = '#374151';
                        borderColor = '#ef4444';
                        labelColor = '#9ca3af';
                        smBox.style.boxShadow = 'none';
                    }
                } else {
                    // Future wave
                    bgColor = '#1f2937';
                    borderColor = '#374151';
                    labelColor = '#6b7280';
                    smBox.style.transform = 'scale(1)';
                    smBox.style.boxShadow = 'none';
                }

                smBox.style.background = bgColor;
                smBox.style.borderColor = borderColor;
                tileLabel.style.color = labelColor;

                if (wave !== currentWave || tileNum >= this.config.numTiles) {
                    smBox.style.transform = 'scale(1)';
                    smBox.style.boxShadow = 'none';
                }
            }
        }
    }

    animate() {
        if (this.state.isPlaying) {
            this.state.currentTime++;

            const totalTime = this.config.tileTime * this.config.numWaves;
            if (this.state.currentTime >= totalTime) {
                this.state.currentTime = totalTime;
                this.stopAnimation();
            }

            this.updateVisualization();
        }
    }

    startAnimation() {
        this.state.isPlaying = true;
        this.state.animationTimer = setInterval(() => this.animate(), this.config.animationInterval);

        document.getElementById('waveQuantPlayIcon').textContent = '⏸';
        document.getElementById('waveQuantPlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('waveQuantPlayIcon').textContent = '▶';
        document.getElementById('waveQuantPlayText').textContent = 'Play';
    }

    resetAnimation() {
        this.stopAnimation();
        this.state.currentTime = 0;
        this.updateVisualization();
    }

    togglePlay() {
        const totalTime = this.config.tileTime * this.config.numWaves;
        if (!this.state.isPlaying && this.state.currentTime >= totalTime) {
            this.resetAnimation();
            setTimeout(() => this.startAnimation(), 100);
        } else if (this.state.isPlaying) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
}

// Auto-initialize wave quantization viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('wave-quantization-viz');
    if (container) {
        new WaveQuantizationViz('wave-quantization-viz');
    }
});

// ============================================================================
// Split-K Visualization
// ============================================================================

class SplitKViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        this.config = {
            numSMs: 4,
            baseTiles: 9,
            splitFactor: 2, // Split each tile into 2 K-slices
            tileTime: 25, // Time for compute per K-slice
            barrierTime: 5, // Time for barrier wait
            reduceTime: 8, // Time for reduction
            animationInterval: 50,
        };

        // Total work units = baseTiles * splitFactor = 18
        this.config.numTiles = this.config.baseTiles * this.config.splitFactor;
        this.config.numWaves = Math.ceil(this.config.numTiles / this.config.numSMs);

        this.state = {
            isPlaying: false,
            currentTime: 0,
            animationTimer: null,
        };

        this.init();
    }

    init() {
        const timePerWave = this.config.tileTime + this.config.barrierTime + this.config.reduceTime;
        const totalTime = timePerWave * this.config.numWaves;

        this.container.innerHTML = commonPipelineStyles + `
            <div class="pipeline-container">
                <h1 class="pipeline-title">Split-K Schedule</h1>
                <p class="pipeline-subtitle">4 SMs, 9 tiles split into 2 K-slices each = 18 work units</p>

                <div class="pipeline-controls">
                    <button id="splitKPlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="splitKPlayIcon">▶</span>
                        <span id="splitKPlayText">Play</span>
                    </button>
                    <button id="splitKResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <div class="pipeline-time-display">
                    <div class="pipeline-time-value">
                        Wave: <span class="current" id="splitKCurrentWave">1</span> / ${this.config.numWaves}
                    </div>
                </div>

                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">SM Timeline with Synchronization</h3>
                    <div style="margin-bottom: 1rem; display: flex; gap: 1.5rem; justify-content: center; font-size: 0.85rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 20px; height: 20px; background: #8b5cf6; border-radius: 3px;"></div>
                            <span>Compute</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 20px; height: 20px; background: #f59e0b; border-radius: 3px;"></div>
                            <span>Arrive (K=0)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 20px; height: 20px; background: #3b82f6; border-radius: 3px;"></div>
                            <span>Reduce (K=1)</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 20px; height: 20px; background: #22c55e; border-radius: 3px;"></div>
                            <span>Complete</span>
                        </div>
                    </div>
                    <div id="splitKSMGrid"></div>
                </div>
            </div>
        `;

        this.initializeSMGrid();

        document.getElementById('splitKPlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('splitKResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    initializeSMGrid() {
        const grid = document.getElementById('splitKSMGrid');
        grid.innerHTML = '';

        const gridContainer = document.createElement('div');
        gridContainer.style.cssText = 'display: flex; flex-direction: column; gap: 1.5rem; margin-top: 1rem;';

        for (let wave = 0; wave < this.config.numWaves; wave++) {
            const waveDiv = document.createElement('div');

            const waveRow = document.createElement('div');
            waveRow.style.cssText = 'display: flex; gap: 1.5rem; align-items: stretch;';

            // Left side: SM boxes
            const smContainer = document.createElement('div');
            smContainer.style.cssText = 'flex: 2; display: flex; flex-direction: column; gap: 0.5rem;';

            const waveLabel = document.createElement('div');
            waveLabel.style.cssText = 'color: #ffffff; font-size: 0.85rem; font-weight: bold;';
            waveLabel.textContent = `Wave ${wave + 1}`;
            smContainer.appendChild(waveLabel);

            const smRow = document.createElement('div');
            smRow.style.cssText = 'display: flex; gap: 1rem; align-items: stretch;';

            for (let sm = 0; sm < this.config.numSMs; sm++) {
                const tileNum = wave * this.config.numSMs + sm;
                const smCol = document.createElement('div');
                smCol.style.cssText = 'flex: 1; display: flex; flex-direction: column; gap: 0.3rem;';

                // Compute block
                const computeBox = document.createElement('div');
                computeBox.id = `splitK-w${wave}-sm${sm}-compute`;
                computeBox.style.cssText = `
                    height: 45px;
                    background: #1f2937;
                    border: 2px solid #374151;
                    border-radius: 0.4rem;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s;
                    font-size: 0.75rem;
                    color: #6b7280;
                `;
                let kSlice, isEvenSlice;
                if (tileNum < this.config.numTiles) {
                    const baseTile = Math.floor(tileNum / this.config.splitFactor);
                    kSlice = tileNum % this.config.splitFactor;
                    isEvenSlice = (kSlice === 0);
                    computeBox.innerHTML = `<div style="font-weight: bold;">T${baseTile}[${kSlice}]</div><div style="font-size: 0.65rem;">SM ${sm}</div>`;
                } else {
                    computeBox.innerHTML = `<div style="font-weight: bold;">—</div><div style="font-size: 0.65rem;">SM ${sm}</div>`;
                    isEvenSlice = true; // Default for idle SMs
                }

                // Barrier block (only for even K-slices)
                const barrierBox = document.createElement('div');
                barrierBox.id = `splitK-w${wave}-sm${sm}-barrier`;
                barrierBox.style.cssText = `
                    height: 15px;
                    background: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 0.3rem;
                    transition: all 0.3s;
                    font-size: 0.6rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #6b7280;
                `;
                barrierBox.textContent = (tileNum < this.config.numTiles && isEvenSlice) ? 'Arrive' : '';

                // Reduce block (only for odd K-slices)
                const reduceBox = document.createElement('div');
                reduceBox.id = `splitK-w${wave}-sm${sm}-reduce`;
                reduceBox.style.cssText = `
                    height: 20px;
                    background: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 0.3rem;
                    transition: all 0.3s;
                    font-size: 0.65rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #6b7280;
                `;
                reduceBox.textContent = (tileNum < this.config.numTiles && !isEvenSlice) ? 'Reduce' : '';

                smCol.appendChild(computeBox);
                smCol.appendChild(barrierBox);
                smCol.appendChild(reduceBox);
                smRow.appendChild(smCol);
            }

            smContainer.appendChild(smRow);
            waveRow.appendChild(smContainer);

            // Right side: Wave info
            const infoCard = document.createElement('div');
            const numActiveSMs = Math.min(this.config.numSMs, this.config.numTiles - wave * this.config.numSMs);
            const utilization = (numActiveSMs / this.config.numSMs * 100).toFixed(0);
            const isPartial = numActiveSMs < this.config.numSMs;

            infoCard.style.cssText = `
                flex: 1;
                background: #1f2937;
                border: 2px solid ${isPartial ? '#ef4444' : '#22c55e'};
                border-radius: 0.5rem;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 0.5rem;
            `;

            const tilesInWave = [];
            for (let sm = 0; sm < this.config.numSMs; sm++) {
                const tileNum = wave * this.config.numSMs + sm;
                if (tileNum < this.config.numTiles) {
                    const baseTile = Math.floor(tileNum / this.config.splitFactor);
                    const kSlice = tileNum % this.config.splitFactor;
                    tilesInWave.push(`T${baseTile}[${kSlice}]`);
                }
            }

            infoCard.innerHTML = `
                <div style="font-size: 0.9rem; font-weight: bold; color: ${isPartial ? '#ef4444' : '#22c55e'};">
                    ${isPartial ? '⚠️ Partial Wave' : '✓ Full Wave'}
                </div>
                <div style="font-size: 0.8rem; color: #d1d5db;">
                    <strong>SMs:</strong> ${numActiveSMs} / ${this.config.numSMs} (${utilization}%)
                </div>
                <div style="font-size: 0.75rem; color: #d1d5db;">
                    <strong>K-slices:</strong> ${tilesInWave.join(', ')}
                </div>
                ${isPartial ? `<div style="font-size: 0.75rem; color: #fca5a5; margin-top: 0.25rem;"><strong>${this.config.numSMs - numActiveSMs} SMs idle</strong></div>` : ''}
            `;

            waveRow.appendChild(infoCard);
            waveDiv.appendChild(waveRow);
            gridContainer.appendChild(waveDiv);
        }

        grid.appendChild(gridContainer);
    }

    updateVisualization() {
        const timePerWave = this.config.tileTime + this.config.barrierTime + this.config.reduceTime;
        const currentWave = Math.floor(this.state.currentTime / timePerWave);
        const timeInWave = this.state.currentTime % timePerWave;

        document.getElementById('splitKCurrentWave').textContent = Math.min(currentWave + 1, this.config.numWaves);

        for (let wave = 0; wave < this.config.numWaves; wave++) {
            for (let sm = 0; sm < this.config.numSMs; sm++) {
                const tileNum = wave * this.config.numSMs + sm;
                const computeBox = document.getElementById(`splitK-w${wave}-sm${sm}-compute`);
                const barrierBox = document.getElementById(`splitK-w${wave}-sm${sm}-barrier`);
                const reduceBox = document.getElementById(`splitK-w${wave}-sm${sm}-reduce`);

                if (!computeBox || !barrierBox || !reduceBox) continue;

                let computeColor, barrierColor, reduceColor;
                let computeBorder, barrierBorder, reduceBorder;

                const kSlice = tileNum % this.config.splitFactor;
                const isEvenSlice = (kSlice === 0);

                if (wave < currentWave) {
                    // Completed wave
                    if (tileNum < this.config.numTiles) {
                        computeColor = barrierColor = reduceColor = '#22c55e';
                        computeBorder = barrierBorder = reduceBorder = '#16a34a';
                    } else {
                        computeColor = barrierColor = reduceColor = '#1f2937';
                        computeBorder = barrierBorder = reduceBorder = '#374151';
                    }
                } else if (wave === currentWave) {
                    // Active wave
                    if (tileNum < this.config.numTiles) {
                        if (timeInWave < this.config.tileTime) {
                            // Computing
                            computeColor = '#8b5cf6';
                            computeBorder = '#7c3aed';
                            barrierColor = reduceColor = '#1f2937';
                            barrierBorder = reduceBorder = '#374151';
                            computeBox.style.boxShadow = '0 0 15px rgba(139, 92, 246, 0.5)';
                        } else if (timeInWave < this.config.tileTime + this.config.barrierTime) {
                            // Barrier/Reduce phase - even slices arrive, odd slices reduce
                            computeColor = '#22c55e';
                            computeBorder = '#16a34a';

                            if (isEvenSlice) {
                                // Even K-slice (K=0): arrive at barrier and wait
                                barrierColor = '#f59e0b';
                                barrierBorder = '#d97706';
                                reduceColor = '#1f2937';
                                reduceBorder = '#374151';
                                barrierBox.style.boxShadow = '0 0 15px rgba(245, 158, 11, 0.5)';
                            } else {
                                // Odd K-slice (K=1): reduce previous results
                                barrierColor = '#1f2937';
                                barrierBorder = '#374151';
                                reduceColor = '#3b82f6';
                                reduceBorder = '#2563eb';
                                reduceBox.style.boxShadow = '0 0 15px rgba(59, 130, 246, 0.5)';
                            }
                        } else {
                            // Final phase - all complete
                            computeColor = '#22c55e';
                            computeBorder = '#16a34a';
                            barrierColor = '#22c55e';
                            barrierBorder = '#16a34a';
                            reduceColor = '#22c55e';
                            reduceBorder = '#16a34a';
                        }
                    } else {
                        // Idle SM
                        computeColor = barrierColor = reduceColor = '#374151';
                        computeBorder = '#ef4444';
                        barrierBorder = reduceBorder = '#374151';
                    }
                } else {
                    // Future wave
                    computeColor = barrierColor = reduceColor = '#1f2937';
                    computeBorder = barrierBorder = reduceBorder = '#374151';
                }

                computeBox.style.background = computeColor;
                computeBox.style.borderColor = computeBorder;
                computeBox.style.color = (computeColor === '#1f2937') ? '#6b7280' : '#ffffff';
                barrierBox.style.background = barrierColor;
                barrierBox.style.borderColor = barrierBorder;
                barrierBox.style.color = (barrierColor === '#1f2937') ? '#6b7280' : '#ffffff';
                reduceBox.style.background = reduceColor;
                reduceBox.style.borderColor = reduceBorder;
                reduceBox.style.color = (reduceColor === '#1f2937') ? '#6b7280' : '#ffffff';

                if (wave !== currentWave || tileNum >= this.config.numTiles) {
                    computeBox.style.boxShadow = 'none';
                    barrierBox.style.boxShadow = 'none';
                    reduceBox.style.boxShadow = 'none';
                }
            }
        }
    }

    animate() {
        if (this.state.isPlaying) {
            this.state.currentTime++;

            const timePerWave = this.config.tileTime + this.config.barrierTime + this.config.reduceTime;
            const totalTime = timePerWave * this.config.numWaves;

            if (this.state.currentTime >= totalTime) {
                this.state.currentTime = totalTime;
                this.stopAnimation();
            }

            this.updateVisualization();
        }
    }

    startAnimation() {
        this.state.isPlaying = true;
        this.state.animationTimer = setInterval(() => this.animate(), this.config.animationInterval);

        document.getElementById('splitKPlayIcon').textContent = '⏸';
        document.getElementById('splitKPlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('splitKPlayIcon').textContent = '▶';
        document.getElementById('splitKPlayText').textContent = 'Play';
    }

    resetAnimation() {
        this.stopAnimation();
        this.state.currentTime = 0;
        this.updateVisualization();
    }

    togglePlay() {
        const timePerWave = this.config.tileTime + this.config.barrierTime + this.config.reduceTime;
        const totalTime = timePerWave * this.config.numWaves;

        if (!this.state.isPlaying && this.state.currentTime >= totalTime) {
            this.resetAnimation();
            setTimeout(() => this.startAnimation(), 100);
        } else if (this.state.isPlaying) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
}

// Auto-initialize Split-K viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('split-k-viz');
    if (container) {
        new SplitKViz('split-k-viz');
    }
});

// ============================================================================
// Stream-K Visualization
// ============================================================================

class StreamKViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        this.config = {
            numSMs: 4,
            totalTiles: 9,
            tilesPerSM: 2.25,
            tileTime: 30,
            animationInterval: 50,
        };

        // Define fractional tile assignments for each SM
        // SM0: T0, T1, T2 (25%), Reduce
        // SM1: T2 (75%), Barrier, T3, T4 (50%), Reduce
        // SM2: T4 (50%), Barrier, T5, T6 (75%), Reduce
        // SM3: T6 (25%), Barrier, T7, T8
        this.assignments = [
            [
                { type: 'tile', tile: 0, fraction: 1.0, color: '#8b5cf6' },
                { type: 'tile', tile: 1, fraction: 1.0, color: '#a78bfa' },
                { type: 'tile', tile: 2, fraction: 0.25, color: '#c4b5fd' },
                { type: 'sync', syncOp: 'reduce', fraction: 0.15 }
            ],
            [
                { type: 'tile', tile: 2, fraction: 0.75, color: '#c4b5fd' },
                { type: 'sync', syncOp: 'barrier', fraction: 0.15 },
                { type: 'tile', tile: 3, fraction: 1.0, color: '#06b6d4' },
                { type: 'tile', tile: 4, fraction: 0.5, color: '#22d3ee' },
                { type: 'sync', syncOp: 'reduce', fraction: 0.15 }
            ],
            [
                { type: 'tile', tile: 4, fraction: 0.5, color: '#22d3ee' },
                { type: 'sync', syncOp: 'barrier', fraction: 0.15 },
                { type: 'tile', tile: 5, fraction: 1.0, color: '#10b981' },
                { type: 'tile', tile: 6, fraction: 0.75, color: '#34d399' },
                { type: 'sync', syncOp: 'reduce', fraction: 0.15 }
            ],
            [
                { type: 'tile', tile: 6, fraction: 0.25, color: '#34d399' },
                { type: 'sync', syncOp: 'barrier', fraction: 0.15 },
                { type: 'tile', tile: 7, fraction: 1.0, color: '#f59e0b' },
                { type: 'tile', tile: 8, fraction: 1.0, color: '#fbbf24' }
            ]
        ];

        this.state = {
            isPlaying: false,
            currentTime: 0,
            animationTimer: null,
        };

        this.init();
    }

    getTotalTime() {
        // Calculate the actual total time based on the longest SM assignment
        let maxTime = 0;
        for (let sm = 0; sm < this.config.numSMs; sm++) {
            let smTime = 0;
            this.assignments[sm].forEach(item => {
                smTime += item.fraction * this.config.tileTime;
            });
            maxTime = Math.max(maxTime, smTime);
        }
        return maxTime;
    }

    init() {
        const totalTime = this.getTotalTime();

        this.container.innerHTML = commonPipelineStyles + `
            <div class="pipeline-container">
                <h1 class="pipeline-title">Stream-K Schedule</h1>
                <p class="pipeline-subtitle">4 SMs, 9 tiles → 2.25 tiles/SM (fractional assignment)</p>

                <div class="pipeline-controls">
                    <button id="streamKPlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="streamKPlayIcon">▶</span>
                        <span id="streamKPlayText">Play</span>
                    </button>
                    <button id="streamKResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <div class="pipeline-time-display">
                    <div class="pipeline-time-value">
                        Progress: <span class="current" id="streamKProgress">0</span>%
                    </div>
                </div>

                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">Fractional Tile Assignment per SM</h3>
                    <div style="display: flex; gap: 1.5rem; margin-top: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 24px; height: 24px; background: #8b5cf6; border-radius: 0.25rem; display: flex; align-items: center; justify-content: center; font-size: 0.75rem;">T</div>
                            <span style="color: #d1d5db; font-size: 0.875rem;">Compute Tile</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 24px; height: 24px; background: #f59e0b; border-radius: 0.25rem; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">🚧</div>
                            <span style="color: #d1d5db; font-size: 0.875rem;">Barrier</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <div style="width: 24px; height: 24px; background: #3b82f6; border-radius: 0.25rem; display: flex; align-items: center; justify-content: center; font-size: 0.85rem;">🔄</div>
                            <span style="color: #d1d5db; font-size: 0.875rem;">Reduce</span>
                        </div>
                    </div>
                    <div id="streamKSMGrid" style="margin-top: 1rem;"></div>
                </div>
            </div>
        `;

        this.initializeSMGrid();

        document.getElementById('streamKPlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('streamKResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    initializeSMGrid() {
        const grid = document.getElementById('streamKSMGrid');
        grid.innerHTML = '';

        const gridContainer = document.createElement('div');
        gridContainer.style.cssText = 'display: flex; flex-direction: column; gap: 1rem;';

        for (let sm = 0; sm < this.config.numSMs; sm++) {
            const smRow = document.createElement('div');
            smRow.style.cssText = 'display: flex; gap: 0.75rem; align-items: center;';

            // SM label
            const smLabel = document.createElement('div');
            smLabel.style.cssText = 'min-width: 50px; font-weight: bold; color: #9ca3af; font-size: 0.9rem;';
            smLabel.textContent = `SM ${sm}`;
            smRow.appendChild(smLabel);

            // Tile bars container (single row with all tiles and sync ops)
            const tilesContainer = document.createElement('div');
            tilesContainer.style.cssText = 'flex: 1; display: flex; gap: 0.25rem; height: 50px;';

            const assignment = this.assignments[sm];
            let totalFraction = 0;
            assignment.forEach(item => totalFraction += item.fraction);

            assignment.forEach((item, idx) => {
                const bar = document.createElement('div');
                bar.id = `streamK-sm${sm}-item${idx}`;
                const widthPercent = (item.fraction / totalFraction) * 100;

                bar.style.cssText = `
                    width: ${widthPercent}%;
                    background: #1f2937;
                    border: 2px solid #374151;
                    border-radius: 0.4rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.75rem;
                    font-weight: bold;
                    color: #6b7280;
                    transition: all 0.3s;
                    position: relative;
                `;

                if (item.type === 'tile') {
                    const label = item.fraction === 1.0
                        ? `T${item.tile}`
                        : `T${item.tile} (${(item.fraction * 100).toFixed(0)}%)`;
                    bar.textContent = label;
                    bar.dataset.type = 'tile';
                    bar.dataset.color = item.color;
                } else if (item.type === 'sync') {
                    bar.textContent = item.syncOp === 'barrier' ? '🚧' : '🔄';
                    bar.style.fontSize = '1.5rem';
                    bar.dataset.type = 'sync';
                    bar.dataset.syncOp = item.syncOp;
                } else if (item.type === 'padding') {
                    bar.style.opacity = '0.3';
                    bar.dataset.type = 'padding';
                }

                tilesContainer.appendChild(bar);
            });

            smRow.appendChild(tilesContainer);
            gridContainer.appendChild(smRow);
        }

        grid.appendChild(gridContainer);
    }

    updateVisualization() {
        const totalTime = this.getTotalTime();
        const progress = (this.state.currentTime / totalTime) * 100;

        document.getElementById('streamKProgress').textContent = Math.min(100, Math.floor(progress));

        for (let sm = 0; sm < this.config.numSMs; sm++) {
            const assignment = this.assignments[sm];
            let cumulativeTime = 0;

            assignment.forEach((item, idx) => {
                const bar = document.getElementById(`streamK-sm${sm}-item${idx}`);
                if (!bar) return;

                const itemStartTime = cumulativeTime;
                const itemDuration = item.fraction * this.config.tileTime;
                const itemEndTime = itemStartTime + itemDuration;

                let bgColor, borderColor, textColor;

                if (item.type === 'tile') {
                    if (this.state.currentTime >= itemEndTime) {
                        // Tile completed
                        bgColor = '#22c55e';
                        borderColor = '#16a34a';
                        textColor = '#ffffff';
                        bar.style.boxShadow = 'none';
                    } else if (this.state.currentTime >= itemStartTime) {
                        // Tile active
                        bgColor = item.color;
                        borderColor = item.color;
                        textColor = '#ffffff';
                        bar.style.boxShadow = `0 0 15px ${item.color}80`;
                    } else {
                        // Tile pending
                        bgColor = '#1f2937';
                        borderColor = '#374151';
                        textColor = '#6b7280';
                        bar.style.boxShadow = 'none';
                    }
                } else if (item.type === 'sync') {
                    if (this.state.currentTime >= itemEndTime) {
                        // Sync operation completed
                        if (item.syncOp === 'barrier') {
                            bgColor = '#f59e0b';
                            borderColor = '#d97706';
                            textColor = '#ffffff';
                        } else {
                            bgColor = '#3b82f6';
                            borderColor = '#2563eb';
                            textColor = '#ffffff';
                        }
                        bar.style.boxShadow = 'none';
                    } else if (this.state.currentTime >= itemStartTime) {
                        // Sync operation active
                        if (item.syncOp === 'barrier') {
                            bgColor = '#f59e0b';
                            borderColor = '#d97706';
                            textColor = '#ffffff';
                            bar.style.boxShadow = '0 0 15px rgba(245, 158, 11, 0.5)';
                        } else {
                            bgColor = '#3b82f6';
                            borderColor = '#2563eb';
                            textColor = '#ffffff';
                            bar.style.boxShadow = '0 0 15px rgba(59, 130, 246, 0.5)';
                        }
                    } else {
                        // Sync pending
                        bgColor = '#1f2937';
                        borderColor = '#374151';
                        textColor = '#6b7280';
                        bar.style.boxShadow = 'none';
                    }
                } else if (item.type === 'padding') {
                    bgColor = '#1f2937';
                    borderColor = '#374151';
                    textColor = '#6b7280';
                    bar.style.boxShadow = 'none';
                }

                bar.style.background = bgColor;
                bar.style.borderColor = borderColor;
                bar.style.color = textColor;

                cumulativeTime = itemEndTime;
            });
        }
    }

    animate() {
        if (this.state.isPlaying) {
            this.state.currentTime += 0.5;

            const totalTime = this.getTotalTime();
            if (this.state.currentTime >= totalTime) {
                // Ensure we go slightly past totalTime to show final reduce ops as completed
                this.state.currentTime = totalTime + 0.1;
                this.updateVisualization();
                this.stopAnimation();
            } else {
                this.updateVisualization();
            }
        }
    }

    startAnimation() {
        this.state.isPlaying = true;
        this.state.animationTimer = setInterval(() => this.animate(), this.config.animationInterval);

        document.getElementById('streamKPlayIcon').textContent = '⏸';
        document.getElementById('streamKPlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('streamKPlayIcon').textContent = '▶';
        document.getElementById('streamKPlayText').textContent = 'Play';
    }

    resetAnimation() {
        this.stopAnimation();
        this.state.currentTime = 0;
        this.updateVisualization();
    }

    togglePlay() {
        const totalTime = this.getTotalTime();

        if (!this.state.isPlaying && this.state.currentTime >= totalTime) {
            this.resetAnimation();
            setTimeout(() => this.startAnimation(), 100);
        } else if (this.state.isPlaying) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
}

// Auto-initialize Stream-K viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('stream-k-viz');
    if (container) {
        new StreamKViz('stream-k-viz');
    }
});

// ============================================================================
// Ping Pong Kernel Visualization (1 Producer + 2 Consumers)
// ============================================================================

class PingPongKernelViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        // Configuration
        this.config = {
            maxTime: 400,
            timeScale: 2.5,
            numTiles: 6,
            animationInterval: 40,
        };

        this.opTypes = {
            TMA_LOAD: { name: 'TMA Load (A+B)', color: 'pipeline-bg-blue-500', duration: 30, icon: '📥' },
            WGMMA: { name: 'WGMMA', color: 'pipeline-bg-purple-500', duration: 35, icon: '⚡' },
            EPILOGUE: { name: 'Epilogue', color: 'pipeline-bg-orange-500', duration: 15, icon: '🔧' },
        };

        // 1 Producer + 2 Consumer warps
        this.streams = [
            { id: 'producer', name: 'Producer Warp', type: 'producer' },
            { id: 'consumer_0', name: 'Consumer Warp 0', type: 'consumer' },
            { id: 'consumer_1', name: 'Consumer Warp 1', type: 'consumer' }
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
        const totalTime = this.getTotalTime();

        this.container.innerHTML = commonPipelineStyles + `
            <div class="pipeline-container">
                <h1 class="pipeline-title">Ping Pong Kernel Timeline</h1>
                <p class="pipeline-subtitle">1 Producer Warp + 2 Consumer Warps → Alternating MMA + Epilogue</p>

                <!-- Controls -->
                <div class="pipeline-controls">
                    <button id="pingPongPlayBtn" class="pipeline-btn pipeline-btn-primary">
                        <span id="pingPongPlayIcon">▶</span>
                        <span id="pingPongPlayText">Play</span>
                    </button>
                    <button id="pingPongResetBtn" class="pipeline-btn pipeline-btn-secondary">
                        <span>↻</span>
                        <span>Reset</span>
                    </button>
                </div>

                <!-- Legend -->
                <div class="pipeline-card">
                    <h3 class="pipeline-card-title">Operation Types</h3>
                    <div class="pipeline-legend-grid" id="pingPongLegendGrid"></div>
                </div>

                <!-- Timeline -->
                <div class="pipeline-timeline-container">
                    <div class="pipeline-timeline-wrapper">
                        <div class="pipeline-time-axis">
                            <div class="pipeline-time-label">Time →</div>
                            <div class="pipeline-time-track" id="pingPongTimeTrack"></div>
                        </div>
                        <div id="pingPongStreamLanes"></div>
                    </div>
                </div>
            </div>
        `;

        this.state.operations = this.initializeOperations();
        this.initializeLegend();
        this.initializeTimeTrack();
        this.initializeStreamLanes();

        // Event listeners
        document.getElementById('pingPongPlayBtn').addEventListener('click', () => this.togglePlay());
        document.getElementById('pingPongResetBtn').addEventListener('click', () => this.resetAnimation());
    }

    getTotalTime() {
        // Calculate based on all tiles completing
        const lastTileStart = (this.config.numTiles - 1) * 40;
        const lastTileEnd = lastTileStart + this.opTypes.TMA_LOAD.duration +
            this.opTypes.WGMMA.duration +
            this.opTypes.EPILOGUE.duration + 10;
        return lastTileEnd;
    }

    initializeOperations() {
        const ops = [];
        let opId = 0;

        for (let tile = 0; tile < this.config.numTiles; tile++) {
            const baseTime = tile * 40; // Spacing between tile loads

            // Producer: TMA Load for this tile
            ops.push({
                id: opId++,
                streamId: 'producer',
                type: 'TMA_LOAD',
                startTime: baseTime,
                endTime: baseTime + this.opTypes.TMA_LOAD.duration,
                tile: tile,
                label: `TMA${tile}`,
            });

            // Determine which consumer handles this tile (alternating)
            const consumerIdx = tile % 2;
            const consumerId = `consumer_${consumerIdx}`;

            // WGMMA starts after TMA load completes
            const wgmmaStart = baseTime + this.opTypes.TMA_LOAD.duration + 2;
            ops.push({
                id: opId++,
                streamId: consumerId,
                type: 'WGMMA',
                startTime: wgmmaStart,
                endTime: wgmmaStart + this.opTypes.WGMMA.duration,
                tile: tile,
                label: `MMA${tile}`,
            });

            // Epilogue runs on same consumer after WGMMA
            const epilogueStart = wgmmaStart + this.opTypes.WGMMA.duration + 1;
            ops.push({
                id: opId++,
                streamId: consumerId,
                type: 'EPILOGUE',
                startTime: epilogueStart,
                endTime: epilogueStart + this.opTypes.EPILOGUE.duration,
                tile: tile,
                label: `Epi${tile}`,
            });
        }

        return ops;
    }

    initializeLegend() {
        const legendGrid = document.getElementById('pingPongLegendGrid');
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
        const timeTrack = document.getElementById('pingPongTimeTrack');
        const totalTime = this.getTotalTime();
        const numMarkers = Math.floor(totalTime / 40) + 1;

        timeTrack.innerHTML = '';

        // Time markers
        for (let i = 0; i < numMarkers; i++) {
            const marker = document.createElement('div');
            marker.className = 'pipeline-time-marker';
            marker.style.left = `${i * 40 * this.config.timeScale}px`;
            marker.textContent = i * 40;
            timeTrack.appendChild(marker);
        }

        // Current time marker
        const currentMarker = document.createElement('div');
        currentMarker.id = 'pingPongCurrentTimeMarker';
        currentMarker.className = 'pipeline-current-time-marker';
        currentMarker.style.left = '0px';
        timeTrack.appendChild(currentMarker);

        // Set width
        timeTrack.style.minWidth = `${totalTime * this.config.timeScale}px`;
    }

    initializeStreamLanes() {
        const streamLanes = document.getElementById('pingPongStreamLanes');
        streamLanes.innerHTML = '';

        this.streams.forEach(stream => {
            const lane = document.createElement('div');
            lane.className = 'pipeline-stream-lane';
            const totalTime = this.getTotalTime();
            lane.innerHTML = `
                <div class="pipeline-stream-label pipeline-${stream.type}">
                    <div class="pipeline-stream-name">${stream.name}</div>
                    <div class="pipeline-stream-type">${stream.type === 'producer' ? 'TMA Loads' : 'Compute'}</div>
                </div>
                <div class="pipeline-lane-track" id="pingPong-lane-${stream.id}" style="min-width: ${totalTime * this.config.timeScale}px"></div>
            `;
            streamLanes.appendChild(lane);
        });

        // Create operation elements
        this.state.operations.forEach(op => {
            const opType = this.opTypes[op.type];
            const lane = document.getElementById(`pingPong-lane-${op.streamId}`);

            const opElement = document.createElement('div');
            opElement.id = `pingPong-op-${op.id}`;
            opElement.className = `pipeline-operation ${opType.color}`;
            opElement.style.left = `${op.startTime * this.config.timeScale}px`;
            opElement.style.width = `${(op.endTime - op.startTime) * this.config.timeScale}px`;
            opElement.style.display = 'none'; // Initially hidden
            opElement.innerHTML = `
                <div class="pipeline-operation-progress" id="pingPong-progress-${op.id}" style="width: 0%"></div>
                <div class="pipeline-operation-content">
                    <div class="pipeline-operation-icon">${opType.icon}</div>
                </div>
            `;

            lane.appendChild(opElement);
        });
    }

    updateVisualization() {
        const currentMarker = document.getElementById('pingPongCurrentTimeMarker');

        if (currentMarker) {
            currentMarker.style.left = `${this.state.currentTime * this.config.timeScale}px`;
        }

        this.state.operations.forEach(op => {
            const isActive = this.state.currentTime >= op.startTime && this.state.currentTime <= op.endTime;
            const isVisible = this.state.currentTime >= op.startTime;

            // Cache DOM elements on first access
            if (!op._cachedElements) {
                op._cachedElements = {
                    opElement: document.getElementById(`pingPong-op-${op.id}`),
                    progressBar: document.getElementById(`pingPong-progress-${op.id}`)
                };
            }

            const { opElement, progressBar } = op._cachedElements;

            if (opElement) {
                // Track previous state to avoid unnecessary updates
                const prevActive = op._prevActive;
                const prevVisible = op._prevVisible;

                // Only update if state changed
                if (isVisible !== prevVisible) {
                    opElement.style.display = isVisible ? 'flex' : 'none';
                    op._prevVisible = isVisible;
                }

                if (isVisible && isActive !== prevActive) {
                    opElement.style.opacity = isActive ? '1' : '0.6';

                    if (isActive) {
                        opElement.classList.add('active');
                        if (!op._pulseElement) {
                            op._pulseElement = document.createElement('div');
                            op._pulseElement.className = 'pipeline-operation-pulse';
                            opElement.appendChild(op._pulseElement);
                        }
                    } else {
                        opElement.classList.remove('active');
                        if (op._pulseElement) {
                            op._pulseElement.remove();
                            op._pulseElement = null;
                        }
                    }
                    op._prevActive = isActive;
                }

                if (progressBar && isActive) {
                    const progress = (this.state.currentTime - op.startTime) / (op.endTime - op.startTime);
                    progressBar.style.width = `${progress * 100}%`;
                } else if (progressBar && this.state.currentTime > op.endTime && op._prevProgressComplete !== true) {
                    progressBar.style.width = '100%';
                    op._prevProgressComplete = true;
                } else if (progressBar && !isVisible && op._prevProgressComplete !== false) {
                    progressBar.style.width = '0%';
                    op._prevProgressComplete = false;
                }
            }
        });
    }

    animate() {
        if (this.state.isPlaying) {
            this.state.currentTime++;

            const totalTime = this.getTotalTime();
            if (this.state.currentTime >= totalTime) {
                this.state.currentTime = totalTime;
                this.stopAnimation();
            }

            this.updateVisualization();
        }
    }

    startAnimation() {
        this.state.isPlaying = true;
        this.state.animationTimer = setInterval(() => this.animate(), this.config.animationInterval);

        document.getElementById('pingPongPlayIcon').textContent = '⏸';
        document.getElementById('pingPongPlayText').textContent = 'Pause';
    }

    stopAnimation() {
        this.state.isPlaying = false;
        if (this.state.animationTimer) {
            clearInterval(this.state.animationTimer);
            this.state.animationTimer = null;
        }

        document.getElementById('pingPongPlayIcon').textContent = '▶';
        document.getElementById('pingPongPlayText').textContent = 'Play';
    }

    resetAnimation() {
        this.stopAnimation();
        this.state.currentTime = 0;
        this.updateVisualization();
    }

    togglePlay() {
        const totalTime = this.getTotalTime();

        if (!this.state.isPlaying && this.state.currentTime >= totalTime) {
            this.resetAnimation();
            setTimeout(() => this.startAnimation(), 100);
        } else if (this.state.isPlaying) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }
}

// Auto-initialize Ping Pong viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('ping-pong-kernel-viz');
    if (container) {
        new PingPongKernelViz('ping-pong-kernel-viz');
    }
});

// ============================================================================
// 32-Byte Swizzle Visualization
// ============================================================================

class SwizzleViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        this.config = {
            rows: 8,
            cols: 8,
            elementSize: 4,
            numBanks: 32,
            bankWidth: 4
        };

        this.colors = [
            '#e57373', '#81c784', '#64b5f6', '#ffb74d',
            '#ba68c8', '#4dd0e1', '#fff176', '#a1887f'
        ];

        this.selectedCells = new Set();

        this.init();
    }

    getBankColor(bank) {
        return this.colors[bank % 8];
    }

    getBank(byteAddress) {
        return Math.floor(byteAddress / this.config.bankWidth) % this.config.numBanks;
    }

    getOriginalAddress(row, col) {
        return (row * this.config.cols * this.config.elementSize) + (col * this.config.elementSize);
    }

    getSwizzledAddress(row, col) {
        const baseAddr = row * this.config.cols * this.config.elementSize;
        const colOffset = col * this.config.elementSize;
        const swizzleBits = (row & 0x7) << 2;
        const swizzledOffset = colOffset ^ swizzleBits;
        return baseAddr + swizzledOffset;
    }

    init() {
        this.container.innerHTML = commonPipelineStyles + `
            <style>
                .swizzle-container {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background-color: #111827;
                    color: #ffffff;
                    padding: 1rem 1.5rem 1.5rem;
                    border-radius: 0.5rem;
                    max-width: 100%;
                    overflow-x: hidden;
                }

                @media (max-width: 768px) {
                    .swizzle-container {
                        padding: 1rem;
                    }
                }

                .swizzle-title {
                    font-size: 1.75rem;
                    font-weight: bold;
                    text-align: center;
                    margin-top: 0;
                    margin-bottom: 0.5rem;
                    background: linear-gradient(to right, #c084fc, #ec4899);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }

                @media (max-width: 768px) {
                    .swizzle-title {
                        font-size: 1.5rem;
                    }
                }

                .swizzle-subtitle {
                    text-align: center;
                    color: #9ca3af;
                    margin-bottom: 2rem;
                    font-size: 0.9rem;
                }

                .swizzle-grids {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap:  1.5rem;
                    margin-bottom: 2rem;
                }

                @media (max-width: 1024px) {
                    .swizzle-grids {
                        grid-template-columns: 1fr;
                    }
                }

                .swizzle-grid-container {
                    background-color: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    max-width: 100%;
                    overflow: hidden;
                }

                @media (max-width: 768px) {
                    .swizzle-grid-container {
                        padding: 0.75rem;
                    }
                }

                .swizzle-grid-title {
                    font-size: 1.1rem;
                    font-weight: bold;
                    margin-bottom: 0.75rem;
                }

                @media (max-width: 768px) {
                    .swizzle-grid-title {
                        font-size: 1rem;
                    }
                }

                .swizzle-grid {
                    display: grid;
                    gap: 2px;
                    margin-top: 15px;
                    max-width: 100%;
                    margin-left: auto;
                    margin-right: auto;
                }

                @media (min-width: 768px) {
                    .swizzle-grid {
                        max-width: 400px;
                    }
                }

                .swizzle-cell {
                    aspect-ratio: 1;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: all 0.2s;
                    font-size: 0.65rem;
                    font-weight: 600;
                    position: relative;
                    border: 2px solid transparent;
                    color: #1e1e1e;
                    text-shadow: 0 0 2px rgba(255,255,255,0.3);
                    min-width: 0;
                }

                @media (max-width: 768px) {
                    .swizzle-cell {
                        font-size: 0.5rem;
                    }
                }

                .swizzle-cell:hover {
                    transform: scale(1.1);
                    z-index: 10;
                    border-color: white;
                    box-shadow: 0 0 8px rgba(255,255,255,0.5);
                }

                .swizzle-cell.selected {
                    border: 3px solid #fff;
                    box-shadow: 0 0 15px rgba(255,255,255,0.8);
                }

                .swizzle-address {
                    font-size: 0.6rem;
                    opacity: 0.95;
                }

                .swizzle-bank {
                    font-size: 0.55rem;
                    opacity: 0.8;
                }

                .swizzle-stats {
                    background-color: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 1.5rem;
                    overflow-x: auto;
                }

                @media (max-width: 768px) {
                    .swizzle-stats {
                        padding: 0.75rem;
                    }
                }

                .swizzle-stats h3 {
                    color: #ff9800;
                    margin-top: 0;
                    margin-bottom: 1rem;
                }

                .swizzle-tables {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 2rem;
                }

                @media (max-width: 1024px) {
                    .swizzle-tables {
                        grid-template-columns: 1fr;
                    }
                }

                .swizzle-table-container h4 {
                    margin-bottom: 1rem;
                    font-size: 1rem;
                }

                .swizzle-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.75rem;
                }

                @media (max-width: 768px) {
                    .swizzle-table {
                        font-size: 0.65rem;
                    }
                }

                .swizzle-table thead {
                    background-color: #374151;
                }

                .swizzle-table th {
                    padding: 0.4rem;
                    text-align: left;
                    font-weight: 600;
                    color: #e5e7eb;
                }

                @media (max-width: 768px) {
                    .swizzle-table th {
                        padding: 0.3rem;
                        font-size: 0.65rem;
                    }
                }

                .swizzle-table td {
                    padding: 0.4rem;
                    border-bottom: 1px solid #374151;
                    color: #d1d5db;
                    word-break: break-word;
                    max-width: 150px;
                }

                @media (max-width: 768px) {
                    .swizzle-table td {
                        padding: 0.3rem;
                        font-size: 0.6rem;
                        max-width: 100px;
                    }
                }

                .swizzle-table tr:hover {
                    background-color: #374151;
                    cursor: pointer;
                }

                .swizzle-table tr.highlighted {
                    background-color: #4a4a00;
                }

                .conflict-value {
                    font-weight: bold;
                    padding: 4px 8px;
                    border-radius: 3px;
                    display: inline-block;
                }

                .conflict-good {
                    background: #2e7d32;
                    color: #81c784;
                }

                .conflict-bad {
                    background: #c62828;
                    color: #e57373;
                }
            </style>

            <div class="swizzle-container">
                <h1 class="swizzle-title">32-Byte Swizzle Mode</h1>

                <div style="background-color: #1f2937; border: 1px solid #4b5563; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; max-width: 600px; margin-left: auto; margin-right: auto;">
                    <div style="display: flex; align-items: center; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                        <div style="background-color: #374151; border: 2px solid #60a5fa; border-radius: 0.375rem; padding: 0.75rem; min-width: 120px;">
                            <div style="font-family: 'Courier New', monospace; color: #60a5fa; font-size: 0.9rem; font-weight: bold; text-align: center; margin-bottom: 0.25rem;">42</div>
                            <div style="font-family: 'Courier New', monospace; color: #93c5fd; font-size: 0.85rem; text-align: center;">B10</div>
                        </div>
                        <div style="color: #d1d5db; font-size: 0.9rem; max-width: 400px;">
                            <div style="margin-bottom: 0.25rem;"><span style="color: #60a5fa; font-weight: bold;">Top number:</span> Memory address (byte offset)</div>
                            <div><span style="color: #93c5fd; font-weight: bold;">Bottom (B#):</span> Memory bank number (0-31)</div>
                        </div>
                    </div>
                </div>

                <div class="swizzle-grids">
                    <div class="swizzle-grid-container">
                        <h2 class="swizzle-grid-title">Original Layout (No Swizzle)</h2>
                        <div id="swizzle-original-grid" class="swizzle-grid"></div>
                    </div>

                    <div class="swizzle-grid-container">
                        <h2 class="swizzle-grid-title">32-Byte Swizzled Layout</h2>
                        <div id="swizzle-swizzled-grid" class="swizzle-grid"></div>
                    </div>
                </div>

                <div class="swizzle-stats">
                    <div style="overflow-x: auto;">
                        <table class="swizzle-table swizzle-combined-table">
                            <thead>
                                <tr>
                                    <th rowspan="2" style="vertical-align: middle;">Column</th>
                                    <th colspan="2" style="text-align: center; border-bottom: 1px solid #555; color: #e57373;">❌ Original (No Swizzle)</th>
                                    <th colspan="2" style="text-align: center; border-bottom: 1px solid #555; color: #81c784;">✅ Swizzled</th>
                                </tr>
                                <tr>
                                    <th>Banks Hit</th>
                                    <th style="text-align: center;">Conflict</th>
                                    <th>Banks Hit</th>
                                    <th style="text-align: center;">Conflict</th>
                                </tr>
                            </thead>
                            <tbody id="swizzle-combined-tbody"></tbody>
                        </table>
                    </div>
                </div>
                </div>
            </div>
        `;

        this.createGrid('swizzle-original-grid', false);
        this.createGrid('swizzle-swizzled-grid', true);
        this.buildConflictTables();

        // Auto-select column 0 on load
        setTimeout(() => this.selectColumn(0), 500);
    }

    createGrid(gridId, isSwizzled) {
        const grid = document.getElementById(gridId);
        grid.style.gridTemplateColumns = `repeat(${this.config.cols}, 1fr)`;
        grid.innerHTML = '';

        for (let row = 0; row < this.config.rows; row++) {
            for (let col = 0; col < this.config.cols; col++) {
                const address = isSwizzled ?
                    this.getSwizzledAddress(row, col) :
                    this.getOriginalAddress(row, col);
                const bank = this.getBank(address);
                const color = this.getBankColor(bank);

                const cell = document.createElement('div');
                cell.className = 'swizzle-cell';
                cell.style.backgroundColor = color;
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.dataset.address = address;
                cell.dataset.bank = bank;
                cell.dataset.swizzled = isSwizzled;

                cell.innerHTML = `
                    <div class="swizzle-address">${address}</div>
                    <div class="swizzle-bank">B${bank}</div>
                `;

                cell.onclick = () => this.selectColumn(col);

                grid.appendChild(cell);
            }
        }
    }

    selectColumn(col) {
        this.selectedCells.clear();
        for (let row = 0; row < this.config.rows; row++) {
            this.selectedCells.add(`${row},${col}`);
        }
        this.updateSelection();
    }

    updateSelection() {
        document.querySelectorAll('.swizzle-cell').forEach(cell => {
            const key = `${cell.dataset.row},${cell.dataset.col}`;
            if (this.selectedCells.has(key)) {
                cell.classList.add('selected');
            } else {
                cell.classList.remove('selected');
            }
        });

        // Highlight table rows
        document.querySelectorAll('.swizzle-table tr').forEach(row => {
            row.classList.remove('highlighted');
        });

        if (this.selectedCells.size > 0) {
            const firstKey = Array.from(this.selectedCells)[0];
            const col = parseInt(firstKey.split(',')[1]);
            if (Array.from(this.selectedCells).every(key => key.split(',')[1] === col.toString())) {
                document.querySelectorAll(`[data-column="${col}"]`).forEach(row => {
                    row.classList.add('highlighted');
                });
            }
        }
    }

    buildConflictTables() {
        const combinedData = [];

        for (let col = 0; col < this.config.cols; col++) {
            const originalBanks = new Set();
            const swizzledBanks = new Set();

            for (let row = 0; row < this.config.rows; row++) {
                const origAddr = this.getOriginalAddress(row, col);
                const origBank = this.getBank(origAddr);
                originalBanks.add(origBank);

                const swizAddr = this.getSwizzledAddress(row, col);
                const swizBank = this.getBank(swizAddr);
                swizzledBanks.add(swizBank);
            }

            const origConflict = this.config.rows / originalBanks.size;
            const swizConflict = this.config.rows / swizzledBanks.size;

            combinedData.push({
                col,
                originalBanks: Array.from(originalBanks).sort((a, b) => a - b),
                origConflict: origConflict,
                swizzledBanks: Array.from(swizzledBanks).sort((a, b) => a - b),
                swizConflict: swizConflict
            });
        }

        // Populate combined table
        const combinedTbody = document.getElementById('swizzle-combined-tbody');
        combinedTbody.innerHTML = '';
        combinedData.forEach(data => {
            const row = document.createElement('tr');
            row.dataset.column = data.col;
            row.onclick = () => this.selectColumn(data.col);

            const origConflictClass = data.origConflict === 1 ? 'conflict-good' : 'conflict-bad';
            const swizConflictClass = data.swizConflict === 1 ? 'conflict-good' : 'conflict-bad';

            row.innerHTML = `
                <td><strong>Col ${data.col}</strong></td>
                <td style="font-size: 0.65rem;">${data.originalBanks.map(b => `B${b}`).join(', ')}</td>
                <td style="text-align: center;"><span class="conflict-value ${origConflictClass}">${data.origConflict}x</span></td>
                <td style="font-size: 0.65rem;">${data.swizzledBanks.map(b => `B${b}`).join(', ')}</td>
                <td style="text-align: center;"><span class="conflict-value ${swizConflictClass}">${data.swizConflict}x</span></td>
            `;
            combinedTbody.appendChild(row);
        });
    }
}

// Auto-initialize Swizzle viz on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('swizzle-viz');
    if (container) {
        new SwizzleViz('swizzle-viz');
    }
});
// Hopper Kernel Results Explorer
// ============================================================================

class HopperResultsExplorer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        this.data = [];
        this.selectedSize = null;
        this.topN = 10;

        this.init();
    }

    async init() {
        await this.loadData();
        this.render();
    }

    async loadData() {
        try {
            const response = await fetch('/assets/data/autotune_hopper_cache_bfloat16.json');
            const jsonData = await response.json();
            this.data = this.parseJSON(jsonData);
        } catch (error) {
            console.error('Error loading data:', error);
            this.container.innerHTML = '<p style="color: red;">Error loading data</p>';
        }
    }

    parseJSON(jsonData) {
        const results = [];

        for (const sizeResult of jsonData.results) {
            const size = sizeResult.size;
            const pytorchTflops = sizeResult.pytorch_tflops;

            for (const config of sizeResult.all_results) {
                if (config.status === 'success') {
                    // Parse config name: "128x256x64_Heuristic_RasterH_Swz1_Spl1"
                    const parts = config.config_name.split('_');
                    const tileConfig = parts[0]; // e.g., "128x256x64"
                    const decomposition = parts[1]; // e.g., "Heuristic", "DataParallel", "SplitK", "StreamK"
                    const rasterPart = parts[2]; // e.g., "RasterH", "RasterN", "RasterM"
                    const swizzlePart = parts[3]; // e.g., "Swz1", "Swz2", "Swz4", "Swz8"

                    // Map raster to full names
                    const rasterMap = {
                        'RasterH': 'Heuristic',
                        'RasterN': 'Along N',
                        'RasterM': 'Along M'
                    };
                    const Rasterization = rasterMap[rasterPart] || rasterPart;
                    const Swizzle = parseInt(swizzlePart.replace('Swz', ''));

                    // Extract split factor if present
                    let Splits = 1;
                    if (parts.length > 4 && parts[4].startsWith('Spl')) {
                        Splits = parseInt(parts[4].replace('Spl', ''));
                    }

                    results.push({
                        M: size,
                        N: size,
                        K: size,
                        TileConfig: tileConfig,
                        Rasterization: Rasterization,
                        Swizzle: Swizzle,
                        Mode: decomposition,
                        Splits: Splits,
                        TFLOPS: config.tflops,
                        PyTorchTFLOPS: pytorchTflops,
                        AvgRuntime_ms: config.median_time_ms,
                        ConfigName: config.config_name
                    });
                }
            }
        }

        return results;
    }

    getUniqueSizes() {
        const sizes = [...new Set(this.data.map(d => d.M))].sort((a, b) => a - b);
        return sizes;
    }

    getTopConfigs(size, n = 10) {
        const sizeData = this.data.filter(d => d.M === size);
        return sizeData.sort((a, b) => b.TFLOPS - a.TFLOPS).slice(0, n);
    }

    getParameterStats(size) {
        const sizeData = this.data.filter(d => d.M === size);

        const stats = {
            Raster: {},
            Swizzle: {},
            Decomposition: {},
            Splits: {}
        };

        sizeData.forEach(row => {
            ['Raster', 'Decomposition'].forEach(param => {
                const key = row[param];
                if (!stats[param][key]) {
                    stats[param][key] = { total: 0, count: 0, max: 0, configs: [] };
                }
                stats[param][key].total += row.TFLOPS;
                stats[param][key].count += 1;
                stats[param][key].max = Math.max(stats[param][key].max, row.TFLOPS);
                stats[param][key].configs.push(row);
            });

            ['Swizzle', 'Splits'].forEach(param => {
                const key = row[param];
                if (!stats[param][key]) {
                    stats[param][key] = { total: 0, count: 0, max: 0, configs: [] };
                }
                stats[param][key].total += row.TFLOPS;
                stats[param][key].count += 1;
                stats[param][key].max = Math.max(stats[param][key].max, row.TFLOPS);
                stats[param][key].configs.push(row);
            });
        });

        // Calculate averages
        Object.keys(stats).forEach(param => {
            Object.keys(stats[param]).forEach(key => {
                stats[param][key].avg = stats[param][key].total / stats[param][key].count;
            });
        });

        return stats;
    }

    render() {
        const sizes = this.getUniqueSizes();
        this.selectedSize = sizes[sizes.length - 1]; // Default to largest size

        this.container.innerHTML = `
            <style>
                .results-explorer {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background-color: #111827;
                    color: #ffffff;
                    padding: 1rem 2rem 2rem 2rem;
                    border-radius: 0.5rem;
                }

                .results-title {
                    font-size: 2rem;
                    font-weight: bold;
                    text-align: center;
                    margin-bottom: 0.5rem;
                    background: linear-gradient(to right, #c084fc, #ec4899);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }

                .results-subtitle {
                    text-align: center;
                    color: #9ca3af;
                    margin-bottom: 2rem;
                    font-size: 0.9rem;
                }

                .size-selector {
                    display: flex;
                    justify-content: center;
                    gap: 0.5rem;
                    margin-bottom: 2rem;
                    flex-wrap: wrap;
                }

                .size-btn {
                    padding: 0.5rem 1rem;
                    border: 2px solid #374151;
                    border-radius: 0.5rem;
                    background-color: #1f2937;
                    color: #9ca3af;
                    cursor: pointer;
                    transition: all 0.2s;
                    font-size: 0.9rem;
                }

                .size-btn:hover {
                    border-color: #9333ea;
                    color: #ffffff;
                }

                .size-btn.active {
                    border-color: #9333ea;
                    background-color: #9333ea;
                    color: #ffffff;
                }

                .results-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }

                @media (max-width: 1024px) {
                    .results-grid {
                        grid-template-columns: 1fr;
                    }
                }

                .results-card {
                    background-color: #1f2937;
                    border: 1px solid #374151;
                    border-radius: 0.5rem;
                    padding: 1.5rem;
                }

                .card-title {
                    font-size: 1.25rem;
                    font-weight: bold;
                    margin-bottom: 1rem;
                    color: #a78bfa;
                }

                .config-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.85rem;
                }

                .config-table th {
                    background-color: #374151;
                    padding: 0.5rem;
                    text-align: left;
                    font-weight: 600;
                    color: #e5e7eb;
                }

                .config-table td {
                    padding: 0.5rem;
                    border-bottom: 1px solid #374151;
                    color: #d1d5db;
                }

                .config-table tr:hover {
                    background-color: #374151;
                }

                .rank-badge {
                    display: inline-block;
                    width: 1.5rem;
                    height: 1.5rem;
                    line-height: 1.5rem;
                    text-align: center;
                    border-radius: 50%;
                    font-weight: bold;
                    font-size: 0.75rem;
                }

                .rank-1 { background-color: #fbbf24; color: #000; }
                .rank-2 { background-color: #d1d5db; color: #000; }
                .rank-3 { background-color: #cd7f32; color: #fff; }
                .rank-other { background-color: #4b5563; color: #fff; }

                .param-bar-container {
                    margin-bottom: 1rem;
                }

                .param-label {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 0.25rem;
                    font-size: 0.85rem;
                }

                .param-name {
                    color: #e5e7eb;
                    font-weight: 500;
                }

                .param-value {
                    color: #9ca3af;
                    font-size: 0.8rem;
                }

                .param-bar {
                    height: 1.5rem;
                    background-color: #374151;
                    border-radius: 0.25rem;
                    overflow: hidden;
                    position: relative;
                }

                .param-bar-fill {
                    height: 100%;
                    background: linear-gradient(to right, #9333ea, #ec4899);
                    transition: width 0.3s;
                    display: flex;
                    align-items: center;
                    padding-left: 0.5rem;
                    font-size: 0.75rem;
                    font-weight: bold;
                }

                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 1rem;
                }

                .stat-box {
                    background-color: #374151;
                    padding: 1rem;
                    border-radius: 0.5rem;
                }

                .stat-label {
                    font-size: 0.75rem;
                    color: #9ca3af;
                    margin-bottom: 0.25rem;
                }

                .stat-value {
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #a78bfa;
                }

                .full-width {
                    grid-column: 1 / -1;
                }
            </style>

            <div class="results-explorer">
                <h1 class="results-title">Hopper Kernel Configuration Explorer</h1>
                <p class="results-subtitle">Explore performance across different matrix sizes and configurations</p>

                <div class="size-selector" id="sizeSelector"></div>

                <div id="resultsContent"></div>
            </div>
        `;

        this.renderSizeSelector();
        this.renderResults();
    }

    renderSizeSelector() {
        const sizes = this.getUniqueSizes();
        const selector = document.getElementById('sizeSelector');

        selector.innerHTML = sizes.map(size => `
            <button class="size-btn ${size === this.selectedSize ? 'active' : ''}"
                    data-size="${size}">
                ${size}×${size}×${size}
            </button>
        `).join('');

        selector.querySelectorAll('.size-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectedSize = parseInt(e.target.dataset.size);
                this.renderSizeSelector();
                this.renderResults();
            });
        });
    }

    renderResults() {
        const topConfigs = this.getTopConfigs(this.selectedSize, this.topN);
        const stats = this.getParameterStats(this.selectedSize);
        const maxTFLOPS = Math.max(...topConfigs.map(c => c.TFLOPS));

        const content = document.getElementById('resultsContent');
        content.innerHTML = `
            <div class="results-grid">
                <!-- Top Configurations -->
                <div class="results-card full-width">
                    <h2 class="card-title">Top ${this.topN} Configurations</h2>
                    <div style="overflow-x: auto;">
                        <table class="config-table">
                            <thead>
                                <tr>
                                    <th>Rank</th>
                                    <th>TFLOPS</th>
                                    <th>Tile</th>
                                    <th>Raster</th>
                                    <th>Swizzle</th>
                                    <th>Decomposition</th>
                                    <th>Splits</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${topConfigs.map((config, idx) => `
                                    <tr>
                                        <td>
                                            <span class="rank-badge rank-${idx < 3 ? idx + 1 : 'other'}">
                                                ${idx + 1}
                                            </span>
                                        </td>
                                        <td><strong>${config.TFLOPS.toFixed(2)}</strong></td>
                                        <td><code style="font-size: 0.75rem; color: #c084fc;">${config.TileConfig}</code></td>
                                        <td>${config.Rasterization}</td>
                                        <td>${config.Swizzle}</td>
                                        <td>${config.Mode}</td>
                                        <td>${config.Splits}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }

    renderParameterBars(paramStats, maxTFLOPS) {
        const sortedParams = Object.entries(paramStats)
            .sort((a, b) => b[1].max - a[1].max);

        return sortedParams.map(([key, value]) => {
            const percentage = (value.max / maxTFLOPS) * 100;
            return `
                <div class="param-bar-container">
                    <div class="param-label">
                        <span class="param-name">${key}</span>
                        <span class="param-value">
                            max: ${value.max.toFixed(2)} | avg: ${value.avg.toFixed(2)} | count: ${value.count}
                        </span>
                    </div>
                    <div class="param-bar">
                        <div class="param-bar-fill" style="width: ${percentage}%;">
                            ${value.max.toFixed(2)} TFLOPS
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    getBestParameter(paramStats) {
        const best = Object.entries(paramStats)
            .sort((a, b) => b[1].max - a[1].max)[0];
        return best ? `${best[0]} (${best[1].max.toFixed(2)} TFLOPS)` : 'N/A';
    }
}

// Auto-initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('hopper-results-explorer');
    if (container) {
        new HopperResultsExplorer('hopper-results-explorer');
    }
});

// ============================================================================
// YOLO Benchmark Results Interactive Charts with Plotly
// ============================================================================

class BenchmarkChartsManager {
    constructor() {
        this.data = null;
        this.pytorchData = {
            64: 0.02069993676659908,
            96: 0.07139573748566991,
            128: 0.16771848717623417,
            256: 1.3530012514633205,
            512: 10.894296327637464,
            768: 35.97401791627472,
            1024: 84.30761729186132,
            1536: 246.45529145304488,
            2048: 413.29554556148776,
            3072: 627.7288305990529,
            4096: 722.5111130154563,
            6144: 709.0085919418082, // Interpolated between 4096 and 8192
            8192: 695.50607086816
        };
    }

    async init() {
        await this.loadPlotly();
        await this.loadData();
        this.createSwizzleRasterChart();
        this.createModeChart();
        this.createFinalBestChart();
    }

    async loadPlotly() {
        if (typeof Plotly !== 'undefined') return;

        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    async loadData() {
        try {
            const response = await fetch('/assets/data/autotune_hopper_cache_bfloat16.json');
            const jsonData = await response.json();
            this.data = this.parseJSON(jsonData);
            this.pytorchData = this.extractPytorchData(jsonData);
            console.log('Loaded', this.data.length, 'benchmark results');
        } catch (error) {
            console.error('Failed to load benchmark data:', error);
        }
    }

    parseJSON(jsonData) {
        const results = [];

        for (const sizeResult of jsonData.results) {
            const size = sizeResult.size;
            const pytorchTflops = sizeResult.pytorch_tflops;

            for (const config of sizeResult.all_results) {
                if (config.status === 'success') {
                    // Parse config name: "128x256x64_Heuristic_RasterH_Swz1_Spl1"
                    const parts = config.config_name.split('_');
                    const tileConfig = parts[0]; // e.g., "128x256x64"
                    const decomposition = parts[1]; // e.g., "Heuristic", "DataParallel", "SplitK", "StreamK"
                    const rasterPart = parts[2]; // e.g., "RasterH", "RasterN", "RasterM"
                    const swizzlePart = parts[3]; // e.g., "Swz1", "Swz2", "Swz4", "Swz8"

                    // Map raster to full names
                    const rasterMap = {
                        'RasterH': 'Heuristic',
                        'RasterN': 'Along N',
                        'RasterM': 'Along M'
                    };
                    const raster = rasterMap[rasterPart] || rasterPart;
                    const swizzle = parseInt(swizzlePart.replace('Swz', ''));

                    // Extract split factor if present
                    let splitFactor = 1;
                    if (parts.length > 4 && parts[4].startsWith('Spl')) {
                        splitFactor = parseInt(parts[4].replace('Spl', ''));
                    }

                    results.push({
                        M: size,  // Keep M for compatibility
                        size: size,
                        tile_config: tileConfig,
                        rasterization: raster,
                        swizzle: swizzle,
                        mode: decomposition,
                        split_k: splitFactor,
                        TFLOPS: config.tflops,
                        pytorch_tflops: pytorchTflops,
                        config_name: config.config_name
                    });
                }
            }
        }

        return results;
    }

    extractPytorchData(jsonData) {
        const pytorchData = {};
        for (const sizeResult of jsonData.results) {
            pytorchData[sizeResult.size] = sizeResult.pytorch_tflops;
        }
        return pytorchData;
    }

    getUniqueMatrixSizes() {
        return [...new Set(this.data.map(d => d.M))].sort((a, b) => a - b);
    }

    createSwizzleRasterChart() {
        const container = document.getElementById('swizzle-raster-chart');
        if (!container) return;

        const matrixSizes = this.getUniqueMatrixSizes();

        // Get best TFLOPS for each size
        const bestBySize = {};
        matrixSizes.forEach(size => {
            const sizeData = this.data.filter(d => d.M === size);
            bestBySize[size] = Math.max(...sizeData.map(d => d.TFLOPS));
        });

        const rasterColors = {
            'Heuristic': '#8b5cf6',
            'Along N': '#ec4899',
            'Along M': '#06b6d4'
        };

        this.renderPlotlyChart(
            'swizzle-raster-chart',
            'Rasterization & Swizzle Strategy Performance',
            matrixSizes,
            bestBySize,
            this.data,
            (d) => d.rasterization,
            rasterColors,
            (d) => `Size: ${d.M}³<br>Tile: ${d.tile_config}<br>TFLOPS: ${d.TFLOPS.toFixed(2)}<br>Raster: ${d.rasterization}<br>Swizzle: ${d.swizzle}<br>Decomp: ${d.mode}<br>Splits: ${d.split_k}`
        );
    }

    createModeChart() {
        const container = document.getElementById('mode-chart');
        if (!container) return;

        const matrixSizes = this.getUniqueMatrixSizes();

        const bestBySize = {};
        matrixSizes.forEach(size => {
            const sizeData = this.data.filter(d => d.M === size);
            bestBySize[size] = Math.max(...sizeData.map(d => d.TFLOPS));
        });

        const decompColors = {
            'Heuristic': '#f59e0b',
            'StreamK': '#10b981',
            'SplitK': '#3b82f6',
            'DataParallel': '#ef4444'
        };

        this.renderPlotlyChart(
            'mode-chart',
            'Decomposition Mode & Split Strategy Performance',
            matrixSizes,
            bestBySize,
            this.data,
            (d) => d.mode,
            decompColors,
            (d) => `Size: ${d.M}³<br>Tile: ${d.tile_config}<br>TFLOPS: ${d.TFLOPS.toFixed(2)}<br>Decomposition: ${d.mode}<br>Splits: ${d.split_k}<br>Raster: ${d.rasterization}<br>Swizzle: ${d.swizzle}`
        );
    }

    createFinalBestChart() {
        const container = document.getElementById('final-best-chart');
        if (!container) return;

        const matrixSizes = this.getUniqueMatrixSizes();

        const bestBySize = {};
        matrixSizes.forEach(size => {
            const sizeData = this.data.filter(d => d.M === size);
            bestBySize[size] = Math.max(...sizeData.map(d => d.TFLOPS));
        });

        const grayColors = {
            'All Configs': '#6b7280'
        };

        this.renderPlotlyChart(
            'final-best-chart',
            'Best Performance Across All Configurations',
            matrixSizes,
            bestBySize,
            this.data,
            () => 'All Configs',
            grayColors,
            (d) => `Size: ${d.M}³<br>Tile: ${d.tile_config}<br>TFLOPS: ${d.TFLOPS.toFixed(2)}<br>Raster: ${d.rasterization}<br>Swizzle: ${d.swizzle}<br>Decomposition: ${d.mode}<br>Splits: ${d.split_k}`
        );
    }

    renderPlotlyChart(containerId, chartTitle, matrixSizes, bestBySize, data, groupKeyFn, colors, hoverTextFn) {
        const traces = [];

        // Group data by color key
        const groupedData = {};
        data.forEach(d => {
            const key = groupKeyFn(d);
            if (!groupedData[key]) {
                groupedData[key] = [];
            }
            groupedData[key].push(d);
        });

        // Create scatter traces for each group
        Object.entries(groupedData).forEach(([key, points]) => {
            const color = colors[key] || '#6b7280';
            traces.push({
                x: points.map(d => d.M),
                y: points.map(d => d.TFLOPS),
                mode: 'markers',
                type: 'scatter',
                name: key,
                marker: {
                    color: color,
                    size: 6,
                    opacity: 0.6,
                    line: {
                        color: color,
                        width: 1
                    }
                },
                hovertemplate: points.map(d => hoverTextFn(d) + '<extra></extra>'),
                showlegend: true
            });
        });

        // PyTorch line
        traces.push({
            x: matrixSizes,
            y: matrixSizes.map(size => this.pytorchData[size]),
            mode: 'lines+markers',
            type: 'scatter',
            name: 'PyTorch',
            line: {
                color: '#22c55e',
                width: 3,
                dash: 'dash'
            },
            marker: {
                color: '#22c55e',
                size: 8,
                symbol: 'diamond'
            },
            hovertemplate: matrixSizes.map(size =>
                `Size: ${size}³<br>PyTorch: ${this.pytorchData[size].toFixed(2)} TFLOPS<extra></extra>`
            )
        });

        // Best CUTLASS line
        traces.push({
            x: matrixSizes,
            y: matrixSizes.map(size => bestBySize[size]),
            mode: 'lines+markers',
            type: 'scatter',
            name: 'Best CUTLASS',
            line: {
                color: '#ffffff',
                width: 3
            },
            marker: {
                color: '#ffffff',
                size: 10,
                symbol: 'circle',
                line: {
                    color: '#8b5cf6',
                    width: 2
                }
            },
            hovertemplate: matrixSizes.map(size => {
                const percentage = ((bestBySize[size] / this.pytorchData[size]) * 100).toFixed(1);
                return `Size: ${size}³<br>Best CUTLASS: ${bestBySize[size].toFixed(2)} TFLOPS<br>vs PyTorch: ${percentage}%<extra></extra>`;
            }),
            text: matrixSizes.map(size => {
                const percentage = ((bestBySize[size] / this.pytorchData[size]) * 100).toFixed(0);
                return `${bestBySize[size].toFixed(0)} (${percentage}%)`;
            }),
            textposition: 'top center',
            textfont: {
                color: '#ffffff',
                size: 11,
                family: 'monospace'
            },
            mode: 'lines+markers+text'
        });

        // 100% reference line trace (for legend only - actual line drawn via shape)
        traces.push({
            x: [matrixSizes[0]],
            y: [100],
            mode: 'lines',
            type: 'scatter',
            name: '100% (PyTorch baseline)',
            yaxis: 'y2',
            line: {
                color: '#6b7280',
                width: 2,
                dash: 'dash'
            },
            hoverinfo: 'skip',
            showlegend: true
        });

        // Percentage line on secondary axis
        traces.push({
            x: matrixSizes,
            y: matrixSizes.map(size => (bestBySize[size] / this.pytorchData[size]) * 100),
            mode: 'lines',
            type: 'scatter',
            name: '% of PyTorch',
            yaxis: 'y2',
            line: {
                color: '#fbbf24',
                width: 2,
                dash: 'dot'
            },
            hovertemplate: matrixSizes.map(size => {
                const percentage = ((bestBySize[size] / this.pytorchData[size]) * 100).toFixed(1);
                return `Size: ${size}³<br>Performance: ${percentage}% of PyTorch<extra></extra>`;
            }),
            showlegend: true
        });

        const layout = {
            title: {
                text: chartTitle,
                font: {
                    size: 18,
                    color: '#fff',
                    family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                },
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: {
                title: {
                    text: 'Matrix Size (M³)',
                    font: { color: '#9ca3af', size: 14 }
                },
                type: 'log',
                tickvals: matrixSizes,
                ticktext: matrixSizes.map(s => `${s}³`),
                gridcolor: '#374151',
                color: '#e5e7eb',
                showline: true,
                linecolor: '#4b5563',
                linewidth: 2
            },
            yaxis: {
                title: {
                    text: 'TFLOPS',
                    font: { color: '#9ca3af', size: 14 }
                },
                gridcolor: '#374151',
                color: '#9ca3af',
                showline: true,
                linecolor: '#4b5563',
                linewidth: 2
            },
            yaxis2: {
                title: {
                    text: '% of PyTorch Performance',
                    font: { color: '#fbbf24', size: 14 }
                },
                overlaying: 'y',
                side: 'right',
                range: [0, 200],
                tickvals: [0, 25, 50, 75, 100, 125, 150, 175, 200],
                ticktext: ['0%', '25%', '50%', '75%', '100%', '125%', '150%', '175%', '200%'],
                color: '#fbbf24',
                showline: true,
                linecolor: '#fbbf24',
                linewidth: 2,
                showgrid: false
            },
            plot_bgcolor: '#111827',
            paper_bgcolor: '#1f2937',
            font: {
                color: '#e5e7eb',
                family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
            },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                x: 0.5,
                y: -0.15,
                xanchor: 'center',
                yanchor: 'top',
                orientation: 'h',
                bgcolor: '#1f2937',
                bordercolor: '#4b5563',
                borderwidth: 1,
                font: { color: '#e5e7eb', size: 11 }
            },
            margin: { l: 80, r: 120, t: 80, b: 120 },
            height: 600,
            autosize: true,
            shapes: [
                {
                    type: 'line',
                    xref: 'paper',
                    yref: 'y2',
                    x0: 0,
                    x1: 1,
                    y0: 100,
                    y1: 100,
                    line: {
                        color: '#6b7280',
                        width: 2,
                        dash: 'dash'
                    }
                }
            ]
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: containerId,
                height: 800,
                width: 1200,
                scale: 2
            }
        };

        Plotly.newPlot(containerId, traces, layout, config);
    }
}

const benchmarkStyles = document.createElement('style');
benchmarkStyles.textContent = `
#swizzle-raster-chart,
#mode-chart,
#final-best-chart {
    margin: 2rem 0;
    width: 100%;
}
`;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBenchmarkCharts);
} else {
    initBenchmarkCharts();
}

async function initBenchmarkCharts() {
    document.head.appendChild(benchmarkStyles);

    const hasCharts = document.getElementById('swizzle-raster-chart') ||
        document.getElementById('mode-chart') ||
        document.getElementById('final-best-chart');

    if (hasCharts) {
        const manager = new BenchmarkChartsManager();
        await manager.init();
    }
}
