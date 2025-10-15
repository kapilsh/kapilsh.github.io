// Common styles and utilities
const commonStyles = `
    <style>
        .viz-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 100%;
            margin: 20px auto;
            padding: 25px;
            background: #1a1a1a;
            border-radius: 12px;
            color: #e0e0e0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .viz-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #4CAF50;
            text-align: center;
        }
        .viz-controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
        }
        .viz-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            background: #4CAF50;
            color: white;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
            font-size: 14px;
        }
        .viz-btn:hover {
            background: #45a049;
            transform: translateY(-2px);
        }
        .viz-btn:disabled {
            background: #333;
            cursor: not-allowed;
            transform: none;
        }
        .viz-canvas {
            background: #252525;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            position: relative;
            overflow: hidden;
        }
        .viz-info {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            font-size: 13px;
            line-height: 1.6;
        }
        .viz-info h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
            font-size: 14px;
        }
        .viz-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .viz-stat {
            background: #1a1a1a;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #4CAF50;
        }
        .viz-stat-label {
            font-size: 11px;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .viz-stat-value {
            font-size: 16px;
            font-weight: 600;
            color: #4CAF50;
            margin-top: 5px;
        }
        .matrix-cell {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
            border: 1px solid #333;
            transition: all 0.3s;
        }
        .matrix-cell.active {
            background: #4CAF50 !important;
            transform: scale(1.1);
            z-index: 10;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
        }
        .matrix-cell.highlight {
            background: #FFA726 !important;
            border-color: #FF6F00;
        }
        .legend {
            display: flex;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        .legend-box {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid #333;
        }
    </style>
`;

// Utility function for creating matrix grid
function createMatrixGrid(rows, cols, cellSize, label, container) {
    const grid = document.createElement('div');
    grid.style.display = 'inline-block';
    grid.style.margin = '10px';

    const title = document.createElement('div');
    title.textContent = label;
    title.style.textAlign = 'center';
    title.style.marginBottom = '10px';
    title.style.fontSize = '14px';
    title.style.fontWeight = '600';
    title.style.color = '#999';

    const matrixContainer = document.createElement('div');
    matrixContainer.style.display = 'grid';
    matrixContainer.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
    matrixContainer.style.gap = '2px';

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.style.width = `${cellSize}px`;
            cell.style.height = `${cellSize}px`;
            cell.style.background = '#333';
            cell.dataset.row = i;
            cell.dataset.col = j;
            matrixContainer.appendChild(cell);
        }
    }

    grid.appendChild(title);
    grid.appendChild(matrixContainer);
    container.appendChild(grid);

    return matrixContainer;
}

// Sleep utility
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================================
// Kernel 1: Naive Implementation Visualization
// ============================================================================
class NaiveKernelViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.matrixSize = 8;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Naive Kernel: One Thread Per Output</div>
                <div class="viz-info">
                    <div id="naiveStats"></div>
                </div>
                <br>
                <div class="viz-controls">
                    <button class="viz-btn" id="naiveAnimate">Animate</button>
                    <button class="viz-btn" id="naiveReset">Reset</button>
                </div>
                <div class="viz-canvas" id="naiveCanvas"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>Current Thread</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FFA726;"></div>
                        <span>Accessed Memory</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #2196F3;"></div>
                        <span>Computed Result</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('naiveCanvas');
        this.renderMatrices();

        document.getElementById('naiveAnimate').addEventListener('click', () => this.animate());
        document.getElementById('naiveReset').addEventListener('click', () => this.reset());
    }

    renderMatrices() {
        this.canvas.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;"></div>';
        const wrapper = this.canvas.firstChild;

        this.matrixA = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix A', wrapper);
        this.matrixB = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix B', wrapper);
        this.matrixC = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix C (Output)', wrapper);
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('naiveAnimate').disabled = true;
        document.getElementById('naiveStats').innerHTML = '';

        let totalMemoryAccesses = 0;

        for (let i = 0; i < this.matrixSize; i++) {
            for (let j = 0; j < this.matrixSize; j++) {
                // Highlight current output cell
                const outputCell = this.matrixC.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                outputCell.classList.add('active');

                // Highlight row in A
                for (let k = 0; k < this.matrixSize; k++) {
                    const cellA = this.matrixA.querySelector(`[data-row="${i}"][data-col="${k}"]`);
                    cellA.classList.add('highlight');
                }

                // Highlight column in B
                for (let k = 0; k < this.matrixSize; k++) {
                    const cellB = this.matrixB.querySelector(`[data-row="${k}"][data-col="${j}"]`);
                    cellB.classList.add('highlight');
                }

                totalMemoryAccesses += 2 * this.matrixSize;
                this.updateStats(i * this.matrixSize + j + 1, totalMemoryAccesses);

                await sleep(200);

                // Clear highlights
                this.matrixA.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
                this.matrixB.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
                outputCell.classList.remove('active');
                outputCell.style.background = '#2196F3';
            }
        }

        document.getElementById('naiveAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateStats(threadsCompleted, memoryAccesses) {
        const totalThreads = this.matrixSize * this.matrixSize;
        const totalOps = totalThreads * this.matrixSize * 2;

        document.getElementById('naiveStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Threads Completed</div>
                    <div class="viz-stat-value">${threadsCompleted}/${totalThreads}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Memory Accesses</div>
                    <div class="viz-stat-value">${memoryAccesses}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Parallelism</div>
                    <div class="viz-stat-value">Low (Poor Memory Pattern)</div>
                </div>
            </div>
        `;
    }

    reset() {
        this.renderMatrices();
        document.getElementById('naiveStats').innerHTML = '';
    }
}

// ============================================================================
// Kernel 2: Memory Coalescing Visualization
// ============================================================================
class CoalescingViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Memory Coalescing: Warp-Level Access Patterns</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="coalescedAnimate">Show Uncoalesced</button>
                    <button class="viz-btn" id="coalescedAnimateGood">Show Coalesced</button>
                </div>
                <div class="viz-canvas" id="coalescedCanvas">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 14px; color: #999; margin-bottom: 10px;">Warp (32 threads accessing memory)</div>
                        <div id="warpThreads" style="display: flex; justify-content: center; gap: 3px; margin-bottom: 20px;"></div>
                        <div style="font-size: 14px; color: #999; margin-bottom: 10px;">Memory Addresses (128-byte cache line)</div>
                        <div id="memoryAddresses" style="display: flex; justify-content: center; gap: 3px;"></div>
                    </div>
                </div>
                <div class="viz-info">
                    <h4>Memory Coalescing Explained</h4>
                    <p><strong>Uncoalesced:</strong> Threads access scattered memory locations → Multiple transactions required</p>
                    <p><strong>Coalesced:</strong> Threads access consecutive memory locations → Single transaction</p>
                    <div id="coalescedStats"></div>
                </div>
            </div>
        `;

        this.renderMemoryLayout();

        document.getElementById('coalescedAnimate').addEventListener('click', () => this.animateUncoalesced());
        document.getElementById('coalescedAnimateGood').addEventListener('click', () => this.animateCoalesced());
    }

    renderMemoryLayout() {
        // Create 32 thread boxes
        const warpContainer = document.getElementById('warpThreads');
        warpContainer.innerHTML = '';
        for (let i = 0; i < 32; i++) {
            const thread = document.createElement('div');
            thread.id = `thread-${i}`;
            thread.style.width = '20px';
            thread.style.height = '30px';
            thread.style.background = '#666';
            thread.style.border = '1px solid #333';
            thread.style.borderRadius = '3px';
            thread.style.display = 'flex';
            thread.style.alignItems = 'center';
            thread.style.justifyContent = 'center';
            thread.style.fontSize = '10px';
            thread.style.transition = 'all 0.3s';
            thread.textContent = i;
            warpContainer.appendChild(thread);
        }

        // Create 32 memory address boxes (128 bytes = 32 floats)
        const memContainer = document.getElementById('memoryAddresses');
        memContainer.innerHTML = '';
        for (let i = 0; i < 32; i++) {
            const addr = document.createElement('div');
            addr.id = `addr-${i}`;
            addr.style.width = '20px';
            addr.style.height = '30px';
            addr.style.background = '#333';
            addr.style.border = '1px solid #555';
            addr.style.borderRadius = '3px';
            addr.style.display = 'flex';
            addr.style.alignItems = 'center';
            addr.style.justifyContent = 'center';
            addr.style.fontSize = '9px';
            addr.style.transition = 'all 0.3s';
            addr.textContent = i;
            memContainer.appendChild(addr);
        }
    }

    async animateUncoalesced() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMemoryLayout();

        let transactions = 0;

        // Simulate scattered access pattern
        for (let i = 0; i < 32; i++) {
            const thread = document.getElementById(`thread-${i}`);
            const memAddr = (i * 7) % 32; // Scattered pattern
            const addr = document.getElementById(`addr-${memAddr}`);

            thread.style.background = '#FF5722';
            addr.style.background = '#FF5722';

            transactions++;
            this.updateCoalescedStats(i + 1, transactions, 'Uncoalesced');

            await sleep(100);
        }

        this.isAnimating = false;
    }

    async animateCoalesced() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMemoryLayout();

        // All threads access consecutive addresses in one transaction
        const threads = [];
        const addrs = [];

        for (let i = 0; i < 32; i++) {
            threads.push(document.getElementById(`thread-${i}`));
            addrs.push(document.getElementById(`addr-${i}`));
        }

        // Animate all at once (coalesced)
        threads.forEach(t => t.style.background = '#4CAF50');
        addrs.forEach(a => a.style.background = '#4CAF50');

        this.updateCoalescedStats(32, 1, 'Coalesced');

        this.isAnimating = false;
    }

    updateCoalescedStats(threadsProcessed, transactions, mode) {
        const efficiency = mode === 'Coalesced' ? '100%' : `${(32 / transactions * 100).toFixed(1)}%`;
        const speedup = mode === 'Coalesced' ? '32×' : '1×';

        document.getElementById('coalescedStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Mode</div>
                    <div class="viz-stat-value">${mode}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Memory Transactions</div>
                    <div class="viz-stat-value">${transactions}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Efficiency</div>
                    <div class="viz-stat-value">${efficiency}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Theoretical Speedup</div>
                    <div class="viz-stat-value">${speedup}</div>
                </div>
            </div>
        `;
    }
}

// ============================================================================
// Coalesced Kernel: Matrix Access Pattern Visualization
// ============================================================================
class CoalescedMatrixViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.matrixSize = 8;
        this.blockSize = 4;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Coalesced Kernel: Warp-Level Matrix Access</div>
                <div class="viz-info">
                    <div id="coalescedMatrixStats"></div>
                </div>
                <br>
                <div class="viz-controls">
                    <button class="viz-btn" id="coalescedMatrixAnimate">Animate Warp Access</button>
                    <button class="viz-btn" id="coalescedMatrixReset">Reset</button>
                </div>
                <div class="viz-canvas" id="coalescedMatrixCanvas"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>Current Warp (4 threads)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FFA726;"></div>
                        <span>Accessed Memory (Coalesced)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #2196F3;"></div>
                        <span>Computed Results</span>
                    </div>
                </div>
                <div class="viz-info" style="margin-top: 20px;">
                    <h4>Key Difference from Naive</h4>
                    <p><strong>Coalesced:</strong> Threads in the same warp (with consecutive threadIdx.x) compute outputs in the same row,
                    accessing consecutive elements from matrix A → Memory coalescing!</p>
                    <p><strong>Naive:</strong> Threads in the same warp compute outputs in different rows,
                    accessing scattered elements from A → No coalescing.</p>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('coalescedMatrixCanvas');
        this.renderMatrices();

        document.getElementById('coalescedMatrixAnimate').addEventListener('click', () => this.animate());
        document.getElementById('coalescedMatrixReset').addEventListener('click', () => this.reset());
    }

    renderMatrices() {
        this.canvas.innerHTML = '<div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap;"></div>';
        const wrapper = this.canvas.firstChild;

        this.matrixA = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix A', wrapper);
        this.matrixB = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix B', wrapper);
        this.matrixC = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix C (Output)', wrapper);
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('coalescedMatrixAnimate').disabled = true;
        document.getElementById('coalescedMatrixStats').innerHTML = '';

        let totalMemoryAccesses = 0;
        let coalescedAccesses = 0;

        // Process in warps (4 threads per warp in our simplified example)
        const warpSize = this.blockSize;
        const totalWarps = (this.matrixSize * this.matrixSize) / warpSize;

        for (let warp = 0; warp < totalWarps; warp++) {
            // Calculate which row this warp is working on
            const row = Math.floor((warp * warpSize) / this.matrixSize);

            // Highlight the warp of threads (4 consecutive output elements in same row)
            const warpThreads = [];
            for (let t = 0; t < warpSize; t++) {
                const globalThreadIdx = warp * warpSize + t;
                const i = Math.floor(globalThreadIdx / this.matrixSize);
                const j = globalThreadIdx % this.matrixSize;

                if (i < this.matrixSize && j < this.matrixSize) {
                    const outputCell = this.matrixC.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                    outputCell.classList.add('active');
                    warpThreads.push({ i, j });
                }
            }

            // Highlight row in A (coalesced access - same row for all threads in warp)
            for (let k = 0; k < this.matrixSize; k++) {
                const cellA = this.matrixA.querySelector(`[data-row="${row}"][data-col="${k}"]`);
                if (cellA) cellA.classList.add('highlight');
            }

            // Highlight columns in B (one per thread)
            for (let t = 0; t < warpThreads.length; t++) {
                const { j } = warpThreads[t];
                for (let k = 0; k < this.matrixSize; k++) {
                    const cellB = this.matrixB.querySelector(`[data-row="${k}"][data-col="${j}"]`);
                    if (cellB) cellB.classList.add('highlight');
                }
            }

            totalMemoryAccesses += warpSize * this.matrixSize * 2; // Reads from A and B
            coalescedAccesses++;

            this.updateStats(warp + 1, totalWarps, coalescedAccesses, totalMemoryAccesses);

            await sleep(400);

            // Clear highlights and mark as computed
            this.matrixA.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
            this.matrixB.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
            this.matrixC.querySelectorAll('.active').forEach(el => {
                el.classList.remove('active');
                el.style.background = '#2196F3';
            });
        }

        document.getElementById('coalescedMatrixAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateStats(warpsCompleted, totalWarps, coalescedAccesses, memoryAccesses) {
        document.getElementById('coalescedMatrixStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Warps Completed</div>
                    <div class="viz-stat-value">${warpsCompleted}/${totalWarps}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Coalesced Memory Transactions</div>
                    <div class="viz-stat-value">${coalescedAccesses} (grouped accesses)</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Access Pattern</div>
                    <div class="viz-stat-value">Consecutive (Coalesced! 🚀)</div>
                </div>
            </div>
        `;
    }

    reset() {
        this.renderMatrices();
        document.getElementById('coalescedMatrixStats').innerHTML = '';
    }
}

// ============================================================================
// GPU Memory Hierarchy Visualization
// ============================================================================
class MemoryHierarchyViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <style>
                .mem-layer {
                    margin: 15px auto;
                    padding: 20px;
                    border-radius: 10px;
                    position: relative;
                    transition: all 0.3s;
                    cursor: pointer;
                }
                .mem-layer:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                }
                .mem-layer-title {
                    font-size: 18px;
                    font-weight: 700;
                    margin-bottom: 8px;
                }
                .mem-layer-stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                    margin-top: 10px;
                }
                .mem-stat {
                    font-size: 12px;
                }
                .mem-stat-label {
                    color: rgba(255,255,255,0.7);
                    font-size: 11px;
                }
                .mem-stat-value {
                    font-size: 16px;
                    font-weight: 600;
                    margin-top: 2px;
                }
                .arrow-down {
                    text-align: center;
                    font-size: 24px;
                    color: #666;
                    margin: 5px 0;
                }
                .shared-mem-detail {
                    margin-top: 15px;
                    padding: 15px;
                    background: rgba(0,0,0,0.2);
                    border-radius: 8px;
                    border-left: 4px solid #FFA726;
                }
                .thread-block {
                    display: inline-block;
                    width: 40px;
                    height: 40px;
                    background: #4CAF50;
                    border-radius: 4px;
                    margin: 3px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 10px;
                    font-weight: bold;
                }
            </style>
            <div class="viz-container">
                <div class="viz-title">RTX 4090 Memory Hierarchy</div>
                <div style="max-width: 700px; margin: 0 auto;">
                    <!-- Memory Stack -->
                    <div style="display: flex; flex-direction: column; gap: 0;">
                        <!-- Registers -->
                        <div class="mem-layer" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%; border-radius: 10px 10px 0 0;">
                            <div class="mem-layer-title">📦 Registers</div>
                            <div class="mem-layer-stats">
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Size/SM</div>
                                    <div class="mem-stat-value">256 KB</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Bandwidth</div>
                                    <div class="mem-stat-value">~19 TB/s</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Latency</div>
                                    <div class="mem-stat-value">~1 cycle</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Scope</div>
                                    <div class="mem-stat-value">Per Thread</div>
                                </div>
                            </div>
                        </div>

                        <!-- L1 Cache / Shared Memory -->
                        <div class="mem-layer" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; width: 100%; border-radius: 0;">
                            <div class="mem-layer-title">🔄 L1 Cache / Shared Memory</div>
                            <div class="mem-layer-stats">
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Size/SM</div>
                                    <div class="mem-stat-value">128 KB</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Bandwidth</div>
                                    <div class="mem-stat-value">~14 TB/s</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Latency</div>
                                    <div class="mem-stat-value">~20-30 cycles</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Scope</div>
                                    <div class="mem-stat-value">Per Thread Block</div>
                                </div>
                            </div>
                        </div>

                        <!-- L2 Cache -->
                        <div class="mem-layer" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; width: 100%; border-radius: 0;">
                            <div class="mem-layer-title">💾 L2 Cache</div>
                            <div class="mem-layer-stats">
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Total Size</div>
                                    <div class="mem-stat-value">72 MB</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Bandwidth</div>
                                    <div class="mem-stat-value">~3.5 TB/s</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Latency</div>
                                    <div class="mem-stat-value">~200 cycles</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Scope</div>
                                    <div class="mem-stat-value">All SMs</div>
                                </div>
                            </div>
                        </div>

                        <!-- Global Memory (HBM) -->
                        <div class="mem-layer" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; width: 100%; border-radius: 0 0 10px 10px;">
                            <div class="mem-layer-title">🌐 Global Memory (GDDR6X)</div>
                            <div class="mem-layer-stats">
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Total Size</div>
                                    <div class="mem-stat-value">24 GB</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Bandwidth</div>
                                    <div class="mem-stat-value">~1 TB/s</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Latency</div>
                                    <div class="mem-stat-value">~400-800 cycles</div>
                                </div>
                                <div class="mem-stat">
                                    <div class="mem-stat-label">Scope</div>
                                    <div class="mem-stat-value">Device-wide</div>
                                </div>
                            </div>
                        </div>
                    </div>


                    <!-- Key Insights -->
                    <div class="viz-info" style="margin-top: 20px;">
                        <h4>🎯 Key Insights for GEMM Optimization</h4>
                        <div style="line-height: 1.8;">
                            <strong style="color: #4CAF50;">1. Shared Memory is Critical:</strong>
                            14 TB/s vs 1 TB/s global memory = 14× faster!<br>

                            <strong style="color: #FFA726;">2. Minimize Global Memory Access:</strong>
                            Load data once, reuse in shared memory/registers<br>

                            <strong style="color: #42A5F5;">3. Maximize Register Usage:</strong>
                            19 TB/s bandwidth - keep hot data in registers<br>

                            <strong style="color: #AB47BC;">4. Coalescing Matters:</strong>
                            Even with caching, coalesced access patterns maximize throughput
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
}

// ============================================================================
// Kernel 3: Shared Memory Caching Visualization
// ============================================================================
class SharedMemoryViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.tileSize = 4;
        this.matrixSize = 8;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Shared Memory Caching: Tile-Based Computation</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="sharedAnimate">Animate Tiling</button>
                    <button class="viz-btn" id="sharedReset">Reset</button>
                </div>
                <div class="viz-info" style="margin-bottom: 15px;">
                    <h4>How Tiling Works:</h4>
                    <ol style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                        <li><strong>Load tiles</strong> from global memory into fast shared memory</li>
                        <li><strong>Compute each C element</strong> one at a time using the cached tiles</li>
                        <li><strong>Slide to next K tile</strong> and accumulate partial results</li>
                        <li><strong>Move to next output tile</strong> and repeat</li>
                    </ol>
                    <p style="margin-top: 10px; color: #FFA726;"><strong>Key:</strong> Each C element requires multiple K tiles - we accumulate partial sums!</p>
                </div>
                <div class="viz-canvas" id="sharedCanvas"></div>
                <div class="viz-info">
                    <div id="sharedStats"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #9C27B0;"></div>
                        <span>Loading Tile</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF9800;"></div>
                        <span>In Shared Memory</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>Computing</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF6B6B;"></div>
                        <span>Partial Result</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #2196F3;"></div>
                        <span>Complete Result</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('sharedCanvas');
        this.renderMatrices();

        document.getElementById('sharedAnimate').addEventListener('click', () => this.animate());
        document.getElementById('sharedReset').addEventListener('click', () => this.reset());
    }

    renderMatrices() {
        this.canvas.innerHTML = `
            <div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap;">
                <div id="sharedMatrixA"></div>
                <div id="sharedMatrixB"></div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div id="sharedMemoryA" style="margin-bottom: 20px;"></div>
                    <div id="sharedMemoryB"></div>
                </div>
                <div id="sharedMatrixC"></div>
            </div>
        `;

        this.matrixA = createMatrixGrid(this.matrixSize, this.matrixSize, 30, 'Matrix A (Global)', document.getElementById('sharedMatrixA'));
        this.matrixB = createMatrixGrid(this.matrixSize, this.matrixSize, 30, 'Matrix B (Global)', document.getElementById('sharedMatrixB'));
        this.sharedA = createMatrixGrid(this.tileSize, this.tileSize, 40, 'Shared Memory A', document.getElementById('sharedMemoryA'));
        this.sharedB = createMatrixGrid(this.tileSize, this.tileSize, 40, 'Shared Memory B', document.getElementById('sharedMemoryB'));
        this.matrixC = createMatrixGrid(this.matrixSize, this.matrixSize, 30, 'Matrix C (Output)', document.getElementById('sharedMatrixC'));
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('sharedAnimate').disabled = true;
        document.getElementById('sharedStats').innerHTML = '';

        let globalLoads = 0;
        let tilesProcessed = 0;
        let computationsCompleted = 0;
        const numTiles = Math.ceil(this.matrixSize / this.tileSize);

        // For each output tile
        for (let blockRow = 0; blockRow < numTiles; blockRow++) {
            for (let blockCol = 0; blockCol < numTiles; blockCol++) {

                // Loop over K dimension in tiles
                for (let tileK = 0; tileK < numTiles; tileK++) {
                    // Highlight tile from A being loaded
                    for (let i = 0; i < this.tileSize; i++) {
                        for (let k = 0; k < this.tileSize; k++) {
                            const row = blockRow * this.tileSize + i;
                            const col = tileK * this.tileSize + k;

                            if (row < this.matrixSize && col < this.matrixSize) {
                                const cellA = this.matrixA.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                                if (cellA) cellA.style.background = '#9C27B0';
                            }
                        }
                    }

                    // Highlight tile from B being loaded
                    for (let k = 0; k < this.tileSize; k++) {
                        for (let j = 0; j < this.tileSize; j++) {
                            const row = tileK * this.tileSize + k;
                            const col = blockCol * this.tileSize + j;

                            if (row < this.matrixSize && col < this.matrixSize) {
                                const cellB = this.matrixB.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                                if (cellB) cellB.style.background = '#9C27B0';
                            }
                        }
                    }

                    await sleep(300);

                    // Show data in shared memory
                    for (let i = 0; i < this.tileSize; i++) {
                        for (let j = 0; j < this.tileSize; j++) {
                            const sharedCellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                            const sharedCellB = this.sharedB.querySelector(`[data-row="${i}"][data-col="${j}"]`);

                            sharedCellA.style.background = '#FF9800';
                            sharedCellB.style.background = '#FF9800';
                        }
                    }

                    globalLoads += 2 * this.tileSize * this.tileSize;

                    await sleep(300);

                    // Compute each element of the output tile one at a time
                    for (let i = 0; i < this.tileSize; i++) {
                        for (let j = 0; j < this.tileSize; j++) {
                            const row = blockRow * this.tileSize + i;
                            const col = blockCol * this.tileSize + j;

                            if (row < this.matrixSize && col < this.matrixSize) {
                                // Highlight the row in shared memory A
                                for (let k = 0; k < this.tileSize; k++) {
                                    const cellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${k}"]`);
                                    if (cellA) cellA.classList.add('active');
                                }

                                // Highlight the column in shared memory B
                                for (let k = 0; k < this.tileSize; k++) {
                                    const cellB = this.sharedB.querySelector(`[data-row="${k}"][data-col="${j}"]`);
                                    if (cellB) cellB.classList.add('active');
                                }

                                await sleep(150);

                                // Update the output cell
                                const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                                if (cellC) {
                                    // Color it partially complete if not the last K tile
                                    if (tileK < numTiles - 1) {
                                        cellC.style.background = '#FF6B6B'; // Red/orange for partial
                                    } else {
                                        cellC.style.background = '#2196F3'; // Bright blue for complete
                                    }
                                }

                                computationsCompleted++;

                                // Clear active highlights
                                this.sharedA.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                                this.sharedB.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                            }
                        }
                    }

                    this.updateSharedStats(tilesProcessed, globalLoads, computationsCompleted, tileK + 1, numTiles);

                    await sleep(300);

                    // Clear shared memory visualization
                    this.sharedA.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                    this.sharedB.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                    this.matrixA.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                    this.matrixB.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                }

                tilesProcessed++;
            }
        }

        document.getElementById('sharedAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateSharedStats(tilesProcessed, globalLoads, computationsCompleted, currentKTile, totalKTiles) {
        const totalTiles = Math.pow(Math.ceil(this.matrixSize / this.tileSize), 2);
        const naiveLoads = this.matrixSize * this.matrixSize * 2 * this.matrixSize;
        const reduction = ((naiveLoads - globalLoads) / naiveLoads * 100).toFixed(1);
        const totalComputations = this.matrixSize * this.matrixSize;

        document.getElementById('sharedStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Output Tiles</div>
                    <div class="viz-stat-value">${tilesProcessed}/${totalTiles}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">K Dimension Progress</div>
                    <div class="viz-stat-value">${currentKTile}/${totalKTiles}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">C Elements Computed</div>
                    <div class="viz-stat-value">${computationsCompleted} partial sums</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Global Memory Loads</div>
                    <div class="viz-stat-value">${globalLoads}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Memory Traffic Reduction</div>
                    <div class="viz-stat-value">${reduction}%</div>
                </div>
            </div>
        `;
    }

    reset() {
        this.renderMatrices();
        document.getElementById('sharedStats').innerHTML = '';
    }
}

// ============================================================================
// Kernel 4: 1D Block Tiling Visualization
// ============================================================================
class Tiling1DViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.tileSize = 4; // Shared memory tile size (like before)
        this.TM = 4; // Each thread computes TM outputs
        this.matrixSize = 8;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">1D Block Tiling: Register-Level Optimization</div>
                <div class="viz-info" style="margin-bottom: 15px;">
                    <h4>Building on Shared Memory:</h4>
                    <p style="margin: 10px 0; line-height: 1.8;">
                        We still use shared memory tiles (<code>tile_a</code>, <code>tile_b</code>), but now each thread computes <strong>TM = ${this.TM} outputs</strong> instead of just 1!
                    </p>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; padding: 15px; background: #2a2a2a; border-radius: 8px;">
                        <div style="border-right: 2px solid #444; padding-right: 15px;">
                            <div style="font-weight: 600; color: #FF9800; margin-bottom: 10px;">❌ Previous (Shared Memory Only)</div>
                            <ul style="margin: 0; padding-left: 20px; line-height: 1.8; font-size: 13px;">
                                <li>Load 1 value from <code>tile_b</code></li>
                                <li>Load 1 value from <code>tile_a</code></li>
                                <li>Compute 1 output</li>
                                <li><strong style="color: #FF6B6B;">No register reuse!</strong></li>
                            </ul>
                            <div style="margin-top: 10px; padding: 8px; background: #1a1a1a; border-radius: 4px; font-size: 12px;">
                                <strong>Ratio:</strong> 2 SMEM reads → 1 output<br>
                                <strong>Efficiency:</strong> 0.5 outputs/read
                            </div>
                        </div>
                        <div style="padding-left: 15px;">
                            <div style="font-weight: 600; color: #4CAF50; margin-bottom: 10px;">✅ Now (1D Block Tiling)</div>
                            <ul style="margin: 0; padding-left: 20px; line-height: 1.8; font-size: 13px;">
                                <li>Load 1 value into <code style="color: #00BCD4;">b_tmp</code> register</li>
                                <li>Load TM values from <code>tile_a</code></li>
                                <li><strong style="color: #8BC34A;">Compute TM outputs!</strong></li>
                                <li><strong style="color: #00BCD4;">b_tmp reused ${this.TM}×</strong></li>
                            </ul>
                            <div style="margin-top: 10px; padding: 8px; background: #1a1a1a; border-radius: 4px; font-size: 12px;">
                                <strong>Ratio:</strong> (1+TM) SMEM reads → TM outputs<br>
                                <strong>Efficiency:</strong> ${(this.TM / (1 + this.TM)).toFixed(2)} outputs/read (${((this.TM / (1 + this.TM)) / 0.5).toFixed(1)}× better!)
                            </div>
                        </div>
                    </div>

                    <p style="margin-top: 10px; color: #FFA726;"><strong>Key Insight:</strong> By caching <code>b_tmp</code> in a register and reusing it ${this.TM} times, we reduce shared memory traffic and increase arithmetic intensity!</p>
                </div>
                <div class="viz-controls">
                    <button class="viz-btn" id="tiling1dAnimate">Animate</button>
                    <button class="viz-btn" id="tiling1dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="tiling1dCanvas"></div>
                <div class="viz-info">
                    <div id="tiling1dStats"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF9800;"></div>
                        <span>tile_a[], tile_b[] (Shared Memory)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #00BCD4;"></div>
                        <span>b_tmp (Register)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #E91E63;"></div>
                        <span>Computing</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #8BC34A;"></div>
                        <span>thread_results[] (Complete)</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('tiling1dCanvas');
        this.renderMatrices();

        document.getElementById('tiling1dAnimate').addEventListener('click', () => this.animate());
        document.getElementById('tiling1dReset').addEventListener('click', () => this.reset());
    }

    renderMatrices() {
        this.canvas.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 20px;">
                <!-- Previous Kernel (Shared Memory Only) -->
                <div style="border: 2px solid #FF6B6B; border-radius: 8px; padding: 15px; background: #1a1a1a;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #FF6B6B; margin-bottom: 5px;">❌ Previous: Shared Memory Only</div>
                        <div style="font-size: 12px; color: #999;">Each thread computes 1 output</div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                        <div style="display: flex; gap: 15px;">
                            <div>
                                <div style="font-size: 11px; color: #FF9800; margin-bottom: 5px; text-align: center;">tile_a</div>
                                <div id="prevSharedA"></div>
                            </div>
                            <div>
                                <div style="font-size: 11px; color: #FF9800; margin-bottom: 5px; text-align: center;">tile_b</div>
                                <div id="prevSharedB"></div>
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #999; margin-bottom: 5px; text-align: center;">Output (1 per thread)</div>
                            <div id="prevMatrixC"></div>
                        </div>
                    </div>
                </div>

                <!-- Current Kernel (1D Block Tiling) -->
                <div style="border: 2px solid #4CAF50; border-radius: 8px; padding: 15px; background: #1a1a1a;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #4CAF50; margin-bottom: 5px;">✅ Now: 1D Block Tiling</div>
                        <div style="font-size: 12px; color: #999;">Each thread computes TM=${this.TM} outputs</div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                        <div style="display: flex; gap: 15px;">
                            <div>
                                <div style="font-size: 11px; color: #FF9800; margin-bottom: 5px; text-align: center;">tile_a</div>
                                <div id="tiling1dSharedA"></div>
                            </div>
                            <div>
                                <div style="font-size: 11px; color: #FF9800; margin-bottom: 5px; text-align: center;">tile_b</div>
                                <div id="tiling1dSharedB"></div>
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #00BCD4; margin-bottom: 5px; text-align: center;">Registers (b_tmp)</div>
                            <div id="tiling1dRegisters" style="display: flex; gap: 5px;"></div>
                        </div>
                        <div>
                            <div style="font-size: 11px; color: #8BC34A; margin-bottom: 5px; text-align: center;">Output (TM per thread)</div>
                            <div id="tiling1dMatrixC"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Previous kernel visualization (smaller)
        this.prevSharedA = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('prevSharedA'));
        this.prevSharedB = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('prevSharedB'));
        this.prevMatrixC = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('prevMatrixC'));

        // Current kernel visualization
        this.sharedA = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('tiling1dSharedA'));
        this.sharedB = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('tiling1dSharedB'));
        this.matrixC = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('tiling1dMatrixC'));

        // Create register visualization - simpler for comparison view
        const regContainer = document.getElementById('tiling1dRegisters');
        regContainer.innerHTML = `
            <div id="regN" style="width: 80px; height: 30px; background: #333; border: 2px solid #555;
                 border-radius: 6px; display: flex; align-items: center; justify-content: center;
                 font-size: 10px; transition: all 0.3s;">b_tmp</div>
        `;
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('tiling1dAnimate').disabled = true;
        document.getElementById('tiling1dStats').innerHTML = '';

        let prevSMEMReads = 0;
        let currentSMEMReads = 0;
        let computationsCompleted = 0;

        // Show tiles loaded into shared memory (both sides)
        for (let i = 0; i < this.tileSize; i++) {
            for (let j = 0; j < this.tileSize; j++) {
                // Previous kernel
                const prevCellA = this.prevSharedA.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                const prevCellB = this.prevSharedB.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                prevCellA.style.background = '#FF9800';
                prevCellB.style.background = '#FF9800';

                // Current kernel
                const cellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                const cellB = this.sharedB.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                cellA.style.background = '#FF9800';
                cellB.style.background = '#FF9800';
            }
        }

        await sleep(500);

        // Process one column at a time to show comparison
        const threadsPerRow = this.tileSize / this.TM;

        for (let threadRow = 0; threadRow < threadsPerRow; threadRow++) {
            for (let col = 0; col < this.tileSize; col++) {
                for (let k = 0; k < this.tileSize; k++) {

                    // ===== PREVIOUS KERNEL (Left side): Read 2 values, compute 1 output =====
                    for (let tm = 0; tm < this.TM; tm++) {
                        const row = threadRow * this.TM + tm;

                        // Read from tile_a
                        const prevCellA = this.prevSharedA.querySelector(`[data-row="${row}"][data-col="${k}"]`);
                        if (prevCellA) prevCellA.classList.add('active');
                        prevSMEMReads++;

                        // Read from tile_b
                        const prevCellB = this.prevSharedB.querySelector(`[data-row="${k}"][data-col="${col}"]`);
                        if (prevCellB) prevCellB.classList.add('active');
                        prevSMEMReads++;

                        await sleep(150);

                        // Compute 1 output
                        const prevCellC = this.prevMatrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                        if (prevCellC) {
                            prevCellC.style.background = '#E91E63'; // Computing
                        }

                        await sleep(80);

                        if (prevCellC) {
                            if (k < this.tileSize - 1) {
                                prevCellC.style.background = '#FF6B6B'; // Partial
                            } else {
                                prevCellC.style.background = '#4A9B6C'; // Complete (muted green)
                            }
                        }

                        // Clear highlights
                        this.prevSharedA.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                        this.prevSharedB.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                    }

                    // ===== CURRENT KERNEL (Right side): Cache b_tmp, compute TM outputs =====
                    // Load TM values from tile_a
                    for (let tm = 0; tm < this.TM; tm++) {
                        const row = threadRow * this.TM + tm;
                        const cellA = this.sharedA.querySelector(`[data-row="${row}"][data-col="${k}"]`);
                        if (cellA) cellA.classList.add('active');
                    }
                    currentSMEMReads += this.TM;

                    // Load 1 value from tile_b into b_tmp register (REUSED!)
                    const cellB = this.sharedB.querySelector(`[data-row="${k}"][data-col="${col}"]`);
                    if (cellB) cellB.classList.add('active');

                    const regN = document.getElementById('regN');
                    regN.style.background = '#00BCD4';
                    regN.style.borderColor = '#00BCD4';
                    regN.textContent = '🔄 b_tmp';

                    currentSMEMReads += 1;

                    await sleep(200);

                    // Compute TM outputs using the SAME b_tmp (register reuse!)
                    for (let tm = 0; tm < this.TM; tm++) {
                        const row = threadRow * this.TM + tm;
                        const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);

                        // Show b_tmp being reused
                        regN.style.background = '#E91E63';
                        regN.textContent = `✅ ×${tm + 1}`;

                        if (cellC) {
                            cellC.style.background = '#E91E63'; // Computing
                        }

                        await sleep(100);

                        if (cellC) {
                            if (k < this.tileSize - 1) {
                                cellC.style.background = '#FF6B6B'; // Partial
                            } else {
                                cellC.style.background = '#8BC34A'; // Complete
                                if (tm === this.TM - 1) computationsCompleted += this.TM;
                            }
                        }
                    }

                    // Reset register
                    regN.style.background = '#333';
                    regN.style.borderColor = '#555';
                    regN.textContent = 'b_tmp';

                    // Clear active highlights from shared memory
                    this.sharedA.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                    this.sharedB.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                }

                this.updateTiling1DStats(prevSMEMReads, currentSMEMReads, computationsCompleted, col + 1);
            }
        }

        document.getElementById('tiling1dAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateTiling1DStats(prevSMEMReads, currentSMEMReads, outputsComplete, colsProcessed) {
        const totalOutputs = this.tileSize * this.tileSize;
        const smemTrafficReduction = prevSMEMReads > 0 ?
            ((1 - (currentSMEMReads / prevSMEMReads)) * 100).toFixed(1) : 0;

        document.getElementById('tiling1dStats').innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                <div style="background: #2a2a2a; padding: 12px; border-radius: 6px; border-left: 4px solid #FF6B6B;">
                    <div style="font-weight: 600; color: #FF6B6B; margin-bottom: 8px;">❌ Previous Kernel</div>
                    <div style="font-size: 13px; line-height: 1.6;">
                        <div><strong>SMEM Reads:</strong> ${prevSMEMReads}</div>
                        <div><strong>Pattern:</strong> 2 reads per output</div>
                        <div><strong>Outputs:</strong> ${outputsComplete}/${totalOutputs}</div>
                    </div>
                </div>
                <div style="background: #2a2a2a; padding: 12px; border-radius: 6px; border-left: 4px solid #4CAF50;">
                    <div style="font-weight: 600; color: #4CAF50; margin-bottom: 8px;">✅ Current Kernel</div>
                    <div style="font-size: 13px; line-height: 1.6;">
                        <div><strong>SMEM Reads:</strong> ${currentSMEMReads}</div>
                        <div><strong>Pattern:</strong> 1 b_tmp + ${this.TM} tile_a per ${this.TM} outputs</div>
                        <div><strong>Outputs:</strong> ${outputsComplete}/${totalOutputs}</div>
                    </div>
                </div>
            </div>
            <div class="viz-stats">
                <div class="viz-stat" style="border-left: 3px solid #FF6B6B;">
                    <div class="viz-stat-label">Previous SMEM Traffic</div>
                    <div class="viz-stat-value" style="color: #FF6B6B;">${prevSMEMReads} reads</div>
                </div>
                <div class="viz-stat" style="border-left: 3px solid #4CAF50;">
                    <div class="viz-stat-label">Current SMEM Traffic</div>
                    <div class="viz-stat-value" style="color: #4CAF50;">${currentSMEMReads} reads</div>
                </div>
                <div class="viz-stat" style="border-left: 3px solid #00BCD4;">
                    <div class="viz-stat-label">b_tmp Reuse Factor</div>
                    <div class="viz-stat-value" style="color: #00BCD4;">${this.TM}× per load</div>
                </div>
                <div class="viz-stat" style="border-left: 3px solid #FFA726;">
                    <div class="viz-stat-label">SMEM Traffic Reduction</div>
                    <div class="viz-stat-value" style="color: #FFA726;">${smemTrafficReduction}%</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 12px; background: #2a2a2a; border-radius: 6px; border-left: 4px solid #FFA726;">
                <div style="font-size: 13px; line-height: 1.6;">
                    <strong style="color: #FFA726;">📊 Side-by-Side Comparison:</strong><br>
                    <span style="color: #FF6B6B;">Left (Previous):</span> Reads 2 values, computes 1 output (no register reuse)<br>
                    <span style="color: #4CAF50;">Right (1D Tiling):</span> Caches <code>b_tmp</code> in register, reuses it ${this.TM}× for ${this.TM} outputs!<br>
                    <span style="color: #00BCD4;">Result:</span> ${smemTrafficReduction}% less shared memory traffic = faster execution!
                </div>
            </div>
        `;
    }

    reset() {
        this.renderMatrices();
        document.getElementById('tiling1dStats').innerHTML = '';
    }
}

// ============================================================================
// Kernel 4: 1D Block Tiling Pipeline Visualization
// Full Matrix → Shared Memory → Thread Blocks → Thread Tiles → CUDA Cores
// ============================================================================
class Tiling1DPipelineViz {
    constructor(containerId) {
        console.log('Tiling1DPipelineViz initializing with containerId:', containerId);
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        console.log('Tiling1DPipelineViz container found, initializing...');
        this.isAnimating = false;
        // Configuration matching typical 1D block tiling
        this.matrixSizeM = 16;  // Full matrix rows
        this.matrixSizeN = 16;  // Full matrix cols
        this.matrixSizeK = 16;  // Common dimension
        this.BM = 8;  // Block tile M
        this.BN = 8;  // Block tile N
        this.BK = 4;  // Block tile K
        this.TM = 4;  // Thread tile M (each thread computes TM outputs)
        this.currentStep = 0;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">1D Block Tiling: Complete Memory Hierarchy Pipeline</div>
                <div class="viz-info" style="margin-bottom: 15px;">
                    <h4>Full Data Flow: Global Memory → Shared Memory → Registers → CUDA Cores</h4>
                    <p style="margin: 10px 0; line-height: 1.8;">
                        This visualization shows how data flows through the entire GPU memory hierarchy:
                        <strong>Full Matrices (GMEM)</strong> → <strong>Thread Block Tiles (SMEM)</strong> →
                        <strong>Thread Tiles (Registers)</strong> → <strong>Computation (CUDA Cores)</strong>
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 15px 0; padding: 15px; background: #2a2a2a; border-radius: 8px; font-size: 12px;">
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #4CAF50; margin-bottom: 5px;">1️⃣ Global Memory</div>
                            <div style="color: #999;">Full ${this.matrixSizeM}×${this.matrixSizeK} & ${this.matrixSizeK}×${this.matrixSizeN} matrices</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #FF9800; margin-bottom: 5px;">2️⃣ Shared Memory</div>
                            <div style="color: #999;">Block tiles: ${this.BM}×${this.BK} & ${this.BK}×${this.BN}</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #00BCD4; margin-bottom: 5px;">3️⃣ Registers</div>
                            <div style="color: #999;">Thread tiles: TM=${this.TM} per thread</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #E91E63; margin-bottom: 5px;">4️⃣ CUDA Cores</div>
                            <div style="color: #999;">FMA operations</div>
                        </div>
                    </div>
                </div>
                <div class="viz-controls">
                    <button class="viz-btn" id="pipeline1dAnimate">Animate Pipeline</button>
                    <button class="viz-btn" id="pipeline1dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="pipeline1dCanvas"></div>
                <div class="viz-info">
                    <div id="pipeline1dStats"></div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('pipeline1dCanvas');
        this.renderPipeline();

        document.getElementById('pipeline1dAnimate').addEventListener('click', () => this.animate());
        document.getElementById('pipeline1dReset').addEventListener('click', () => this.reset());
    }

    renderPipeline() {
        this.canvas.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 30px;">
                <!-- Stage 1: Global Memory (Full Matrices) -->
                <div style="background: #1a1a1a; padding: 20px; border-radius: 8px; border: 2px solid #4CAF50;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #4CAF50; margin-bottom: 5px;">
                            1️⃣ Global Memory (GMEM)
                        </div>
                        <div style="font-size: 12px; color: #999;">Full matrices stored in GPU DRAM</div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <div>
                                <div style="font-size: 11px; color: #4CAF50; margin-bottom: 5px; text-align: center;">
                                    Matrix A (${this.matrixSizeM}×${this.matrixSizeK})
                                </div>
                                <div id="globalMatrixA"></div>
                            </div>
                            <div style="font-size: 20px; color: #666;">×</div>
                            <div>
                                <div style="font-size: 11px; color: #4CAF50; margin-bottom: 5px; text-align: center;">
                                    Matrix B (${this.matrixSizeK}×${this.matrixSizeN})
                                </div>
                                <div id="globalMatrixB"></div>
                            </div>
                        </div>
                        <div style="font-size: 20px; color: #666;">‖</div>
                        <div>
                            <div style="font-size: 11px; color: #4CAF50; margin-bottom: 5px; text-align: center;">
                                Matrix C (${this.matrixSizeM}×${this.matrixSizeN})
                            </div>
                            <div id="globalMatrixC"></div>
                        </div>
                    </div>
                </div>

                <!-- Arrow Down -->
                <div style="text-align: center; color: #FF9800; font-size: 32px; height: 20px;">↓</div>

                <!-- Stage 2: Shared Memory (Thread Block Tiles) -->
                <div style="background: #1a1a1a; padding: 20px; border-radius: 8px; border: 2px solid #FF9800;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #FF9800; margin-bottom: 5px;">
                            2️⃣ Shared Memory (SMEM) - Thread Block Tiles
                        </div>
                        <div style="font-size: 12px; color: #999;">Each thread block loads tiles from global memory</div>
                    </div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">
                        <div>
                            <div style="font-size: 11px; color: #FF9800; margin-bottom: 5px; text-align: center;">
                                tile_a[${this.BM}×${this.BK}]
                            </div>
                            <div id="sharedTileA"></div>
                        </div>
                        <div style="font-size: 24px; color: #666;">×</div>
                        <div>
                            <div style="font-size: 11px; color: #FF9800; margin-bottom: 5px; text-align: center;">
                                tile_b[${this.BK}×${this.BN}]
                            </div>
                            <div id="sharedTileB"></div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; text-align: center; font-size: 11px; color: #FFA726;">
                        <strong>All ${(this.BM / this.TM) * this.BN} threads in block cooperate to load tiles</strong>
                    </div>
                </div>

                <!-- Arrow Down -->
                <div style="text-align: center; color: #00BCD4; font-size: 32px; height: 20px;">↓</div>

                <!-- Stage 3: Registers (Thread Tiles) -->
                <div style="background: #1a1a1a; padding: 20px; border-radius: 8px; border: 2px solid #00BCD4;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #00BCD4; margin-bottom: 5px;">
                            3️⃣ Registers (Thread Tiles) - Per Thread
                        </div>
                        <div style="font-size: 12px; color: #999;">Each thread caches data for its TM=${this.TM} outputs</div>
                    </div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">
                        <div>
                            <div style="font-size: 11px; color: #00BCD4; margin-bottom: 5px; text-align: center;">
                                thread_tile_a[TM=${this.TM}]
                            </div>
                            <div id="registerTileA"></div>
                        </div>
                        <div style="font-size: 24px; color: #666;">×</div>
                        <div>
                            <div style="font-size: 11px; color: #00BCD4; margin-bottom: 5px; text-align: center;">
                                b_tmp (register)
                            </div>
                            <div id="registerB"></div>
                        </div>
                        <div style="font-size: 24px; color: #666;">=</div>
                        <div>
                            <div style="font-size: 11px; color: #00BCD4; margin-bottom: 5px; text-align: center;">
                                thread_results[TM=${this.TM}]
                            </div>
                            <div id="registerResults"></div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; text-align: center; font-size: 11px; color: #00BCD4;">
                        <strong>b_tmp reused ${this.TM}× for ${this.TM} FMA operations</strong>
                    </div>
                </div>

                <!-- Arrow Down -->
                <div style="text-align: center; color: #E91E63; font-size: 32px; height: 20px;">↓</div>

                <!-- Stage 4: CUDA Cores -->
                <div style="background: #1a1a1a; padding: 20px; border-radius: 8px; border: 2px solid #E91E63;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #E91E63; margin-bottom: 5px;">
                            4️⃣ CUDA Cores - Computation
                        </div>
                        <div style="font-size: 12px; color: #999;">FMA: result[i] += thread_tile_a[i] * b_tmp</div>
                    </div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                        <div id="cudaCores" style="display: flex; gap: 10px;"></div>
                    </div>
                    <div style="margin-top: 15px; text-align: center; font-size: 11px; color: #E91E63;">
                        <strong>Each thread executes TM=${this.TM} FMA operations per K iteration</strong>
                    </div>
                </div>
            </div>
        `;

        // Render matrices at each stage
        this.globalA = createMatrixGrid(this.matrixSizeM, this.matrixSizeK, 14, '', document.getElementById('globalMatrixA'));
        this.globalB = createMatrixGrid(this.matrixSizeK, this.matrixSizeN, 14, '', document.getElementById('globalMatrixB'));
        this.globalC = createMatrixGrid(this.matrixSizeM, this.matrixSizeN, 14, '', document.getElementById('globalMatrixC'));

        this.sharedA = createMatrixGrid(this.BM, this.BK, 30, '', document.getElementById('sharedTileA'));
        this.sharedB = createMatrixGrid(this.BK, this.BN, 30, '', document.getElementById('sharedTileB'));

        // Registers - vertical arrays
        const regAContainer = document.getElementById('registerTileA');
        regAContainer.style.display = 'flex';
        regAContainer.style.flexDirection = 'column';
        regAContainer.style.gap = '5px';
        for (let i = 0; i < this.TM; i++) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.style.width = '80px';
            cell.style.height = '35px';
            cell.style.background = '#333';
            cell.style.fontSize = '10px';
            cell.setAttribute('data-reg-a', i);
            cell.innerHTML = `<div style="font-weight: 600;">a[${i}]</div><div style="font-size: 9px; color: #999;"></div>`;
            regAContainer.appendChild(cell);
        }

        const regBContainer = document.getElementById('registerB');
        const bCell = document.createElement('div');
        bCell.className = 'matrix-cell';
        bCell.style.width = '80px';
        bCell.style.height = '35px';
        bCell.style.background = '#333';
        bCell.style.fontSize = '10px';
        bCell.setAttribute('data-reg-b', '0');
        bCell.innerHTML = `<div style="font-weight: 600;">b_tmp</div><div style="font-size: 9px; color: #999;"></div>`;
        regBContainer.appendChild(bCell);

        const regResultsContainer = document.getElementById('registerResults');
        regResultsContainer.style.display = 'flex';
        regResultsContainer.style.flexDirection = 'column';
        regResultsContainer.style.gap = '5px';
        for (let i = 0; i < this.TM; i++) {
            const cell = document.createElement('div');
            cell.className = 'matrix-cell';
            cell.style.width = '60px';
            cell.style.height = '30px';
            cell.style.background = '#333';
            cell.setAttribute('data-result', i);
            cell.textContent = `r[${i}]`;
            regResultsContainer.appendChild(cell);
        }

        // CUDA Cores visualization
        const coresContainer = document.getElementById('cudaCores');
        for (let i = 0; i < this.TM; i++) {
            const core = document.createElement('div');
            core.style.cssText = `
                width: 50px; height: 50px; background: #333; border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 10px; font-weight: 600; color: #999;
                border: 2px solid #555;
            `;
            core.setAttribute('data-core', i);
            core.textContent = `FMA${i}`;
            coresContainer.appendChild(core);
        }

        this.updateStats();
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        document.getElementById('pipeline1dAnimate').disabled = true;

        this.reset();
        await sleep(500);

        const numKIterations = this.matrixSizeK / this.BK; // 16 / 4 = 4 iterations
        const totalSteps = numKIterations;

        // Process all K iterations
        for (let kIter = 0; kIter < numKIterations; kIter++) {
            const kOffset = kIter * this.BK;
            const isLastIteration = (kIter === numKIterations - 1);

            this.updateStats(`<strong style="color: #FFA726;">📍 K Iteration ${kIter + 1}/${numKIterations}</strong> - Processing K tiles [${kOffset}:${kOffset + this.BK}]`);
            await sleep(300);

            // Step 1: Highlight GMEM blocks being loaded
            this.updateStats(`<strong style="color: #4CAF50;">1️⃣ GMEM → SMEM</strong>: Loading blocks from Global Memory`);
            await this.highlightGMEMBlocks(kIter);
            await sleep(700);

            // Step 2: Show blocks loaded in SMEM
            this.updateStats(`<strong style="color: #FF9800;">2️⃣ SMEM Ready</strong>: tile_a[${this.BM}×${this.BK}] ⊗ tile_b[${this.BK}×${this.BN}]`);
            await this.loadSharedMemory(kIter);
            await sleep(700);

            // Step 3: Highlight thread's tile in SMEM and load to registers
            this.updateStats(`<strong style="color: #00BCD4;">3️⃣ SMEM → Registers</strong>: Thread loads ${this.TM} values + b_tmp`);
            await this.loadRegisters(kIter);
            await sleep(900);

            // Step 4: Compute FMA operations
            this.updateStats(`<strong style="color: #E91E63;">4️⃣ Compute</strong>: ${this.TM} FMA ops (b_tmp reused ${this.TM}×)`);
            await this.computeCudaCores(kIter);
            await sleep(700);

            // Step 5: Show partial results (pink) or final results (green)
            if (!isLastIteration) {
                this.updateStats(`<strong style="color: #FF1493;">5️⃣ Partial Results</strong>: Accumulating... (${kIter + 1}/${numKIterations} complete)`);
                await this.showPartialResults(kIter);
                await sleep(600);
            } else {
                this.updateStats(`<strong style="color: #8BC34A;">5️⃣ Final Results</strong>: Writing ${this.TM} outputs to Global Memory`);
                await this.showFinalResults();
                await sleep(800);
            }

            // Mark processed GMEM cells as complete (green)
            await this.markGMEMComplete(kIter, isLastIteration);
            await sleep(400);
        }

        this.updateStats(`✅ <strong>Complete!</strong> All ${numKIterations} K tiles processed. Matrix multiplication done!`);

        this.isAnimating = false;
        document.getElementById('pipeline1dAnimate').disabled = false;
    }

    async highlightGMEMBlocks(kIter = 0) {
        const kOffset = kIter * this.BK;

        // Highlight the block being loaded in global A (rows 0-BM, cols kOffset to kOffset+BK)
        for (let i = 0; i < this.BM; i++) {
            for (let j = 0; j < this.BK; j++) {
                const cell = this.globalA.querySelector(`[data-row="${i}"][data-col="${kOffset + j}"]`);
                if (cell && cell.style.background !== 'rgb(139, 195, 74)') { // Don't override green (complete)
                    cell.style.background = '#4CAF50';
                    cell.style.border = '2px solid #8BC34A';
                }
            }
        }

        // Highlight the block being loaded in global B (rows kOffset to kOffset+BK, cols 0-BN)
        for (let i = 0; i < this.BK; i++) {
            for (let j = 0; j < this.BN; j++) {
                const cell = this.globalB.querySelector(`[data-row="${kOffset + i}"][data-col="${j}"]`);
                if (cell && cell.style.background !== 'rgb(139, 195, 74)') {
                    cell.style.background = '#4CAF50';
                    cell.style.border = '2px solid #8BC34A';
                }
            }
        }
    }

    async markGMEMComplete(kIter, isLastIteration) {
        const kOffset = kIter * this.BK;

        // Mark processed cells in A as complete (green)
        for (let i = 0; i < this.BM; i++) {
            for (let j = 0; j < this.BK; j++) {
                const cell = this.globalA.querySelector(`[data-row="${i}"][data-col="${kOffset + j}"]`);
                if (cell) {
                    cell.style.background = '#8BC34A';
                    cell.style.border = '1px solid #333';
                }
            }
        }

        // Mark processed cells in B as complete (green)
        for (let i = 0; i < this.BK; i++) {
            for (let j = 0; j < this.BN; j++) {
                const cell = this.globalB.querySelector(`[data-row="${kOffset + i}"][data-col="${j}"]`);
                if (cell) {
                    cell.style.background = '#8BC34A';
                    cell.style.border = '1px solid #333';
                }
            }
        }

        // Mark output cells in C
        if (isLastIteration) {
            for (let i = 0; i < this.TM; i++) {
                const cell = this.globalC.querySelector(`[data-row="${i}"][data-col="0"]`);
                if (cell) {
                    cell.style.background = '#8BC34A';
                    cell.style.border = '2px solid #AED581';
                }
            }
        }
    }

    async showPartialResults(kIter) {
        // Show partial sums in pink/magenta
        for (let i = 0; i < this.TM; i++) {
            const resultCell = document.querySelector(`[data-result="${i}"]`);
            resultCell.style.background = '#FF1493'; // Deep pink for partial
            resultCell.style.border = '2px solid #FF69B4';

            const cell = this.globalC.querySelector(`[data-row="${i}"][data-col="0"]`);
            if (cell) {
                cell.style.background = '#FFB6C1'; // Light pink
                cell.style.border = '2px solid #FF1493';
            }
        }
    }

    async showFinalResults() {
        // Show final accumulated results in green
        for (let i = 0; i < this.TM; i++) {
            const resultCell = document.querySelector(`[data-result="${i}"]`);
            resultCell.style.background = '#8BC34A';
            resultCell.style.border = '2px solid #AED581';

            const cell = this.globalC.querySelector(`[data-row="${i}"][data-col="0"]`);
            if (cell) {
                cell.style.background = '#8BC34A';
                cell.style.border = '2px solid #AED581';
            }
        }
    }

    async loadSharedMemory(kIter = 0) {
        this.currentStep = 2;

        for (let i = 0; i < this.BM; i++) {
            for (let j = 0; j < this.BK; j++) {
                const cell = this.sharedA.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                cell.style.background = '#FF9800';
                cell.style.border = '2px solid #FFA726';
            }
        }

        for (let i = 0; i < this.BK; i++) {
            for (let j = 0; j < this.BN; j++) {
                const cell = this.sharedB.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                cell.style.background = '#FF9800';
                cell.style.border = '2px solid #FFA726';
            }
        }
    }

    async loadRegisters(kIter = 0) {
        this.currentStep = 3;

        const kOffset = kIter * this.BK;
        const threadCol = 0;

        // Load thread_tile_a values from shared memory
        for (let i = 0; i < this.TM; i++) {
            const cell = this.sharedA.querySelector(`[data-row="${i}"][data-col="${threadCol}"]`);
            cell.style.background = '#00BCD4';
            cell.style.border = '2px solid #4DD0E1';
            await sleep(50);

            const regCell = document.querySelector(`[data-reg-a="${i}"]`);
            regCell.style.background = '#00BCD4';
            regCell.style.border = '2px solid #4DD0E1';
            // Show GMEM and SMEM indices
            const gmemIdx = `A[${i},${kOffset + threadCol}]`;
            const smemIdx = `tile_a[${i},${threadCol}]`;
            regCell.innerHTML = `<div style="font-weight: 600;">a[${i}]</div><div style="font-size: 8px; color: #4DD0E1;">${gmemIdx}<br/>${smemIdx}</div>`;
        }

        // Load b_tmp from shared memory
        const bCell = this.sharedB.querySelector(`[data-row="${threadCol}"][data-col="0"]`);
        bCell.style.background = '#00BCD4';
        bCell.style.border = '2px solid #4DD0E1';

        const regBCell = document.querySelector(`[data-reg-b="0"]`);
        regBCell.style.background = '#00BCD4';
        regBCell.style.border = '2px solid #4DD0E1';
        // Show GMEM and SMEM indices for b_tmp
        const gmemIdxB = `B[${kOffset + threadCol},0]`;
        const smemIdxB = `tile_b[${threadCol},0]`;
        regBCell.innerHTML = `<div style="font-weight: 600;">b_tmp</div><div style="font-size: 8px; color: #4DD0E1;">${gmemIdxB}<br/>${smemIdxB}</div>`;
    }

    async computeCudaCores(kIter = 0) {
        this.currentStep = 4;

        for (let i = 0; i < this.TM; i++) {
            const core = document.querySelector(`[data-core="${i}"]`);
            core.style.background = '#E91E63';
            core.style.border = '2px solid #F06292';
            core.textContent = '⚡FMA';

            await sleep(100);

            const resultCell = document.querySelector(`[data-result="${i}"]`);
            // Keep accumulating (darker green for more iterations)
            const intensity = Math.min(255, 139 + kIter * 20);
            resultCell.style.background = `rgb(${intensity - 100}, ${intensity}, ${intensity - 85})`;
            resultCell.style.border = '2px solid #AED581';
        }
    }

    async writeResults() {
        this.currentStep = 5;
        this.updateStats(`Results written back - ${this.TM} outputs computed with high arithmetic intensity!`);

        // Highlight output region in global C
        for (let i = 0; i < this.TM; i++) {
            const cell = this.globalC.querySelector(`[data-row="${i}"][data-col="0"]`);
            cell.style.background = '#8BC34A';
            cell.style.border = '2px solid #AED581';
        }
    }

    updateStats(message = '') {
        const stats = document.getElementById('pipeline1dStats');
        const stepsInfo = [
            '⏸️  Ready to start',
            '1️⃣  Loading block tiles from global memory (GMEM → SMEM)',
            '2️⃣  Block tiles cached in shared memory',
            '3️⃣  Thread loading data into registers (SMEM → Registers)',
            '4️⃣  Computing FMA operations on CUDA cores',
            '5️⃣  ✅ Complete - Results written back to global memory'
        ];

        stats.innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Current Step</div>
                    <div class="viz-stat-value">${this.currentStep}/5</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Block Tile Size</div>
                    <div class="viz-stat-value">${this.BM}×${this.BN}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Thread Tile Size</div>
                    <div class="viz-stat-value">TM=${this.TM}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Register Reuse</div>
                    <div class="viz-stat-value">${this.TM}×</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 15px; background: #2a2a2a; border-radius: 6px; border-left: 4px solid #4CAF50;">
                <div style="font-size: 14px; font-weight: 600; color: #4CAF50; margin-bottom: 10px;">
                    ${stepsInfo[this.currentStep]}
                </div>
                ${message ? `<div style="font-size: 13px; color: #999;">${message}</div>` : ''}
            </div>
            <div style="margin-top: 15px; padding: 15px; background: #2a2a2a; border-radius: 6px;">
                <div style="font-size: 13px; line-height: 1.8; color: #e0e0e0;">
                    <strong style="color: #FFA726;">Key Insight:</strong><br>
                    • <strong>${(this.BM / this.TM) * this.BN} threads</strong> cooperate to load ${this.BM}×${this.BK} + ${this.BK}×${this.BN} = ${this.BM * this.BK + this.BK * this.BN} values into SMEM<br>
                    • Each thread loads <strong>${this.TM} + 1</strong> values into registers (${this.TM} from tile_a, 1 b_tmp from tile_b)<br>
                    • b_tmp is <strong>reused ${this.TM}×</strong> → ${this.TM} FMA operations per b_tmp load<br>
                    • <strong>Arithmetic Intensity:</strong> ${this.TM} FLOPs / (${this.TM + 1} SMEM reads) = ${(this.TM / (this.TM + 1)).toFixed(2)} FLOPs/read
                </div>
            </div>
        `;
    }

    reset() {
        this.currentStep = 0;
        this.renderPipeline();
    }
}

// ============================================================================
// Kernel 5: 2D Block Tiling Visualization
// ============================================================================
class Tiling2DViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.tileSize = 4; // Shared memory tile size
        this.TM = 4; // Each thread computes TM outputs in M dimension
        this.TN = 4; // Each thread computes TN outputs in N dimension
        this.matrixSize = 8;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">2D Block Tiling: Outer Product Register Blocking</div>
                <div class="viz-info" style="margin-bottom: 15px;">
                    <h4>Building on 1D Tiling:</h4>
                    <p style="margin: 10px 0; line-height: 1.8;">
                        Now each thread computes <strong>TM × TN = ${this.TM}×${this.TN} outputs</strong> instead of just TM!
                    </p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; padding: 15px; background: #2a2a2a; border-radius: 8px;">
                        <div style="border-right: 2px solid #444; padding-right: 15px;">
                            <div style="font-weight: 600; color: #FFA726; margin-bottom: 10px;">🟠 1D Block Tiling</div>
                            <ul style="margin: 0; padding-left: 20px; line-height: 1.8; font-size: 13px;">
                                <li>Cache 1 <code>b_tmp</code> value</li>
                                <li>Load TM values from <code>tile_a</code></li>
                                <li>Compute TM outputs</li>
                                <li><strong>TM reuse per b_tmp</strong></li>
                            </ul>
                        </div>
                        <div style="padding-left: 15px;">
                            <div style="font-weight: 600; color: #4CAF50; margin-bottom: 10px;">✅ 2D Block Tiling</div>
                            <ul style="margin: 0; padding-left: 20px; line-height: 1.8; font-size: 13px;">
                                <li>Cache <code style="color: #9C27B0;">register_m[TM]</code> values</li>
                                <li>Cache <code style="color: #00BCD4;">register_n[TN]</code> values</li>
                                <li><strong style="color: #4CAF50;">Compute TM×TN outputs via outer product!</strong></li>
                                <li><strong>Bidirectional reuse: TM×TN per register pair</strong></li>
                            </ul>
                        </div>
                    </div>
                    <p style="margin-top: 10px; color: #FFA726;"><strong>Key Insight:</strong> Outer product pattern: each <code>register_m[i]</code> × <code>register_n[j]</code> generates one output, creating a ${this.TM}×${this.TN} tile!</p>
                </div>
                <div class="viz-controls">
                    <button class="viz-btn" id="tiling2dAnimate">Animate</button>
                    <button class="viz-btn" id="tiling2dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="tiling2dCanvas"></div>
                <div class="viz-info">
                    <div id="tiling2dStats"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF9800;"></div>
                        <span>tile_a[], tile_b[] (Shared Memory)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #9C27B0;"></div>
                        <span>register_m[TM] (from tile_a)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #00BCD4;"></div>
                        <span>register_n[TN] (from tile_b)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #E91E63;"></div>
                        <span>Computing Outer Product</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>thread_results[TM×TN]</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('tiling2dCanvas');
        this.renderMatrices();

        document.getElementById('tiling2dAnimate').addEventListener('click', () => this.animate());
        document.getElementById('tiling2dReset').addEventListener('click', () => this.reset());
    }

    renderMatrices() {
        this.canvas.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 25px;">
                <!-- Memory Hierarchy Flow -->
                <div style="display: flex; gap: 30px; align-items: flex-start; justify-content: center;">
                    <!-- Shared Memory Section -->
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #FF9800; margin-bottom: 10px; font-weight: 600;">tile_a[] (Shared Memory)</div>
                            <div id="tiling2dSharedA"></div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #FF9800; margin-bottom: 10px; font-weight: 600;">tile_b[] (Shared Memory)</div>
                            <div id="tiling2dSharedB"></div>
                        </div>
                    </div>

                    <!-- Arrow -->
                    <div style="display: flex; align-items: center; justify-content: center; font-size: 40px; color: #FFA726;">
                        →
                    </div>

                    <!-- Registers Section -->
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #9C27B0; margin-bottom: 8px; font-weight: 600;">register_m[TM]</div>
                            <div id="tiling2dRegM" style="display: flex; gap: 3px;"></div>
                            <div style="font-size: 10px; color: #999; margin-top: 5px;">Cached row from tile_a</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #00BCD4; margin-bottom: 8px; font-weight: 600;">register_n[TN]</div>
                            <div id="tiling2dRegN" style="display: flex; gap: 3px;"></div>
                            <div style="font-size: 10px; color: #999; margin-top: 5px;">Cached col from tile_b</div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; padding: 10px; background: #2a2a2a; border-radius: 6px; border: 2px solid #4CAF50;">
                            <div style="font-size: 14px; color: #4CAF50; font-weight: 600;">Outer Product</div>
                            <div style="font-size: 20px; margin: 5px 0;">⊗</div>
                            <div style="font-size: 11px; color: #999;">TM × TN = ${this.TM}×${this.TN}</div>
                        </div>
                    </div>

                    <!-- Arrow -->
                    <div style="display: flex; align-items: center; justify-content: center; font-size: 40px; color: #FFA726;">
                        →
                    </div>

                    <!-- Output Section -->
                    <div style="text-align: center;">
                        <div style="font-size: 12px; color: #4CAF50; margin-bottom: 8px; font-weight: 600;">thread_results[TM×TN]</div>
                        <div id="tiling2dMatrixC"></div>
                        <div style="font-size: 10px; color: #999; margin-top: 5px;">One thread computes<br>${this.TM}×${this.TN} outputs!</div>
                    </div>
                </div>
            </div>
        `;

        // Create shared memory tiles
        this.sharedA = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('tiling2dSharedA'));
        this.sharedB = createMatrixGrid(this.tileSize, this.tileSize, 30, '', document.getElementById('tiling2dSharedB'));

        // Create output tile
        this.matrixC = createMatrixGrid(this.TM, this.TN, 35, '', document.getElementById('tiling2dMatrixC'));

        // Create register_m visualization (horizontal now, like array)
        const regMContainer = document.getElementById('tiling2dRegM');
        regMContainer.innerHTML = '';
        for (let i = 0; i < this.TM; i++) {
            const reg = document.createElement('div');
            reg.id = `reg2d-m-${i}`;
            reg.style.width = '50px';
            reg.style.height = '50px';
            reg.style.background = '#333';
            reg.style.border = '2px solid #555';
            reg.style.borderRadius = '4px';
            reg.style.display = 'flex';
            reg.style.alignItems = 'center';
            reg.style.justifyContent = 'center';
            reg.style.fontSize = '10px';
            reg.style.transition = 'all 0.3s';
            reg.textContent = `m[${i}]`;
            regMContainer.appendChild(reg);
        }

        // Create register_n visualization (horizontal)
        const regNContainer = document.getElementById('tiling2dRegN');
        regNContainer.innerHTML = '';
        for (let j = 0; j < this.TN; j++) {
            const reg = document.createElement('div');
            reg.id = `reg2d-n-${j}`;
            reg.style.width = '50px';
            reg.style.height = '50px';
            reg.style.background = '#333';
            reg.style.border = '2px solid #555';
            reg.style.borderRadius = '4px';
            reg.style.display = 'flex';
            reg.style.alignItems = 'center';
            reg.style.justifyContent = 'center';
            reg.style.fontSize = '10px';
            reg.style.transition = 'all 0.3s';
            reg.textContent = `n[${j}]`;
            regNContainer.appendChild(reg);
        }
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('tiling2dAnimate').disabled = true;
        document.getElementById('tiling2dStats').innerHTML = '';

        let smemReads = 0;
        let computationsCompleted = 0;

        // Step 1: Load tiles into shared memory
        for (let i = 0; i < this.tileSize; i++) {
            for (let j = 0; j < this.tileSize; j++) {
                const cellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                const cellB = this.sharedB.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                cellA.style.background = '#FF9800';
                cellB.style.background = '#FF9800';
            }
        }

        await sleep(500);

        // Simulate one K iteration to show the outer product pattern
        for (let k = 0; k < this.tileSize; k++) {
            // Step 2: Load register_m[TM] from tile_a
            for (let i = 0; i < this.TM; i++) {
                const cellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${k}"]`);
                if (cellA) cellA.classList.add('active');

                const regM = document.getElementById(`reg2d-m-${i}`);
                regM.style.background = '#9C27B0';
                regM.style.borderColor = '#9C27B0';
            }

            smemReads += this.TM;
            await sleep(300);

            // Step 3: Load register_n[TN] from tile_b
            for (let j = 0; j < this.TN; j++) {
                const cellB = this.sharedB.querySelector(`[data-row="${k}"][data-col="${j}"]`);
                if (cellB) cellB.classList.add('active');

                const regN = document.getElementById(`reg2d-n-${j}`);
                regN.style.background = '#00BCD4';
                regN.style.borderColor = '#00BCD4';
            }

            smemReads += this.TN;
            await sleep(300);

            // Step 4: Compute outer product TM × TN
            // Show each element being computed
            for (let i = 0; i < this.TM; i++) {
                for (let j = 0; j < this.TN; j++) {
                    // Highlight registers being used
                    const regM = document.getElementById(`reg2d-m-${i}`);
                    const regN = document.getElementById(`reg2d-n-${j}`);
                    regM.style.background = '#E91E63';
                    regN.style.background = '#E91E63';

                    // Highlight output cell
                    const cellC = this.matrixC.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                    if (cellC) cellC.style.background = '#E91E63';

                    await sleep(50);

                    // Update output
                    if (cellC) {
                        if (k < this.tileSize - 1) {
                            cellC.style.background = '#FF6B6B'; // Partial
                        } else {
                            cellC.style.background = '#4CAF50'; // Complete
                            computationsCompleted++;
                        }
                    }

                    // Reset register colors
                    regM.style.background = '#9C27B0';
                    regN.style.background = '#00BCD4';
                }
            }

            this.updateTiling2DStats(smemReads, computationsCompleted, k + 1);

            await sleep(200);

            // Clear active highlights from shared memory
            this.sharedA.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
            this.sharedB.querySelectorAll('.active').forEach(el => el.classList.remove('active'));

            // Reset registers
            for (let i = 0; i < this.TM; i++) {
                const regM = document.getElementById(`reg2d-m-${i}`);
                regM.style.background = '#333';
                regM.style.borderColor = '#555';
            }
            for (let j = 0; j < this.TN; j++) {
                const regN = document.getElementById(`reg2d-n-${j}`);
                regN.style.background = '#333';
                regN.style.borderColor = '#555';
            }
        }

        document.getElementById('tiling2dAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateTiling2DStats(smemReads, computationsCompleted, kProgress) {
        const totalOutputs = this.TM * this.TN;
        const totalSMEMReads = this.tileSize * (this.TM + this.TN);
        const reusePerLoad = (totalOutputs / (this.TM + this.TN)).toFixed(2);

        document.getElementById('tiling2dStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">K Progress</div>
                    <div class="viz-stat-value">${kProgress}/${this.tileSize}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">SMEM Reads (So Far)</div>
                    <div class="viz-stat-value">${smemReads}/${totalSMEMReads}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Outputs Completed</div>
                    <div class="viz-stat-value">${computationsCompleted}/${totalOutputs}</div>
                </div>
                <div class="viz-stat" style="border-left: 3px solid #4CAF50;">
                    <div class="viz-stat-label">Outer Product Size</div>
                    <div class="viz-stat-value" style="color: #4CAF50;">${this.TM}×${this.TN} = ${totalOutputs}</div>
                </div>
                <div class="viz-stat" style="border-left: 3px solid #9C27B0;">
                    <div class="viz-stat-label">Register Reuse Factor</div>
                    <div class="viz-stat-value" style="color: #9C27B0;">${reusePerLoad}× per register</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 12px; background: #2a2a2a; border-radius: 6px; border-left: 4px solid #FFA726;">
                <div style="font-size: 13px; line-height: 1.6;">
                    <strong style="color: #FFA726;">🎯 Outer Product Pattern:</strong><br>
                    <span style="color: #9C27B0;">register_m[${this.TM}]</span> ⊗ <span style="color: #00BCD4;">register_n[${this.TN}]</span> = <span style="color: #4CAF50;">${this.TM}×${this.TN} outputs</span><br>
                    Each iteration: Load <strong>${this.TM + this.TN}</strong> values, compute <strong>${this.TM * this.TN}</strong> outputs!<br>
                    <span style="color: #4CAF50;">Efficiency:</span> ${reusePerLoad}× more computation per SMEM read than 1D tiling!
                </div>
            </div>
        `;
    }

    reset() {
        this.renderMatrices();
        document.getElementById('tiling2dStats').innerHTML = '';
    }
}

// ============================================================================
// Kernel 6: Vectorized Memory Access Visualization
// ============================================================================
class VectorizedViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Vectorized Memory Access: float4 vs float</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="vectorScalar">Scalar Loads (4× float)</button>
                    <button class="viz-btn" id="vectorVectorized">Vectorized Load (1× float4)</button>
                </div>
                <div class="viz-canvas" id="vectorCanvas">
                    <div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap; margin-top: 20px;">
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #999; margin-bottom: 10px;">Memory (128-bit cache line)</div>
                            <div id="vectorMemory" style="display: flex; gap: 2px; justify-content: center;"></div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 14px; color: #999; margin-bottom: 10px;">GPU Registers</div>
                            <div id="vectorRegisters"></div>
                        </div>
                    </div>
                </div>
                <div class="viz-info">
                    <h4>Vectorized Loading Benefits</h4>
                    <p><strong>Scalar (4× float):</strong> 4 separate load instructions, potentially 4 memory transactions</p>
                    <p><strong>Vectorized (1× float4):</strong> 1 load instruction, 1 memory transaction, guaranteed alignment</p>
                    <div id="vectorStats"></div>
                </div>
            </div>
        `;

        this.renderMemory();

        document.getElementById('vectorScalar').addEventListener('click', () => this.animateScalar());
        document.getElementById('vectorVectorized').addEventListener('click', () => this.animateVectorized());
    }

    renderMemory() {
        const memContainer = document.getElementById('vectorMemory');
        memContainer.innerHTML = '';

        for (let i = 0; i < 16; i++) {
            const byte = document.createElement('div');
            byte.id = `mem-byte-${i}`;
            byte.style.width = '30px';
            byte.style.height = '40px';
            byte.style.background = '#333';
            byte.style.border = '1px solid #555';
            byte.style.display = 'flex';
            byte.style.alignItems = 'center';
            byte.style.justifyContent = 'center';
            byte.style.fontSize = '10px';
            byte.style.transition = 'all 0.3s';

            if (i % 4 === 0) {
                byte.style.borderLeft = '3px solid #4CAF50';
            }

            memContainer.appendChild(byte);
        }

        const regContainer = document.getElementById('vectorRegisters');
        regContainer.innerHTML = '';

        for (let i = 0; i < 4; i++) {
            const reg = document.createElement('div');
            reg.id = `reg-float-${i}`;
            reg.style.width = '100px';
            reg.style.height = '50px';
            reg.style.background = '#333';
            reg.style.border = '2px solid #555';
            reg.style.borderRadius = '6px';
            reg.style.margin = '5px';
            reg.style.display = 'flex';
            reg.style.alignItems = 'center';
            reg.style.justifyContent = 'center';
            reg.style.fontSize = '12px';
            reg.style.transition = 'all 0.3s';
            reg.textContent = `float[${i}]`;
            regContainer.appendChild(reg);
        }
    }

    async animateScalar() {
        this.renderMemory();

        let instructions = 0;

        for (let i = 0; i < 4; i++) {
            // Highlight 4 bytes being loaded
            for (let b = 0; b < 4; b++) {
                const byte = document.getElementById(`mem-byte-${i * 4 + b}`);
                byte.style.background = '#FFC107';
            }

            await sleep(300);

            // Show in register
            const reg = document.getElementById(`reg-float-${i}`);
            reg.style.background = '#FFC107';
            reg.style.borderColor = '#FFC107';

            instructions++;
            this.updateVectorStats('Scalar', instructions, i + 1);

            await sleep(300);

            // Clear memory highlight
            for (let b = 0; b < 4; b++) {
                const byte = document.getElementById(`mem-byte-${i * 4 + b}`);
                byte.style.background = '#333';
            }
        }
    }

    async animateVectorized() {
        this.renderMemory();

        // Highlight all 16 bytes at once
        for (let i = 0; i < 16; i++) {
            const byte = document.getElementById(`mem-byte-${i}`);
            byte.style.background = '#4CAF50';
        }

        await sleep(300);

        // Load all 4 floats at once
        for (let i = 0; i < 4; i++) {
            const reg = document.getElementById(`reg-float-${i}`);
            reg.style.background = '#4CAF50';
            reg.style.borderColor = '#4CAF50';
            reg.textContent = `float4.${['x', 'y', 'z', 'w'][i]}`;
        }

        this.updateVectorStats('Vectorized (float4)', 1, 4);

        await sleep(500);

        // Clear highlights
        for (let i = 0; i < 16; i++) {
            const byte = document.getElementById(`mem-byte-${i}`);
            byte.style.background = '#333';
        }
    }

    updateVectorStats(mode, instructions, floatsLoaded) {
        const instructionEfficiency = mode === 'Scalar' ? '1×' : '4×';
        const bandwidth = mode === 'Scalar' ? 'Up to 4 transactions' : '1 transaction (guaranteed)';

        document.getElementById('vectorStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Mode</div>
                    <div class="viz-stat-value">${mode}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Instructions</div>
                    <div class="viz-stat-value">${instructions}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Floats Loaded</div>
                    <div class="viz-stat-value">${floatsLoaded}/4</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Memory Transactions</div>
                    <div class="viz-stat-value">${bandwidth}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Instruction Efficiency</div>
                    <div class="viz-stat-value">${instructionEfficiency}</div>
                </div>
            </div>
        `;
    }
}

// ============================================================================
// Performance Comparison Chart
// ============================================================================
class PerformanceComparison {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.init();
    }

    init() {
        const kernels = [
            { name: 'Naive', gflops: 309, percent: 1.3, color: '#E53935' },
            { name: 'Coalesced', gflops: 1987, percent: 8.5, color: '#FB8C00' },
            { name: 'Shared Mem', gflops: 2980, percent: 12.8, color: '#FFB300' },
            { name: '1D Tiling', gflops: 8475, percent: 36.5, color: '#7CB342' },
            { name: '2D Tiling', gflops: 15972, percent: 68.7, color: '#00897B' },
            { name: 'Vectorized', gflops: 18237, percent: 78.4, color: '#1E88E5' },
            { name: 'Autotuned', gflops: 19721, percent: 84.8, color: '#5E35B1' },
            { name: 'cuBLAS', gflops: 23250, percent: 100, color: '#4CAF50' }
        ];

        const maxGflops = 23250;

        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Performance Progression: 309 → 19,721 GFLOPs</div>
                <div class="viz-canvas" style="padding: 30px;">
                    <div id="perfChart"></div>
                </div>
                <div class="viz-info">
                    <h4>Key Insights</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li><strong>64× speedup</strong> from naive to final implementation</li>
                        <li><strong>Memory coalescing</strong> gave biggest single improvement (6.4×)</li>
                        <li><strong>2D tiling</strong> achieved major performance breakthrough</li>
                        <li><strong>15% gap</strong> to cuBLAS due to Tensor Cores and advanced techniques</li>
                    </ul>
                    <div id="perfStats"></div>
                </div>
            </div>
        `;

        const chartContainer = document.getElementById('perfChart');

        kernels.forEach((kernel, idx) => {
            const bar = document.createElement('div');
            bar.style.display = 'flex';
            bar.style.alignItems = 'center';
            bar.style.marginBottom = '15px';
            bar.style.gap = '15px';

            const label = document.createElement('div');
            label.textContent = kernel.name;
            label.style.width = '120px';
            label.style.fontSize = '13px';
            label.style.fontWeight = '600';
            label.style.color = '#e0e0e0';

            const barContainer = document.createElement('div');
            barContainer.style.flex = '1';
            barContainer.style.background = '#333';
            barContainer.style.borderRadius = '8px';
            barContainer.style.overflow = 'hidden';
            barContainer.style.position = 'relative';
            barContainer.style.height = '40px';

            const barFill = document.createElement('div');
            barFill.style.width = '0%';
            barFill.style.height = '100%';
            barFill.style.background = kernel.color;
            barFill.style.transition = 'width 1s ease-out';
            barFill.style.display = 'flex';
            barFill.style.alignItems = 'center';
            barFill.style.justifyContent = 'flex-end';
            barFill.style.paddingRight = '10px';
            barFill.style.fontSize = '12px';
            barFill.style.fontWeight = '600';
            barFill.style.color = 'white';

            const value = document.createElement('div');
            value.textContent = `${kernel.gflops.toLocaleString()} GFLOPs (${kernel.percent}%)`;
            value.style.width = '200px';
            value.style.fontSize = '12px';
            value.style.fontWeight = '600';
            value.style.color = kernel.color;
            value.style.textAlign = 'right';

            barContainer.appendChild(barFill);
            bar.appendChild(label);
            bar.appendChild(barContainer);
            bar.appendChild(value);
            chartContainer.appendChild(bar);

            // Animate bars
            setTimeout(() => {
                barFill.style.width = `${kernel.percent}%`;
            }, idx * 100);
        });

        // Stats
        document.getElementById('perfStats').innerHTML = `
            <div class="viz-stats" style="margin-top: 20px;">
                <div class="viz-stat">
                    <div class="viz-stat-label">Total Speedup</div>
                    <div class="viz-stat-value">64×</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Best Single Optimization</div>
                    <div class="viz-stat-value">Memory Coalescing (6.4×)</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Final vs cuBLAS</div>
                    <div class="viz-stat-value">84.8%</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Remaining Gap</div>
                    <div class="viz-stat-value">Tensor Cores + Advanced Techniques</div>
                </div>
            </div>
        `;
    }
}

// ============================================================================
// Index Transformation Visualization (Naive vs Coalesced)
// ============================================================================
class IndexTransformViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;
        this.blockSize = 4;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Index Transformation: Naive vs Coalesced</div>
                <div class="viz-info">
                    <p>Compare how thread indices map to matrix positions in naive vs coalesced kernels.</p>
                    <p><strong>Block Size:</strong> ${this.blockSize}×${this.blockSize} | <strong>Threads per block:</strong> ${this.blockSize * this.blockSize}</p>
                </div>
                <br>
                <div class="viz-controls">
                    <button class="viz-btn" id="transformAnimate">Animate Thread Mapping</button>
                    <button class="viz-btn" id="transformReset">Reset</button>
                </div>
                <div class="viz-canvas" id="transformCanvas" style="overflow-x: auto;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; min-width: 700px;">
                        <!-- Naive Kernel -->
                        <div style="border: 2px solid #e74c3c; padding: 15px; border-radius: 8px;">
                            <h3 style="text-align: center; color: #e74c3c; margin-bottom: 12px; font-size: 16px;">🐌 Naive Kernel</h3>
                            <div style="background: #2a2a2a; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 11px;">
                                <div>x = blockIdx.x * ${this.blockSize} + threadIdx.x</div>
                                <div>y = blockIdx.y * ${this.blockSize} + threadIdx.y</div>
                            </div>
                            <div id="naiveGrid" style="display: grid; grid-template-columns: repeat(${this.blockSize}, 65px); column-gap: 8px; row-gap: 16px; justify-content: center;"></div>
                            <div id="naiveMemAccess" style="margin-top: 12px; padding: 8px; background: #2a2a2a; border-radius: 6px; min-height: 70px; font-size: 12px;"></div>
                        </div>
                        <!-- Coalesced Kernel -->
                        <div style="border: 2px solid #27ae60; padding: 15px; border-radius: 8px;">
                            <h3 style="text-align: center; color: #27ae60; margin-bottom: 12px; font-size: 16px;">🚀 Coalesced Kernel</h3>
                            <div style="background: #2a2a2a; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 11px;">
                                <div>x = blockIdx.x * ${this.blockSize} + (threadIdx.x / ${this.blockSize})</div>
                                <div>y = blockIdx.y * ${this.blockSize} + (threadIdx.x % ${this.blockSize})</div>
                            </div>
                            <div id="coalescedGrid" style="display: grid; grid-template-columns: repeat(${this.blockSize}, 65px); column-gap: 8px; row-gap: 16px; justify-content: center;"></div>
                            <div id="coalescedMemAccess" style="margin-top: 12px; padding: 8px; background: #2a2a2a; border-radius: 6px; min-height: 70px; font-size: 12px;"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Wait for DOM to be ready
        setTimeout(() => {
            this.renderGrids();

            const animateBtn = document.getElementById('transformAnimate');
            const resetBtn = document.getElementById('transformReset');

            if (animateBtn) {
                animateBtn.addEventListener('click', () => this.animate());
            }
            if (resetBtn) {
                resetBtn.addEventListener('click', () => this.reset());
            }
        }, 0);
    }

    renderGrids() {
        const naiveGrid = document.getElementById('naiveGrid');
        const coalescedGrid = document.getElementById('coalescedGrid');

        if (!naiveGrid || !coalescedGrid) {
            console.error('Grid containers not found');
            return;
        }

        naiveGrid.innerHTML = '';
        coalescedGrid.innerHTML = '';

        // Create cells for both grids
        for (let i = 0; i < this.blockSize * this.blockSize; i++) {
            const naiveCell = this.createCell(i, 'naive');
            const coalescedCell = this.createCell(i, 'coalesced');
            naiveGrid.appendChild(naiveCell);
            coalescedGrid.appendChild(coalescedCell);
        }
    }

    createCell(threadIdx, type) {
        const cell = document.createElement('div');
        cell.className = 'matrix-cell';
        cell.style.width = '65px';
        cell.style.height = '75px';
        cell.style.background = '#333';
        cell.style.display = 'flex';
        cell.style.flexDirection = 'column';
        cell.style.alignItems = 'center';
        cell.style.justifyContent = 'space-around';
        cell.style.padding = '5px 3px';
        cell.dataset.threadIdx = threadIdx;
        cell.dataset.type = type;

        let x, y, row;
        if (type === 'naive') {
            x = threadIdx % this.blockSize;
            y = Math.floor(threadIdx / this.blockSize);
            row = x;
        } else {
            x = Math.floor(threadIdx / this.blockSize);
            y = threadIdx % this.blockSize;
            row = x;
        }

        cell.innerHTML = `
            <div style="font-weight: bold; color: #4CAF50; font-size: 9px;">T${threadIdx}</div>
            <div style="display: flex; gap: 3px; justify-content: center;">
                <div style="background: #1a1a1a; padding: 2px 4px; border-radius: 3px; border: 1px solid #555;">
                    <div style="font-size: 7px; color: #999;">x</div>
                    <div style="font-size: 13px; font-weight: bold; color: #FFA726; line-height: 1;">${x}</div>
                </div>
                <div style="background: #1a1a1a; padding: 2px 4px; border-radius: 3px; border: 1px solid #555;">
                    <div style="font-size: 7px; color: #999;">y</div>
                    <div style="font-size: 13px; font-weight: bold; color: #42A5F5; line-height: 1;">${y}</div>
                </div>
            </div>
            <div style="font-size: 8px; color: #888;">A[${row}][k]</div>
        `;

        return cell;
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        document.getElementById('transformAnimate').disabled = true;

        this.renderGrids();
        document.getElementById('naiveMemAccess').innerHTML = '';
        document.getElementById('coalescedMemAccess').innerHTML = '';

        const totalThreads = this.blockSize * this.blockSize;

        for (let i = 0; i < totalThreads; i++) {
            // Highlight current thread in both kernels
            const naiveCell = document.querySelector(`[data-thread-idx="${i}"][data-type="naive"]`);
            const coalescedCell = document.querySelector(`[data-thread-idx="${i}"][data-type="coalesced"]`);

            naiveCell.classList.add('active');
            coalescedCell.classList.add('active');

            // Calculate memory access patterns
            const naiveX = i % this.blockSize;
            const naiveRow = naiveX;

            const coalescedX = Math.floor(i / this.blockSize);
            const coalescedY = i % this.blockSize;

            // Update memory access info
            const prevNaiveX = (i - 1) % this.blockSize;
            const sameRowNaive = i > 0 && naiveRow === prevNaiveX ? '✅' : '❌';
            const prevCoalescedX = i > 0 ? Math.floor((i - 1) / this.blockSize) : -1;
            const sameRowCoalesced = prevCoalescedX === coalescedX ? '✅' : '❌';

            document.getElementById('naiveMemAccess').innerHTML = `
                <div style="font-weight: bold; margin-bottom: 8px; color: #4CAF50;">Thread ${i} Memory Access</div>
                <div style="margin-bottom: 4px;">Matrix Position: <strong>A[${naiveRow}][k]</strong></div>
                ${i > 0 ? `
                    <div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px;">
                        ${naiveRow !== prevNaiveX ?
                        `❌ <strong style="color: #e74c3c;">Different row</strong> from Thread ${i - 1}<br>
                             <span style="color: #999;">Stride: K elements (scattered)</span>` :
                        `✅ <strong style="color: #27ae60;">Same row</strong> as Thread ${i - 1}`}
                    </div>
                ` : '<div style="color: #999; margin-top: 8px;">First thread in warp</div>'}
            `;

            document.getElementById('coalescedMemAccess').innerHTML = `
                <div style="font-weight: bold; margin-bottom: 8px; color: #4CAF50;">Thread ${i} Memory Access</div>
                <div style="margin-bottom: 4px;">Matrix Position: <strong>A[${coalescedX}][${coalescedY}]</strong></div>
                ${i > 0 ? `
                    <div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px;">
                        ${coalescedX === prevCoalescedX ?
                        `✅ <strong style="color: #27ae60;">Same row</strong> as Thread ${i - 1}<br>
                             <span style="color: #4CAF50;">Stride: 1 element (coalesced! 🚀)</span>` :
                        `⚠️ <strong style="color: #FFA726;">New row</strong> (warp ${Math.floor(i / this.blockSize)})`}
                    </div>
                ` : '<div style="color: #999; margin-top: 8px;">First thread in warp</div>'}
            `;

            await sleep(700);

            naiveCell.classList.remove('active');
            coalescedCell.classList.remove('active');
        }

        // Show completion message
        document.getElementById('naiveMemAccess').innerHTML = `
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 14px; color: #e74c3c; font-weight: bold;">❌ Non-coalesced Access</div>
                <div style="margin-top: 8px; color: #999;">→ ${this.blockSize} memory transactions for first ${this.blockSize} threads</div>
            </div>
        `;

        document.getElementById('coalescedMemAccess').innerHTML = `
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 14px; color: #27ae60; font-weight: bold;">✅ Coalesced Access</div>
                <div style="margin-top: 8px; color: #999;">→ 1 memory transaction for first ${this.blockSize} threads</div>
            </div>
        `;

        document.getElementById('transformAnimate').disabled = false;
        this.isAnimating = false;
    }

    reset() {
        this.renderGrids();
        document.getElementById('naiveMemAccess').innerHTML = '';
        document.getElementById('coalescedMemAccess').innerHTML = '';
    }
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.NaiveKernelViz = NaiveKernelViz;
    window.CoalescingViz = CoalescingViz;
    window.CoalescedMatrixViz = CoalescedMatrixViz;
    window.MemoryHierarchyViz = MemoryHierarchyViz;
    window.IndexTransformViz = IndexTransformViz;
    window.SharedMemoryViz = SharedMemoryViz;
    window.Tiling1DViz = Tiling1DViz;
    window.Tiling1DPipelineViz = Tiling1DPipelineViz;
    window.Tiling2DViz = Tiling2DViz;
    window.VectorizedViz = VectorizedViz;
    window.PerformanceComparison = PerformanceComparison;
}
