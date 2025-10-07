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
                <div class="viz-canvas" id="sharedCanvas"></div>
                <div class="viz-info">
                    <h4>Shared Memory Benefits</h4>
                    <p>Instead of each thread loading from global memory independently:</p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Threads cooperatively load tiles into fast shared memory</li>
                        <li>All threads in block reuse the cached data</li>
                        <li>Reduces global memory traffic by tile_size factor</li>
                    </ul>
                    <div id="sharedStats"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #9C27B0;"></div>
                        <span>Current Tile</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF9800;"></div>
                        <span>Shared Memory</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #2196F3;"></div>
                        <span>Computed Output</span>
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
        const numTiles = Math.ceil(this.matrixSize / this.tileSize);

        for (let tileRow = 0; tileRow < numTiles; tileRow++) {
            for (let tileCol = 0; tileCol < numTiles; tileCol++) {
                // Highlight tiles being loaded
                for (let i = 0; i < this.tileSize; i++) {
                    for (let j = 0; j < this.tileSize; j++) {
                        const row = tileRow * this.tileSize + i;
                        const col = tileCol * this.tileSize + j;

                        if (row < this.matrixSize && col < this.matrixSize) {
                            const cellA = this.matrixA.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                            const cellB = this.matrixB.querySelector(`[data-row="${row}"][data-col="${col}"]`);

                            if (cellA) cellA.style.background = '#9C27B0';
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

                // Compute output tile
                for (let i = 0; i < this.tileSize; i++) {
                    for (let j = 0; j < this.tileSize; j++) {
                        const row = tileRow * this.tileSize + i;
                        const col = tileCol * this.tileSize + j;

                        if (row < this.matrixSize && col < this.matrixSize) {
                            const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                            if (cellC) cellC.style.background = '#2196F3';
                        }
                    }
                }

                tilesProcessed++;
                this.updateSharedStats(tilesProcessed, globalLoads);

                await sleep(300);

                // Clear shared memory visualization
                this.sharedA.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                this.sharedB.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                this.matrixA.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                this.matrixB.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
            }
        }

        document.getElementById('sharedAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateSharedStats(tilesProcessed, globalLoads) {
        const totalTiles = Math.pow(Math.ceil(this.matrixSize / this.tileSize), 2);
        const naiveLoads = this.matrixSize * this.matrixSize * 2 * this.matrixSize;
        const reduction = ((naiveLoads - globalLoads) / naiveLoads * 100).toFixed(1);

        document.getElementById('sharedStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Tiles Processed</div>
                    <div class="viz-stat-value">${tilesProcessed}/${totalTiles}</div>
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
        this.threadTileSize = 4; // TM = 4
        this.matrixSize = 8;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">1D Block Tiling: Multiple Outputs Per Thread</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="tiling1dAnimate">Animate</button>
                    <button class="viz-btn" id="tiling1dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="tiling1dCanvas"></div>
                <div class="viz-info">
                    <h4>Thread-Level Tiling</h4>
                    <p>Each thread now computes multiple output elements (TM = ${this.threadTileSize}):</p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Load TM elements from matrix A into registers</li>
                        <li>Load 1 element from matrix B</li>
                        <li>Compute TM outputs using the same B element</li>
                        <li>Higher arithmetic intensity (more compute per byte loaded)</li>
                    </ul>
                    <div id="tiling1dStats"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #E91E63;"></div>
                        <span>Thread Working</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #00BCD4;"></div>
                        <span>Register Cache</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #8BC34A;"></div>
                        <span>Computed Outputs</span>
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
            <div style="display: flex; justify-content: space-around; align-items: flex-start; flex-wrap: wrap;">
                <div id="tiling1dMatrixA"></div>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 12px; color: #999; margin-bottom: 10px;">Thread Registers</div>
                    <div id="tiling1dRegisters"></div>
                </div>
                <div id="tiling1dMatrixB"></div>
                <div id="tiling1dMatrixC"></div>
            </div>
        `;

        this.matrixA = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix A', document.getElementById('tiling1dMatrixA'));
        this.matrixB = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix B', document.getElementById('tiling1dMatrixB'));
        this.matrixC = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix C', document.getElementById('tiling1dMatrixC'));

        // Create register visualization
        const regContainer = document.getElementById('tiling1dRegisters');
        regContainer.innerHTML = '';
        for (let i = 0; i < this.threadTileSize; i++) {
            const reg = document.createElement('div');
            reg.id = `reg-${i}`;
            reg.style.width = '80px';
            reg.style.height = '40px';
            reg.style.background = '#333';
            reg.style.border = '2px solid #555';
            reg.style.borderRadius = '6px';
            reg.style.margin = '5px';
            reg.style.display = 'flex';
            reg.style.alignItems = 'center';
            reg.style.justifyContent = 'center';
            reg.style.fontSize = '11px';
            reg.style.transition = 'all 0.3s';
            reg.textContent = `regM[${i}]`;
            regContainer.appendChild(reg);
        }
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('tiling1dAnimate').disabled = true;
        document.getElementById('tiling1dStats').innerHTML = '';

        const numThreads = Math.ceil(this.matrixSize / this.threadTileSize);
        let threadsCompleted = 0;
        let totalOutputs = 0;

        for (let threadIdx = 0; threadIdx < numThreads; threadIdx++) {
            for (let col = 0; col < this.matrixSize; col++) {
                // Highlight TM rows being loaded into registers
                for (let i = 0; i < this.threadTileSize; i++) {
                    const row = threadIdx * this.threadTileSize + i;
                    if (row < this.matrixSize) {
                        const cellA = this.matrixA.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                        if (cellA) cellA.style.background = '#E91E63';

                        const reg = document.getElementById(`reg-${i}`);
                        reg.style.background = '#00BCD4';
                        reg.style.borderColor = '#00BCD4';
                    }
                }

                // Highlight column element from B
                const cellB = this.matrixB.querySelector(`[data-row="${col}"][data-col="${col}"]`);
                if (cellB) cellB.style.background = '#E91E63';

                await sleep(200);

                // Compute TM outputs
                for (let i = 0; i < this.threadTileSize; i++) {
                    const row = threadIdx * this.threadTileSize + i;
                    if (row < this.matrixSize) {
                        const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                        if (cellC) {
                            cellC.style.background = '#8BC34A';
                            totalOutputs++;
                        }
                    }
                }

                await sleep(200);

                // Clear highlights
                this.matrixA.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                this.matrixB.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                for (let i = 0; i < this.threadTileSize; i++) {
                    const reg = document.getElementById(`reg-${i}`);
                    reg.style.background = '#333';
                    reg.style.borderColor = '#555';
                }
            }

            threadsCompleted++;
            this.updateTiling1DStats(threadsCompleted, totalOutputs);
        }

        document.getElementById('tiling1dAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateTiling1DStats(threadsCompleted, totalOutputs) {
        const totalThreads = Math.ceil(this.matrixSize / this.threadTileSize) * this.matrixSize;
        const outputsPerThread = this.threadTileSize;
        const arithmeticIntensity = (2 * outputsPerThread).toFixed(1);

        document.getElementById('tiling1dStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Threads Used</div>
                    <div class="viz-stat-value">${threadsCompleted}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Outputs Per Thread</div>
                    <div class="viz-stat-value">${outputsPerThread}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Total Outputs Computed</div>
                    <div class="viz-stat-value">${totalOutputs}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Arithmetic Intensity</div>
                    <div class="viz-stat-value">${arithmeticIntensity}× Better</div>
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
        this.TM = 4;
        this.TN = 4;
        this.matrixSize = 8;
        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">2D Block Tiling: TM × TN Outputs Per Thread</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="tiling2dAnimate">Animate</button>
                    <button class="viz-btn" id="tiling2dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="tiling2dCanvas"></div>
                <div class="viz-info">
                    <h4>2D Thread-Level Tiling</h4>
                    <p>Each thread computes a ${this.TM}×${this.TN} grid of outputs:</p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Load TM elements from A into regM[]</li>
                        <li>Load TN elements from B into regN[]</li>
                        <li>Compute outer product: TM × TN results</li>
                        <li>Maximum register reuse and arithmetic intensity</li>
                    </ul>
                    <div id="tiling2dStats"></div>
                </div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #673AB7;"></div>
                        <span>Loaded in Registers</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF6D00;"></div>
                        <span>Computing</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #00E676;"></div>
                        <span>TM×TN Output Tile</span>
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
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                <div id="tiling2dMatrixA"></div>
                <div style="text-align: center;">
                    <div style="font-size: 14px; color: #999; margin-bottom: 10px;">Outer Product</div>
                    <div style="font-size: 24px; color: #4CAF50; margin: 20px 0;">regM ⊗ regN</div>
                    <div style="font-size: 12px; color: #999;">= ${this.TM}×${this.TN} outputs</div>
                </div>
                <div id="tiling2dMatrixB"></div>
                <div id="tiling2dMatrixC"></div>
            </div>
        `;

        this.matrixA = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix A', document.getElementById('tiling2dMatrixA'));
        this.matrixB = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix B', document.getElementById('tiling2dMatrixB'));
        this.matrixC = createMatrixGrid(this.matrixSize, this.matrixSize, 35, 'Matrix C', document.getElementById('tiling2dMatrixC'));
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('tiling2dAnimate').disabled = true;
        document.getElementById('tiling2dStats').innerHTML = '';

        const numThreadsRow = Math.ceil(this.matrixSize / this.TM);
        const numThreadsCol = Math.ceil(this.matrixSize / this.TN);
        let threadsCompleted = 0;

        for (let threadRow = 0; threadRow < numThreadsRow; threadRow++) {
            for (let threadCol = 0; threadCol < numThreadsCol; threadCol++) {
                // Highlight TM rows from A
                for (let i = 0; i < this.TM; i++) {
                    const row = threadRow * this.TM + i;
                    for (let k = 0; k < this.matrixSize; k++) {
                        if (row < this.matrixSize) {
                            const cellA = this.matrixA.querySelector(`[data-row="${row}"][data-col="${k}"]`);
                            if (cellA) cellA.style.background = '#673AB7';
                        }
                    }
                }

                // Highlight TN columns from B
                for (let j = 0; j < this.TN; j++) {
                    const col = threadCol * this.TN + j;
                    for (let k = 0; k < this.matrixSize; k++) {
                        if (col < this.matrixSize) {
                            const cellB = this.matrixB.querySelector(`[data-row="${k}"][data-col="${col}"]`);
                            if (cellB) cellB.style.background = '#673AB7';
                        }
                    }
                }

                await sleep(300);

                // Show computation (highlight output tile)
                for (let i = 0; i < this.TM; i++) {
                    for (let j = 0; j < this.TN; j++) {
                        const row = threadRow * this.TM + i;
                        const col = threadCol * this.TN + j;

                        if (row < this.matrixSize && col < this.matrixSize) {
                            const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                            if (cellC) cellC.style.background = '#FF6D00';
                        }
                    }
                }

                await sleep(300);

                // Mark outputs as complete
                for (let i = 0; i < this.TM; i++) {
                    for (let j = 0; j < this.TN; j++) {
                        const row = threadRow * this.TM + i;
                        const col = threadCol * this.TN + j;

                        if (row < this.matrixSize && col < this.matrixSize) {
                            const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                            if (cellC) cellC.style.background = '#00E676';
                        }
                    }
                }

                threadsCompleted++;
                this.updateTiling2DStats(threadsCompleted);

                await sleep(200);

                // Clear input highlights
                this.matrixA.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
                this.matrixB.querySelectorAll('.matrix-cell').forEach(el => el.style.background = '#333');
            }
        }

        document.getElementById('tiling2dAnimate').disabled = false;
        this.isAnimating = false;
    }

    updateTiling2DStats(threadsCompleted) {
        const totalThreads = Math.ceil(this.matrixSize / this.TM) * Math.ceil(this.matrixSize / this.TN);
        const outputsPerThread = this.TM * this.TN;
        const totalOutputs = Math.min(threadsCompleted * outputsPerThread, this.matrixSize * this.matrixSize);
        const arithmeticIntensity = (outputsPerThread / (this.TM + this.TN)).toFixed(2);

        document.getElementById('tiling2dStats').innerHTML = `
            <div class="viz-stats">
                <div class="viz-stat">
                    <div class="viz-stat-label">Threads Completed</div>
                    <div class="viz-stat-value">${threadsCompleted}/${totalThreads}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Outputs Per Thread</div>
                    <div class="viz-stat-value">${this.TM}×${this.TN} = ${outputsPerThread}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Total Outputs</div>
                    <div class="viz-stat-value">${totalOutputs}</div>
                </div>
                <div class="viz-stat">
                    <div class="viz-stat-label">Register Efficiency</div>
                    <div class="viz-stat-value">${arithmeticIntensity} FLOP/load</div>
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
    window.IndexTransformViz = IndexTransformViz;
    window.SharedMemoryViz = SharedMemoryViz;
    window.Tiling1DViz = Tiling1DViz;
    window.Tiling2DViz = Tiling2DViz;
    window.VectorizedViz = VectorizedViz;
    window.PerformanceComparison = PerformanceComparison;
}
