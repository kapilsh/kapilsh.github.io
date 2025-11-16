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
                <div class="viz-title">Naive Kernel</div>
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

    reset() {
        this.renderMatrices();
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
                <div class="viz-title">Global Memory Coalesced Kernel</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="coalescedMatrixAnimate">Animate</button>
                    <button class="viz-btn" id="coalescedMatrixReset">Reset</button>
                </div>
                <div class="viz-canvas" id="coalescedMatrixCanvas"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>Current Warp</span>
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

        // Iterate through columns of B (outer loop)
        for (let colB = 0; colB < this.matrixSize; colB++) {
            // Iterate through rows of A (inner loop)
            for (let rowA = 0; rowA < this.matrixSize; rowA++) {
                // Highlight one row from A
                for (let k = 0; k < this.matrixSize; k++) {
                    const cellA = this.matrixA.querySelector(`[data-row="${rowA}"][data-col="${k}"]`);
                    if (cellA) cellA.classList.add('highlight');
                }

                // Highlight one column from B
                for (let k = 0; k < this.matrixSize; k++) {
                    const cellB = this.matrixB.querySelector(`[data-row="${k}"][data-col="${colB}"]`);
                    if (cellB) cellB.classList.add('highlight');
                }

                // Mark output element (rowA, colB) as active
                const outputCell = this.matrixC.querySelector(`[data-row="${rowA}"][data-col="${colB}"]`);
                if (outputCell) outputCell.classList.add('active');

                await sleep(300);

                // Clear highlights
                this.matrixA.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));
                this.matrixB.querySelectorAll('.highlight').forEach(el => el.classList.remove('highlight'));

                // Mark as computed
                if (outputCell) {
                    outputCell.classList.remove('active');
                    outputCell.style.background = '#2196F3';
                }
            }
        }

        document.getElementById('coalescedMatrixAnimate').disabled = false;
        this.isAnimating = false;
    }

    reset() {
        this.renderMatrices();
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

    reset() {
        this.renderMatrices();
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
                <div class="viz-controls">
                    <button class="viz-btn" id="tiling1dAnimate">Animate</button>
                    <button class="viz-btn" id="tiling1dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="tiling1dCanvas"></div>
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
                        <div class="legend-box" style="background: #FF6B6B;"></div>
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
                        <div style="font-size: 16px; font-weight: 700; color: #FF6B6B; margin-bottom: 5px;">Previous: Shared Memory Only</div>
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
                        <div style="font-size: 16px; font-weight: 700; color: #4CAF50; margin-bottom: 5px;">Now: 1D Block Tiling</div>
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

        // Create register visualization - larger and with index display
        const regContainer = document.getElementById('tiling1dRegisters');
        regContainer.innerHTML = `
            <div id="regN" style="width: 120px; height: 80px; background: #333; border: 2px solid #555;
                 border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center;
                 font-size: 12px; transition: all 0.3s; padding: 10px; text-align: center;">
                <div style="font-weight: bold;">b_tmp</div>
                <div id="regIndex" style="font-size: 13px; color: #999; margin-top: 5px;">—</div>
            </div>
        `;
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderMatrices();

        document.getElementById('tiling1dAnimate').disabled = true;

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
                        const prevCellA = this.prevSharedA.querySelector(`[data-row="${k}"][data-col="${row}"]`);
                        if (prevCellA) prevCellA.classList.add('active');
                        prevSMEMReads++;

                        // Read from tile_b
                        const prevCellB = this.prevSharedB.querySelector(`[data-row="${k}"][data-col="${col}"]`);
                        if (prevCellB) prevCellB.classList.add('active');
                        prevSMEMReads++;

                        await sleep(150);

                        // Compute 1 output
                        const prevCellC = this.prevMatrixC.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                        if (prevCellC) prevCellC.style.background = '#FF6B6B'; // Computing
                        await sleep(80);

                        if (prevCellC) {
                            if (k < this.tileSize - 1) {
                                prevCellC.style.background = '#FF6B6B'; // Partial
                            } else {
                                prevCellC.style.background = '#8BC34A'; // Complete (muted green)
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
                    const regIndex = document.getElementById('regIndex');
                    regN.style.background = '#00BCD4';
                    regN.style.borderColor = '#00BCD4';
                    regIndex.textContent = `B[${k}][${col}]`;
                    regIndex.style.color = '#00BCD4';

                    currentSMEMReads += 1;

                    await sleep(200);

                    // Compute TM outputs using the SAME b_tmp (register reuse!)
                    for (let tm = 0; tm < this.TM; tm++) {
                        const row = threadRow * this.TM + tm;
                        const cellC = this.matrixC.querySelector(`[data-row="${row}"][data-col="${k}"]`);

                        // Show b_tmp being reused
                        regN.style.background = '#FF6B6B';
                        regIndex.textContent = `×${tm + 1}`;
                        regIndex.style.color = '#FF6B6B';

                        if (cellC) cellC.style.background = '#FF6B6B';
                        await sleep(100);

                        if (col == this.tileSize - 1) {
                            cellC.style.background = '#8BC34A'; // Complete result (final k)
                        }
                    }

                    // Reset register
                    regN.style.background = '#333';
                    regN.style.borderColor = '#555';
                    regIndex.textContent = '—';
                    regIndex.style.color = '#999';

                    // Clear active highlights from shared memory
                    this.sharedA.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                    this.sharedB.querySelectorAll('.active').forEach(el => el.classList.remove('active'));
                }
            }
        }

        document.getElementById('tiling1dAnimate').disabled = false;
        this.isAnimating = false;
    }

    reset() {
        this.renderMatrices();
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
                            <div style="font-weight: 600; color: #4CAF50; margin-bottom: 5px;">1. Global Memory</div>
                            <div style="color: #999;">Full ${this.matrixSizeM}×${this.matrixSizeK} & ${this.matrixSizeK}×${this.matrixSizeN} matrices</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #FF9800; margin-bottom: 5px;">2. Shared Memory</div>
                            <div style="color: #999;">Block tiles: ${this.BM}×${this.BK} & ${this.BK}×${this.BN}</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #00BCD4; margin-bottom: 5px;">3. Registers</div>
                            <div style="color: #999;">Thread tiles: TM=${this.TM} per thread</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: #1a1a1a; border-radius: 6px;">
                            <div style="font-weight: 600; color: #FF6B6B; margin-bottom: 5px;">4. CUDA Cores</div>
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
                            1. Global Memory (GMEM)
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
                            2. Shared Memory (SMEM) - Thread Block Tiles
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
                            3. Registers (Thread Tiles) - Per Thread
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
                <div style="text-align: center; color: #FF6B6B; font-size: 32px; height: 20px;">↓</div>

                <!-- Stage 4: CUDA Cores -->
                <div style="background: #1a1a1a; padding: 20px; border-radius: 8px; border: 2px solid #FF6B6B;">
                    <div style="text-align: center; margin-bottom: 15px;">
                        <div style="font-size: 16px; font-weight: 700; color: #FF6B6B; margin-bottom: 5px;">
                            4. CUDA Cores - Computation
                        </div>
                        <div style="font-size: 12px; color: #999;">FMA: result[i] += thread_tile_a[i] * b_tmp</div>
                    </div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 20px;">
                        <div id="cudaCores" style="display: flex; gap: 10px;"></div>
                    </div>
                    <div style="margin-top: 15px; text-align: center; font-size: 11px; color: #FF6B6B;">
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

            this.updateStats(`<strong style="color: #FFA726;">K Iteration ${kIter + 1}/${numKIterations}</strong> - Processing K tiles [${kOffset}:${kOffset + this.BK}]`);
            await sleep(300);

            // Step 1: Highlight GMEM blocks being loaded
            this.updateStats(`<strong style="color: #4CAF50;">1. GMEM → SMEM</strong>: Loading blocks from Global Memory`);
            await this.highlightGMEMBlocks(kIter);
            await sleep(700);

            // Step 2: Show blocks loaded in SMEM
            this.updateStats(`<strong style="color: #FF9800;">2. SMEM Ready</strong>: tile_a[${this.BM}×${this.BK}] ⊗ tile_b[${this.BK}×${this.BN}]`);
            await this.loadSharedMemory(kIter);
            await sleep(700);

            // Step 3: Highlight thread's tile in SMEM and load to registers
            this.updateStats(`<strong style="color: #00BCD4;">3. SMEM → Registers</strong>: Thread loads ${this.TM} values + b_tmp`);
            await this.loadRegisters(kIter);
            await sleep(900);

            // Step 4: Compute FMA operations
            this.updateStats(`<strong style="color: #FF6B6B;">4. Compute</strong>: ${this.TM} FMA ops (b_tmp reused ${this.TM}×)`);
            await this.computeCudaCores(kIter);
            await sleep(700);

            // Step 5: Show partial results (pink) or final results (green)
            if (!isLastIteration) {
                this.updateStats(`<strong style="color: #FF1493;">5. Partial Results</strong>: Accumulating... (${kIter + 1}/${numKIterations} complete)`);
                await this.showPartialResults(kIter);
                await sleep(600);
            } else {
                this.updateStats(`<strong style="color: #8BC34A;">5. Final Results</strong>: Writing ${this.TM} outputs to Global Memory`);
                await this.showFinalResults();
                await sleep(800);
            }

            // Mark processed GMEM cells as complete (green)
            await this.markGMEMComplete(kIter, isLastIteration);
            await sleep(400);
        }

        this.updateStats(`<strong>Complete!</strong> All ${numKIterations} K tiles processed. Matrix multiplication done!`);

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
            core.style.background = '#FF6B6B';
            core.style.border = '2px solid #F06292';
            core.textContent = 'FMA';

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
            'Ready to start',
            '1. Loading block tiles from global memory (GMEM → SMEM)',
            '2. Block tiles cached in shared memory',
            '3. Thread loading data into registers (SMEM → Registers)',
            '4. Computing FMA operations on CUDA cores',
            '5. Complete - Results written back to global memory'
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
                <div class="viz-title">2D Block Tiling: Outer Product</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="tiling2dAnimate">Animate</button>
                    <button class="viz-btn" id="tiling2dReset">Reset</button>
                </div>
                <div class="viz-canvas" id="tiling2dCanvas"></div>
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
                        <div class="legend-box" style="background: #FF6B6B;"></div>
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
                    regM.style.background = '#FF6B6B';
                    regN.style.background = '#FF6B6B';

                    // Highlight output cell
                    const cellC = this.matrixC.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                    if (cellC) cellC.style.background = '#FF6B6B';

                    await sleep(50);

                    // Update output
                    if (cellC) {
                        if (k < this.tileSize - 1) {
                            cellC.style.background = '#FF6B6B';
                        } else {
                            cellC.style.background = '#4CAF50';
                            computationsCompleted++;
                        }
                    }

                    // Reset register colors
                    regM.style.background = '#9C27B0';
                    regN.style.background = '#00BCD4';
                }
            }

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

    reset() {
        this.renderMatrices();
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
                <div class="viz-title"> Thread To Index/Memory Mapping: Naive vs Coalesced</div>
                <div class="viz-controls">
                    <button class="viz-btn" id="transformAnimate">Animate</button>
                    <button class="viz-btn" id="transformReset">Reset</button>
                </div>
                <div class="viz-canvas" id="transformCanvas" style="overflow-x: auto;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; min-width: 700px;">
                        <!-- Naive Kernel -->
                        <div style="border: 2px solid #e74c3c; padding: 15px; border-radius: 8px;">
                            <h3 style="text-align: center; color: #e74c3c; margin-bottom: 12px; font-size: 16px;">Naive Kernel</h3>
                            <div style="background: #2a2a2a; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-family: monospace; font-size: 11px;">
                                <div>x = blockIdx.x * ${this.blockSize} + (threadIdx.x % ${this.blockSize})</div>
                                <div>y = blockIdx.y * ${this.blockSize} + (threadIdx.x / ${this.blockSize})</div>
                            </div>
                            <div id="naiveGrid" style="display: grid; grid-template-columns: repeat(${this.blockSize}, 65px); column-gap: 8px; row-gap: 16px; justify-content: center;"></div>
                            <div id="naiveMemAccess" style="margin-top: 12px; padding: 8px; background: #2a2a2a; border-radius: 6px; min-height: 70px; font-size: 12px;"></div>
                        </div>
                        <!-- Coalesced Kernel -->
                        <div style="border: 2px solid #27ae60; padding: 15px; border-radius: 8px;">
                            <h3 style="text-align: center; color: #27ae60; margin-bottom: 12px; font-size: 16px;">Coalesced Kernel</h3>
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
            // For coalesced, group threads by row (all 4 threads in a row load together)
            const isFirstInRow = (i % this.blockSize) === 0;

            // Highlight current thread in naive, or the whole row in coalesced
            const naiveCell = document.querySelector(`[data-thread-idx="${i}"][data-type="naive"]`);
            naiveCell.classList.add('active');

            if (isFirstInRow) {
                // Highlight all 4 threads in the current row for coalesced
                for (let j = 0; j < this.blockSize; j++) {
                    const idx = i + j;
                    const coalescedCell = document.querySelector(`[data-thread-idx="${idx}"][data-type="coalesced"]`);
                    if (coalescedCell) coalescedCell.classList.add('active');
                }
            }

            // Calculate memory access patterns
            const naiveX = i % this.blockSize;
            const naiveRow = naiveX;

            const coalescedX = Math.floor(i / this.blockSize);
            const coalescedY = i % this.blockSize;

            // Update memory access info
            const prevNaiveX = (i - 1) % this.blockSize;

            document.getElementById('naiveMemAccess').innerHTML = `
                <div style="font-weight: bold; margin-bottom: 8px; color: #4CAF50;">Thread ${i} Memory Access</div>
                <div style="margin-bottom: 4px;">Matrix Position: <strong>A[${naiveRow}][k]</strong></div>
                ${i > 0 ? `
                    <div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px;">
                        ${naiveRow !== prevNaiveX ?
                        `<strong style="color: #e74c3c;">Different row</strong> from Thread ${i - 1}<br>
                             <span style="color: #999;">Stride: K elements (scattered)</span>` :
                        `<strong style="color: #27ae60;">Same row</strong> as Thread ${i - 1}`}
                    </div>
                ` : '<div style="color: #999; margin-top: 8px;">First thread in warp</div>'}
            `;

            document.getElementById('coalescedMemAccess').innerHTML = `
                <div style="font-weight: bold; margin-bottom: 8px; color: #4CAF50;">Threads ${Math.floor(i / this.blockSize) * this.blockSize}-${Math.floor(i / this.blockSize) * this.blockSize + this.blockSize - 1} Memory Access</div>
                <div style="margin-bottom: 4px;">Matrix Row: <strong>A[${coalescedX}][0..3]</strong></div>
                <div style="margin-top: 8px; padding: 8px; background: #1a1a1a; border-radius: 4px;">
                    <strong style="color: #27ae60;">All 4 threads load row together</strong><br>
                    <span style="color: #4CAF50;">1 coalesced memory transaction (128 bits)</span><br>
                    <span style="font-size: 11px; color: #999; margin-top: 4px;">Thread gets A[${coalescedX}][${coalescedY}]</span>
                </div>
            `;

            await sleep(700);

            // Remove active from naive
            naiveCell.classList.remove('active');

            // Remove active from coalesced row when moving to next row
            if (isFirstInRow) {
                for (let j = 0; j < this.blockSize; j++) {
                    const idx = i + j;
                    const coalescedCell = document.querySelector(`[data-thread-idx="${idx}"][data-type="coalesced"]`);
                    if (coalescedCell) coalescedCell.classList.remove('active');
                }
            }
        }

        // Show completion message
        document.getElementById('naiveMemAccess').innerHTML = `
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 14px; color: #e74c3c; font-weight: bold;">Non-coalesced Access</div>
                <div style="margin-top: 8px; color: #999;">→ ${this.blockSize} separate memory transactions for ${this.blockSize} threads</div>
            </div>
        `;

        document.getElementById('coalescedMemAccess').innerHTML = `
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 14px; color: #27ae60; font-weight: bold;">Coalesced Access</div>
                <div style="margin-top: 8px; color: #999;">→ 1 memory transaction per row of ${this.blockSize} threads (128-bit load)</div>
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

// ============================================================================
// Warp Tiling Visualization
// ============================================================================
class WarpTilingViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;

        // Configuration matching typical warp tiling parameters
        this.BM = 8;   // Block tile M
        this.BN = 8;   // Block tile N
        this.BK = 8;   // Block tile K
        this.WM = 32;  // Warp tile M (half of block)
        this.WN = 32;  // Warp tile N (half of block)
        this.TM = 8;   // Thread tile M
        this.TN = 4;   // Thread tile N
        this.WARPSIZE = 32;

        // For visualization, use smaller sizes
        this.vizBM = 8;
        this.vizBN = 8;
        this.vizWM = 8;
        this.vizWN = 8;
        this.vizTM = 4;
        this.vizTN = 4;

        this.init();
    }

    init() {
        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">Warp Tiling</div>

                <div class="viz-controls">
                    <button class="viz-btn" id="warpAnimate">Animate</button>
                    <button class="viz-btn" id="warpReset">Reset</button>
                </div>
                <div class="viz-info" style="margin-bottom: 15px;">
                    <div style="padding: 12px; background: #1a1a1a; border-radius: 6px; margin-top: 10px; border: 2px solid #9C27B0;">
                        <strong style="color: #9C27B0;">Tile Dimensions:</strong>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 8px; font-size: 12px;">
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #FF9800;">
                                <div style="color: #999; font-size: 10px;">Block Tile</div>
                                <div style="color: #FF9800; font-weight: 600;">BM=${this.BM}, BN=${this.BN}, BK=${this.BK}</div>
                            </div>
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #9C27B0;">
                                <div style="color: #999; font-size: 10px;">Warp Tile</div>
                                <div style="color: #9C27B0; font-weight: 600;">WM=${this.vizWM}, WN=${this.vizWN}</div>
                            </div>
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #00BCD4;">
                                <div style="color: #999; font-size: 10px;">Thread Tile</div>
                                <div style="color: #00BCD4; font-weight: 600;">TM=${this.vizTM}, TN=${this.vizTN}</div>
                            </div>
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #4CAF50;">
                                <div style="color: #999; font-size: 10px;">Warp Threads</div>
                                <div style="color: #4CAF50; font-weight: 600;">${this.WARPSIZE} threads</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="viz-canvas" id="warpCanvas"></div>

                <div class="legend">
                                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF9800;"></div>
                        <span>Shared Memory (Block Tile)</span>
                    </div>
                    
                    <div class="legend-item">
                        <div class="legend-box" style="background: #9C27B0;"></div>
                        <span>Warp Fragment (register_m)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #00BCD4;"></div>
                        <span>Warp Fragment (register_n)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF6B6B;"></div>
                        <span>Computing Thread Tile</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>Thread Results (TM×TN)</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('warpCanvas');
        this.renderHierarchy();

        document.getElementById('warpAnimate').addEventListener('click', () => this.animate());
        document.getElementById('warpReset').addEventListener('click', () => this.reset());
    }

    renderHierarchy() {
        this.canvas.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 30px;">
                <!-- Level 1: Shared Memory -->
                <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 16px; color: #FF9800; margin-bottom: 10px; font-weight: 600;">
                            Shared Memory (Block Tiles ${this.vizBM}×${this.vizBN})
                        </div>
                        <div style="font-size: 11px; color: #999; margin-bottom: 8px;">
                            All warps in the block share this data
                        </div>
                    </div>
                    <div style="display: flex; gap: 30px; align-items: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #FF9800; margin-bottom: 8px;">tile_a[${this.vizBM}×${this.BK}]</div>
                            <div id="warpSharedA"></div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #FF9800; margin-bottom: 8px;">tile_b[${this.BK}×${this.vizBN}]</div>
                            <div id="warpSharedB"></div>
                        </div>
                    </div>
                </div>

                <!-- Arrow down -->
                <div style="font-size: 30px; color: #FFA726;">↓</div>
                <div style="font-size: 11px; color: #9C27B0; font-weight: 600; margin-top: -25px;">
                    Warp loads WM×WN fragment into registers
                </div>

                <!-- Level 2: Warp Registers -->
                <div style="display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 20px; background: #2a2a2a; border-radius: 8px; border: 2px solid #9C27B0;">
                    <div style="text-align: center;">
                        <div style="font-size: 16px; color: #9C27B0; margin-bottom: 10px; font-weight: 600;">
                            Warp-Level Register Cache (${this.vizWM}×1 and 1x${this.vizWN})
                        </div>
                        <div style="font-size: 11px; color: #999; margin-bottom: 8px;">
                            One warp (32 threads) collectively holds these fragments
                        </div>
                    </div>
                    <div style="display: flex; gap: 30px; align-items: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #9C27B0; margin-bottom: 8px;">register_m[WM]</div>
                            <div id="warpRegM" style="display: flex; flex-direction: column; gap: 8px;"></div>
                        </div>
                        <div style="font-size: 30px; color: #FF6B6B;">⊗</div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #00BCD4; margin-bottom: 8px;">register_n[WN]</div>
                            <div id="warpRegN" style="display: flex; gap: 8px;"></div>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #999; text-align: center; margin-top: 5px;">
                        Each thread holds ${this.vizTM}×${this.vizTN} portion of the warp tile
                    </div>
                </div>

                <!-- Arrow down -->
                <div style="font-size: 30px; color: #FFA726;">↓</div>
                <div style="font-size: 11px; color: #4CAF50; font-weight: 600; margin-top: -25px;">
                    Each thread computes its TM×TN tile from warp fragments
                </div>

                <!-- Level 3: Thread Output -->
                <div style="display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 20px; background: #2a2a2a; border-radius: 8px; border: 2px solid #4CAF50;">
                    <div style="text-align: center;">
                        <div style="font-size: 16px; color: #4CAF50; margin-bottom: 10px; font-weight: 600;">
                            Thread Output Tile (${this.vizTM}×${this.vizTN})
                        </div>
                        <div style="font-size: 11px; color: #999; margin-bottom: 8px;">
                            Each of 32 threads computes this
                        </div>
                    </div>
                    <div id="warpThreadOutput"></div>
                    <div style="font-size: 11px; color: #999; text-align: center; margin-top: 5px;">
                        Total per warp: ${this.vizWM}×${this.vizWN} = ${this.vizWM * this.vizWN} outputs
                    </div>
                </div>
            </div>
        `;

        // Create shared memory tiles (smaller for visualization)
        const cellSize = 35;
        this.sharedA = createMatrixGrid(this.vizWM, this.BK, cellSize, '', document.getElementById('warpSharedA'));
        this.sharedB = createMatrixGrid(this.BK, this.vizWN, cellSize, '', document.getElementById('warpSharedB'));

        // Create warp register fragments for register_m with thread tile grouping
        const regMContainer = document.getElementById('warpRegM');
        regMContainer.innerHTML = '';
        const numThreadTiles = this.vizWM / this.vizTM;
        for (let threadIdx = 0; threadIdx < numThreadTiles; threadIdx++) {
            const groupBox = document.createElement('div');
            groupBox.style.padding = '8px';
            groupBox.style.border = '2px solid #9C27B0';
            groupBox.style.borderRadius = '6px';
            groupBox.style.display = 'flex';
            groupBox.style.flexDirection = 'column';
            groupBox.style.gap = '2px';

            for (let i = 0; i < this.vizTM; i++) {
                const regIdx = threadIdx * this.vizTM + i;
                const reg = document.createElement('div');
                reg.id = `warp-reg-m-${regIdx}`;
                reg.style.width = '40px';
                reg.style.height = '20px';
                reg.style.background = '#333';
                reg.style.border = '2px solid #555';
                reg.style.borderRadius = '4px';
                reg.style.display = 'flex';
                reg.style.alignItems = 'center';
                reg.style.justifyContent = 'center';
                reg.style.fontSize = '9px';
                reg.style.transition = 'all 0.3s';
                reg.style.color = '#999';
                reg.textContent = `m${regIdx}`;
                groupBox.appendChild(reg);
            }
            regMContainer.appendChild(groupBox);
        }

        // Create warp register fragments for register_n with thread tile grouping
        const regNContainer = document.getElementById('warpRegN');
        regNContainer.innerHTML = '';
        const numThreadTilesN = this.vizWN / this.vizTN;
        for (let threadIdx = 0; threadIdx < numThreadTilesN; threadIdx++) {
            const groupBox = document.createElement('div');
            groupBox.style.padding = '8px';
            groupBox.style.border = '2px solid #00BCD4';
            groupBox.style.borderRadius = '6px';
            groupBox.style.display = 'flex';
            groupBox.style.gap = '2px';

            for (let j = 0; j < this.vizTN; j++) {
                const regIdx = threadIdx * this.vizTN + j;
                const reg = document.createElement('div');
                reg.id = `warp-reg-n-${regIdx}`;
                reg.style.width = '40px';
                reg.style.height = '40px';
                reg.style.background = '#333';
                reg.style.border = '2px solid #555';
                reg.style.borderRadius = '4px';
                reg.style.display = 'flex';
                reg.style.alignItems = 'center';
                reg.style.justifyContent = 'center';
                reg.style.fontSize = '9px';
                reg.style.transition = 'all 0.3s';
                reg.style.color = '#999';
                reg.textContent = `n${regIdx}`;
                groupBox.appendChild(reg);
            }
            regNContainer.appendChild(groupBox);
        }

        // Create thread output tile
        this.threadOutput = createMatrixGrid(this.vizTM, this.vizTN, 45, '', document.getElementById('warpThreadOutput'));
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.renderHierarchy();

        document.getElementById('warpAnimate').disabled = true;

        // Step 1: Highlight block tile in shared memory
        for (let i = 0; i < this.vizWM; i++) {
            for (let k = 0; k < this.BK; k++) {
                const cellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${k}"]`);
                if (cellA) {
                    cellA.style.background = '#FF9800';
                    cellA.style.borderColor = '#FF9800';
                }
            }
        }
        for (let k = 0; k < this.BK; k++) {
            for (let j = 0; j < this.vizWN; j++) {
                const cellB = this.sharedB.querySelector(`[data-row="${k}"][data-col="${j}"]`);
                if (cellB) {
                    cellB.style.background = '#FF9800';
                    cellB.style.borderColor = '#FF9800';
                }
            }
        }
        await sleep(800);

        // Iterate over K dimension
        for (let k = 0; k < this.BK; k++) {
            // Step 2: Load warp fragment from shared memory to registers

            // Highlight column k from tile_a
            for (let i = 0; i < this.vizWM; i++) {
                const cellA = this.sharedA.querySelector(`[data-row="${i}"][data-col="${k}"]`);
                if (cellA) {
                    cellA.classList.add('active');
                    cellA.style.background = '#9C27B0';
                }

                const regM = document.getElementById(`warp-reg-m-${i}`);
                if (regM) {
                    regM.style.background = '#9C27B0';
                    regM.style.borderColor = '#9C27B0';
                    regM.style.color = '#fff';
                }
            }
            await sleep(400);

            // Highlight column k from tile_b
            for (let j = 0; j < this.vizWN; j++) {
                const cellB = this.sharedB.querySelector(`[data-row="${k}"][data-col="${j}"]`);
                if (cellB) {
                    cellB.classList.add('active');
                    cellB.style.background = '#00BCD4';
                }

                const regN = document.getElementById(`warp-reg-n-${j}`);
                if (regN) {
                    regN.style.background = '#00BCD4';
                    regN.style.borderColor = '#00BCD4';
                    regN.style.color = '#fff';
                }
            }
            await sleep(400);

            // Step 3: Compute thread tile using outer product of warp fragments
            for (let i = 0; i < this.vizTM; i++) {
                for (let j = 0; j < this.vizTN; j++) {
                    // Highlight the registers being used
                    const regM = document.getElementById(`warp-reg-m-${i}`);
                    const regN = document.getElementById(`warp-reg-n-${j}`);
                    if (regM) regM.style.background = '#FF6B6B';
                    if (regN) regN.style.background = '#FF6B6B';

                    // Highlight output cell
                    const cellOut = this.threadOutput.querySelector(`[data-row="${i}"][data-col="${j}"]`);
                    if (cellOut) {
                        cellOut.style.background = '#FF6B6B';
                        cellOut.style.borderColor = '#FF6B6B';
                    }

                    await sleep(80);

                    // Update output cell
                    if (cellOut) {
                        if (k < this.BK - 1) {
                            cellOut.style.background = '#FF6B6B'; // Partial result
                            cellOut.textContent = `~${k + 1}`;
                        } else {
                            cellOut.style.background = '#4CAF50'; // Final result
                            cellOut.textContent = '';
                        }
                        cellOut.style.borderColor = '#4CAF50';
                    }

                    // Reset register colors
                    if (regM) regM.style.background = '#9C27B0';
                    if (regN) regN.style.background = '#00BCD4';
                }
            }

            await sleep(300);

            // Clear active highlights
            this.sharedA.querySelectorAll('.active').forEach(el => {
                el.classList.remove('active');
                el.style.background = '#FF9800';
            });
            this.sharedB.querySelectorAll('.active').forEach(el => {
                el.classList.remove('active');
                el.style.background = '#FF9800';
            });

            // Reset register colors for next iteration
            for (let i = 0; i < this.vizWM; i++) {
                const regM = document.getElementById(`warp-reg-m-${i}`);
                if (regM) {
                    regM.style.background = '#333';
                    regM.style.borderColor = '#555';
                    regM.style.color = '#999';
                }
            }
            for (let j = 0; j < this.vizWN; j++) {
                const regN = document.getElementById(`warp-reg-n-${j}`);
                if (regN) {
                    regN.style.background = '#333';
                    regN.style.borderColor = '#555';
                    regN.style.color = '#999';
                }
            }
        }

        document.getElementById('warpAnimate').disabled = false;
        this.isAnimating = false;
    }

    reset() {
        this.renderHierarchy();
    }
}

// ============================================================================
// WMMA TensorCore Visualization
// ============================================================================
class WmmaTensorcoreViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }
        this.isAnimating = false;

        // Real kernel parameters from the code
        this.BLOCK_ROW_WARPS = 4;
        this.BLOCK_COL_WARPS = 4;
        this.WARP_ROW_TILES = 4;
        this.WARP_COL_TILES = 2;
        this.WMMA_M = 16;
        this.WMMA_N = 16;
        this.WMMA_K = 16;

        // Calculate block dimensions
        this.BLOCK_ROW_TILES = this.WARP_ROW_TILES * this.BLOCK_ROW_WARPS;
        this.BLOCK_COL_TILES = this.WARP_COL_TILES * this.BLOCK_COL_WARPS;
        this.BM = this.BLOCK_ROW_TILES * this.WMMA_M; // 256
        this.BN = this.BLOCK_COL_TILES * this.WMMA_N; // 128
        this.BK = this.WMMA_K; // 16

        this.init();
    }

    init() {
        // For visualization, use smaller representative sizes (matching WarpTilingViz style)
        this.vizBM = 32;   // Visualize as 32 instead of 256 (2 WMMA tiles)
        this.vizBN = 32;   // Visualize as 32 instead of 128 (2 WMMA tiles)
        this.vizBK = 16;   // Keep K the same
        this.vizBlockRowWarps = 2;  // Show 2×2 warp grid instead of 4×4
        this.vizBlockColWarps = 2;
        this.vizWarpRowTiles = 1;   // Show 1×1 WMMA tiles per warp instead of 4×2
        this.vizWarpColTiles = 1;

        this.container.innerHTML = commonStyles + `
            <div class="viz-container">
                <div class="viz-title">WMMA TensorCore Warp-Tiled Kernel</div>

                <div class="viz-controls">
                    <button class="viz-btn" id="wmmaAnimate">Animate</button>
                    <button class="viz-btn" id="wmmaReset">Reset</button>
                </div>

                <div class="viz-info" style="margin-bottom: 15px;">
                    <div style="padding: 12px; background: #1a1a1a; border-radius: 6px; margin-top: 10px; border: 2px solid #FF6B6B;">
                        <strong style="color: #FF6B6B;">Real Kernel Dimensions:</strong>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 8px; font-size: 12px;">
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #FF9800;">
                                <div style="color: #999; font-size: 10px;">Block Tile</div>
                                <div style="color: #FF9800; font-weight: 600;">BM=${this.BM}, BN=${this.BN}, BK=${this.BK}</div>
                            </div>
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #9C27B0;">
                                <div style="color: #999; font-size: 10px;">Warp Grid</div>
                                <div style="color: #9C27B0; font-weight: 600;">${this.BLOCK_ROW_WARPS}×${this.BLOCK_COL_WARPS} = ${this.BLOCK_ROW_WARPS * this.BLOCK_COL_WARPS} warps</div>
                            </div>
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #00BCD4;">
                                <div style="color: #999; font-size: 10px;">Warp Tiles</div>
                                <div style="color: #00BCD4; font-weight: 600;">${this.WARP_ROW_TILES}×${this.WARP_COL_TILES} WMMA ops per warp</div>
                            </div>
                            <div style="padding: 8px; background: #2a2a2a; border-radius: 4px; border-left: 3px solid #FF6B6B;">
                                <div style="color: #999; font-size: 10px;">WMMA Tile</div>
                                <div style="color: #FF6B6B; font-weight: 600;">${this.WMMA_M}×${this.WMMA_N}×${this.WMMA_K}</div>
                            </div>
                        </div>
                        <div style="font-size: 11px; color: #999; margin-top: 10px; text-align: center; font-style: italic;">
                            Visualization shows scaled-down version for clarity
                        </div>
                    </div>
                </div>

                <div class="viz-canvas" id="wmmaCanvas"></div>

                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF9800;"></div>
                        <span>Block Tile (Shared Memory)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #9C27B0;"></div>
                        <span>Warp Processing Area</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #FF6B6B;"></div>
                        <span>WMMA Fragment (16×16)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-box" style="background: #4CAF50;"></div>
                        <span>Active TensorCore Operation</span>
                    </div>
                </div>
            </div>
        `;

        this.canvas = document.getElementById('wmmaCanvas');
        this.renderHierarchy();

        document.getElementById('wmmaAnimate').addEventListener('click', () => this.animate());
        document.getElementById('wmmaReset').addEventListener('click', () => this.reset());
    }

    renderHierarchy() {
        // Use reasonable cell sizes for visualization
        const cellSize = 12;

        this.canvas.innerHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; gap: 30px;">
                <!-- Level 1: Block Level (Shared Memory) -->
                <div style="display: flex; flex-direction: column; align-items: center; gap: 15px;">
                    <div style="text-align: center;">
                        <div style="font-size: 16px; color: #FF9800; margin-bottom: 10px; font-weight: 600;">
                            Block Level: Shared Memory Tiles
                        </div>
                        <div style="font-size: 11px; color: #999; margin-bottom: 8px;">
                            ${this.vizBlockRowWarps * this.vizBlockColWarps} warps in a ${this.vizBlockRowWarps}×${this.vizBlockColWarps} grid cooperatively load and process
                        </div>
                    </div>
                    <div style="display: flex; gap: 30px; align-items: center; flex-wrap: wrap; justify-content: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #FF9800; margin-bottom: 8px;">tile_a[${this.vizBM}×${this.vizBK}]</div>
                            <div id="wmmaBlockA" style="position: relative;"></div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #FF9800; margin-bottom: 8px;">tile_b[${this.vizBK}×${this.vizBN}]</div>
                            <div id="wmmaBlockB" style="position: relative;"></div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #999; margin-bottom: 8px;">Output C[${this.vizBM}×${this.vizBN}]</div>
                            <div id="wmmaBlockC" style="position: relative;"></div>
                        </div>
                    </div>
                </div>

                <!-- Arrow down -->
                <div style="font-size: 30px; color: #FFA726;">↓</div>
                <div style="font-size: 11px; color: #9C27B0; font-weight: 600; margin-top: -25px;">
                    Each warp processes ${this.vizWarpRowTiles}×${this.vizWarpColTiles} WMMA tiles
                </div>

                <!-- Level 2: Warp Level -->
                <div style="display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 20px; background: #2a2a2a; border-radius: 8px; border: 2px solid #9C27B0;">
                    <div style="text-align: center;">
                        <div style="font-size: 16px; color: #9C27B0; margin-bottom: 10px; font-weight: 600;">
                            Warp Level: WMMA Tile Grid (${this.vizWarpRowTiles * this.WMMA_M}×${this.vizWarpColTiles * this.WMMA_N})
                        </div>
                        <div style="font-size: 11px; color: #999; margin-bottom: 8px;">
                            Each warp computes ${this.vizWarpRowTiles}×${this.vizWarpColTiles} = ${this.vizWarpRowTiles * this.vizWarpColTiles} WMMA operations
                        </div>
                    </div>
                    <div id="wmmaWarpGrid" style="position: relative;"></div>
                    <div style="font-size: 11px; color: #999; text-align: center; margin-top: 5px;">
                        Warp output: ${this.vizWarpRowTiles * this.WMMA_M}×${this.vizWarpColTiles * this.WMMA_N} = ${this.vizWarpRowTiles * this.WMMA_M * this.vizWarpColTiles * this.WMMA_N} elements
                    </div>
                </div>

                <!-- Arrow down -->
                <div style="font-size: 30px; color: #FFA726;">↓</div>
                <div style="font-size: 11px; color: #FF6B6B; font-weight: 600; margin-top: -25px;">
                    TensorCore executes D = A × B + C (16×16×16 matrix multiply-accumulate)
                </div>

                <!-- Level 3: TensorCore Operation -->
                <div style="display: flex; flex-direction: column; align-items: center; gap: 15px; padding: 20px; background: #2a2a2a; border-radius: 8px; border: 2px solid #FF6B6B;">
                    <div style="text-align: center;">
                        <div style="font-size: 16px; color: #FF6B6B; margin-bottom: 10px; font-weight: 600;">
                            TensorCore WMMA Operation (${this.WMMA_M}×${this.WMMA_N}×${this.WMMA_K})
                        </div>
                        <div style="font-size: 11px; color: #999; margin-bottom: 8px;">
                            32 threads collectively hold matrix fragments
                        </div>
                    </div>
                    <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap; justify-content: center;">
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #FF6B6B; margin-bottom: 8px;">A_frag (${this.WMMA_M}×${this.WMMA_K})</div>
                            <div id="wmmaTcA"></div>
                        </div>
                        <div style="font-size: 24px; color: #FF6B6B;">×</div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #FF6B6B; margin-bottom: 8px;">B_frag (${this.WMMA_K}×${this.WMMA_N})</div>
                            <div id="wmmaTcB"></div>
                        </div>
                        <div style="font-size: 24px; color: #FF6B6B;">+</div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #00BCD4; margin-bottom: 8px;">ACC_frag (${this.WMMA_M}×${this.WMMA_N})</div>
                            <div id="wmmaTcAcc"></div>
                        </div>
                        <div style="font-size: 24px; color: #4CAF50;">=</div>
                        <div style="text-align: center;">
                            <div style="font-size: 12px; color: #4CAF50; margin-bottom: 8px;">D_frag (${this.WMMA_M}×${this.WMMA_N})</div>
                            <div id="wmmaTcD"></div>
                        </div>
                    </div>
                    <div style="font-size: 11px; color: #999; text-align: center; margin-top: 5px;">
                        Single warp-synchronous mma_sync() instruction
                    </div>
                </div>
            </div>
        `;

        // Create block-level tiles with reasonable cell size
        this.blockA = createMatrixGrid(this.vizBM, this.vizBK, cellSize, '', document.getElementById('wmmaBlockA'));
        this.blockB = createMatrixGrid(this.vizBK, this.vizBN, cellSize, '', document.getElementById('wmmaBlockB'));
        this.blockC = createMatrixGrid(this.vizBM, this.vizBN, cellSize, '', document.getElementById('wmmaBlockC'));

        // Add warp grid overlay to block C
        this.addWarpGridOverlay(this.blockC, this.vizBM, this.vizBN, cellSize);

        // Create warp-level WMMA tile grid
        const warpM = this.vizWarpRowTiles * this.WMMA_M;
        const warpN = this.vizWarpColTiles * this.WMMA_N;
        const warpCellSize = 18;
        this.warpGrid = createMatrixGrid(warpM, warpN, warpCellSize, '', document.getElementById('wmmaWarpGrid'));
        this.addWmmaTileOverlay(this.warpGrid, warpM, warpN, warpCellSize);

        // Create TensorCore fragment visualizations
        const tcCellSize = 15;
        this.tcA = createMatrixGrid(this.WMMA_M, this.WMMA_K, tcCellSize, '', document.getElementById('wmmaTcA'));
        this.tcB = createMatrixGrid(this.WMMA_K, this.WMMA_N, tcCellSize, '', document.getElementById('wmmaTcB'));
        this.tcAcc = createMatrixGrid(this.WMMA_M, this.WMMA_N, tcCellSize, '', document.getElementById('wmmaTcAcc'));
        this.tcD = createMatrixGrid(this.WMMA_M, this.WMMA_N, tcCellSize, '', document.getElementById('wmmaTcD'));
    }

    addWarpGridOverlay(grid, vizBM, vizBN, cellSize) {
        const container = grid.parentElement;
        container.style.position = 'relative';

        const warpHeight = (this.vizWarpRowTiles * this.WMMA_M) * (cellSize + 2);
        const warpWidth = (this.vizWarpColTiles * this.WMMA_N) * (cellSize + 2);

        for (let wr = 0; wr < this.vizBlockRowWarps; wr++) {
            for (let wc = 0; wc < this.vizBlockColWarps; wc++) {
                const overlay = document.createElement('div');
                overlay.className = `warp-overlay-${wr}-${wc}`;
                overlay.style.position = 'absolute';
                overlay.style.top = `${wr * warpHeight}px`;
                overlay.style.left = `${wc * warpWidth}px`;
                overlay.style.width = `${warpWidth}px`;
                overlay.style.height = `${warpHeight}px`;
                overlay.style.border = '2px solid #9C27B0';
                overlay.style.borderRadius = '2px';
                overlay.style.pointerEvents = 'none';
                overlay.style.opacity = '0.5';
                container.appendChild(overlay);
            }
        }
    }

    addWmmaTileOverlay(grid, warpM, warpN, cellSize) {
        const container = grid.parentElement;
        container.style.position = 'relative';

        const tileHeight = this.WMMA_M * (cellSize + 2);
        const tileWidth = this.WMMA_N * (cellSize + 2);

        for (let tr = 0; tr < this.vizWarpRowTiles; tr++) {
            for (let tc = 0; tc < this.vizWarpColTiles; tc++) {
                const overlay = document.createElement('div');
                overlay.className = `wmma-tile-overlay-${tr}-${tc}`;
                overlay.style.position = 'absolute';
                overlay.style.top = `${tr * tileHeight}px`;
                overlay.style.left = `${tc * tileWidth}px`;
                overlay.style.width = `${tileWidth}px`;
                overlay.style.height = `${tileHeight}px`;
                overlay.style.border = '2px solid #FF6B6B';
                overlay.style.borderRadius = '2px';
                overlay.style.pointerEvents = 'none';
                overlay.style.opacity = '0.6';
                container.appendChild(overlay);

                // Add label
                const label = document.createElement('div');
                label.textContent = `${tr},${tc}`;
                label.style.position = 'absolute';
                label.style.top = '2px';
                label.style.left = '2px';
                label.style.fontSize = '9px';
                label.style.color = '#FF6B6B';
                label.style.fontWeight = '700';
                label.style.background = 'rgba(0,0,0,0.7)';
                label.style.padding = '1px 3px';
                label.style.borderRadius = '2px';
                overlay.appendChild(label);
            }
        }
    }

    async animate() {
        if (this.isAnimating) return;
        this.isAnimating = true;

        const animateBtn = document.getElementById('wmmaAnimate');
        animateBtn.disabled = true;

        // Reset
        this.reset();

        // Step 1: Load shared memory tiles
        await this.highlightMatrix(this.blockA, '#FF9800');
        await sleep(300);
        await this.highlightMatrix(this.blockB, '#FF9800');
        await sleep(500);

        // Step 2: Highlight a specific warp area
        const warpRow = 0;
        const warpCol = 0;
        const warpOverlay = document.querySelector(`.warp-overlay-${warpRow}-${warpCol}`);
        if (warpOverlay) {
            warpOverlay.style.border = '3px solid #9C27B0';
            warpOverlay.style.opacity = '1';
            warpOverlay.style.background = 'rgba(156, 39, 176, 0.2)';
        }
        await sleep(500);

        // Step 3: Highlight warp tiles
        await this.highlightMatrix(this.warpGrid, '#9C27B0');
        await sleep(500);

        // Step 4: Animate WMMA operations for this warp
        for (let i = 0; i < this.vizWarpRowTiles; i++) {
            for (let j = 0; j < this.vizWarpColTiles; j++) {
                // Highlight the WMMA tile being processed
                const wmmaTileOverlay = document.querySelector(`.wmma-tile-overlay-${i}-${j}`);
                if (wmmaTileOverlay) {
                    wmmaTileOverlay.style.border = '3px solid #4CAF50';
                    wmmaTileOverlay.style.background = 'rgba(76, 175, 80, 0.3)';
                }

                // Animate TensorCore operation
                await this.highlightMatrix(this.tcA, '#FF6B6B');
                await sleep(200);
                await this.highlightMatrix(this.tcB, '#FF6B6B');
                await sleep(200);
                await this.highlightMatrix(this.tcAcc, '#00BCD4');
                await sleep(200);
                await this.highlightMatrix(this.tcD, '#4CAF50');
                await sleep(300);

                // Reset TensorCore matrices
                this.resetMatrix(this.tcA);
                this.resetMatrix(this.tcB);
                this.resetMatrix(this.tcAcc);
                this.resetMatrix(this.tcD);

                // Mark this warp tile as complete
                if (wmmaTileOverlay) {
                    wmmaTileOverlay.style.background = 'rgba(76, 175, 80, 0.5)';
                }

                await sleep(100);
            }
        }

        // Step 5: Show final output in block C
        await this.highlightWarpInBlockC(warpRow, warpCol);
        await sleep(500);

        this.isAnimating = false;
        animateBtn.disabled = false;
    }

    async highlightMatrix(matrix, color) {
        const cells = matrix.children;
        for (let cell of cells) {
            cell.style.background = color;
        }
    }

    resetMatrix(matrix) {
        const cells = matrix.children;
        for (let cell of cells) {
            cell.style.background = '#333';
        }
    }

    async highlightWarpInBlockC(warpRow, warpCol) {
        const cells = this.blockC.children;
        const startRow = warpRow * this.vizWarpRowTiles * this.WMMA_M;
        const startCol = warpCol * this.vizWarpColTiles * this.WMMA_N;
        const endRow = startRow + this.vizWarpRowTiles * this.WMMA_M;
        const endCol = startCol + this.vizWarpColTiles * this.WMMA_N;

        for (let cell of cells) {
            const row = parseInt(cell.dataset.row);
            const col = parseInt(cell.dataset.col);
            if (row >= startRow && row < endRow && col >= startCol && col < endCol) {
                cell.style.background = '#4CAF50';
            }
        }
    }

    reset() {
        this.renderHierarchy();
    }
}

// Roofline Model Visualization
class RooflineViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) return;

        // RTX 4090 specifications
        this.peakFLOPS = 82.6; // TFLOPS for FP32
        this.peakBandwidth = 1.008; // TB/s (1008 GB/s)
        this.ridgePoint = this.peakFLOPS / this.peakBandwidth; // ~82 FLOP/byte

        // Kernel performance data (for 4096x4096 matrices)
        // Only showing PyTorch for reference
        this.kernels = [
            { name: 'PyTorch', tflops: 84.19, bandwidth: 123.33, color: '#32CD32' }
        ];

        // Calculate arithmetic intensity for each kernel
        // AI = FLOPS / Bandwidth (in FLOP/byte)
        this.kernels.forEach(k => {
            k.arithmeticIntensity = k.tflops * 1000 / k.bandwidth; // Convert TFLOPS to GFLOPS
        });

        this.init();
    }

    init() {
        this.container.innerHTML = `
            ${commonStyles}
            <div class="viz-container">
                <div class="viz-title">Roofline Model - RTX 4090 GEMM Performance</div>
                <canvas id="roofline-canvas" width="1200" height="900" style="width: 100%; height: auto;"></canvas>
            </div>
        `;

        this.canvas = document.getElementById('roofline-canvas');
        this.ctx = this.canvas.getContext('2d');

        // Default settings
        this.showLabels = true;
        this.showGrid = true;

        // Tooltip handling
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseout', () => this.handleMouseOut());

        this.hoveredKernel = null;
        this.draw();
    }

    draw() {
        const canvas = this.canvas;
        const ctx = this.ctx;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#252525';
        ctx.fillRect(0, 0, width, height);

        // Margins
        const margin = { top: 40, right: 150, bottom: 150, left: 150 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;

        // Scales (logarithmic)
        const minAI = 0.1;
        const maxAI = 1000;
        const minPerf = 0.1;
        const maxPerf = 100;

        const xScale = (ai) => {
            return margin.left + (Math.log10(ai) - Math.log10(minAI)) /
                (Math.log10(maxAI) - Math.log10(minAI)) * plotWidth;
        };

        const yScale = (perf) => {
            return height - margin.bottom - (Math.log10(perf) - Math.log10(minPerf)) /
                (Math.log10(maxPerf) - Math.log10(minPerf)) * plotHeight;
        };

        // Draw grid
        if (this.showGrid) {
            this.drawGrid(ctx, xScale, yScale, margin, plotWidth, plotHeight, minAI, maxAI, minPerf, maxPerf);
        }

        // Draw roofline
        this.drawRoofline(ctx, xScale, yScale, margin, plotWidth, plotHeight);

        // Draw kernels
        this.drawKernels(ctx, xScale, yScale, this.showLabels);

        // Draw axes
        this.drawAxes(ctx, xScale, yScale, margin, width, height, plotWidth, plotHeight, minAI, maxAI, minPerf, maxPerf);

        // Draw legend
        this.drawLegend(ctx, width, margin);

        // Draw tooltip (must be last so it appears on top)
        this.drawTooltip(ctx);
    }

    drawGrid(ctx, xScale, yScale, margin, plotWidth, plotHeight, minAI, maxAI, minPerf, maxPerf) {
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);

        // Vertical grid lines (AI)
        const aiValues = [0.1, 1, 10, 100, 1000];
        aiValues.forEach(ai => {
            if (ai >= minAI && ai <= maxAI) {
                const x = xScale(ai);
                ctx.beginPath();
                ctx.moveTo(x, margin.top);
                ctx.lineTo(x, margin.top + plotHeight);
                ctx.stroke();
            }
        });

        // Horizontal grid lines (Performance)
        const perfValues = [0.1, 1, 10, 100];
        perfValues.forEach(perf => {
            if (perf >= minPerf && perf <= maxPerf) {
                const y = yScale(perf);
                ctx.beginPath();
                ctx.moveTo(margin.left, y);
                ctx.lineTo(margin.left + plotWidth, y);
                ctx.stroke();
            }
        });

        ctx.setLineDash([]);
    }

    drawRoofline(ctx, xScale, yScale, margin, plotWidth, plotHeight) {
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 3;
        ctx.beginPath();

        // Memory-bound region (diagonal line)
        const startAI = 0.1;
        const ridgeAI = this.ridgePoint;
        const startPerf = startAI * this.peakBandwidth;

        ctx.moveTo(xScale(startAI), yScale(startPerf));
        ctx.lineTo(xScale(ridgeAI), yScale(this.peakFLOPS));

        // Compute-bound region (horizontal line)
        ctx.lineTo(xScale(1000), yScale(this.peakFLOPS));

        ctx.stroke();

        // Draw Peak FLOPS horizontal line more prominently
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(xScale(0.1), yScale(this.peakFLOPS));
        ctx.lineTo(xScale(1000), yScale(this.peakFLOPS));
        ctx.stroke();
        ctx.setLineDash([]);

        // Label the peak FLOPS on the left
        ctx.fillStyle = '#FF6B6B';
        ctx.font = 'bold 26px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Peak FP32: ${this.peakFLOPS} TFLOPS`, xScale(0.15), yScale(this.peakFLOPS) - 15);

        // Label the roofline
        ctx.fillStyle = '#4CAF50';
        ctx.font = 'bold 28px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Roofline', xScale(200), yScale(this.peakFLOPS) + 35);
    }

    drawKernels(ctx, xScale, yScale, showLabels) {
        this.kernels.forEach((kernel, idx) => {
            const x = xScale(kernel.arithmeticIntensity);
            const y = yScale(kernel.tflops);

            // Draw point
            ctx.fillStyle = kernel.color;
            ctx.beginPath();
            const radius = this.hoveredKernel === idx ? 12 : 10;
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();

            // Draw border
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3;
            ctx.stroke();

            // Draw label
            if (showLabels || this.hoveredKernel === idx) {
                ctx.fillStyle = '#e0e0e0';
                ctx.font = this.hoveredKernel === idx ? 'bold 22px Arial' : '20px Arial';
                const labelY = y - 20;
                const labelX = x + 15;
                ctx.fillText(kernel.name, labelX, labelY);
            }

            // Store position for hover detection
            kernel.x = x;
            kernel.y = y;
        });
    }

    drawAxes(ctx, xScale, yScale, margin, width, height, plotWidth, plotHeight, minAI, maxAI, minPerf, maxPerf) {
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;

        // X-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, height - margin.bottom);
        ctx.lineTo(margin.left + plotWidth, height - margin.bottom);
        ctx.stroke();

        // Y-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, height - margin.bottom);
        ctx.stroke();

        // X-axis ticks
        ctx.font = '22px Arial';
        ctx.textAlign = 'center';
        ctx.fillStyle = '#e0e0e0';
        const aiTicks = [0.1, 1, 10, 100, 1000];
        aiTicks.forEach(ai => {
            if (ai >= minAI && ai <= maxAI) {
                const x = xScale(ai);
                ctx.fillText(ai.toString(), x, height - margin.bottom + 50);

                // Tick mark
                ctx.beginPath();
                ctx.moveTo(x, height - margin.bottom);
                ctx.lineTo(x, height - margin.bottom + 10);
                ctx.stroke();
            }
        });

        // Y-axis ticks
        ctx.textAlign = 'right';
        const perfTicks = [0.1, 1, 10, 100];
        perfTicks.forEach(perf => {
            if (perf >= minPerf && perf <= maxPerf) {
                const y = yScale(perf);
                ctx.fillText(perf.toString(), margin.left - 30, y + 8);

                // Tick mark
                ctx.beginPath();
                ctx.moveTo(margin.left - 10, y);
                ctx.lineTo(margin.left, y);
                ctx.stroke();
            }
        });

        // X-axis label (positioned below tick labels with proper spacing)
        ctx.fillStyle = '#e0e0e0';
        ctx.font = 'bold 28px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Arithmetic Intensity (FLOP/byte)', width / 2, height - 30);

        // Y-axis label (positioned to the left of tick labels with proper spacing)
        ctx.save();
        ctx.translate(40, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Performance (TFLOPS)', 0, 0);
        ctx.restore();
    }

    drawLegend(ctx, width, margin) {
        const legendX = width - margin.right + 10;
        let legendY = margin.top + 20;

        ctx.font = 'bold 24px Arial';
        ctx.fillStyle = '#e0e0e0';
        ctx.textAlign = 'left';
        ctx.fillText('Kernels:', legendX, legendY);

        legendY += 30;

        // Show top performing kernels in legend
        const topKernels = [...this.kernels].sort((a, b) => b.tflops - a.tflops).slice(0, 6);

        topKernels.forEach(kernel => {
            // Color box
            ctx.fillStyle = kernel.color;
            ctx.fillRect(legendX, legendY - 12, 16, 16);

            // Name
            ctx.fillStyle = '#e0e0e0';
            ctx.font = '20px Arial';
            ctx.fillText(kernel.name, legendX + 24, legendY + 2);

            legendY += 28;
        });
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const mouseX = (e.clientX - rect.left) * scaleX;
        const mouseY = (e.clientY - rect.top) * scaleY;

        let found = false;
        this.kernels.forEach((kernel, idx) => {
            const dx = mouseX - kernel.x;
            const dy = mouseY - kernel.y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 15) {
                this.hoveredKernel = idx;
                this.hoveredX = mouseX;
                this.hoveredY = mouseY;
                found = true;
                this.canvas.style.cursor = 'pointer';
            }
        });

        if (!found && this.hoveredKernel !== null) {
            this.hoveredKernel = null;
            this.canvas.style.cursor = 'default';
        }

        if (found || this.hoveredKernel === null) {
            this.draw();
        }
    }

    handleMouseOut() {
        if (this.hoveredKernel !== null) {
            this.hoveredKernel = null;
            this.canvas.style.cursor = 'default';
            this.draw();
        }
    }

    drawTooltip(ctx) {
        if (this.hoveredKernel === null) return;

        const kernel = this.kernels[this.hoveredKernel];
        const padding = 25;
        const lineHeight = 40;
        const tooltipWidth = 500;
        const tooltipHeight = 180;

        // Position tooltip near the mouse
        let tooltipX = this.hoveredX + 30;
        let tooltipY = this.hoveredY - tooltipHeight / 2;

        // Keep tooltip within canvas bounds
        if (tooltipX + tooltipWidth > this.canvas.width - 30) {
            tooltipX = this.hoveredX - tooltipWidth - 30;
        }
        if (tooltipY < 30) tooltipY = 30;
        if (tooltipY + tooltipHeight > this.canvas.height - 30) {
            tooltipY = this.canvas.height - tooltipHeight - 30;
        }

        // Draw tooltip background
        ctx.fillStyle = 'rgba(26, 26, 26, 0.95)';
        ctx.strokeStyle = kernel.color;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 12);
        ctx.fill();
        ctx.stroke();

        // Draw tooltip text
        ctx.fillStyle = '#e0e0e0';
        ctx.font = 'bold 28px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(kernel.name, tooltipX + padding, tooltipY + padding + 24);

        ctx.font = '24px Arial';
        ctx.fillText(`Performance: ${kernel.tflops.toFixed(2)} TFLOPS`, tooltipX + padding, tooltipY + padding + 24 + lineHeight);
        ctx.fillText(`Matrix Size: 4096×4096`, tooltipX + padding, tooltipY + padding + 24 + lineHeight * 2);
        ctx.fillText(`Arithmetic Intensity: ${kernel.arithmeticIntensity.toFixed(1)} FLOP/byte`, tooltipX + padding, tooltipY + padding + 24 + lineHeight * 3);
    }
}

// Export for use in HTML
if (typeof window !== 'undefined') {
    window.NaiveKernelViz = NaiveKernelViz;
    window.CoalescingViz = CoalescingViz;
    window.CoalescedMatrixViz = CoalescedMatrixViz;
    window.IndexTransformViz = IndexTransformViz;
    window.SharedMemoryViz = SharedMemoryViz;
    window.Tiling1DViz = Tiling1DViz;
    window.Tiling1DPipelineViz = Tiling1DPipelineViz;
    window.Tiling2DViz = Tiling2DViz;
    window.VectorizedViz = VectorizedViz;
    window.WarpTilingViz = WarpTilingViz;
    window.WmmaTensorcoreViz = WmmaTensorcoreViz;
    window.PerformanceComparison = PerformanceComparison;
    window.RooflineViz = RooflineViz;
}

// Auto-initialize all visualizations when DOM is ready
document.addEventListener('DOMContentLoaded', function () {
    if (document.getElementById('roofline-viz')) new RooflineViz('roofline-viz');
    if (document.getElementById('naive-viz')) new NaiveKernelViz('naive-viz');
    if (document.getElementById('coalesced-matrix-viz')) new CoalescedMatrixViz('coalesced-matrix-viz');
    if (document.getElementById('index-transform-viz')) new IndexTransformViz('index-transform-viz');
    if (document.getElementById('shared-memory-viz')) new SharedMemoryViz('shared-memory-viz');
    if (document.getElementById('1d-tiling-viz')) new Tiling1DViz('1d-tiling-viz');
    if (document.getElementById('1d-pipeline-viz')) new Tiling1DPipelineViz('1d-pipeline-viz');
    if (document.getElementById('2d-tiling-viz')) new Tiling2DViz('2d-tiling-viz');
    if (document.getElementById('vectorized-viz')) new VectorizedViz('vectorized-viz');
    if (document.getElementById('warp-tiling-viz')) new WarpTilingViz('warp-tiling-viz');
    if (document.getElementById('wmma-tensorcore-viz')) new WmmaTensorcoreViz('wmma-tensorcore-viz');
    if (document.getElementById('performance-comparison')) new PerformanceComparison('performance-comparison');
});
