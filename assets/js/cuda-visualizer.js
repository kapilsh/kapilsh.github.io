(function() {
  'use strict';

  class CUDAVisualizer {
    constructor(containerId) {
      this.container = document.getElementById(containerId);
      if (!this.container) {
        console.error(`Container ${containerId} not found`);
        return;
      }

      // State
      this.isPlaying = false;
      this.currentStep = 0;
      this.activeBlock = 0;
      this.activeWarp = 0;

      // Kernel configuration
      this.gridDim = { x: 2, y: 2 };
      this.blockDim = { x: 16, y: 8 };
      this.warpSize = 32;

      this.totalBlocks = this.gridDim.x * this.gridDim.y;
      this.threadsPerBlock = this.blockDim.x * this.blockDim.y;
      this.warpsPerBlock = this.threadsPerBlock / this.warpSize;
      this.totalThreads = this.totalBlocks * this.threadsPerBlock;

      // Kernel instructions
      this.instructions = [
        "int tid = blockIdx.x * blockDim.x + threadIdx.x;",
        "float a_val = a[tid];  // Load from global memory",
        "float b_val = b[tid];  // Load from global memory",
        "float result = a_val + b_val;  // Compute",
        "c[tid] = result;  // Store to global memory"
      ];

      this.totalStepsPerBlock = this.instructions.length * this.warpsPerBlock;
      this.totalSteps = this.totalStepsPerBlock * this.totalBlocks;

      this.timer = null;

      this.init();
    }

    init() {
      this.render();
      this.attachEventListeners();
    }

    getWarpColor(warpIdx, isActive, isComplete) {
      if (isComplete) return 'warp-complete';
      if (isActive) return 'warp-active';
      return 'warp-idle';
    }

    getBlockColor(blockIdx) {
      const colors = ['block-blue', 'block-purple', 'block-cyan', 'block-pink'];
      return colors[blockIdx % colors.length];
    }

    getBlockStatus(blockIdx) {
      if (this.currentStep === 0) return 'waiting';
      if (this.currentStep >= this.totalStepsPerBlock) return 'complete';
      return 'active';
    }

    handlePlay() {
      this.isPlaying = !this.isPlaying;
      if (this.isPlaying) {
        this.startAnimation();
      } else {
        this.stopAnimation();
      }
      this.updateControls();
    }

    handleReset() {
      this.isPlaying = false;
      this.currentStep = 0;
      this.activeBlock = 0;
      this.activeWarp = 0;
      this.stopAnimation();
      this.render();
    }

    startAnimation() {
      this.timer = setInterval(() => {
        this.currentStep++;
        this.activeWarp = Math.floor(this.currentStep / this.instructions.length);

        if (this.currentStep >= this.totalStepsPerBlock) {
          this.isPlaying = false;
          this.stopAnimation();
        }

        this.render();
      }, 1000);
    }

    stopAnimation() {
      if (this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    }

    updateControls() {
      const playBtn = this.container.querySelector('.play-btn');
      if (playBtn) {
        playBtn.innerHTML = this.isPlaying
          ? '<span class="icon">⏸</span> Pause'
          : '<span class="icon">▶</span> Play';
      }
    }

    render() {
      const currentInstruction = this.currentStep % this.instructions.length;

      this.container.innerHTML = `
        <style>
          .cuda-viz {
            background: #111827;
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          }
          .cuda-viz h1 { font-size: 1.875rem; margin-bottom: 0.5rem; font-weight: bold; }
          .cuda-viz h3 { font-weight: bold; margin-bottom: 0.75rem; font-size: 1.125rem; }
          .cuda-viz .kernel-info { color: #9ca3af; margin-bottom: 1rem; }
          .cuda-viz .kernel-code { font-family: monospace; color: #34d399; }
          .cuda-viz .control-panel { background: #1f2937; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1.5rem; }
          .cuda-viz .controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 1rem; }
          .cuda-viz button {
            padding: 0.5rem 1.5rem;
            border-radius: 0.375rem;
            border: none;
            cursor: pointer;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
          }
          .cuda-viz .play-btn { background: #2563eb; color: white; }
          .cuda-viz .play-btn:hover { background: #1d4ed8; }
          .cuda-viz .play-btn:disabled { background: #4b5563; cursor: not-allowed; }
          .cuda-viz .reset-btn { background: #374151; color: white; }
          .cuda-viz .reset-btn:hover { background: #4b5563; }
          .cuda-viz .step-info { font-size: 0.875rem; }
          .cuda-viz .step-info span { font-family: monospace; }
          .cuda-viz .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            padding-top: 1rem;
            border-top: 1px solid #374151;
          }
          .cuda-viz .stat-card { background: #374151; padding: 0.75rem; border-radius: 0.375rem; }
          .cuda-viz .stat-label { font-size: 0.75rem; color: #9ca3af; margin-bottom: 0.25rem; }
          .cuda-viz .stat-value { font-family: monospace; font-size: 1.125rem; }
          .cuda-viz .stat-formula { font-size: 0.75rem; color: #9ca3af; margin-top: 0.25rem; }
          .cuda-viz .grid-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
          }
          .cuda-viz .section { background: #1f2937; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; }
          .cuda-viz .blocks-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem; margin-bottom: 0.75rem; }
          .cuda-viz .block-card { padding: 1rem; text-align: center; border-radius: 0.5rem; transition: all 0.3s; }
          .cuda-viz .block-blue { background: #2563eb; }
          .cuda-viz .block-purple { background: #7c3aed; }
          .cuda-viz .block-cyan { background: #0891b2; }
          .cuda-viz .block-pink { background: #db2777; }
          .cuda-viz .block-complete { background: #059669; }
          .cuda-viz .block-waiting { background: #4b5563; }
          .cuda-viz .block-active { box-shadow: 0 0 0 2px #facc15; }
          .cuda-viz .block-title { font-weight: bold; font-size: 1.125rem; }
          .cuda-viz .block-coords { font-size: 0.75rem; margin-top: 0.25rem; opacity: 0.9; }
          .cuda-viz .block-status { font-size: 0.75rem; margin-top: 0.5rem; }
          .cuda-viz .warps-container { display: flex; gap: 0.5rem; }
          .cuda-viz .warp-group { flex: 1; }
          .cuda-viz .warp-label { font-size: 0.75rem; text-align: center; margin-bottom: 0.5rem; font-weight: 600; }
          .cuda-viz .threads-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.25rem; }
          .cuda-viz .thread-cell {
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 0.25rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.5rem;
            font-family: monospace;
            transition: all 0.3s;
          }
          .cuda-viz .warp-idle { background: #4b5563; }
          .cuda-viz .warp-active { background: #facc15; color: #111827; font-weight: bold; box-shadow: 0 0 0 2px white; }
          .cuda-viz .warp-complete { background: #10b981; }
          .cuda-viz .warp-status-list { display: flex; flex-direction: column; gap: 0.5rem; }
          .cuda-viz .warp-status-item {
            padding: 0.75rem;
            border-radius: 0.375rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
          }
          .cuda-viz .warp-status-idle { background: #374151; color: #9ca3af; }
          .cuda-viz .warp-status-active { background: #facc15; color: #111827; font-weight: bold; }
          .cuda-viz .warp-status-complete { background: #10b981; color: white; font-weight: 600; }
          .cuda-viz .execution-status { background: #1f2937; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem; }
          .cuda-viz .status-card { padding: 1rem; border-radius: 0.375rem; text-align: center; }
          .cuda-viz .status-ready { background: #374151; border: 2px solid #4b5563; }
          .cuda-viz .status-blocks-active { background: #581c87; border: 2px solid #7c3aed; margin-bottom: 1rem; }
          .cuda-viz .status-warp-active { background: #facc15; color: #111827; margin-bottom: 1rem; }
          .cuda-viz .status-instruction { background: #374151; border: 2px solid #10b981; margin-bottom: 1rem; }
          .cuda-viz .status-complete { background: #064e3b; border: 2px solid #10b981; }
          .cuda-viz .status-icon { font-size: 2rem; margin-bottom: 0.5rem; }
          .cuda-viz .status-title { font-weight: bold; font-size: 1.25rem; }
          .cuda-viz .status-subtitle { font-size: 0.875rem; margin-top: 0.5rem; }
          .cuda-viz .status-label { font-size: 0.875rem; margin-bottom: 0.25rem; opacity: 0.8; }
          .cuda-viz .instruction-list { display: flex; flex-direction: column; gap: 0.5rem; font-family: monospace; font-size: 0.875rem; }
          .cuda-viz .instruction-item { padding: 0.5rem; border-radius: 0.375rem; transition: all 0.3s; }
          .cuda-viz .instruction-idle { background: #374151; color: #9ca3af; }
          .cuda-viz .instruction-active { background: #059669; color: white; font-weight: bold; box-shadow: 0 0 0 2px #34d399; }
          .cuda-viz .info-box { font-size: 0.875rem; background: #1e3a8a; padding: 0.5rem; border-radius: 0.375rem; border: 1px solid #3b82f6; }
          .cuda-viz .info-box-alt { font-size: 0.875rem; background: #374151; padding: 0.75rem; border-radius: 0.375rem; }
          .cuda-viz .key-concepts { background: #1f2937; border-radius: 0.5rem; padding: 1rem; }
          .cuda-viz .concept-list { display: flex; flex-direction: column; gap: 0.75rem; font-size: 0.875rem; color: #d1d5db; }
          .cuda-viz .concept-title { color: white; font-weight: bold; }
          @media (max-width: 768px) {
            .cuda-viz .grid-container { grid-template-columns: 1fr; }
            .cuda-viz .stats-grid { grid-template-columns: repeat(2, 1fr); }
          }
        </style>

        <div class="cuda-viz">
          <h1>CUDA Grid & Warp Execution</h1>
          <div class="kernel-info">
            Kernel: <span class="kernel-code">addVectors&lt;&lt;&lt;dim3(2,2), dim3(16,8)&gt;&gt;&gt;(a, b, c)</span>
          </div>

          <div class="control-panel">
            <div class="controls">
              <button class="play-btn" ${this.currentStep >= this.totalSteps ? 'disabled' : ''}>
                <span class="icon">${this.isPlaying ? '⏸' : '▶'}</span>
                ${this.isPlaying ? 'Pause' : 'Play'}
              </button>
              <button class="reset-btn">
                <span class="icon">↻</span>
                Reset
              </button>
              <div class="step-info">
                <span style="color: #9ca3af;">Step:</span>
                <span style="color: white;">${this.currentStep}</span> /
                <span style="color: #6b7280;">${this.totalStepsPerBlock}</span>
              </div>
            </div>

            <div class="stats-grid">
              <div class="stat-card">
                <div class="stat-label">Total Blocks</div>
                <div class="stat-value">
                  ${this.gridDim.x} × ${this.gridDim.y} = <span style="color: #a78bfa;">${this.totalBlocks}</span>
                </div>
                <div class="stat-formula">gridDim.x × gridDim.y</div>
              </div>

              <div class="stat-card">
                <div class="stat-label">Threads per Block</div>
                <div class="stat-value">
                  ${this.blockDim.x} × ${this.blockDim.y} = <span style="color: #60a5fa;">${this.threadsPerBlock}</span>
                </div>
                <div class="stat-formula">blockDim.x × blockDim.y</div>
              </div>

              <div class="stat-card">
                <div class="stat-label">Total Threads</div>
                <div class="stat-value">
                  ${this.totalBlocks} × ${this.threadsPerBlock} = <span style="color: #22d3ee;">${this.totalThreads}</span>
                </div>
                <div class="stat-formula">blocks × threads/block</div>
              </div>

              <div class="stat-card">
                <div class="stat-label">Warps per Block</div>
                <div class="stat-value">
                  ${this.threadsPerBlock} ÷ ${this.warpSize} = <span style="color: #fbbf24;">${this.warpsPerBlock}</span>
                </div>
                <div class="stat-formula">threads ÷ warp size</div>
              </div>
            </div>
          </div>

          <div class="grid-container">
            <div>
              ${this.renderGridBlocks()}
              ${this.renderWarpVisualization()}
              ${this.renderWarpStatus()}
            </div>

            <div>
              ${this.renderExecutionStatus(currentInstruction)}
              ${this.renderInstructionList(currentInstruction)}
              ${this.renderKeyConcepts()}
            </div>
          </div>
        </div>
      `;

      this.attachEventListeners();
    }

    renderGridBlocks() {
      const blocksHtml = Array.from({ length: this.totalBlocks }).map((_, blockIdx) => {
        const status = this.getBlockStatus(blockIdx);
        const blockX = blockIdx % this.gridDim.x;
        const blockY = Math.floor(blockIdx / this.gridDim.x);

        let blockClass = '';
        if (status === 'complete') {
          blockClass = 'block-complete';
        } else if (status === 'active') {
          blockClass = this.getBlockColor(blockIdx) + ' block-active';
        } else {
          blockClass = 'block-waiting';
        }

        let statusText = '';
        if (status === 'complete') statusText = '✓ Complete';
        else if (status === 'active') statusText = '⚡ Running';
        else statusText = 'Ready';

        return `
          <div class="block-card ${blockClass}">
            <div class="block-title">Block ${blockIdx}</div>
            <div class="block-coords">(${blockX}, ${blockY})</div>
            <div class="block-status">${statusText}</div>
          </div>
        `;
      }).join('');

      return `
        <div class="section">
          <h3>Grid: ${this.totalBlocks} Blocks Executing in Parallel</h3>
          <div class="blocks-grid">${blocksHtml}</div>
          <div class="info-box">
            <strong>Real GPU behavior:</strong> All blocks execute simultaneously on different SMs. Each processes its own data chunk independently.
          </div>
        </div>
      `;
    }

    renderWarpVisualization() {
      const warpsHtml = Array.from({ length: this.warpsPerBlock }).map((_, warpIdx) => {
        const threadsHtml = Array.from({ length: this.warpSize }).map((_, threadInWarp) => {
          const threadIdx = warpIdx * this.warpSize + threadInWarp;
          const isActive = warpIdx === this.activeWarp && this.currentStep > 0 && this.currentStep <= this.totalStepsPerBlock;
          const isComplete = this.currentStep > 0 && this.currentStep >= (warpIdx + 1) * this.instructions.length;

          return `
            <div class="thread-cell ${this.getWarpColor(warpIdx, isActive, isComplete)}"
                 title="Thread ${threadIdx}">
              ${threadInWarp}
            </div>
          `;
        }).join('');

        return `
          <div class="warp-group">
            <div class="warp-label">Warp ${warpIdx}</div>
            <div class="threads-grid">${threadsHtml}</div>
          </div>
        `;
      }).join('');

      return `
        <div class="section">
          <h3>All Blocks: Warp Execution (showing Block 0)</h3>
          <div class="warps-container">${warpsHtml}</div>
          <div class="info-box-alt">
            Same warp pattern in all ${this.totalBlocks} blocks executing simultaneously. Block 0 threads: 0-127, Block 1: 128-255, etc.
          </div>
        </div>
      `;
    }

    renderWarpStatus() {
      const warpsHtml = Array.from({ length: this.warpsPerBlock }).map((_, warpIdx) => {
        const isActive = warpIdx === this.activeWarp && this.currentStep > 0 && this.currentStep <= this.totalStepsPerBlock;
        const isComplete = this.currentStep >= (warpIdx + 1) * this.instructions.length;

        let statusClass = 'warp-status-idle';
        let statusText = '';

        if (isActive) {
          statusClass = 'warp-status-active';
          statusText = '⚡ Executing';
        } else if (isComplete) {
          statusClass = 'warp-status-complete';
          statusText = '✓ Complete';
        }

        return `
          <div class="warp-status-item ${statusClass}">
            <span>Warp ${warpIdx} (Threads ${warpIdx * 32}-${(warpIdx + 1) * 32 - 1})</span>
            ${statusText ? `<span style="font-size: 0.875rem;">${statusText}</span>` : ''}
          </div>
        `;
      }).join('');

      return `
        <div class="section">
          <h3>Warp Execution (Same in All Blocks)</h3>
          <div class="warp-status-list">${warpsHtml}</div>
        </div>
      `;
    }

    renderExecutionStatus(currentInstruction) {
      let content = '';

      if (this.currentStep === 0) {
        content = `
          <div class="status-card status-ready">
            <div class="status-icon">▶️</div>
            <div class="status-title">Ready to Execute</div>
            <div class="status-subtitle">
              Press Play to start. All ${this.totalBlocks} blocks will run in parallel.
            </div>
          </div>
        `;
      } else if (this.currentStep > 0 && this.currentStep <= this.totalStepsPerBlock) {
        content = `
          <div class="status-card status-blocks-active">
            <div class="status-label">All Blocks Active</div>
            <div class="status-title">${this.totalBlocks} Blocks Running</div>
            <div class="status-subtitle">Each processing ${this.threadsPerBlock} array elements</div>
          </div>

          <div class="status-card status-warp-active">
            <div class="status-label">Current Warp (in all blocks)</div>
            <div class="status-title">Warp ${this.activeWarp}</div>
            <div class="status-subtitle">
              ${this.totalBlocks} × 32 = ${this.totalBlocks * 32} threads executing this instruction
            </div>
          </div>

          <div class="status-card status-instruction">
            <div class="status-label">Instruction ${currentInstruction + 1} of ${this.instructions.length}</div>
            <div style="font-family: monospace; font-size: 0.875rem; color: #34d399; line-height: 1.5;">
              ${this.instructions[currentInstruction]}
            </div>
          </div>

          <div class="info-box-alt">
            <strong>Parallel SIMT:</strong> Each block's Warp ${this.activeWarp} executes this instruction simultaneously across different SMs.
          </div>
        `;
      } else {
        content = `
          <div class="status-card status-complete">
            <div class="status-icon">✓</div>
            <div class="status-title">All Blocks Complete!</div>
            <div class="status-subtitle">
              ${this.totalBlocks} blocks executed in parallel
            </div>
            <div class="status-subtitle">
              Total: ${this.totalThreads} threads processed ${this.totalThreads} array elements
            </div>
          </div>
        `;
      }

      return `
        <div class="execution-status">
          <h3>Current Execution</h3>
          ${content}
        </div>
      `;
    }

    renderInstructionList(currentInstruction) {
      const instructionsHtml = this.instructions.map((instr, idx) => {
        const isActive = idx === currentInstruction && this.currentStep > 0 && this.currentStep <= this.totalSteps;
        return `
          <div class="instruction-item ${isActive ? 'instruction-active' : 'instruction-idle'}">
            ${idx + 1}. ${instr}
          </div>
        `;
      }).join('');

      return `
        <div class="section">
          <h3>Kernel Instructions</h3>
          <div class="instruction-list">${instructionsHtml}</div>
        </div>
      `;
    }

    renderKeyConcepts() {
      return `
        <div class="key-concepts">
          <h3>Key Concepts</h3>
          <div class="concept-list">
            <div>
              <span class="concept-title">📦 Grid:</span> Launches ${this.totalBlocks} blocks. Each block processes a chunk of data independently.
            </div>
            <div>
              <span class="concept-title">🔄 Blocks:</span> Can execute in any order or in parallel on different SMs (Streaming Multiprocessors).
            </div>
            <div>
              <span class="concept-title">⚡ Warps:</span> Each block's threads are organized into warps of 32. Warps are the unit of execution.
            </div>
            <div>
              <span class="concept-title">🎯 Scalability:</span> More blocks = better GPU utilization across multiple SMs.
            </div>
          </div>
        </div>
      `;
    }

    attachEventListeners() {
      const playBtn = this.container.querySelector('.play-btn');
      const resetBtn = this.container.querySelector('.reset-btn');

      if (playBtn) {
        playBtn.addEventListener('click', () => this.handlePlay());
      }

      if (resetBtn) {
        resetBtn.addEventListener('click', () => this.handleReset());
      }
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      new CUDAVisualizer('cuda-visualizer');
    });
  } else {
    new CUDAVisualizer('cuda-visualizer');
  }
})();
