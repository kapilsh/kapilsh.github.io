---
# the default layout is 'page'
title: Triton IR Visualizer
icon: fas fa-project-diagram
order: 5
---

<style>
#triton-ir-visualizer-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
}

#triton-ir-visualizer-container * {
    box-sizing: border-box;
}

#triton-ir-visualizer-container .container {
    max-width: 1600px;
    margin: 0 auto;
    background: white;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
}

@media (max-width: 768px) {
    #triton-ir-visualizer-container .container {
        padding: 20px;
        border-radius: 15px;
        margin: 0 5px;
    }
}

@media (max-width: 500px) {
    #triton-ir-visualizer-container .container {
        padding: 15px;
        border-radius: 10px;
    }
}

#triton-ir-visualizer-container .header {
    text-align: center;
    margin-bottom: 30px;
}

#triton-ir-visualizer-container .header h1 {
    font-size: 2.5em;
    font-weight: 700;
    margin-bottom: 10px;
    color: #2c3e50;
}

#triton-ir-visualizer-container .header p {
    font-size: 1.2em;
    color: #6c757d;
    margin-bottom: 5px;
}

#triton-ir-visualizer-container .subtitle {
    font-size: 1em;
    color: #adb5bd;
    font-style: italic;
}

#triton-ir-visualizer-container .info-box {
    background: #e7f3ff;
    padding: 15px;
    border-left: 4px solid #007bff;
    margin-bottom: 20px;
    border-radius: 4px;
}

#triton-ir-visualizer-container .info-box h3 {
    margin-top: 0;
    color: #0056b3;
    font-size: 1.2em;
}

#triton-ir-visualizer-container .info-box ol {
    margin: 10px 0 10px 20px;
}

#triton-ir-visualizer-container .info-box ul {
    margin: 5px 0 5px 20px;
}

#triton-ir-visualizer-container .input-section {
    margin-top: 20px;
    background: #f8f9fa;
    padding: 20px;
    border-radius: 8px;
    border: 2px solid #e9ecef;
}

#triton-ir-visualizer-container .input-section h2 {
    font-size: 1.3em;
    color: #2c3e50;
    margin-bottom: 15px;
}

#triton-ir-visualizer-container textarea {
    width: 100%;
    height: 300px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    resize: vertical;
}

#triton-ir-visualizer-container .visualize-btn {
    width: 100%;
    padding: 12px;
    margin-top: 10px;
    background: #28a745;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
}

#triton-ir-visualizer-container .visualize-btn:hover {
    background: #218838;
}

#triton-ir-visualizer-container .output-section {
    background: white;
    padding: 0;
    border-radius: 8px;
    border: 2px solid #e9ecef;
    min-height: 600px;
    overflow: auto;
    max-height: 800px;
}

#triton-ir-visualizer-container .legend {
    margin-bottom: 20px;
    padding: 15px;
    background: white;
    border-radius: 8px;
}

#triton-ir-visualizer-container .legend h3 {
    margin-top: 0;
    color: #333;
    font-size: 1.1em;
}

#triton-ir-visualizer-container .legend-items {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
}

#triton-ir-visualizer-container .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9em;
}

#triton-ir-visualizer-container .legend-color {
    width: 20px;
    height: 20px;
    border-radius: 3px;
    border: 1px solid #999;
    flex-shrink: 0;
}

#triton-ir-visualizer-container #controls-container {
    margin-top: 20px;
}

#triton-ir-visualizer-container .ir-controls {
    margin: 20px 0;
    padding: 15px;
    background: #f5f5f5;
    border-radius: 8px;
}

#triton-ir-visualizer-container .ir-controls .step-info {
    margin-bottom: 10px;
    font-weight: bold;
    color: #2c3e50;
}

#triton-ir-visualizer-container .ir-controls button {
    padding: 8px 16px;
    cursor: pointer;
    border: none;
    background: #007bff;
    color: white;
    border-radius: 4px;
    font-size: 14px;
}

#triton-ir-visualizer-container .ir-controls button:hover {
    background: #0056b3;
}

#triton-ir-visualizer-container .ir-controls input[type="range"] {
    vertical-align: middle;
    margin-left: 10px;
}

#triton-ir-visualizer-container svg {
    border: 1px solid #ddd;
    background: white;
    border-radius: 8px;
}

#triton-ir-visualizer-container .node {
    cursor: pointer;
}

#triton-ir-visualizer-container .info-panel {
    background: #f8f9fa;
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #e9ecef;
}

#triton-ir-visualizer-container .info-panel h4 {
    color: #2c3e50;
    margin-bottom: 15px;
    font-weight: 600;
}

#triton-ir-visualizer-container .info-panel p {
    line-height: 1.6;
    color: #5a6c7d;
    margin-bottom: 10px;
}
</style>

<div id="triton-ir-visualizer-container">
    <div class="container">
        <div class="header">
            <h1>🔬 Triton GPU IR Visualizer</h1>
            <p>Interactive Step-by-Step SSA Instruction Flow</p>
            <div class="subtitle">Visualize how Triton GPU kernels process data through SSA instructions</div>
        </div>

        <div class="info-box">
            <h3>How to use:</h3>
            <ol>
                <li>Paste your Triton IR code in the text area below (or use the sample provided)</li>
                <li>Click "Visualize IR" to generate the data flow graph</li>
                <li>Use the controls to step through the execution:
                    <ul>
                        <li><strong>Next/Previous:</strong> Step through instructions one at a time</li>
                        <li><strong>Reset:</strong> Go back to the beginning</li>
                        <li><strong>Play All:</strong> Automatically step through all instructions</li>
                        <li><strong>Speed slider:</strong> Adjust animation speed for auto-play</li>
                    </ul>
                </li>
                <li>Hover over nodes to see the full instruction text</li>
                <li>Nodes are color-coded by operation type and arranged hierarchically by dependencies</li>
            </ol>
        </div>

        <div class="input-section">
            <h2>Input: Triton IR Code</h2>
            <textarea id="ir-input" placeholder="Paste your Triton IR here...">
</textarea>
            <button class="visualize-btn" onclick="visualizeTritonIR()">Visualize IR</button>

            <div class="legend">
                <h3>Operation Types:</h3>
                <div class="legend-items">
                    <div class="legend-item">
                        <div class="legend-color" style="background: #c8e6c9;"></div>
                        <span>Input</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #e3f2fd;"></div>
                        <span>Constant</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #fff9c4;"></div>
                        <span>Range</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #b3e5fc;"></div>
                        <span>Arithmetic</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #dcedc8;"></div>
                        <span>Pointer</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffe0b2;"></div>
                        <span>Broadcast</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffab91;"></div>
                        <span>Compare</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ffccbc;"></div>
                        <span>Load</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #ce93d8;"></div>
                        <span>Reshape</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f48fb1;"></div>
                        <span>Reduce</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #f8bbd0;"></div>
                        <span>Store</span>
                    </div>
                </div>
            </div>

            <div id="controls-container"></div>
        </div>

        <div class="output-section">
            <div id="visualization"></div>
        </div>

        <div class="info-panel">
            <h4>ℹ️ About Triton IR</h4>
            <p><strong>Triton IR (Intermediate Representation)</strong> is a low-level representation of GPU kernels written in Triton. It uses Static Single Assignment (SSA) form where each variable is assigned exactly once, making data flow explicit and easier to analyze.</p>
            <p><strong>Key Concepts:</strong> Each instruction produces a result (like %0, %1, etc.) that can be used by subsequent instructions. The visualization shows these dependencies as arrows, helping you understand how data flows through the computation.</p>
            <p><strong>Use Cases:</strong> Understanding Triton IR is essential for debugging GPU kernels, optimizing performance, and learning how high-level Triton code translates to GPU operations.</p>
        </div>
    </div>
</div>

<script src="{{ '/assets/js/triton-ir-visualizer.js' | relative_url }}"></script>
<script>
let tritonVisualizer = null;

const sampleIR = `#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#loc = loc("/home/user/triton/naive.py":28:0)
#loc1 = loc(unknown)
#loc16 = loc("/home/user/triton/naive.py":56:17)
#loc22 = loc(callsite(#loc15 at #loc16))
#loc23 = loc(callsite(#loc1 at #loc16))
#loc24 = loc(callsite(#loc17 at #loc22))
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_naive(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %cst = arith.constant dense<512> : tensor<512xi32, #blocked> loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32, #blocked> loc(#loc1)
    %c512_i32 = arith.constant 512 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = tt.get_program_id y : i32 loc(#loc3)
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked> loc(#loc4)
    %3 = arith.muli %0, %c512_i32 : i32 loc(#loc5)
    %4 = tt.addptr %arg0, %3 : !tt.ptr<f32>, i32 loc(#loc6)
    %5 = tt.splat %4 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked> loc(#loc7)
    %6 = tt.addptr %5, %2 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked> loc(#loc7)
    %7 = arith.muli %2, %cst : tensor<512xi32, #blocked> loc(#loc8)
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked> loc(#loc9)
    %9 = tt.addptr %8, %7 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked> loc(#loc9)
    %10 = tt.splat %1 : i32 -> tensor<512xi32, #blocked> loc(#loc10)
    %11 = tt.addptr %9, %10 : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked> loc(#loc10)
    %12 = arith.cmpi slt, %2, %cst : tensor<512xi32, #blocked> loc(#loc11)
    %13 = tt.load %6, %12, %cst_0 : tensor<512x!tt.ptr<f32>, #blocked> loc(#loc12)
    %14 = tt.load %11, %12, %cst_0 : tensor<512x!tt.ptr<f32>, #blocked> loc(#loc13)
    %15 = arith.mulf %13, %14 : tensor<512xf32, #blocked> loc(#loc14)
    %16 = tt.reshape %15 allow_reorder : tensor<512xf32, #blocked> -> tensor<512xf32, #blocked1> loc(#loc22)
    %17 = "tt.reduce"(%16) <{axis = 0 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %20 = arith.addf %arg3, %arg4 : f32 loc(#loc24)
      tt.reduce.return %20 : f32 loc(#loc22)
    }) : (tensor<512xf32, #blocked1>) -> f32 loc(#loc22)
    %18 = tt.addptr %arg2, %3 : !tt.ptr<f32>, i32 loc(#loc18)
    %19 = tt.addptr %18, %1 : !tt.ptr<f32>, i32 loc(#loc19)
    tt.store %19, %17 : !tt.ptr<f32> loc(#loc20)
    tt.return loc(#loc21)
  } loc(#loc)
} loc(#loc)
#loc2 = loc("/home/user/triton/naive.py":39:26)
#loc3 = loc("/home/user/triton/naive.py":40:26)
#loc4 = loc("/home/user/triton/naive.py":44:26)
#loc5 = loc("/home/user/triton/naive.py":49:29)
#loc6 = loc("/home/user/triton/naive.py":49:21)
#loc7 = loc("/home/user/triton/naive.py":49:33)
#loc8 = loc("/home/user/triton/naive.py":50:30)
#loc9 = loc("/home/user/triton/naive.py":50:21)
#loc10 = loc("/home/user/triton/naive.py":50:34)
#loc11 = loc("/home/user/triton/naive.py":52:38)
#loc12 = loc("/home/user/triton/naive.py":52:16)
#loc13 = loc("/home/user/triton/naive.py":53:16)
#loc14 = loc("/home/user/triton/naive.py":56:36)
#loc15 = loc("/home/user/triton/standard.py":290:36)
#loc17 = loc("/home/user/triton/standard.py":260:15)
#loc18 = loc("/home/user/triton/naive.py":59:27)
#loc19 = loc("/home/user/triton/naive.py":59:39)
#loc20 = loc("/home/user/triton/naive.py":60:27)
#loc21 = loc("/home/user/triton/naive.py":60:4)
#loc22 = loc(callsite(#loc15 at #loc16))
#loc24 = loc(callsite(#loc17 at #loc22))`;

function visualizeTritonIR() {
    const irText = document.getElementById('ir-input').value;
    if (!irText.trim()) {
        alert('Please enter Triton IR code');
        return;
    }

    const vizContainer = document.getElementById('visualization');
    vizContainer.innerHTML = '';

    tritonVisualizer = initTritonIRVisualizer('visualization', 'controls-container', irText);
}

// Auto-load sample and visualize on page load
window.addEventListener('load', () => {
    document.getElementById('ir-input').value = sampleIR;
    visualizeTritonIR();
});
</script>
