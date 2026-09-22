---
layout: wide-page
title: MXFP4 Visualizer
icon: fas fa-calculator
order: 3
---

<link rel="stylesheet" href="{{ '/assets/css/viz-panel.css' | relative_url }}">

<style>
/* Widget-specific layout only. Palette, cards, buttons, inputs, tiles and chips
   all come from assets/css/viz-panel.css, shared with the RoPE visualizer and
   matching the standalone apps. */

/* One card per row, full width. The steps here are a sequence -- values in,
   bits out, what it cost, how it got there -- so reading top to bottom follows
   the argument, where two columns asked you to zig-zag. It also gives the
   sixteen-wide rows the room to lay out honestly. */
#mxfp4-visualizer-container .vz-grid-2 { grid-template-columns: 1fr; }

/* Column counts are set by what has to fit: a signed decimal, a 16-character
   bit string, and a 4-character one are three different widths. */
#mxfp4-visualizer-container .input-grid,
#mxfp4-visualizer-container .bits-grid {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 6px;
}

/* All sixteen codes on one line: the block is the unit, and breaking it across
   rows invites reading it as two blocks of eight. */
#mxfp4-visualizer-container .code-grid {
    display: grid;
    grid-template-columns: repeat(16, 1fr);
    gap: 5px;
}

@media (max-width: 900px) {
    #mxfp4-visualizer-container .input-grid,
    #mxfp4-visualizer-container .bits-grid { grid-template-columns: repeat(4, 1fr); }
    #mxfp4-visualizer-container .code-grid { grid-template-columns: repeat(8, 1fr); }
}

@media (max-width: 560px) {
    #mxfp4-visualizer-container .input-grid,
    #mxfp4-visualizer-container .bits-grid { grid-template-columns: repeat(2, 1fr); }
    #mxfp4-visualizer-container .code-grid { grid-template-columns: repeat(4, 1fr); }
}

#mxfp4-visualizer-container .buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 12px 0;
}

/* The E2M1 codebook: sixteen values, the active ones lit. */
#mxfp4-visualizer-container .lut-grid {
    display: grid;
    grid-template-columns: repeat(16, 1fr);
    gap: 5px;
}

@media (max-width: 900px) {
    #mxfp4-visualizer-container .lut-grid { grid-template-columns: repeat(8, 1fr); }
}

@media (max-width: 560px) {
    #mxfp4-visualizer-container .lut-grid { grid-template-columns: repeat(4, 1fr); }
}

#mxfp4-visualizer-container .lut-cell {
    background: var(--vz-elevated);
    border: 1px solid var(--vz-border);
    border-radius: 7px;
    padding: 8px 4px;
    text-align: center;
    color: var(--vz-dim);
    transition: background .15s, border-color .15s, color .15s;
}

#mxfp4-visualizer-container .lut-cell:hover {
    background: var(--vz-card-hover);
    border-color: var(--vz-border-strong);
}

/* A codeword this input actually maps to. */
#mxfp4-visualizer-container .lut-cell.selected {
    background: rgba(118, 185, 0, .14);
    border-color: var(--vz-green);
    color: var(--vz-green);
}

/* One value's trip through the quantizer: divide by scale, round, look up. */
#mxfp4-visualizer-container .quantization-flow {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(104px, 1fr));
    gap: 8px;
}

#mxfp4-visualizer-container .flow-item {
    background: var(--vz-elevated);
    border: 1px solid var(--vz-border);
    border-radius: 8px;
    padding: 9px 6px;
    text-align: center;
    font-variant-numeric: tabular-nums;
}

#mxfp4-visualizer-container .flow-value { font-weight: 650; font-size: 13px; }
#mxfp4-visualizer-container .flow-arrow { color: var(--vz-faint); font-size: 11px; margin: 3px 0; }
#mxfp4-visualizer-container .flow-result { color: var(--vz-green); font-weight: 650; }

#mxfp4-visualizer-container .error-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
    gap: 8px;
}

#mxfp4-visualizer-container .error-item {
    background: var(--vz-elevated);
    border: 1px solid var(--vz-border);
    border-radius: 8px;
    padding: 9px;
    text-align: center;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}

#mxfp4-visualizer-container .scale-info {
    background: var(--vz-elevated);
    border: 1px solid var(--vz-border);
    border-radius: 8px;
    padding: 10px 12px;
    color: var(--vz-dim);
    font-size: 12.5px;
    margin-bottom: 12px;
}
</style>

<div id="mxfp4-visualizer-container" class="viz-panel-root">
    <div class="vz-head">
        <p>
            Sixteen BF16 numbers share one power-of-two scale, then each is squeezed into four bits:
            one sign, two exponent, one mantissa. Change any input to watch what survives the trip
            and what gets rounded away.
        </p>
    </div>

    <div class="vz-grid-2">
        <div class="vz-card">
            <h3>Input</h3>
            <p class="vz-sub">Sixteen BF16 values — one MXFP4 block.</p>
            <div class="input-grid">
                <input type="number" id="val0" step="0.1" value="250.5">
                <input type="number" id="val1" step="0.1" value="-125.25">
                <input type="number" id="val2" step="0.1" value="75.75">
                <input type="number" id="val3" step="0.1" value="400.0">
                <input type="number" id="val4" step="0.1" value="120.2">
                <input type="number" id="val5" step="0.1" value="-80.8">
                <input type="number" id="val6" step="0.1" value="310.1">
                <input type="number" id="val7" step="0.1" value="-230.3">
                <input type="number" id="val8" step="0.1" value="90.9">
                <input type="number" id="val9" step="0.1" value="520.2">
                <input type="number" id="val10" step="0.1" value="-170.7">
                <input type="number" id="val11" step="0.1" value="280.8">
                <input type="number" id="val12" step="0.1" value="-40.4">
                <input type="number" id="val13" step="0.1" value="160.6">
                <input type="number" id="val14" step="0.1" value="-350.5">
                <input type="number" id="val15" step="0.1" value="470.7">
            </div>
            <div class="buttons">
                <button class="vz-btn primary" id="quantizeBtn">Quantize</button>
                <button class="vz-btn" id="randomBtn">Random</button>
                <button class="vz-btn" id="presetBtn">Preset</button>
            </div>
            <p class="vz-note">
                Block size <span id="vecSize" class="num">16</span> values. The scale is chosen from
                the block maximum, so one outlier costs every other value precision.
            </p>
        </div>

        <div class="vz-card">
            <h3>Bits</h3>
            <p class="vz-sub">What goes in, and what comes out.</p>
            <label>BF16 in — 16 bits each</label>
            <div class="bits-grid" style="margin-bottom: 14px;">
                <div class="vz-chip" id="bf16_0">0100000010100000</div>
                <div class="vz-chip" id="bf16_1">1011111110100000</div>
                <div class="vz-chip" id="bf16_2">0011111101100000</div>
                <div class="vz-chip" id="bf16_3">0100000010000000</div>
                <div class="vz-chip" id="bf16_4">0100000010100000</div>
                <div class="vz-chip" id="bf16_5">1011111110100000</div>
                <div class="vz-chip" id="bf16_6">0011111101100000</div>
                <div class="vz-chip" id="bf16_7">0100000010000000</div>
                <div class="vz-chip" id="bf16_8">0100000010100000</div>
                <div class="vz-chip" id="bf16_9">1011111110100000</div>
                <div class="vz-chip" id="bf16_10">0011111101100000</div>
                <div class="vz-chip" id="bf16_11">0100000010000000</div>
                <div class="vz-chip" id="bf16_12">0100000010100000</div>
                <div class="vz-chip" id="bf16_13">1011111110100000</div>
                <div class="vz-chip" id="bf16_14">0011111101100000</div>
                <div class="vz-chip" id="bf16_15">0100000010000000</div>
            </div>
            <label>MXFP4 out — 4 bits each</label>
            <div class="code-grid">
                <div class="vz-chip on" id="code0">0100</div>
                <div class="vz-chip on" id="code1">1010</div>
                <div class="vz-chip on" id="code2">0010</div>
                <div class="vz-chip on" id="code3">0110</div>
                <div class="vz-chip on" id="code4">0011</div>
                <div class="vz-chip on" id="code5">1001</div>
                <div class="vz-chip on" id="code6">0101</div>
                <div class="vz-chip on" id="code7">1100</div>
                <div class="vz-chip on" id="code8">0010</div>
                <div class="vz-chip on" id="code9">0110</div>
                <div class="vz-chip on" id="code10">1011</div>
                <div class="vz-chip on" id="code11">0101</div>
                <div class="vz-chip on" id="code12">1001</div>
                <div class="vz-chip on" id="code13">0011</div>
                <div class="vz-chip on" id="code14">1110</div>
                <div class="vz-chip on" id="code15">0110</div>
            </div>
            <p class="vz-note">
                Shared scale factor <b id="scaleValue" class="num">1.0</b>, stored once for the block.
            </p>
        </div>
    </div>

    <div class="vz-card">
        <h3>What it costs</h3>
        <p class="vz-sub">Memory saved, and the error you pay for it.</p>
        <div class="vz-tiles">
            <div class="vz-tile">
                <div class="k">Original</div>
                <div class="v" id="originalSize">256 bits</div>
                <div class="n">16 values, BF16</div>
            </div>
            <div class="vz-tile">
                <div class="k">Compressed</div>
                <div class="v" id="mxfp4Size">96 bits</div>
                <div class="n">16 codes plus one scale</div>
            </div>
            <div class="vz-tile good">
                <div class="k">Ratio</div>
                <div class="v"><span id="compressionRatio">2.67</span>:1</div>
                <div class="n"><span id="savedPercent">62.5</span>% smaller</div>
            </div>
            <div class="vz-tile">
                <div class="k">Saved</div>
                <div class="v" id="memorySaved">160 bits</div>
                <div class="n"><span id="compressedBits">96</span> kept of <span id="originalBits">256</span></div>
            </div>
            <div class="vz-tile warn">
                <div class="k">Max error</div>
                <div class="v" id="maxErrorValue">0.500</div>
                <div class="n">MSE <span id="mseValue">0.0938</span> · MAE <span id="maeValue">0.250</span></div>
            </div>
        </div>
    </div>

    <div class="vz-grid-2">
        <div class="vz-card">
            <h4>E2M1 codebook</h4>
            <p class="vz-sub">All sixteen representable values. Lit cells are the ones this block uses.</p>
            <div class="lut-grid" id="lutGrid"></div>
        </div>

        <div class="vz-card">
            <h4>Quantization</h4>
            <p class="vz-sub">Each value divided by the scale, rounded to the nearest codeword.</p>
            <div class="scale-info" id="scaleInfo">
                Scale Factor: 1.0 (Max value: 4.0 → MXFP4 range: ±6.0)
            </div>
            <div class="quantization-flow" id="quantFlow"></div>
        </div>
    </div>

    <div class="vz-card">
        <h4>Per-value error</h4>
        <p class="vz-sub">Original against reconstruction, value by value.</p>
        <div class="error-grid" id="errorGrid"></div>
    </div>
</div>

<script src="{{ '/assets/js/mxfp4-visualizer.js' | relative_url }}"></script>