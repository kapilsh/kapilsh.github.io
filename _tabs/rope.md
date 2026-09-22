---
layout: wide-page
title: RoPE Visualizer
icon: fas fa-compass
order: 4
---

<link rel="stylesheet" href="{{ '/assets/css/viz-panel.css' | relative_url }}">

<style>
/* Widget-specific layout only. Palette, cards, controls, tiles and notes come
   from assets/css/viz-panel.css, shared with the MXFP4 visualizer and matching
   the standalone apps. */

#rope-visualizer-container .control-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
}

/* A slider and its read-out belong on one line; the number next to it is the
   point of moving the slider. */
#rope-visualizer-container .control-group .row {
    display: flex;
    align-items: center;
    gap: 10px;
}

#rope-visualizer-container .control-group .val {
    color: var(--vz-green);
    font-family: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
    min-width: 4.5em;
    text-align: right;
}

#rope-visualizer-container .visualization-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 14px;
    align-items: start;
}

/* Plotly draws its own background; these keep its canvas from poking out of
   the rounded card corners. */
#rope-visualizer-container .js-plotly-plot,
#rope-visualizer-container .plot-container {
    border-radius: 8px;
    overflow: hidden;
}

#rope-visualizer-container .info-panel p {
    margin: 0 0 9px;
    color: var(--vz-dim);
    font-size: 13px;
}

#rope-visualizer-container .info-panel p:last-child { margin-bottom: 0; }
#rope-visualizer-container .info-panel strong { color: var(--vz-text); }
</style>

<div id="rope-visualizer-container" class="viz-panel-root">
    <div class="vz-head">
        <p>
            Rotary embeddings turn a token's position into a rotation, one angle per dimension pair.
            YaRN stretches those angles so a model trained on a short context can read a long one.
            Move the scaling factor to see which frequencies it leaves alone and which it bends.
        </p>
    </div>

    <div class="vz-card">
        <h3>Configuration</h3>
        <p class="vz-sub">Everything below recomputes as you change these.</p>
        <div class="control-grid">
            <div class="control-group">
                <label for="scalingFactor">YaRN scaling factor</label>
                <div class="row">
                    <input type="range" id="scalingFactor" min="1" max="8" step="0.5" value="4">
                    <span class="val" id="scalingFactorValue">4.0×</span>
                </div>
            </div>

            <div class="control-group">
                <label for="headDim">Head dimension</label>
                <select id="headDim">
                    <option value="32">32</option>
                    <option value="64" selected>64</option>
                    <option value="128">128</option>
                </select>
            </div>

            <div class="control-group">
                <label for="seqLength">Sequence length</label>
                <div class="row">
                    <input type="range" id="seqLength" min="256" max="8192" step="256" value="2048">
                    <span class="val" id="seqLengthValue">2048</span>
                </div>
            </div>

            <div class="control-group">
                <label for="baseFreq">Base frequency</label>
                <input type="number" id="baseFreq" value="10000" step="1000">
            </div>

            <div class="control-group">
                <label for="initialContext">Initial context length</label>
                <input type="number" id="initialContext" value="2048" step="256">
            </div>
        </div>
    </div>

    <div class="vz-card">
        <h3>What that buys you</h3>
        <p class="vz-sub">The wavelength range is the span of positions the embedding can still tell apart.</p>
        <div class="vz-tiles">
            <div class="vz-tile good">
                <div class="k">Effective context</div>
                <div class="v" id="effectiveContext">8192</div>
                <div class="n">tokens</div>
            </div>
            <div class="vz-tile">
                <div class="k">Frequency pairs</div>
                <div class="v" id="freqPairs">32</div>
                <div class="n">head dim ÷ 2</div>
            </div>
            <div class="vz-tile">
                <div class="k">Max wavelength</div>
                <div class="v" id="maxWavelength">65536</div>
                <div class="n">slowest dimension</div>
            </div>
            <div class="vz-tile">
                <div class="k">Min wavelength</div>
                <div class="v" id="minWavelength">2</div>
                <div class="n">fastest dimension</div>
            </div>
        </div>
    </div>

    <div class="visualization-grid">
        <div class="vz-card">
            <h4>Frequency analysis</h4>
            <p class="vz-sub">Inverse frequency per dimension pair, before and after scaling.</p>
            <div id="frequencyPlot"></div>
        </div>

        <div class="vz-card">
            <h4>Wavelength comparison</h4>
            <p class="vz-sub">How far a position can travel before the angle repeats.</p>
            <div id="wavelengthPlot"></div>
        </div>

        <div class="vz-card">
            <h4>Frequency scaling ratio</h4>
            <p class="vz-sub">Scaled over original, per dimension. Flat means untouched.</p>
            <div id="ratioPlot"></div>
        </div>

        <div class="vz-card">
            <h4>YaRN ramp</h4>
            <p class="vz-sub">The interpolation that decides which dimensions get stretched.</p>
            <div id="rampPlot"></div>
        </div>
    </div>

    <div class="vz-card heatmap-container">
        <h4>Embeddings heatmap</h4>
        <p class="vz-sub">Position against dimension, coloured by the rotation applied.</p>
        <div id="heatmapPlot"></div>
    </div>

    <div class="vz-card info-panel">
        <h4>About RoPE and YaRN</h4>
        <p><strong>Rotary Position Embedding</strong> encodes position by rotating query and key vectors, with a rotation angle that depends on both the position and the dimension pair.</p>
        <p><strong>YaRN</strong> extends the usable context by scaling frequency components unevenly: high-frequency dimensions, which carry local ordering, are left alone, while low-frequency ones are interpolated so distant positions stay distinguishable.</p>
        <p><strong>Why it matters:</strong> the model keeps its behaviour on short sequences while becoming usable on ones far longer than it was trained on.</p>
    </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.26.0/plotly.min.js"></script>
<script src="{{ '/assets/js/rope-visualizer.js' | relative_url }}"></script>