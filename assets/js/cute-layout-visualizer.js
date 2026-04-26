// cute-layout-visualizer.js
// All styling injected from JS. Markdown only has bare <div id="..."> targets.

(function () {
  // ── exact CuTeVis palette ─────────────────────────────────────────────────
  const BG        = '#0a0c10';
  const SURFACE   = '#0f1219';
  const PANEL     = '#141820';
  const BORDER    = '#1e2535';
  const BORDER_BR = '#2a3550';
  const TEXT_DIM  = '#5a6a88';
  const TEXT_MUT  = '#3a4a66';
  const ACCENT    = '#4fc3f7';
  const ACCENT2   = '#7c83f0';

  const COLORS = [
    { bg: '#1a3a5c', border: '#2a5a8c', text: '#4fc3f7' },
    { bg: '#1a4a3a', border: '#2a7a5a', text: '#66bb6a' },
    { bg: '#3a3a1a', border: '#6a6a2a', text: '#ffb74d' },
    { bg: '#3a1a4a', border: '#6a2a8a', text: '#ce93d8' },
    { bg: '#4a1a1a', border: '#8a2a2a', text: '#ef9a9a' },
    { bg: '#1a4a4a', border: '#2a8a8a', text: '#80cbc4' },
    { bg: '#2a4a1a', border: '#4a8a2a', text: '#a5d6a7' },
    { bg: '#4a2a1a', border: '#8a4a2a', text: '#ffcc80' },
  ];

  // Four shades of blue for the Mode 2 (k) dimension — used in heatmap slice labels
  const BLUE_SHADES = [
    { bg: '#0d2236', border: '#183a5c', text: '#4fc3f7' },
    { bg: '#163450', border: '#245a8a', text: '#72d0f8' },
    { bg: '#1e4870', border: '#3070b0', text: '#96dcfa' },
    { bg: '#285e90', border: '#3c88d8', text: '#bae8fc' },
  ];

  // 2-D palette for 3-D grid: [colorFamily][shade]
  // colorFamily = col % 4 (Mode 1), shade = k % 4 (Mode 2)
  const PALETTE_3D = [
    // Blue (col 0)
    [
      { bg: '#0d1f33', border: '#163352', text: '#4fc3f7' },
      { bg: '#163450', border: '#245a8a', text: '#72d0f8' },
      { bg: '#1e4870', border: '#3070b0', text: '#96dcfa' },
      { bg: '#285e90', border: '#3c88d8', text: '#bae8fc' },
    ],
    // Green (col 1)
    [
      { bg: '#0d2e1a', border: '#165228', text: '#66bb6a' },
      { bg: '#163d25', border: '#246a3c', text: '#81c784' },
      { bg: '#1e5230', border: '#308a52', text: '#a5d6a7' },
      { bg: '#286640', border: '#3cac68', text: '#c8e6c9' },
    ],
    // Amber (col 2)
    [
      { bg: '#2e2200', border: '#524000', text: '#ffb74d' },
      { bg: '#3d2e00', border: '#6a5200', text: '#ffc77a' },
      { bg: '#524000', border: '#8a6c00', text: '#ffd98a' },
      { bg: '#664e00', border: '#ac8400', text: '#ffe5a8' },
    ],
    // Purple (col 3)
    [
      { bg: '#1f0d33', border: '#361652', text: '#ce93d8' },
      { bg: '#2d1248', border: '#4c2280', text: '#d9a5e0' },
      { bg: '#3c1860', border: '#6430a8', text: '#e3b8ea' },
      { bg: '#4a1e7a', border: '#7a3ecc', text: '#edcbf0' },
    ],
  ];
  const FAMILY_NAMES = ['Blue', 'Green', 'Amber', 'Purple'];

  // ── inject CSS into <head> ────────────────────────────────────────────────
  // Targets the ID divs directly — they ARE the outer box. No nested wrapper.
  function injectStyles() {
    if (document.getElementById('clv-styles')) return;
    const s = document.createElement('style');
    s.id = 'clv-styles';
    s.textContent = `
      #cute-row-major-viz,
      #cute-col-major-viz,
      #cute-3d-viz,
      #cute-3d-row-major-viz,
      #cute-3d-col-major-hier-viz,
      #cute-3d-row-major-hier-viz,
      #cute-blas-nt-viz,
      #cute-blas-tn-viz,
      #cute-blas-nn-viz,
      #cute-blas-tt-viz,
      #cute-nvfp4-viz,
      #cute-nvfp4-batched-viz {
        background: ${BG} !important;
        border: 1px solid ${BORDER_BR} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        margin: 20px 0 !important;
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
        line-height: 1 !important;
      }
      .clv-header {
        padding: 16px 24px;
        border-bottom: 1px solid ${BORDER};
        background: ${SURFACE};
        display: flex;
        align-items: center;
        gap: 20px;
        flex-shrink: 0;
      }
      .clv-formula {
        font-size: 15px;
        color: ${ACCENT};
        font-weight: 700;
        letter-spacing: 0.5px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-formula .op { color: ${TEXT_DIM}; }
      .clv-body {
        padding: 28px 32px;
        background: ${BG};
        overflow-x: auto;
        display: flex;
        flex-direction: column;
        gap: 20px;
      }
      .clv-section-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: ${TEXT_MUT};
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-grid-container {
        display: flex;
        flex-direction: column;
      }
      .clv-axis-row {
        display: flex;
        gap: 2px;
        margin-left: 28px;
        margin-bottom: 4px;
      }
      .clv-axes-wrap {
        display: flex;
        align-items: flex-start;
      }
      .clv-yaxis {
        display: flex;
        flex-direction: column;
        gap: 2px;
        margin-right: 6px;
      }
      .clv-axlbl {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        color: ${TEXT_MUT};
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        flex-shrink: 0;
      }
      .clv-axlbl-y {
        width: 22px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        font-size: 9px;
        color: ${TEXT_MUT};
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        flex-shrink: 0;
      }
      .clv-index-grid {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }
      .clv-index-row {
        display: flex;
        gap: 2px;
      }
      .clv-index-cell {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 700;
        border-radius: 3px;
        border: 1px solid rgba(255,255,255,0.05);
        transition: all 0.1s;
        cursor: default;
        position: relative;
        flex-shrink: 0;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-index-cell:hover {
        z-index: 10;
        transform: scale(1.15);
        border-color: rgba(255,255,255,0.3) !important;
      }
      .clv-mode-label {
        font-size: 10px;
        color: ${TEXT_DIM};
        margin-top: 6px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-offset-bar {
        display: flex;
        gap: 1px;
        flex-wrap: wrap;
        margin-top: 4px;
      }
      .clv-offset-cell {
        height: 24px;
        min-width: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        font-weight: 700;
        border-radius: 2px;
        padding: 0 4px;
        cursor: default;
        transition: transform 0.1s;
        position: relative;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-offset-cell:hover {
        transform: scale(1.12);
        z-index: 10;
      }
      .clv-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
      }
      .clv-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 10px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-legend-swatch {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        flex-shrink: 0;
      }
      /* 3-D slice column headers */
      .clv-slice-label {
        font-size: 10px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
      }
      .clv-slices-wrap {
        display: flex;
        gap: 24px;
        flex-wrap: wrap;
      }
      /* smaller cells for 3-D (less space needed) */
      .clv-index-cell.sm {
        width: 30px;
        height: 30px;
        font-size: 9px;
      }
      .clv-axlbl.sm   { width: 30px; height: 30px; }
      .clv-axlbl-y.sm { width: 18px; height: 30px; }
      .clv-axis-row.sm { margin-left: 22px; }
    `;
    document.head.appendChild(s);
  }

  // ── HTML helpers ──────────────────────────────────────────────────────────

  function indexCell(text, col, title, sm) {
    const cls = 'clv-index-cell' + (sm ? ' sm' : '');
    return `<div class="${cls}" style="background:${col.bg};border-color:${col.border};color:${col.text};" title="${title}">${text}</div>`;
  }

  function axLbl(text, sm)  { return `<div class="clv-axlbl${sm ? ' sm' : ''}">${text}</div>`; }
  function axLblY(text, sm) { return `<div class="clv-axlbl-y${sm ? ' sm' : ''}">${text}</div>`; }

  // ── 2-D grid (colorDim: 0 = by row, 1 = by column) ───────────────────────
  function grid2D(rows, cols, s0, s1, colorDim, sm) {
    let html = `<div class="clv-axis-row${sm ? ' sm' : ''}">`;
    for (let c = 0; c < cols; c++) html += axLbl(c, sm);
    html += `</div><div class="clv-axes-wrap"><div class="clv-yaxis">`;
    for (let r = 0; r < rows; r++) html += axLblY(r, sm);
    html += `</div><div class="clv-index-grid">`;
    for (let r = 0; r < rows; r++) {
      html += `<div class="clv-index-row">`;
      for (let c = 0; c < cols; c++) {
        const off = r * s0 + c * s1;
        const col = COLORS[(colorDim === 0 ? r : c) % COLORS.length];
        html += indexCell(off, col, `(${r},${c}) \u2192 ${off}`, sm);
      }
      html += `</div>`;
    }
    html += `</div></div>`;
    return html;
  }

  // ── linear memory bar ─────────────────────────────────────────────────────
  function memBar2D(rows, cols, s0, s1, colorDim) {
    const entries = [];
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++)
        entries.push({ r, c, off: r * s0 + c * s1 });
    const maxOff = Math.max(...entries.map(e => e.off));
    const byOff  = {};
    entries.forEach(e => { byOff[e.off] = e; });

    let html = `<div class="clv-offset-bar">`;
    for (let off = 0; off <= maxOff; off++) {
      const e   = byOff[off];
      if (!e) continue;
      const col = COLORS[(colorDim === 0 ? e.r : e.c) % COLORS.length];
      html += `<div class="clv-offset-cell" style="background:${col.bg};border:1px solid ${col.border};color:${col.text};" title="offset ${off}: (${e.r},${e.c})">${off}</div>`;
    }
    html += `</div>`;
    return html;
  }

  // ── legend ────────────────────────────────────────────────────────────────
  function legend(count, label) {
    let html = `<div class="clv-legend">`;
    for (let i = 0; i < count; i++) {
      const col = COLORS[i % COLORS.length];
      html += `<div class="clv-legend-item" style="color:${col.text};">
        <div class="clv-legend-swatch" style="background:${col.bg};border:1px solid ${col.border};"></div>
        ${label} ${i}
      </div>`;
    }
    html += `</div>`;
    return html;
  }

  // ── heatmap canvas (Memory offsets, hue = value) ─────────────────────────
  // Returns a DOM node — cannot be serialized into a template literal.
  function buildHeatmap(rows, cols, s0, s1) {
    const maxOff    = (rows - 1) * s0 + (cols - 1) * s1;
    const cellSize  = Math.max(28, Math.min(48, Math.floor(480 / Math.max(rows, cols))));
    const W         = cols * cellSize;
    const H         = rows * cellSize;

    const wrap = document.createElement('div');

    // grid canvas
    const canvas    = document.createElement('canvas');
    canvas.width    = W;
    canvas.height   = H;
    canvas.style.cssText = 'display:block;border-radius:4px;';
    const ctx       = canvas.getContext('2d');

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const off = r * s0 + c * s1;
        const t   = off / Math.max(maxOff, 1);
        const hue = 200 + t * 120;   // blue → green → yellow (matches original)
        const sat = 70;
        const lit = 25 + t * 20;
        ctx.fillStyle = `hsl(${hue},${sat}%,${lit}%)`;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize - 1, cellSize - 1);
        ctx.fillStyle = `hsl(${hue},${sat}%,${60 + t * 20}%)`;
        ctx.font = `bold ${Math.min(12, cellSize * 0.35)}px JetBrains Mono`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(off, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
      }
    }
    wrap.appendChild(canvas);

    // gradient colour bar
    const gradBar      = document.createElement('canvas');
    gradBar.width      = W;
    gradBar.height     = 12;
    gradBar.style.cssText = `display:block;width:${W}px;border-radius:2px;margin-top:4px;`;
    const gctx         = gradBar.getContext('2d');
    const grad         = gctx.createLinearGradient(0, 0, W, 0);
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      grad.addColorStop(t, `hsl(${200 + t * 120},70%,${35 + t * 20}%)`);
    }
    gctx.fillStyle = grad;
    gctx.fillRect(0, 0, W, 12);
    wrap.appendChild(gradBar);

    // offset range labels
    const lbl = document.createElement('div');
    lbl.style.cssText = `display:flex;justify-content:space-between;font-size:9px;color:${TEXT_MUT};margin-top:2px;font-family:'JetBrains Mono','Fira Code',monospace;`;
    lbl.innerHTML = `<span>offset 0</span><span>offset ${maxOff}</span>`;
    wrap.appendChild(lbl);

    return wrap;
  }

  function attachHeatmap(el, rows, cols, s0, s1) {
    const slot = el.querySelector('.clv-heatmap-slot');
    if (slot) slot.appendChild(buildHeatmap(rows, cols, s0, s1));
  }

  // ── 3-D heatmap: one canvas per k-slice, same hue formula ─────────────────
  function buildHeatmap3D(d0, d1, d2, s0, s1, s2) {
    const maxOff  = (d0 - 1) * s0 + (d1 - 1) * s1 + (d2 - 1) * s2;
    const cs      = 30; // cell size matches the .sm grid cells
    const sliceW  = d1 * cs;

    const container = document.createElement('div');

    // row of slice canvases
    const sliceRow = document.createElement('div');
    sliceRow.style.cssText = 'display:flex;gap:20px;flex-wrap:wrap;';

    for (let k = 0; k < d2; k++) {
      const p       = BLUE_SHADES[k % BLUE_SHADES.length];
      const sliceDiv = document.createElement('div');

      const lbl = document.createElement('div');
      lbl.className  = 'clv-slice-label';
      lbl.style.color = p.text;
      lbl.textContent = `k=${k}`;
      sliceDiv.appendChild(lbl);

      const canvas  = document.createElement('canvas');
      canvas.width  = sliceW;
      canvas.height = d0 * cs;
      canvas.style.cssText = 'display:block;border-radius:4px;';
      const ctx = canvas.getContext('2d');

      for (let r = 0; r < d0; r++) {
        for (let c = 0; c < d1; c++) {
          const off = r * s0 + c * s1 + k * s2;
          const t   = off / Math.max(maxOff, 1);
          const hue = 200 + t * 120;
          ctx.fillStyle = `hsl(${hue},70%,${25 + t * 20}%)`;
          ctx.fillRect(c * cs, r * cs, cs - 1, cs - 1);
          ctx.fillStyle = `hsl(${hue},70%,${60 + t * 20}%)`;
          ctx.font = `bold ${Math.min(11, cs * 0.35)}px JetBrains Mono`;
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(off, c * cs + cs / 2, r * cs + cs / 2);
        }
      }
      sliceDiv.appendChild(canvas);
      sliceRow.appendChild(sliceDiv);
    }
    container.appendChild(sliceRow);

    // gradient bar spanning the full offset range
    const barW    = d2 * sliceW + (d2 - 1) * 20;
    const gradBar = document.createElement('canvas');
    gradBar.width  = barW;
    gradBar.height = 12;
    gradBar.style.cssText = `display:block;width:${barW}px;border-radius:2px;margin-top:8px;`;
    const gctx = gradBar.getContext('2d');
    const grad = gctx.createLinearGradient(0, 0, barW, 0);
    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      grad.addColorStop(t, `hsl(${200 + t * 120},70%,${35 + t * 20}%)`);
    }
    gctx.fillStyle = grad;
    gctx.fillRect(0, 0, barW, 12);
    container.appendChild(gradBar);

    const lbl = document.createElement('div');
    lbl.style.cssText = `display:flex;justify-content:space-between;font-size:9px;color:${TEXT_MUT};margin-top:2px;width:${barW}px;font-family:'JetBrains Mono','Fira Code',monospace;`;
    lbl.innerHTML = `<span>offset 0</span><span>offset ${maxOff}</span>`;
    container.appendChild(lbl);

    return container;
  }

  function attachHeatmap3D(el, d0, d1, d2, s0, s1, s2) {
    const slot = el.querySelector('.clv-heatmap3d-slot');
    if (slot) slot.appendChild(buildHeatmap3D(d0, d1, d2, s0, s1, s2));
  }

  // ── 3-D linear memory bar (colored by k-slice) ────────────────────────────
  function memBar3D(d0, d1, d2, s0, s1, s2) {
    const entries = [];
    for (let r = 0; r < d0; r++)
      for (let c = 0; c < d1; c++)
        for (let k = 0; k < d2; k++)
          entries.push({ r, c, k, off: r * s0 + c * s1 + k * s2 });

    const maxOff = Math.max(...entries.map(e => e.off));
    const byOff  = {};
    entries.forEach(e => { byOff[e.off] = e; });

    let html = `<div class="clv-offset-bar">`;
    for (let off = 0; off <= maxOff; off++) {
      const e = byOff[off];
      if (!e) continue;
      const p = PALETTE_3D[e.c % PALETTE_3D.length][e.k % 4];
      html += `<div class="clv-offset-cell" style="background:${p.bg};border:1px solid ${p.border};color:${p.text};" title="offset ${off}: (${e.r},${e.c},${e.k})">${off}</div>`;
    }
    html += `</div>`;
    return html;
  }

  // ── 3-D mode legend (2D: color family = col, shade = k) ─────────────────
  function legend3D(d1, d2) {
    let html = `<div style="display:flex;flex-direction:column;gap:8px;">`;
    for (let c = 0; c < d1; c++) {
      const fname = FAMILY_NAMES[c % FAMILY_NAMES.length];
      html += `<div class="clv-legend" style="align-items:center;">`;
      html += `<span style="font-size:9px;color:${TEXT_DIM};min-width:80px;font-family:'JetBrains Mono','Fira Code',monospace;">j=${c} (${fname})</span>`;
      for (let k = 0; k < d2; k++) {
        const p = PALETTE_3D[c % PALETTE_3D.length][k % 4];
        html += `<div class="clv-legend-item" style="color:${p.text};">
          <div class="clv-legend-swatch" style="background:${p.bg};border:1px solid ${p.border};"></div>
          k=${k}
        </div>`;
      }
      html += `</div>`;
    }
    html += `</div>`;
    return html;
  }

  // ── renderers ─────────────────────────────────────────────────────────────

  function renderRowMajor(el) {
    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">Layout(<span style="color:${ACCENT}">(4,8)</span><span class="op">,</span>&nbsp;<span style="color:${ACCENT2}">(8,1)</span>)</div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">Grid</div>
          ${grid2D(4, 8, 8, 1, 1)}
          <div class="clv-mode-label">&rarr; Mode 1 (cols) &rarr;</div>
          <div class="clv-mode-label">&darr; Mode 0 (rows)</div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Memory Offsets (hue = value)</div>
          <div class="clv-heatmap-slot"></div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory &rarr;</div>
          ${memBar2D(4, 8, 8, 1, 1)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Mode Legend</div>
          ${legend(8, 'Col')}
        </div>
      </div>`;
    attachHeatmap(el, 4, 8, 8, 1);
  }

  function renderColMajor(el) {
    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">Layout(<span style="color:${ACCENT}">(4,8)</span><span class="op">,</span>&nbsp;<span style="color:${ACCENT2}">(1,4)</span>)</div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">Grid</div>
          ${grid2D(4, 8, 1, 4, 1)}
          <div class="clv-mode-label">&rarr; Mode 1 (cols) &rarr;</div>
          <div class="clv-mode-label">&darr; Mode 0 (rows)</div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Memory Offsets (hue = value)</div>
          <div class="clv-heatmap-slot"></div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory &rarr;</div>
          ${memBar2D(4, 8, 1, 4, 1)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Mode Legend</div>
          ${legend(8, 'Col')}
        </div>
      </div>`;
    attachHeatmap(el, 4, 8, 1, 4);
  }

  // Build one row of slice panels for a given fixed dimension.
  // Colors are always keyed by (j, k) = PALETTE_3D[j%4][k%4] regardless of slice view.
  function sliceGrid3D(fixedDim, d0, d1, d2, s0, s1, s2) {
    // Map (fixedDim) → (sliceCount, rowCount, colCount, strides, axis names)
    const cfg = [
      { sc: d0, rc: d1, cc: d2, ss: s0, rs: s1, cs: s2, sn: 'i', rn: 'j', cn: 'k',
        coords: (sv,r,c) => [sv,r,c], jk: (sv,r,c) => [r,c] },
      { sc: d1, rc: d0, cc: d2, ss: s1, rs: s0, cs: s2, sn: 'j', rn: 'i', cn: 'k',
        coords: (sv,r,c) => [r,sv,c], jk: (sv,r,c) => [sv,c] },
      { sc: d2, rc: d0, cc: d1, ss: s2, rs: s0, cs: s1, sn: 'k', rn: 'i', cn: 'j',
        coords: (sv,r,c) => [r,c,sv], jk: (sv,r,c) => [c,sv] },
    ][fixedDim];

    let html = `<div class="clv-slices-wrap">`;
    for (let sv = 0; sv < cfg.sc; sv++) {
      let s = `<div class="clv-grid-container">`;
      s += `<div class="clv-slice-label" style="color:${TEXT_DIM};">${cfg.sn}=${sv}</div>`;
      s += `<div class="clv-axis-row sm">`;
      for (let c = 0; c < cfg.cc; c++) s += axLbl(c, true);
      s += `</div><div class="clv-axes-wrap"><div class="clv-yaxis">`;
      for (let r = 0; r < cfg.rc; r++) s += axLblY(r, true);
      s += `</div><div class="clv-index-grid">`;
      for (let r = 0; r < cfg.rc; r++) {
        s += `<div class="clv-index-row">`;
        for (let c = 0; c < cfg.cc; c++) {
          const off = sv * cfg.ss + r * cfg.rs + c * cfg.cs;
          const [j, k] = cfg.jk(sv, r, c);
          const p = PALETTE_3D[j % PALETTE_3D.length][k % 4];
          const [ci, cj, ck] = cfg.coords(sv, r, c);
          s += indexCell(off, p, `(${ci},${cj},${ck}) \u2192 ${off}`, true);
        }
        s += `</div>`;
      }
      s += `</div></div></div>`;
      html += s;
    }
    html += `</div>`;
    return html;
  }

  function _renderTensor3DLayout(el, d0, d1, d2, s0, s1, s2) {
    const shapeStr  = `(${d0},${d1},${d2})`;
    const strideStr = `(${s0},${s1},${s2})`;

    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">Layout(<span style="color:${ACCENT}">${shapeStr}</span><span class="op">,</span>&nbsp;<span style="color:${ACCENT2}">${strideStr}</span>)</div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">Grid — fixed i (Mode 0)</div>
          ${sliceGrid3D(0, d0, d1, d2, s0, s1, s2)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Grid — fixed j (Mode 1)</div>
          ${sliceGrid3D(1, d0, d1, d2, s0, s1, s2)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Grid — fixed k (Mode 2)</div>
          ${sliceGrid3D(2, d0, d1, d2, s0, s1, s2)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory &rarr;</div>
          ${memBar3D(d0, d1, d2, s0, s1, s2)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Mode Legend</div>
          ${legend3D(d1, d2)}
        </div>
      </div>`;
  }

  function renderTensor3D(el)         { _renderTensor3DLayout(el, 4, 4, 4,  1,  4, 16); }
  function renderTensor3DRowMajor(el) { _renderTensor3DLayout(el, 4, 4, 4, 16,  4,  1); }

  // ── hierarchical 3-D renderers ────────────────────────────────────────────
  // ((4,4),4):((1,4),16) — Mode 0 = (i,j):(1,4) col-major submatrix, Mode 1 = k:16
  function renderHierColMajor(el) {
    // Layout constants
    const OW = 32, IW = 18, CS = 30, CG = 2, LG = 6, GG = 8;
    // OW: j= label width, IW: i= label width (.clv-axlbl-y.sm), CS: cell size (.sm)
    // CG: cell gap, LG: label-to-grid gap (matches .clv-yaxis margin-right), GG: group gap
    const xOff  = OW + IW + LG;               // x-axis header margin-left
    const innerH = 4 * CS + 3 * CG;           // height of 4 i-rows with spacers = 126px
    const cSpc  = `<div style="width:${CG}px;height:1px;flex-shrink:0;"></div>`;
    const lSpc  = `<div style="width:${LG}px;height:1px;flex-shrink:0;"></div>`;

    let g = `<div style="display:inline-flex;flex-direction:column;">`;

    // x-axis: k = 0..3
    g += `<div style="display:flex;align-items:center;margin-left:${xOff}px;margin-bottom:4px;">`;
    for (let k = 0; k < 4; k++) {
      if (k > 0) g += cSpc;
      g += `<div class="clv-axlbl sm" style="color:${TEXT_DIM};">${k}</div>`;
    }
    g += `</div>`;

    // j groups (Mode 0 outer sub-mode)
    for (let j = 0; j < 4; j++) {
      g += `<div style="display:flex;align-items:flex-start;${j > 0 ? `margin-top:${GG}px;` : ''}">`;
      // j= outer label, vertically centered over the 4 i-rows
      g += `<div style="width:${OW}px;height:${innerH}px;display:flex;align-items:center;` +
           `justify-content:flex-end;padding-right:4px;font-size:9px;color:${TEXT_DIM};` +
           `font-family:'JetBrains Mono','Fira Code',monospace;flex-shrink:0;">j=${j}</div>`;
      // i rows (Mode 0 inner sub-mode)
      g += `<div style="display:flex;flex-direction:column;">`;
      for (let i = 0; i < 4; i++) {
        if (i > 0) g += `<div style="height:${CG}px;"></div>`;
        g += `<div style="display:flex;align-items:center;">`;
        g += `<div class="clv-axlbl-y sm" style="color:${TEXT_MUT};">${i}</div>`;
        g += lSpc;
        for (let k = 0; k < 4; k++) {
          if (k > 0) g += cSpc;
          const off = i * 1 + j * 4 + k * 16;
          const p = PALETTE_3D[j % 4][k % 4];
          g += indexCell(off, p, `(i=${i},j=${j},k=${k}) \u2192 ${off}`, true);
        }
        g += `</div>`;
      }
      g += `</div></div>`;
    }
    g += `</div>`;

    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">Layout(<span style="color:${ACCENT}">((4,4),4)</span><span class="op">,</span>&nbsp;<span style="color:${ACCENT2}">((1,4),16)</span>)</div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">Hierarchical Grid</div>
          <div style="overflow-x:auto;">${g}</div>
          <div class="clv-mode-label" style="margin-top:8px;">&rarr; Mode 1 / k &rarr;</div>
          <div class="clv-mode-label">&darr; Mode 0 = (j outer, i inner) &darr;</div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory &rarr;</div>
          ${memBar3D(4, 4, 4, 1, 4, 16)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Mode Legend</div>
          ${legend3D(4, 4)}
        </div>
      </div>`;
  }

  // (4,(4,4)):(16,(4,1)) — Mode 0 = i:16, Mode 1 = (j,k):(4,1) row-major submatrix
  function renderHierRowMajor(el) {
    const RW = 18, CS = 30, CG = 2, LG = 6, GG = 8;
    // RW: i= label width, CS: cell size, CG: cell gap, LG: label gap, GG: j-group gap
    const xOff = RW + LG;                     // x-axis header margin-left
    const grpW = 4 * CS + 3 * CG;             // width of one j-group = 126px
    const cSpc = `<div style="width:${CG}px;height:1px;flex-shrink:0;"></div>`;
    const lSpc = `<div style="width:${LG}px;height:1px;flex-shrink:0;"></div>`;
    const gSpc = `<div style="width:${GG}px;height:1px;flex-shrink:0;"></div>`;

    let g = `<div style="display:inline-flex;flex-direction:column;">`;

    // x-axis level 1: j group headers
    g += `<div style="display:flex;align-items:center;margin-left:${xOff}px;margin-bottom:2px;">`;
    for (let j = 0; j < 4; j++) {
      if (j > 0) g += gSpc;
      g += `<div style="width:${grpW}px;text-align:center;font-size:9px;color:${TEXT_DIM};` +
           `font-family:'JetBrains Mono','Fira Code',monospace;">j=${j}</div>`;
    }
    g += `</div>`;

    // x-axis level 2: k labels within each j group
    g += `<div style="display:flex;align-items:center;margin-left:${xOff}px;margin-bottom:4px;">`;
    for (let j = 0; j < 4; j++) {
      if (j > 0) g += gSpc;
      for (let k = 0; k < 4; k++) {
        if (k > 0) g += cSpc;
        g += `<div class="clv-axlbl sm" style="color:${TEXT_MUT};">${k}</div>`;
      }
    }
    g += `</div>`;

    // i rows (Mode 0)
    for (let i = 0; i < 4; i++) {
      if (i > 0) g += `<div style="height:${CG}px;"></div>`;
      g += `<div style="display:flex;align-items:center;">`;
      g += `<div class="clv-axlbl-y sm" style="color:${TEXT_DIM};">${i}</div>`;
      g += lSpc;
      for (let j = 0; j < 4; j++) {
        if (j > 0) g += gSpc;
        for (let k = 0; k < 4; k++) {
          if (k > 0) g += cSpc;
          const off = i * 16 + j * 4 + k * 1;
          const p = PALETTE_3D[j % 4][k % 4];
          g += indexCell(off, p, `(i=${i},j=${j},k=${k}) \u2192 ${off}`, true);
        }
      }
      g += `</div>`;
    }
    g += `</div>`;

    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">Layout(<span style="color:${ACCENT}">(4,(4,4))</span><span class="op">,</span>&nbsp;<span style="color:${ACCENT2}">(16,(4,1))</span>)</div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">Hierarchical Grid</div>
          <div style="overflow-x:auto;">${g}</div>
          <div class="clv-mode-label" style="margin-top:8px;">&rarr; Mode 1 = (j outer, k inner) &rarr;</div>
          <div class="clv-mode-label">&darr; Mode 0 / i &darr;</div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory &rarr;</div>
          ${memBar3D(4, 4, 4, 16, 4, 1)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Mode Legend</div>
          ${legend3D(4, 4)}
        </div>
      </div>`;
  }

  // ── nvfp4 layout renderer ────────────────────────────────────────────────
  // M=4, K=16 (one scale block). Color by byte pair (k//2).
  // sfa scale strip (4×1 fp8) shown alongside, aligned with data rows.
  function renderNvfp4Layout(el) {
    const M = 4, K = 16;

    // Data grid: sm cells, colored by byte pair (k//2)
    function dataGrid() {
      let h = `<div class="clv-axis-row sm">`;
      for (let k = 0; k < K; k++) h += axLbl(k, true);
      h += `</div><div class="clv-axes-wrap"><div class="clv-yaxis">`;
      for (let m = 0; m < M; m++) h += axLblY(m, true);
      h += `</div><div class="clv-index-grid">`;
      for (let m = 0; m < M; m++) {
        h += `<div class="clv-index-row">`;
        for (let k = 0; k < K; k++) {
          const off = m * K + k;
          const col = COLORS[Math.floor(k / 2) % COLORS.length];
          const nibble = k % 2 === 0 ? 'lo' : 'hi';
          h += indexCell(off, col,
            `(m=${m},k=${k}) \u2192 elem[${off}], byte[${Math.floor(off/2)}] ${nibble}-nibble`, true);
        }
        h += `</div>`;
      }
      h += `</div></div>`;
      return h;
    }

    // sfa scale strip: M rows × 1 col, header height matches .clv-axis-row.sm (30px + 4px margin)
    function scaleStrip() {
      const SC = { bg: '#0d2540', border: '#1a4a7c', text: '#90caf9' };
      let h = `<div style="display:flex;flex-direction:column;margin-left:14px;">`;
      h += `<div style="height:30px;display:flex;align-items:center;justify-content:center;` +
           `margin-bottom:4px;font-size:9px;color:${TEXT_DIM};` +
           `font-family:'JetBrains Mono','Fira Code',monospace;">sfa<br>fp8</div>`;
      for (let m = 0; m < M; m++) {
        if (m > 0) h += `<div style="height:2px;"></div>`;
        h += `<div style="width:44px;height:30px;display:flex;align-items:center;` +
             `justify-content:center;font-size:9px;font-weight:700;border-radius:3px;` +
             `background:${SC.bg};border:1px solid ${SC.border};color:${SC.text};` +
             `font-family:'JetBrains Mono','Fira Code',monospace;cursor:default;" ` +
             `title="sfa[${m},ks=0] covers k=0..${K-1}">s[${m}]</div>`;
      }
      h += `</div>`;
      return h;
    }

    // Byte-level memory bar: one cell per byte = 2 nvfp4 elements
    function memBar() {
      let html = `<div class="clv-offset-bar">`;
      for (let m = 0; m < M; m++) {
        for (let p = 0; p < K / 2; p++) {
          const byteAddr = m * (K / 2) + p;
          const k0 = p * 2;
          const col = COLORS[p % COLORS.length];
          html += `<div class="clv-offset-cell" style="background:${col.bg};` +
                  `border:1px solid ${col.border};color:${col.text};" ` +
                  `title="byte[${byteAddr}]: (m=${m}, k=${k0}+${k0+1})">${byteAddr}</div>`;
        }
      }
      html += `</div>`;
      return html;
    }

    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">
          nvfp4 <span style="color:${ACCENT}">(4,16):(16,1)</span>
          <span class="op">&nbsp;+&nbsp;</span>
          sfa <span style="color:${ACCENT2}">(4,1):(1,1)</span>
          <span style="font-size:11px;color:${TEXT_DIM};">&nbsp;&nbsp;K-major &middot; BLOCK=16</span>
        </div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">nvfp4 data — B0..B7 = byte pairs (k//2) &nbsp;&middot;&nbsp; 16 elements = 8 bytes = 1 scale block &rarr;</div>
          <div style="display:flex;align-items:flex-start;">
            <div>
              ${dataGrid()}
              <div class="clv-mode-label">&rarr; K (16 nvfp4 elements = 8 bytes) &rarr;</div>
              <div class="clv-mode-label">&darr; M</div>
            </div>
            ${scaleStrip()}
          </div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory (bytes 0–31) &rarr;</div>
          ${memBar()}
          <div style="margin-top:6px;font-size:9px;color:${TEXT_DIM};font-family:'JetBrains Mono','Fira Code',monospace;">
            M&times;K/2 = 32 bytes &nbsp;|&nbsp; each cell = 1 byte (2 nvfp4 elements) &nbsp;|&nbsp; color = byte pair B(k//2)
          </div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Byte-pair Legend (k//2)</div>
          ${legend(K / 2, 'B')}
        </div>
      </div>`;
  }

  // ── batched nvfp4 layout renderer ────────────────────────────────────────
  // (M=2, K=8, L=2):(8,1,16) K-major. Color: PALETTE_3D[l][m] — batch=family, row=shade.
  // Shows L batches side by side, each with its sfa strip; memory bar shows batch boundaries.
  function renderNvfp4BatchedLayout(el) {
    const M = 2, K = 8, L = 2, BLOCK = 8;
    // offset(m, k, l) = l*M*K + m*K + k

    function colorOf(l, m) { return PALETTE_3D[l % PALETTE_3D.length][m % 4]; }

    function batchGrid(l) {
      const lc = colorOf(l, 0);
      let h = `<div>`;
      // Batch label aligned with k-axis (margin-left matches .clv-axis-row.sm)
      h += `<div style="margin-left:22px;font-size:10px;font-weight:700;color:${lc.text};` +
           `margin-bottom:4px;font-family:'JetBrains Mono','Fira Code',monospace;">l=${l}</div>`;
      // k-axis + grid
      h += `<div class="clv-axis-row sm">`;
      for (let k = 0; k < K; k++) h += axLbl(k, true);
      h += `</div><div class="clv-axes-wrap"><div class="clv-yaxis">`;
      for (let m = 0; m < M; m++) h += axLblY(m, true);
      h += `</div><div class="clv-index-grid">`;
      for (let m = 0; m < M; m++) {
        h += `<div class="clv-index-row">`;
        for (let k = 0; k < K; k++) {
          const off = l * M * K + m * K + k;
          const col = colorOf(l, m);
          const nibble = k % 2 === 0 ? 'lo' : 'hi';
          h += indexCell(off, col,
            `(m=${m},k=${k},l=${l}) \u2192 elem[${off}], byte[${Math.floor(off/2)}] ${nibble}`, true);
        }
        h += `</div>`;
      }
      h += `</div></div>`;
      // sfa scale strip below grid
      h += `<div style="margin-top:8px;">`;
      h += `<div style="margin-left:22px;font-size:8px;color:${TEXT_DIM};margin-bottom:4px;` +
           `font-family:'JetBrains Mono','Fira Code',monospace;">sfa[m, ks=0, l=${l}]</div>`;
      h += `<div class="clv-axes-wrap"><div class="clv-yaxis">`;
      for (let m = 0; m < M; m++) h += axLblY(m, true);
      h += `</div><div style="display:flex;flex-direction:column;gap:2px;">`;
      for (let m = 0; m < M; m++) {
        const col = colorOf(l, m);
        h += `<div style="width:48px;height:30px;display:flex;align-items:center;` +
             `justify-content:center;font-size:9px;font-weight:700;border-radius:3px;` +
             `background:${col.bg};border:1px solid ${col.border};color:${col.text};` +
             `font-family:'JetBrains Mono','Fira Code',monospace;cursor:default;" ` +
             `title="sfa[m=${m},ks=0,l=${l}] covers k=0..${K-1}">s[${m}]</div>`;
      }
      h += `</div></div></div>`;
      h += `</div>`;
      return h;
    }

    // Byte-level memory bar: iterate l→m→k, one cell per byte
    function memBar() {
      let html = `<div class="clv-offset-bar">`;
      for (let l = 0; l < L; l++) {
        for (let m = 0; m < M; m++) {
          for (let k = 0; k < K; k += 2) {
            const off = l * M * K + m * K + k;
            const byteAddr = off / 2;
            const col = colorOf(l, m);
            html += `<div class="clv-offset-cell" style="background:${col.bg};` +
                    `border:1px solid ${col.border};color:${col.text};" ` +
                    `title="byte[${byteAddr}]: (l=${l},m=${m},k=${k}+${k+1})">${byteAddr}</div>`;
          }
        }
      }
      html += `</div>`;
      return html;
    }

    function batchLegend() {
      let html = `<div class="clv-legend">`;
      for (let l = 0; l < L; l++) {
        for (let m = 0; m < M; m++) {
          const col = colorOf(l, m);
          html += `<div class="clv-legend-item" style="color:${col.text};">
            <div class="clv-legend-swatch" style="background:${col.bg};border:1px solid ${col.border};"></div>
            l=${l}, m=${m}
          </div>`;
        }
      }
      html += `</div>`;
      return html;
    }

    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">
          nvfp4 batched <span style="color:${ACCENT}">(2,8,2):(8,1,16)</span>
          <span style="font-size:11px;color:${TEXT_DIM};">&nbsp;&nbsp;K-major &middot; BLOCK=8</span>
        </div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">L=2 batches stored sequentially — each (M=2,K=8) K-major · sfa per (m,l) below each grid</div>
          <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:flex-start;">
            ${batchGrid(0)}
            ${batchGrid(1)}
          </div>
          <div class="clv-mode-label" style="margin-top:8px;">&rarr; K (8 nvfp4 elements = 4 bytes = 1 scale block) &rarr;</div>
          <div class="clv-mode-label">&darr; M &nbsp;&middot;&nbsp; batch l starts at byte l&times;M&times;K/2 = l&times;8</div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Linear Memory (bytes 0–15) &rarr;</div>
          ${memBar()}
          <div style="margin-top:6px;font-size:9px;color:${TEXT_DIM};font-family:'JetBrains Mono','Fira Code',monospace;">
            M&times;K&times;L/2 = 16 bytes &nbsp;|&nbsp; batch boundary at byte 8 &nbsp;|&nbsp; color = (batch l, row m)
          </div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">Legend (batch l, row m)</div>
          ${batchLegend()}
        </div>
      </div>`;
  }

  // ── BLAS format renderers ─────────────────────────────────────────────────
  // Shows A(M×K) and B(N×K) grids side by side with individual memory bars.
  // Cells colored by K-column index (colorDim=1) — the shared dimension between A and B.
  function _renderBlasPair(el, blasName, M, K, N, sA0, sA1, aMajorLabel, sB0, sB1, bMajorLabel) {
    const fmtA = `(${M},${K}):(${sA0},${sA1})`;
    const fmtB = `(${N},${K}):(${sB0},${sB1})`;

    el.innerHTML = `
      <div class="clv-header">
        <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
          <div class="clv-formula">BLAS&nbsp;<span style="color:${ACCENT}">${blasName}</span></div>
          <div style="font-size:12px;font-family:'JetBrains Mono','Fira Code',monospace;color:${TEXT_DIM};">
            A&nbsp;=&nbsp;<span style="color:${ACCENT}">${fmtA}</span>
            &nbsp;&nbsp;&nbsp;
            B&nbsp;=&nbsp;<span style="color:${ACCENT2}">${fmtB}</span>
          </div>
        </div>
      </div>
      <div class="clv-body">
        <div style="display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start;">
          <div>
            <div class="clv-section-label">A (${M}×${K}) — ${aMajorLabel}</div>
            ${grid2D(M, K, sA0, sA1, 1)}
            <div class="clv-mode-label">&rarr; K &rarr;</div>
            <div class="clv-mode-label">&darr; M</div>
          </div>
          <div>
            <div class="clv-section-label">B (${N}×${K}) — ${bMajorLabel}</div>
            ${grid2D(N, K, sB0, sB1, 1)}
            <div class="clv-mode-label">&rarr; K &rarr;</div>
            <div class="clv-mode-label">&darr; N</div>
          </div>
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">A Linear Memory &rarr;</div>
          ${memBar2D(M, K, sA0, sA1, 1)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">B Linear Memory &rarr;</div>
          ${memBar2D(N, K, sB0, sB1, 1)}
        </div>
        <div class="clv-grid-container">
          <div class="clv-section-label">K-Column Legend</div>
          ${legend(K, 'k=')}
        </div>
      </div>`;
  }

  function renderBlasNT(el) {
    _renderBlasPair(el, 'NT', 4, 6, 2,
      1, 4, 'M-major (col-major)',
      1, 2, 'N-major (col-major)');
  }

  function renderBlasTN(el) {
    _renderBlasPair(el, 'TN', 4, 6, 2,
      6, 1, 'K-major (row-major)',
      6, 1, 'K-major (row-major)');
  }

  function renderBlasNN(el) {
    _renderBlasPair(el, 'NN', 4, 6, 2,
      1, 4, 'M-major (col-major)',
      6, 1, 'K-major (row-major)');
  }

  function renderBlasTT(el) {
    _renderBlasPair(el, 'TT', 4, 6, 2,
      6, 1, 'K-major (row-major)',
      1, 2, 'N-major (col-major)');
  }

  // ── init ─────────────────────────────────────────────────────────────────
  function init() {
    injectStyles();
    const rm = document.getElementById('cute-row-major-viz');
    if (rm) renderRowMajor(rm);
    const cm = document.getElementById('cute-col-major-viz');
    if (cm) renderColMajor(cm);
    const td = document.getElementById('cute-3d-viz');
    if (td) renderTensor3D(td);
    const tdrm = document.getElementById('cute-3d-row-major-viz');
    if (tdrm) renderTensor3DRowMajor(tdrm);
    const hcm = document.getElementById('cute-3d-col-major-hier-viz');
    if (hcm) renderHierColMajor(hcm);
    const hrm = document.getElementById('cute-3d-row-major-hier-viz');
    if (hrm) renderHierRowMajor(hrm);
    const bnt = document.getElementById('cute-blas-nt-viz');
    if (bnt) renderBlasNT(bnt);
    const btn = document.getElementById('cute-blas-tn-viz');
    if (btn) renderBlasTN(btn);
    const bnn = document.getElementById('cute-blas-nn-viz');
    if (bnn) renderBlasNN(bnn);
    const btt = document.getElementById('cute-blas-tt-viz');
    if (btt) renderBlasTT(btt);
    const fp4 = document.getElementById('cute-nvfp4-viz');
    if (fp4) renderNvfp4Layout(fp4);
    const fp4b = document.getElementById('cute-nvfp4-batched-viz');
    if (fp4b) renderNvfp4BatchedLayout(fp4b);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
