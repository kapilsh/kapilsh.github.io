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

  // ── inject CSS into <head> ────────────────────────────────────────────────
  // Targets the ID divs directly — they ARE the outer box. No nested wrapper.
  function injectStyles() {
    if (document.getElementById('clv-styles')) return;
    const s = document.createElement('style');
    s.id = 'clv-styles';
    s.textContent = `
      #cute-row-major-viz,
      #cute-col-major-viz,
      #cute-3d-viz {
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

  function renderTensor3D(el) {
    const [d0, d1, d2, s0, s1, s2] = [4, 4, 4, 1, 4, 16];

    let slices = `<div class="clv-slices-wrap">`;
    for (let k = 0; k < d2; k++) {
      const col = COLORS[k % COLORS.length];
      let s = `<div class="clv-grid-container">`;
      s += `<div class="clv-slice-label" style="color:${col.text};">Mode 2 = ${k}</div>`;
      s += `<div class="clv-axis-row sm">`;
      for (let c = 0; c < d1; c++) s += axLbl(c, true);
      s += `</div><div class="clv-axes-wrap"><div class="clv-yaxis">`;
      for (let r = 0; r < d0; r++) s += axLblY(r, true);
      s += `</div><div class="clv-index-grid">`;
      for (let r = 0; r < d0; r++) {
        s += `<div class="clv-index-row">`;
        for (let c = 0; c < d1; c++) {
          const off = r * s0 + c * s1 + k * s2;
          s += indexCell(off, col, `(${r},${c},${k}) \u2192 ${off}`, true);
        }
        s += `</div>`;
      }
      s += `</div></div></div>`;
      slices += s;
    }
    slices += `</div>`;

    el.innerHTML = `
      <div class="clv-header">
        <div class="clv-formula">Layout(<span style="color:${ACCENT}">(4,4,4)</span><span class="op">,</span>&nbsp;<span style="color:${ACCENT2}">(1,4,16)</span>)</div>
      </div>
      <div class="clv-body">
        <div class="clv-grid-container">
          <div class="clv-section-label">Grid (sliced by Mode 2)</div>
          ${slices}
        </div>
      </div>`;
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
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
