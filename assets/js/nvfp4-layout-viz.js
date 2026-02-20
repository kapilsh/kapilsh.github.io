// NVFP4 Scale Factor Layout Transform Visualization
// to_blocked: 5-step transform for Blackwell SM100/SM120/SM121
// ============================================================================

(function () {
  // ── Inject styles ──────────────────────────────────────────────────────────
  const css = `
  /* ── CSS variable scope ── */
  .nvfp4-root {
    --bg:      #0c0e14;
    --surface: #131720;
    --surface2:#1a1f2e;
    --border:  #252d42;
    --text:    #c8d4f0;
    --dim:     #5a6580;
    --accent:  #4a9eff;

    --w0: #3d7aff;  --w1: #ff5f87;  --w2: #ffd166;  --w3: #06d6a0;
    --w0d: rgba(61,122,255,0.18);   --w1d: rgba(255,95,135,0.18);
    --w2d: rgba(255,209,102,0.18);  --w3d: rgba(6,214,160,0.18);

    --b1w0: #7b4fff; --b1w1: #ff9f43; --b1w2: #54d0e6; --b1w3: #a8e063;
    --b1w0d: rgba(123,79,255,0.18); --b1w1d: rgba(255,159,67,0.18);
    --b1w2d: rgba(84,208,230,0.18); --b1w3d: rgba(168,224,99,0.18);
  }
  .nvfp4-root * { box-sizing: border-box; }

  /* ── Section card wrapper (matches original .section look) ── */
  .nvfp4-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 4px 0 6px;
    overflow-x: auto;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text);
  }

  /* ── Section header: badge + title ── */
  .nvfp4-section-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
  }
  .nvfp4-step-badge {
    background: var(--accent);
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    white-space: nowrap;
    flex-shrink: 0;
    text-transform: uppercase;
  }
  .nvfp4-section-title {
    font-size: 14px;
    font-weight: 600;
    color: #dde6ff;
    margin: 0;
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* ── Legend title block (left-accent bar) ── */
  .nvfp4-title-block {
    border-left: 3px solid var(--accent);
    padding-left: 16px;
    margin-bottom: 18px;
  }
  .nvfp4-title-block-h {
    font-size: 16px;
    font-weight: 600;
    color: #e8f0ff;
    margin: 0 0 4px;
    letter-spacing: -0.3px;
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .nvfp4-title-block-sub {
    color: var(--dim);
    margin: 0;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Legend ── */
  .nvfp4-legend { display: flex; gap: 14px; flex-wrap: wrap; }
  .nvfp4-legend-group { border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; background: var(--surface2); }
  .nvfp4-legend-group-title { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--dim); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.07em; }
  .nvfp4-legend-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .nvfp4-legend-item { display: flex; align-items: center; gap: 5px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text); }
  .nvfp4-swatch { width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0; }

  /* ── Cells ── */
  .nvfp4-cell { display: flex; align-items: center; justify-content: center; font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 500; border-radius: 1px; cursor: default; }
  .nvfp4-b0w0 { background: var(--w0d);   color: var(--w0);   border: 1px solid rgba(61,122,255,0.3); }
  .nvfp4-b0w1 { background: var(--w1d);   color: var(--w1);   border: 1px solid rgba(255,95,135,0.3); }
  .nvfp4-b0w2 { background: var(--w2d);   color: var(--w2);   border: 1px solid rgba(255,209,102,0.35); }
  .nvfp4-b0w3 { background: var(--w3d);   color: var(--w3);   border: 1px solid rgba(6,214,160,0.3); }
  .nvfp4-b1w0 { background: var(--b1w0d); color: var(--b1w0); border: 1px solid rgba(123,79,255,0.3); }
  .nvfp4-b1w1 { background: var(--b1w1d); color: var(--b1w1); border: 1px solid rgba(255,159,67,0.3); }
  .nvfp4-b1w2 { background: var(--b1w2d); color: var(--b1w2); border: 1px solid rgba(84,208,230,0.3); }
  .nvfp4-b1w3 { background: var(--b1w3d); color: var(--b1w3); border: 1px solid rgba(168,224,99,0.3); }

  .nvfp4-grid { display: grid; gap: 1.5px; background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }

  .nvfp4-matrix-wrap { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 14px; }
  .nvfp4-matrix-container { display: flex; flex-direction: column; align-items: center; gap: 5px; }
  .nvfp4-matrix-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--dim); }
  .nvfp4-matrix-shape { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--accent); background: var(--surface2); padding: 2px 6px; border-radius: 3px; border: 1px solid var(--border); }

  .nvfp4-tile-block { border: 1px solid; border-radius: 4px; padding: 4px; }
  .nvfp4-tile-block.rb0cb0 { border-color: rgba(61,122,255,0.5);  background: rgba(61,122,255,0.04); }
  .nvfp4-tile-block.rb0cb1 { border-color: rgba(255,98,232,0.5);  background: rgba(255,98,232,0.04); }
  .nvfp4-tile-block.rb1cb0 { border-color: rgba(255,159,67,0.5);  background: rgba(255,159,67,0.04); }
  .nvfp4-tile-block.rb1cb1 { border-color: rgba(84,208,230,0.5);  background: rgba(84,208,230,0.04); }

  .nvfp4-tile-label { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; text-align: center; padding: 2px 0 3px; }
  .nvfp4-tile-label.rb0cb0 { color: #7ec8ff; }
  .nvfp4-tile-label.rb0cb1 { color: #ff98e8; }
  .nvfp4-tile-label.rb1cb0 { color: #ffb347; }
  .nvfp4-tile-label.rb1cb1 { color: #54d0e6; }

  .nvfp4-elip { text-align: center; color: var(--dim); font-size: 9px; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }

  .nvfp4-blk-pill { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 3px; margin: 0 2px; }
  .nvfp4-rb0cb0-pill { background: rgba(61,122,255,0.2);  color: #7ec8ff; }
  .nvfp4-rb0cb1-pill { background: rgba(255,98,232,0.2);  color: #ff98e8; }
  .nvfp4-rb1cb0-pill { background: rgba(255,159,67,0.2);  color: #ffb347; }
  .nvfp4-rb1cb1-pill { background: rgba(84,208,230,0.2);  color: #54d0e6; }

  .nvfp4-compare-wrap { display: flex; gap: 32px; flex-wrap: wrap; margin-top: 4px; }
  .nvfp4-compare-col h4 { font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-bottom: 8px; }
  .nvfp4-order-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
  .nvfp4-order-idx { color: var(--dim); min-width: 28px; text-align: right; }
  .nvfp4-order-note { color: var(--dim); font-size: 11px; }
  .nvfp4-bad-label  { color: #ff6b6b; font-size: 11px; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }
  .nvfp4-good-label { color: #50fa7b; font-size: 11px; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }

  /* ── Final tile headers ── */
  .nvfp4-band-header { font-family: 'JetBrains Mono', monospace; font-size: 8px; text-align: center; }
  .nvfp4-col-header  { font-family: 'JetBrains Mono', monospace; font-size: 7px; color: var(--dim); text-align: center; }
  .nvfp4-thread-label { font-family: 'JetBrains Mono', monospace; font-size: 7.5px; color: var(--dim); display: flex; align-items: center; justify-content: flex-end; padding-right: 4px; }

  /* ── Quadrant grid ── */
  .nvfp4-quad-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; width: 230px; }
  .nvfp4-quad-cell  { border-radius: 4px; padding: 7px 9px; }
  .nvfp4-quad-title { font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700; }
  .nvfp4-quad-desc  { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: var(--dim); margin-top: 2px; }
  .nvfp4-quad-note  { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: var(--dim); margin-top: 8px; }

  /* ── Divider between final tiles ── */
  .nvfp4-tile-sep { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
  `;

  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ── Shared constants ───────────────────────────────────────────────────────
  const B0W = ['b0w0', 'b0w1', 'b0w2', 'b0w3'];
  const B1W = ['b1w0', 'b1w1', 'b1w2', 'b1w3'];
  const CW = 28, CH = 17;

  // ── Primitive helpers ──────────────────────────────────────────────────────
  function cell(txt, cls, tip, w, h) {
    w = w || CW; h = h || CH;
    const d = document.createElement('div');
    d.className = `nvfp4-cell nvfp4-${cls}`;
    d.style.width = w + 'px';
    d.style.height = h + 'px';
    d.textContent = txt;
    if (tip) d.title = tip;
    return d;
  }

  function grid(rows, cols, fn, cw, ch) {
    cw = cw || CW; ch = ch || CH;
    const g = document.createElement('div');
    g.className = 'nvfp4-grid';
    g.style.gridTemplateColumns = `repeat(${cols}, ${cw}px)`;
    g.style.gridTemplateRows    = `repeat(${rows}, ${ch}px)`;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) g.appendChild(fn(r, c));
    return g;
  }

  function elipDiv(txt) {
    const d = document.createElement('div');
    d.className = 'nvfp4-elip';
    d.textContent = txt;
    return d;
  }

  function dotRow() {
    const d = document.createElement('div');
    d.style.cssText = `width:${CW}px;height:10px;display:flex;align-items:center;justify-content:center;font-size:9px;color:var(--dim);font-family:'JetBrains Mono',monospace;`;
    d.textContent = '···';
    return d;
  }

  // ── Section wrapper factory ────────────────────────────────────────────────
  // Returns { section, content } — section is the styled card, content is where
  // the visualization body goes.  badge / badgeColor / title match the original
  // .step-badge + h2 pattern from the HTML.
  function makeSection(badge, badgeColor, title) {
    const section = document.createElement('div');
    section.className = 'nvfp4-root nvfp4-section';

    const header = document.createElement('div');
    header.className = 'nvfp4-section-header';

    const b = document.createElement('span');
    b.className = 'nvfp4-step-badge';
    b.style.background = badgeColor;
    b.textContent = badge;

    const h = document.createElement('h3');
    h.className = 'nvfp4-section-title';
    h.textContent = title;

    header.appendChild(b);
    header.appendChild(h);
    section.appendChild(header);

    const content = document.createElement('div');
    section.appendChild(content);

    return { section, content };
  }

  // ── LEGEND ────────────────────────────────────────────────────────────────
  function buildLegend(container) {
    const section = document.createElement('div');
    section.className = 'nvfp4-root nvfp4-section';

    // Title block (left-accent bar)
    const tb = document.createElement('div');
    tb.className = 'nvfp4-title-block';
    const th = document.createElement('div');
    th.className = 'nvfp4-title-block-h';
    th.textContent = 'to_blocked — NVFP4 Scale Factor Layout Transform';
    const ts = document.createElement('div');
    ts.className = 'nvfp4-title-block-sub';
    ts.textContent = 'Example: M=256, K=128 → sf shape (256,8) → n_row_blocks=2, n_col_blocks=2 → 4 tiles of (32×16)';
    tb.appendChild(th);
    tb.appendChild(ts);
    section.appendChild(tb);

    // Legend rows
    const legendWrap = document.createElement('div');
    legendWrap.className = 'nvfp4-legend';

    const groups = [
      {
        title: 'Row-block 0 (rows 0–127) warp bands',
        items: [
          { color: 'var(--w0)',   label: 'band0 r0–31'    },
          { color: 'var(--w1)',   label: 'band1 r32–63'   },
          { color: 'var(--w2)',   label: 'band2 r64–95'   },
          { color: 'var(--w3)',   label: 'band3 r96–127'  },
        ]
      },
      {
        title: 'Row-block 1 (rows 128–255) warp bands',
        items: [
          { color: 'var(--b1w0)', label: 'band0 r128–159' },
          { color: 'var(--b1w1)', label: 'band1 r160–191' },
          { color: 'var(--b1w2)', label: 'band2 r192–223' },
          { color: 'var(--b1w3)', label: 'band3 r224–255' },
        ]
      },
    ];

    groups.forEach(({ title, items }) => {
      const grp = document.createElement('div');
      grp.className = 'nvfp4-legend-group';
      const t = document.createElement('div');
      t.className = 'nvfp4-legend-group-title';
      t.textContent = title;
      grp.appendChild(t);
      const row = document.createElement('div');
      row.className = 'nvfp4-legend-row';
      items.forEach(({ color, label }) => {
        const item = document.createElement('div');
        item.className = 'nvfp4-legend-item';
        const sw = document.createElement('div');
        sw.className = 'nvfp4-swatch';
        sw.style.background = color;
        const sp = document.createElement('span');
        sp.style.color = color;
        sp.textContent = label;
        item.appendChild(sw);
        item.appendChild(sp);
        row.appendChild(item);
      });
      grp.appendChild(row);
      legendWrap.appendChild(grp);
    });

    // Tile identity group
    const tg = document.createElement('div');
    tg.className = 'nvfp4-legend-group';
    const tt = document.createElement('div');
    tt.className = 'nvfp4-legend-group-title';
    tt.textContent = 'Outer tile identity';
    tg.appendChild(tt);
    [
      { pill: 'rb0cb0', label: 'T0 rb0·cb0', note: 'r0–127, k0–3'   },
      { pill: 'rb0cb1', label: 'T1 rb0·cb1', note: 'r0–127, k4–7'   },
      { pill: 'rb1cb0', label: 'T2 rb1·cb0', note: 'r128–255, k0–3' },
      { pill: 'rb1cb1', label: 'T3 rb1·cb1', note: 'r128–255, k4–7' },
    ].forEach(({ pill, label, note }) => {
      const row = document.createElement('div');
      row.className = 'nvfp4-legend-row';
      row.style.marginTop = '4px';
      const p = document.createElement('span');
      p.className = `nvfp4-blk-pill nvfp4-${pill}-pill`;
      p.textContent = label;
      const n = document.createElement('span');
      n.style.cssText = 'font-family:"JetBrains Mono",monospace;font-size:10px;color:var(--dim);';
      n.textContent = note;
      row.appendChild(p);
      row.appendChild(n);
      tg.appendChild(row);
    });
    legendWrap.appendChild(tg);
    section.appendChild(legendWrap);
    container.appendChild(section);
  }

  // ── SETUP VIZ ─────────────────────────────────────────────────────────────
  function buildSetupViz(container) {
    const { section, content } = makeSection(
      'Setup', '#4a9eff',
      'Input: (256, 8) scale factor matrix'
    );

    const wrap = document.createElement('div');
    wrap.className = 'nvfp4-matrix-wrap';

    const SH = 3;
    const totalRows = (SH + 1) * 8 + 1;
    const g = grid(totalRows, 8, (r, c) => {
      const band = Math.floor(r / (SH + 1));
      const ri   = r % (SH + 1);
      if (ri === SH) return dotRow();
      if (band >= 8) return cell('', '');
      const rb = Math.floor(band / 4);
      const wb = band % 4;
      const origRow = rb * 128 + wb * 32 + ri;
      const cls = rb === 0 ? B0W[wb] : B1W[wb];
      return cell(`${origRow},${c}`, cls, `row=${origRow} col=${c}`);
    });

    const mc = document.createElement('div');
    mc.className = 'nvfp4-matrix-container';
    const lbl = document.createElement('div');
    lbl.className = 'nvfp4-matrix-label';
    lbl.textContent = 'Input (256, 8)';
    const shp = document.createElement('div');
    shp.className = 'nvfp4-matrix-shape';
    shp.textContent = '256 rows × 8 k-cols';
    mc.appendChild(lbl);
    mc.appendChild(shp);
    mc.appendChild(g);
    mc.appendChild(elipDiv('256 rows total — showing 3 sample rows per warp-band'));
    wrap.appendChild(mc);

    // Quadrant diagram
    const qdiv = document.createElement('div');
    qdiv.style.paddingTop = '32px';
    const qgrid = document.createElement('div');
    qgrid.className = 'nvfp4-quad-grid';
    [
      { border: 'rgba(61,122,255,0.5)',  bg: 'rgba(61,122,255,0.05)',  titleColor: '#7ec8ff', title: 'rb=0, cb=0', desc: 'rows 0–127\nk-cols 0–3' },
      { border: 'rgba(255,98,232,0.5)',  bg: 'rgba(255,98,232,0.05)',  titleColor: '#ff98e8', title: 'rb=0, cb=1', desc: 'rows 0–127\nk-cols 4–7' },
      { border: 'rgba(255,159,67,0.5)',  bg: 'rgba(255,159,67,0.05)',  titleColor: '#ffb347', title: 'rb=1, cb=0', desc: 'rows 128–255\nk-cols 0–3' },
      { border: 'rgba(84,208,230,0.5)',  bg: 'rgba(84,208,230,0.05)',  titleColor: '#54d0e6', title: 'rb=1, cb=1', desc: 'rows 128–255\nk-cols 4–7' },
    ].forEach(({ border, bg, titleColor, title, desc }) => {
      const qc = document.createElement('div');
      qc.className = 'nvfp4-quad-cell';
      qc.style.border = `1px solid ${border}`;
      qc.style.background = bg;
      const qt = document.createElement('div');
      qt.className = 'nvfp4-quad-title';
      qt.style.color = titleColor;
      qt.textContent = title;
      const qd = document.createElement('div');
      qd.className = 'nvfp4-quad-desc';
      qd.style.whiteSpace = 'pre';
      qd.textContent = desc;
      qc.appendChild(qt);
      qc.appendChild(qd);
      qgrid.appendChild(qc);
    });
    qdiv.appendChild(qgrid);
    const qn = document.createElement('div');
    qn.className = 'nvfp4-quad-note';
    qn.textContent = '4 logical tile regions in the input';
    qdiv.appendChild(qn);
    wrap.appendChild(qdiv);

    content.appendChild(wrap);
    container.appendChild(section);
  }

  // ── STEP 1 VIZ ────────────────────────────────────────────────────────────
  function buildStep1Viz(container) {
    const { section, content } = makeSection(
      'Step 1', '#4a9eff',
      'view(2, 128, 2, 4) — expose all four block boundaries'
    );

    const wrap = document.createElement('div');
    wrap.className = 'nvfp4-matrix-wrap';

    const tiles = [
      { idx: '[0,:,0,:]', cls: 'rb0cb0', ro: 0,   co: 0, wArr: B0W },
      { idx: '[0,:,1,:]', cls: 'rb0cb1', ro: 0,   co: 4, wArr: B0W },
      { idx: '[1,:,0,:]', cls: 'rb1cb0', ro: 128, co: 0, wArr: B1W },
      { idx: '[1,:,1,:]', cls: 'rb1cb1', ro: 128, co: 4, wArr: B1W },
    ];
    const SH = 4;
    tiles.forEach(({ idx, cls, ro, co, wArr }) => {
      const tb = document.createElement('div');
      tb.className = `nvfp4-tile-block ${cls}`;
      const tl = document.createElement('div');
      tl.className = `nvfp4-tile-label ${cls}`;
      tl.textContent = idx;
      tb.appendChild(tl);
      const g = grid(SH + 1, 4, (r, c) => {
        if (r === SH) return dotRow();
        const or = ro + r, oc = co + c;
        return cell(`${or},${oc}`, wArr[Math.floor(r / 32)], `(${or},${oc})`);
      });
      tb.appendChild(g);
      tb.appendChild(elipDiv('128 rows × 4 k-cols'));
      wrap.appendChild(tb);
    });

    content.appendChild(wrap);
    container.appendChild(section);
  }

  // ── STEP 2 VIZ ────────────────────────────────────────────────────────────
  function buildStep2Viz(container) {
    const { section, content } = makeSection(
      'Step 2', '#4a9eff',
      'permute(0, 2, 1, 3) — bring col_block next to row_block'
    );

    const wrap = document.createElement('div');
    wrap.className = 'nvfp4-compare-wrap';

    function orderList(items) {
      const col = document.createElement('div');
      items.forEach(({ rb, cb, cls, note }, i) => {
        const row = document.createElement('div');
        row.className = 'nvfp4-order-row';
        const idx = document.createElement('span');
        idx.className = 'nvfp4-order-idx';
        idx.textContent = `slot ${i}`;
        const pill = document.createElement('span');
        pill.className = `nvfp4-blk-pill nvfp4-${cls}-pill`;
        pill.textContent = `rb=${rb}·cb=${cb}`;
        const n = document.createElement('span');
        n.className = 'nvfp4-order-note';
        n.textContent = note;
        row.appendChild(idx);
        row.appendChild(pill);
        row.appendChild(n);
        col.appendChild(row);
      });
      return col;
    }

    const beforeCol = document.createElement('div');
    beforeCol.className = 'nvfp4-compare-col';
    const bh = document.createElement('h4');
    bh.className = 'nvfp4-bad-label';
    bh.textContent = '✗ Without permute: col_block iterates first';
    beforeCol.appendChild(bh);
    beforeCol.appendChild(orderList([
      { rb: 0, cb: 0, cls: 'rb0cb0', note: 'rows 0–127, k 0–3'   },
      { rb: 1, cb: 0, cls: 'rb1cb0', note: 'rows 128–255, k 0–3' },
      { rb: 0, cb: 1, cls: 'rb0cb1', note: 'rows 0–127, k 4–7'   },
      { rb: 1, cb: 1, cls: 'rb1cb1', note: 'rows 128–255, k 4–7' },
    ]));
    const bwarn = document.createElement('div');
    bwarn.className = 'nvfp4-bad-label';
    bwarn.textContent = '⚠  slots 1 & 2 jump between row-blocks — breaks MMA tile iterator';
    beforeCol.appendChild(bwarn);

    const afterCol = document.createElement('div');
    afterCol.className = 'nvfp4-compare-col';
    const ah = document.createElement('h4');
    ah.className = 'nvfp4-good-label';
    ah.textContent = '✓ With permute(0,2,1,3): row_block iterates first';
    afterCol.appendChild(ah);
    afterCol.appendChild(orderList([
      { rb: 0, cb: 0, cls: 'rb0cb0', note: 'rows 0–127, k 0–3'   },
      { rb: 0, cb: 1, cls: 'rb0cb1', note: 'rows 0–127, k 4–7'   },
      { rb: 1, cb: 0, cls: 'rb1cb0', note: 'rows 128–255, k 0–3' },
      { rb: 1, cb: 1, cls: 'rb1cb1', note: 'rows 128–255, k 4–7' },
    ]));
    const agood = document.createElement('div');
    agood.className = 'nvfp4-good-label';
    agood.textContent = '✓ complete rb0 tiles before rb1 — matches CuTe tile traversal';
    afterCol.appendChild(agood);

    wrap.appendChild(beforeCol);
    wrap.appendChild(afterCol);
    content.appendChild(wrap);
    container.appendChild(section);
  }

  // ── STEP 3 VIZ ────────────────────────────────────────────────────────────
  function buildStep3Viz(container) {
    const { section, content } = makeSection(
      'Step 3', '#4a9eff',
      'reshape(-1, 4, 32, 4) — merge outer block dims; split 128 rows into 4 warp-bands × 32'
    );

    const wrap = document.createElement('div');
    wrap.className = 'nvfp4-matrix-wrap';
    wrap.style.alignItems = 'flex-start';

    const SH = 3;
    const configs = [
      { label: 'Tile 0  rb0·cb0', cls: 'rb0cb0', ro: 0,   co: 0, wArr: B0W },
      { label: 'Tile 1  rb0·cb1', cls: 'rb0cb1', ro: 0,   co: 4, wArr: B0W },
      { label: 'Tile 2  rb1·cb0', cls: 'rb1cb0', ro: 128, co: 0, wArr: B1W },
      { label: 'Tile 3  rb1·cb1', cls: 'rb1cb1', ro: 128, co: 4, wArr: B1W },
    ];
    configs.forEach(({ label, cls, ro, co, wArr }) => {
      const tb = document.createElement('div');
      tb.className = `nvfp4-tile-block ${cls}`;
      const tl = document.createElement('div');
      tl.className = `nvfp4-tile-label ${cls}`;
      tl.textContent = label;
      tb.appendChild(tl);
      const totalRows = 4 * (SH + 1);
      const g = grid(totalRows, 4, (r, c) => {
        const wb = Math.floor(r / (SH + 1)), ri = r % (SH + 1);
        if (ri === SH) return dotRow();
        const or = ro + wb * 32 + ri, oc = co + c;
        return cell(`${or},${oc}`, wArr[wb], `band ${wb}, thread ${ri}`);
      });
      tb.appendChild(g);
      tb.appendChild(elipDiv('4 bands × 32 threads × 4 k-cols'));
      wrap.appendChild(tb);
    });

    content.appendChild(wrap);
    container.appendChild(section);
  }

  // ── FINAL TILES VIZ ───────────────────────────────────────────────────────
  function buildFinalTilesViz(container) {
    const { section, content } = makeSection(
      'Step 5', '#4a9eff',
      'reshape(-1, 32, 16) — 4 output tiles of (32×16), one per outer block'
    );

    const FCW = 25, FCH = 16;
    const configs = [
      {
        label: 'Tile 0  (rb0·cb0)  rows 0–127,   k 0–3',
        cls: 'rb0cb0', ro: 0,   co: 0, wArr: B0W,
        ghColors: ['var(--w0)', 'var(--w1)', 'var(--w2)', 'var(--w3)']
      },
      {
        label: 'Tile 1  (rb0·cb1)  rows 0–127,   k 4–7',
        cls: 'rb0cb1', ro: 0,   co: 4, wArr: B0W,
        ghColors: ['var(--w0)', 'var(--w1)', 'var(--w2)', 'var(--w3)']
      },
      {
        label: 'Tile 2  (rb1·cb0)  rows 128–255, k 0–3',
        cls: 'rb1cb0', ro: 128, co: 0, wArr: B1W,
        ghColors: ['var(--b1w0)', 'var(--b1w1)', 'var(--b1w2)', 'var(--b1w3)']
      },
      {
        label: 'Tile 3  (rb1·cb1)  rows 128–255, k 4–7',
        cls: 'rb1cb1', ro: 128, co: 4, wArr: B1W,
        ghColors: ['var(--b1w0)', 'var(--b1w1)', 'var(--b1w2)', 'var(--b1w3)']
      },
    ];

    configs.forEach(({ label, cls, ro, co, wArr, ghColors }, ti) => {
      if (ti > 0) {
        const sep = document.createElement('hr');
        sep.className = 'nvfp4-tile-sep';
        content.appendChild(sep);
      }

      const lbl = document.createElement('div');
      lbl.className = `nvfp4-tile-label ${cls}`;
      lbl.style.cssText = 'text-align:left;margin-bottom:8px;font-size:11px;';
      lbl.textContent = label;
      content.appendChild(lbl);

      const tw = document.createElement('div');
      tw.style.cssText = 'display:flex;gap:8px;align-items:flex-start;';

      // thread row labels
      const rlc = document.createElement('div');
      rlc.style.cssText = 'display:flex;flex-direction:column;gap:1.5px;';
      const sp = document.createElement('div');
      sp.style.height = (FCH + 2 + FCH + 2) + 'px';
      rlc.appendChild(sp);
      for (let t = 0; t < 32; t++) {
        const rl = document.createElement('div');
        rl.className = 'nvfp4-thread-label';
        rl.style.cssText += `height:${FCH}px;min-width:22px;`;
        rl.textContent = `t${t}`;
        rlc.appendChild(rl);
      }
      tw.appendChild(rlc);

      const tiDiv = document.createElement('div');

      // warp-band group header
      const gh = document.createElement('div');
      gh.style.cssText = 'display:flex;gap:1.5px;margin-bottom:2px;';
      for (let wb = 0; wb < 4; wb++) {
        const d = document.createElement('div');
        const rowStart = ro + wb * 32;
        d.className = 'nvfp4-band-header';
        d.style.cssText = `width:${FCW * 4 + 4 * 1.5}px;color:${ghColors[wb]};`;
        d.textContent = `rows ${rowStart}–${rowStart + 31}`;
        gh.appendChild(d);
      }
      tiDiv.appendChild(gh);

      // k-position sub-header
      const sh = document.createElement('div');
      sh.style.cssText = 'display:flex;gap:1.5px;margin-bottom:2px;';
      for (let c = 0; c < 16; c++) {
        const d = document.createElement('div');
        d.className = 'nvfp4-col-header';
        d.style.width = FCW + 'px';
        d.textContent = `k${co + (c % 4)}`;
        sh.appendChild(d);
      }
      tiDiv.appendChild(sh);

      // tile grid
      const g = document.createElement('div');
      g.className = 'nvfp4-grid';
      g.style.gridTemplateColumns = `repeat(16, ${FCW}px)`;
      g.style.gridTemplateRows    = `repeat(32, ${FCH}px)`;
      for (let t = 0; t < 32; t++) {
        for (let col = 0; col < 16; col++) {
          const wb = Math.floor(col / 4), kp = col % 4;
          const or = ro + wb * 32 + t, oc = co + kp;
          const d = document.createElement('div');
          d.className = `nvfp4-cell nvfp4-${wArr[wb]}`;
          d.style.width = FCW + 'px';
          d.style.height = FCH + 'px';
          d.style.fontSize = '7px';
          d.textContent = `${or},${oc}`;
          d.title = `orig(${or},${oc}) → row=${t} col=${col}`;
          g.appendChild(d);
        }
      }
      tiDiv.appendChild(g);
      tw.appendChild(tiDiv);
      content.appendChild(tw);
    });

    container.appendChild(section);
  }

  // ── Wire up all divs ───────────────────────────────────────────────────────
  function init() {
    const legend     = document.getElementById('nvfp4-legend');
    const setupViz   = document.getElementById('nvfp4-setup-viz');
    const step1Viz   = document.getElementById('nvfp4-step1-viz');
    const step2Viz   = document.getElementById('nvfp4-step2-viz');
    const step3Viz   = document.getElementById('nvfp4-step3-viz');
    const finalTiles = document.getElementById('nvfp4-final-tiles');

    if (legend)     buildLegend(legend);
    if (setupViz)   buildSetupViz(setupViz);
    if (step1Viz)   buildStep1Viz(step1Viz);
    if (step2Viz)   buildStep2Viz(step2Viz);
    if (step3Viz)   buildStep3Viz(step3Viz);
    if (finalTiles) buildFinalTilesViz(finalTiles);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
