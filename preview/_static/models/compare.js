// Contrast two benchmarked models on a model page. Runs only when a reader picks a model:
// loads the two results files (a few KB each) and draws the difference per dataset.
(function () {
  "use strict";
  const sel = document.querySelector("select.fb-compare");
  if (!sel) return;
  const out = document.querySelector(".fb-compare-out");
  const base = new URL("results/", document.currentScript ? document.currentScript.src : location.href);
  const cache = {};
  const get = (n) => (cache[n] = cache[n] || fetch(new URL(`${encodeURIComponent(n)}.json`, base)).then((r) => {
    if (!r.ok) throw new Error(`${n}: HTTP ${r.status}`);
    return r.json();
  }));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const f = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
  const sd = (x, d = 3) => (x == null || Number.isNaN(x) ? "" : (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(d));
  const cls = (d) => (d == null || Number.isNaN(d) || Math.abs(d) < 0.0005 ? "flat" : d > 0 ? "up" : "down");
  const METRICS = {f1: "F1", precision: "Precision", recall: "Recall", mean_iou: "Mask IoU"};
  let metric = "f1";

  function kpis(a, b) {
    const scores = Object.entries(METRICS).map(([k, label]) => {
      const d = a.overall[k] - b.overall[k];
      return `<div><span>${label}</span><b>${f(a.overall[k], 4)}</b><em class="fb-${cls(d)}">${sd(d, 4)} vs ${f(b.overall[k], 4)}</em></div>`;
    }).join("");
    // Speed only compares when both ran on the same GPU.
    const ta = a.timing || {}, tb = b.timing || {};
    let speed = "";
    if (ta.seconds_per_image && tb.seconds_per_image) {
      const r = ta.seconds_per_image / tb.seconds_per_image;
      const same = ta.device && ta.device === tb.device;
      speed = `<div><span>Time / image</span><b>${ta.seconds_per_image.toFixed(2)} s</b><em class="fb-${same ? (r > 1.05 ? "down" : r < 0.95 ? "up" : "flat") : "flat"}">${r.toFixed(1)}× ${r >= 1 ? "slower" : "faster"} than ${tb.seconds_per_image.toFixed(2)} s${same ? "" : " (different GPUs)"}</em></div>`;
    }
    return `<div class="fb-kpis">${scores}${speed}</div>`;
  }

  function chart(a, b, names) {
    const ds = Object.keys(a.datasets).filter((n) => n in b.datasets).map((n) => {
      const va = a.datasets[n][metric], vb = b.datasets[n][metric];
      return {n, va, vb, d: va - vb};
    }).sort((p, q) => (q.d || 0) - (p.d || 0));
    const W = 720, row = 24, left = 200, right = 76, top = 26, H = top + ds.length * row + 28;
    // Axis starts at the 0.1 step below the lowest value (never above 0.5), as on the static chart.
    const vals = ds.flatMap((r) => [r.va, r.vb]).filter((v) => v != null && !Number.isNaN(v));
    const lo = Math.min(0.5, Math.floor(Math.min(...vals, 1) * 10) / 10);
    const x = (v) => left + ((v == null || Number.isNaN(v) ? lo : Math.max(lo, v)) - lo) / (1 - lo) * (W - left - right);
    const ticks = Array.from({length: Math.round((1 - lo) / 0.1) + 1}, (_, i) => +(lo + i * 0.1).toFixed(1));
    let s = `<text x="${left}" y="14" class="fb-legend"><tspan class="fb-c-f">●</tspan> ${esc(names[0])}   <tspan class="fb-c-o">○</tspan> ${esc(names[1])}</text>
      <text x="${W - right + 8}" y="14" class="fb-legend">difference</text>`;
    for (const v of ticks) {
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${top - 4}" y2="${H - 24}" class="fb-grid"/><text x="${x(v)}" y="${H - 8}" class="fb-axis" text-anchor="middle">${v}</text>`;
    }
    ds.forEach((r, i) => {
      const y = top + i * row + row / 2;
      s += `<g><title>${esc(r.n)}: ${f(r.va)} vs ${f(r.vb)} (${sd(r.d)})</title>
        <rect x="0" y="${y - row / 2}" width="${W}" height="${row}" class="fb-band${i % 2 ? " fb-band--alt" : ""}"/>
        <text x="${left - 10}" y="${y + 4}" class="fb-label" text-anchor="end">${esc(r.n)}</text>
        <line x1="${x(r.vb)}" x2="${x(r.va)}" y1="${y}" y2="${y}" class="fb-link fb-link--${cls(r.d)}"/>
        <circle cx="${x(r.vb)}" cy="${y}" r="4.5" class="fb-c-o"/>
        <circle cx="${x(r.va)}" cy="${y}" r="5" class="fb-c-f"/>
        <text x="${W - right + 8}" y="${y + 4}" class="fb-axis fb-${cls(r.d)}">${sd(r.d)}</text></g>`;
    });
    const better = ds.filter((r) => cls(r.d) === "up").length, worse = ds.filter((r) => cls(r.d) === "down").length;
    return `<p>${METRICS[metric]} per dataset, largest gain first: <b>${esc(names[0])}</b> is better on
      <span class="fb-up">${better}</span> and worse on <span class="fb-down">${worse}</span> of ${ds.length} datasets.</p>
      <div class="fb-table-wrap"><svg viewBox="0 0 ${W} ${H}" class="fb-chart" role="img" aria-label="Per-dataset ${METRICS[metric]} of the two models">${s}</svg></div>`;
  }

  function draw() {
    const other = sel.value;
    if (!other) { out.innerHTML = ""; return; }
    out.innerHTML = `<p class="fb-muted">Loading…</p>`;
    Promise.all([get(sel.dataset.self), get(other)]).then(([a, b]) => {
      const tabs = Object.entries(METRICS).map(([k, l]) =>
        `<button type="button" data-m="${k}" class="fb-tab${k === metric ? " is-on" : ""}" aria-pressed="${k === metric}">${l}</button>`).join("");
      out.innerHTML = `${kpis(a, b)}<div class="fb-tabs" role="group" aria-label="Metric">${tabs}</div>${chart(a, b, [sel.dataset.self, other])}`;
      out.querySelectorAll(".fb-tab").forEach((t) => t.addEventListener("click", () => { metric = t.dataset.m; draw(); }));
      if (!ex || !ex.querySelector(".fb-other") || ex.dataset.shown !== other) { ex && (ex.dataset.shown = other); tilesCompare(other); }
    }).catch((e) => { out.innerHTML = `<p class="fb-note">Could not load the results (${esc(e.message)}).</p>`; });
  }

  // ---------------------------------------------------------------- example tiles, side by side
  // This page's model on the left of a divider, the compared model on the right; a slider under
  // each image moves the divider, and a badge on each side names the model shown there.
  const ex = document.querySelector(".fb-examples");
  const BINS = [[0.85, "q4"], [0.75, "q3"], [0.65, "q2"], [0.5, "q1"]];
  const iouClass = (v) => (v == null ? "fb-fp" : "fb-" + (BINS.find(([t]) => v >= t) || BINS[3])[1]);
  const pts = (xy) => { let o = ""; for (let k = 0; k + 1 < xy.length; k += 2) o += `${xy[k]},${xy[k + 1]} `; return o; };
  const NS = "http://www.w3.org/2000/svg";
  const tileCache = {};

  function tilesClear() {
    if (!ex) return;
    delete ex.dataset.shown;
    ex.querySelectorAll(".fb-cmp").forEach((n) => n.remove());
    ex.querySelectorAll(".fb-tile-box.is-cmp").forEach((b) => b.classList.remove("is-cmp"));
    ex.querySelectorAll(".fb-count[data-own]").forEach((c) => { c.innerHTML = c.dataset.own; });
  }

  function tilesCompare(other) {
    if (!ex) return;
    const self = ex.dataset.self;
    const url = new URL(`${encodeURIComponent(other)}.json`, new URL(ex.dataset.tiles, location.href));
    tileCache[other] = tileCache[other] || fetch(url).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
    tileCache[other].then((data) => {
      if (sel.value !== other) return;  // the reader moved on while this loaded
      tilesClear();
      ex.querySelectorAll("figure.fb-tile[data-tile]").forEach((fig) => {
        const t = data.tiles[fig.dataset.tile];
        const box = fig.querySelector(".fb-tile-box");
        const own = box.querySelector("svg");
        if (!t || !own) return;
        // The compared model's layer: the same hand-drawn outlines, marked found or missed by IT.
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", own.getAttribute("viewBox"));
        svg.setAttribute("preserveAspectRatio", "none");
        svg.classList.add("fb-cmp", "fb-other");
        const gt = own.querySelector("g.fb-layer").cloneNode(true);
        gt.querySelectorAll("polygon").forEach((p, i) => {
          const v = t.gt_iou[i];
          p.setAttribute("class", "fb-gt" + (v == null ? " fb-miss" : ""));
          p.querySelector("title").textContent = v == null ? "missed animal" : `animal found, IoU ${v.toFixed(2)}`;
        });
        const pr = document.createElementNS(NS, "g");
        pr.setAttribute("class", "fb-layer");
        pr.innerHTML = t.pred.map((p) => `<polygon points="${pts(p.xy)}" class="fb-pr ${iouClass(p.iou)}"><title>${
          p.iou == null ? "false detection" : `IoU ${p.iou.toFixed(2)}`}, confidence ${p.conf.toFixed(2)}</title></polygon>`).join("");
        svg.append(gt, pr);
        box.append(svg);
        box.insertAdjacentHTML("beforeend",
          `<div class="fb-cmp fb-wipe-line" aria-hidden="true"></div>
           <span class="fb-cmp fb-badge fb-badge--a">◀ ${esc(self)}</span>
           <span class="fb-cmp fb-badge fb-badge--b">${esc(other)} ▶</span>`);
        box.classList.add("is-cmp");
        box.style.setProperty("--x", "50%");
        box.insertAdjacentHTML("afterend",
          `<div class="fb-cmp fb-wipe"><span class="fb-wipe-a">${esc(self)}</span>
           <input type="range" min="0" max="100" value="50" step="1"
             aria-label="Divider between ${esc(self)} (left) and ${esc(other)} (right)">
           <span class="fb-wipe-b">${esc(other)}</span></div>`);
        fig.querySelector(".fb-wipe input").addEventListener("input", (e) => {
          box.style.setProperty("--x", `${e.target.value}%`);
        });
        const c = fig.querySelector(".fb-count");
        if (!c.dataset.own) c.dataset.own = c.innerHTML;
        const found = t.gt_iou.filter((v) => v != null).length;
        const fp = t.pred.filter((p) => p.iou == null).length;
        c.innerHTML = `<span class="fb-wipe-a">${c.dataset.own}</span> <span aria-hidden="true">|</span> <span class="fb-wipe-b">${found}/${t.gt_iou.length} found · ${fp} false</span>`;
      });
    }).catch((e) => {
      const n = document.createElement("p");
      n.className = "fb-note fb-cmp";
      n.textContent = `Could not load ${other}'s example tiles (${e.message}).`;
      ex.prepend(n);
    });
  }

  sel.addEventListener("change", () => { if (!sel.value) tilesClear(); });
  sel.addEventListener("change", draw);
  const want = new URLSearchParams(location.search).get("compare");
  if (want && [...sel.options].some((o) => o.value === want)) { sel.value = want; draw(); }
})();
