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
    }).catch((e) => { out.innerHTML = `<p class="fb-note">Could not load the results (${esc(e.message)}).</p>`; });
  }

  sel.addEventListener("change", draw);
  const want = new URLSearchParams(location.search).get("compare");
  if (want && [...sel.options].some((o) => o.value === want)) { sel.value = want; draw(); }
})();
