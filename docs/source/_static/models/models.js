// Draws the model versions page from models.json. To publish a model or a benchmark result,
// edit models.json; nothing here needs to change.
(function () {
  "use strict";
  const root = document.getElementById("fb-models");
  if (!root) return;
  const here = document.currentScript ? document.currentScript.src : "_static/models/models.js";
  const url = new URL("models.json", here);

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
  const mb = (b) => (b ? `${(b / 1e6).toFixed(0)} MB` : "–");
  const STATUS = {
    default: ["Default", "The model fb_predict and Predictor use unless told otherwise."],
    released: ["Released", "Published in the model zoo."],
    candidate: ["Candidate", "Trained and documented, not yet released."],
    retired: ["Retired", "Superseded; kept for reproducibility."],
  };

  function chip(status) {
    const [label, title] = STATUS[status] || [status, ""];
    return `<span class="fb-chip fb-chip--${esc(status)}" title="${esc(title)}">${esc(label)}</span>`;
  }

  function scoreCell(m) {
    if (m.benchmark && m.benchmark.overall) {
      const o = m.benchmark.overall;
      return `<b>${num(o.f1)}</b><span class="fb-muted"> · P ${num(o.precision)} · R ${num(o.recall)}</span>`;
    }
    return `<span class="fb-muted">pending</span>`;
  }

  function table(models) {
    const rows = models.map((m) => `
      <tr>
        <th scope="row"><a href="#model-${esc(m.name)}"><code>${esc(m.name)}</code></a></th>
        <td>${chip(m.status)}</td>
        <td class="fb-num">${esc(m.date || "–")}</td>
        <td>${esc(m.size || "–")}</td>
        <td class="fb-num">${scoreCell(m)}</td>
        <td class="fb-num">${m.url ? `<a href="${esc(m.url)}">${mb(m.bytes)}</a>` : `<span class="fb-muted">${mb(m.bytes)}</span>`}</td>
      </tr>`).join("");
    return `<div class="fb-table-wrap"><table class="fb-table">
      <thead><tr><th>Model</th><th>Status</th><th>Date</th><th>Size</th><th>Benchmark F1</th><th>Weights</th></tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  function dl(pairs) {
    const items = pairs.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`);
    return items.length ? `<dl class="fb-dl">${items.join("")}</dl>` : "";
  }

  function detail(m) {
    const t = m.training;
    const training = t ? dl([
      ["Code", t.commit ? `<a href="https://github.com/darsa-group/flat-bug/commit/${esc(t.commit)}"><code>${esc(t.commit.slice(0, 12))}</code></a>${t.commit_note ? ` <span class="fb-muted">(${esc(t.commit_note)})</span>` : ""}` : null],
      ["Starting point", t.base_model ? `<code>${esc(t.base_model)}</code>` : null],
      ["Epochs", t.epochs],
      ["Data", t.images ? `${Number(t.images).toLocaleString("en")} images from ${t.datasets} datasets` : null],
      ["Hardware", esc(t.hardware)],
      ["Framework", t.ultralytics ? `ultralytics ${esc(t.ultralytics)}` : null],
    ]) : `<p class="fb-muted">No training record: this model predates training manifests.</p>`;
    const inf = m.inference
      ? `<pre class="fb-pre">${esc(Object.entries(m.inference).map(([k, v]) => `${k}: ${v}`).join("\n"))}</pre>`
      : `<p class="fb-muted">Default predictor settings.</p>`;
    const ie = m.internal_eval;
    const internal = ie ? `<p>F1 <b>${num(ie.f1, 4)}</b> · precision ${num(ie.precision, 4)} · recall ${num(ie.recall, 4)}
      on ${Number(ie.instances).toLocaleString("en")} animals <span class="fb-muted">(${esc(ie.label)})</span></p>` : "";
    return `<section class="fb-model" id="model-${esc(m.name)}">
      <h3><code>${esc(m.name)}</code> ${chip(m.status)}</h3>
      ${m.summary ? `<p>${esc(m.summary)}</p>` : ""}
      ${dl([["Architecture", esc(m.architecture)], [m.status === "candidate" ? "Trained" : "Released", esc(m.date)], ["Weights", m.bytes ? `${mb(m.bytes)}${m.sha256 ? ` · sha256 <code>${esc(m.sha256.slice(0, 16))}…</code>` : ""}` : null]])}
      <h4>How it was trained</h4>${training}
      <h4>Predictor settings</h4>${inf}
      ${internal ? `<h4>Evaluation during training</h4>${internal}` : ""}
      <h4>Benchmark</h4>${m.benchmark ? benchmarkDetail(m.benchmark) : `<p class="fb-muted">Not yet benchmarked.</p>`}
    </section>`;
  }

  // Per-dataset F1 as a dot strip, one row per dataset, one dot per benchmarked model.
  function benchmarkDetail(b) {
    const ds = Object.entries(b.datasets || {}).sort((a, c) => a[0].localeCompare(c[0]));
    if (!ds.length) return "";
    const W = 640, row = 22, left = 190, right = 24, H = ds.length * row + 30;
    const x = (v) => left + v * (W - left - right);
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((v) =>
      `<line x1="${x(v)}" x2="${x(v)}" y1="0" y2="${H - 22}" class="fb-grid"/><text x="${x(v)}" y="${H - 6}" class="fb-axis" text-anchor="middle">${v}</text>`).join("");
    const dots = ds.map(([name, s], i) => {
      const y = i * row + row / 2;
      return `<text x="${left - 8}" y="${y + 4}" class="fb-axis" text-anchor="end">${esc(name)}</text>
        <circle cx="${x(s.f1 || 0)}" cy="${y}" r="5" class="fb-dot"><title>${esc(name)}: F1 ${num(s.f1)} (GT ${s.gt}, pred ${s.pred})</title></circle>`;
    }).join("");
    return `<p>F1 <b>${num(b.overall.f1, 4)}</b> · precision ${num(b.overall.precision, 4)} · recall ${num(b.overall.recall, 4)}
      · mask IoU ${num(b.overall.mean_iou)}</p>
      <div class="fb-table-wrap"><svg viewBox="0 0 ${W} ${H}" class="fb-strip" role="img" aria-label="F1 per benchmark dataset">${ticks}${dots}</svg></div>`;
  }

  fetch(url)
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then((data) => {
      const models = data.models || [];
      const b = data.benchmark || {};
      const pending = !models.some((m) => m.benchmark);
      root.innerHTML = `
        ${table(models)}
        ${pending ? `<p class="fb-note">Benchmark results are being computed for <b>${esc(b.name)}</b> (${esc(b.status)}); they will appear in the table as each model is run.</p>` : ""}
        <h2 id="versions">Versions</h2>
        ${models.map(detail).join("")}
        <p class="fb-muted">Updated ${esc(data.updated)}. Weights are served from the <a href="${esc(data.zoo_url)}">model zoo</a>.</p>`;
    })
    .catch((e) => {
      root.innerHTML = `<p class="fb-note">The model list could not be loaded (${esc(e.message)}). The weights are in the <a href="https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/">model zoo</a>.</p>`;
    });
})();
