"""Build the model pages from the model registry, at documentation build time.

Inputs, all under ``_static/models/``:

``models.json``
    The registry: every model, its status, date, size, download URL and a short summary.
``results/<name>.json``
    A model's benchmark result (the summary of a run_benchmark.py results.json; see
    ``scripts/benchmark/publish_result.py``). Optional: a model without one shows as pending.
``manifests/<name>.yaml``
    The model bundle's full manifest. Optional.

Outputs, regenerated on every build and not under version control:

``_generated/models_table.html``
    The table on the models page: one row per model with its benchmark scores.
``models/<name>.md``
    One page per model: scores, a per-dataset chart and table, how it was trained, the full
    manifest, and a picker to contrast it with another benchmarked model.

Everything is plain HTML and inline SVG, computed here; the only script on the site is the one
behind the comparison picker, which runs when a reader asks for a comparison.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import yaml

STATUS = {
    "default": ("Default", "The model fb_predict and Predictor use unless told otherwise."),
    "released": ("Released", "Published in the model zoo."),
    "candidate": ("Candidate", "Trained and documented, not yet released."),
    "retired": ("Retired", "Superseded; kept for reproducibility."),
}


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def num(x, d: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def mb(b) -> str:
    return f"{b / 1e6:.0f} MB" if b else "–"


def chip(status: str) -> str:
    label, title = STATUS.get(status, (status, ""))
    return f'<span class="fb-chip fb-chip--{esc(status)}" title="{esc(title)}">{esc(label)}</span>'


def load(static: Path):
    reg = json.loads((static / "models.json").read_text())
    for m in reg["models"]:
        r = static / "results" / f"{m['name']}.json"
        m["results"] = json.loads(r.read_text()) if r.exists() else None
        mf = static / "manifests" / f"{m['name']}.yaml"
        m["manifest_text"] = mf.read_text() if mf.exists() else None
    return reg


# ------------------------------------------------------------------ table on the models page
def table(reg: dict) -> str:
    rows = []
    for m in reg["models"]:
        o = (m["results"] or {}).get("overall")
        scores = (f'<td class="fb-num"><b>{num(o["f1"])}</b></td><td class="fb-num">{num(o["precision"])}</td>'
                  f'<td class="fb-num">{num(o["recall"])}</td><td class="fb-num">{num(o["mean_iou"])}</td>'
                  if o else '<td class="fb-muted" colspan="4">not yet benchmarked</td>')
        weights = (f'<a href="{esc(m["url"])}">{mb(m.get("bytes"))}</a>' if m.get("url")
                   else f'<span class="fb-muted">{mb(m.get("bytes"))}</span>')
        page = f'models/{m["name"]}.html'
        rows.append(
            f'<tr class="fb-row" data-href="{page}">'
            f'<th scope="row"><a href="{page}"><code>{esc(m["name"])}</code></a></th>'
            f'<td>{chip(m["status"])}</td><td class="fb-num">{esc(m.get("date") or "–")}</td>'
            f'<td>{esc(m.get("size") or "–")}</td>{scores}<td class="fb-num">{weights}</td></tr>')
    b = reg.get("benchmark", {})
    note = ("" if any(m["results"] for m in reg["models"]) else
            f'<p class="fb-note">Benchmark results are being computed for <b>{esc(b.get("name"))}</b>; '
            f'they will appear here as each model is run.</p>')
    return f"""<div class="fb-table-wrap"><table class="fb-table fb-table--models">
<thead><tr><th>Model</th><th>Status</th><th>Date</th><th>Size</th>
<th title="harmonic mean of precision and recall">F1</th><th title="share of detections that are animals">Precision</th>
<th title="share of animals that are found">Recall</th><th title="how closely the outlines follow the hand-drawn ones">Mask IoU</th>
<th>Weights</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{note}
<script>document.querySelectorAll('tr.fb-row').forEach(r => r.addEventListener('click', e => {{
  if (!e.target.closest('a')) location.href = r.dataset.href; }}));</script>"""


# ------------------------------------------------------------------ per-dataset chart
def axis_start(values) -> float:
    """Where the score axis starts: the 0.1 step below the lowest value, and never above 0.5.

    Scores mostly sit between 0.7 and 1, and a 0-1 axis would squeeze them into a third of the
    width; the start is printed on the axis.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return min(0.5, math.floor(min(vals, default=0) * 10) / 10)


def dataset_chart(datasets: dict) -> str:
    """Dot plot, one row per dataset: precision, recall and F1 on a shared axis."""
    ds = sorted(datasets.items(), key=lambda kv: kv[0].lower())
    W, row, left, right, top = 720, 24, 200, 70, 26
    H = top + len(ds) * row + 28
    lo = axis_start(s[k] for _, s in ds for k in ("precision", "recall", "f1"))

    def x(v):
        v = lo if v is None or math.isnan(v) else max(lo, v)
        return left + (v - lo) / (1 - lo) * (W - left - right)

    parts = [f'<text x="{left}" y="14" class="fb-legend">'
             '<tspan class="fb-c-p">◆</tspan> precision  <tspan class="fb-c-r">▲</tspan> recall  '
             '<tspan class="fb-c-f">●</tspan> F1</text>',
             f'<text x="{W - right + 8}" y="14" class="fb-legend">animals</text>']
    for v in [round(lo + i * 0.1, 1) for i in range(round((1 - lo) / 0.1) + 1)]:
        parts.append(f'<line x1="{x(v)}" x2="{x(v)}" y1="{top - 4}" y2="{H - 24}" class="fb-grid"/>'
                     f'<text x="{x(v)}" y="{H - 8}" class="fb-axis" text-anchor="middle">{v:g}</text>')
    for i, (name, s) in enumerate(ds):
        y = top + i * row + row / 2
        tip = (f"{name}: F1 {num(s['f1'])}, precision {num(s['precision'])}, recall {num(s['recall'])}, "
               f"mask IoU {num(s['mean_iou'])} ({s['gt']} animals, {s['pred']} detections)")
        x0, x1 = sorted([x(s["precision"]), x(s["recall"])])
        parts.append(
            f'<g><title>{esc(tip)}</title>'
            f'<rect x="0" y="{y - row / 2}" width="{W}" height="{row}" class="fb-band{" fb-band--alt" if i % 2 else ""}"/>'
            f'<text x="{left - 10}" y="{y + 4}" class="fb-label" text-anchor="end">{esc(name)}</text>'
            f'<line x1="{x0}" x2="{x1}" y1="{y}" y2="{y}" class="fb-span"/>'
            f'<path d="M{x(s["precision"])},{y - 5} l5,5 l-5,5 l-5,-5 z" class="fb-c-p"/>'
            f'<path d="M{x(s["recall"])},{y - 5} l5,9 l-10,0 z" class="fb-c-r"/>'
            f'<circle cx="{x(s["f1"])}" cy="{y}" r="5" class="fb-c-f"/>'
            f'<text x="{W - right + 8}" y="{y + 4}" class="fb-axis">{s["gt"]:,}</text></g>')
    return (f'<div class="fb-table-wrap"><svg viewBox="0 0 {W} {H}" class="fb-chart" role="img" '
            f'aria-label="Precision, recall and F1 per benchmark dataset">{"".join(parts)}</svg></div>')


def dataset_table(datasets: dict) -> str:
    rows = "".join(
        f'<tr><th scope="row">{esc(n)}</th><td class="fb-num">{s["gt"]:,}</td><td class="fb-num">{s["pred"]:,}</td>'
        f'<td class="fb-num">{num(s["precision"])}</td><td class="fb-num">{num(s["recall"])}</td>'
        f'<td class="fb-num"><b>{num(s["f1"])}</b></td><td class="fb-num">{num(s["mean_iou"])}</td></tr>'
        for n, s in sorted(datasets.items(), key=lambda kv: kv[0].lower()))
    return (f'<div class="fb-table-wrap"><table class="fb-table"><thead><tr><th>Dataset</th><th>Animals</th>'
            f'<th>Detections</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mask IoU</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>')


# ------------------------------------------------------------------ model page
def dl(pairs) -> str:
    items = [f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in pairs if v not in (None, "", "None")]
    return f'<dl class="fb-dl">{"".join(items)}</dl>' if items else ""


def model_page(m: dict, reg: dict) -> str:
    r = m["results"]
    t = m.get("training") or {}
    commit = t.get("commit")
    commit_html = (f'<a href="https://github.com/darsa-group/flat-bug/commit/{esc(commit)}"><code>{esc(commit[:12])}</code></a>'
                   + (f' <span class="fb-muted">({esc(t["commit_note"])})</span>' if t.get("commit_note") else "")
                   if commit else None)
    training = dl([
        ("Code", commit_html),
        ("Starting point", f"<code>{esc(t['base_model'])}</code>" if t.get("base_model") else None),
        ("Epochs", esc(t.get("epochs"))),
        ("Data", f"{t['images']:,} images from {t['datasets']} datasets" if t.get("images") else None),
        ("Hardware", esc(t.get("hardware"))),
        ("Framework", f"ultralytics {esc(t['ultralytics'])}" if t.get("ultralytics") else None),
    ]) or '<p class="fb-muted">No training record: this model predates training manifests.</p>'
    inference = (f'<pre class="fb-pre">{esc(yaml.safe_dump(m["inference"], sort_keys=False).strip())}</pre>'
                 if m.get("inference") else '<p class="fb-muted">Default predictor settings.</p>')

    if r:
        o, code, bench = r["overall"], r["code"], r["benchmark"]
        others = [x for x in reg["models"] if x["results"] and x["name"] != m["name"]
                  and x["results"]["benchmark"]["sha256"] == bench["sha256"]
                  and x["results"]["scorer"]["version"] == r["scorer"]["version"]]
        options = "".join(f'<option value="{esc(x["name"])}">{esc(x["name"])}</option>' for x in others)
        compare = (f"""<h2 id="compare">Compare</h2>
<p><label>Contrast <code>{esc(m['name'])}</code> with
<select class="fb-compare" data-self="{esc(m['name'])}"><option value="">choose a model…</option>{options}</select></label></p>
<div class="fb-compare-out" aria-live="polite"></div>""" if others else "")
        scores = f"""<div class="fb-kpis">
<div><span>F1</span><b>{num(o['f1'], 4)}</b></div><div><span>Precision</span><b>{num(o['precision'], 4)}</b></div>
<div><span>Recall</span><b>{num(o['recall'], 4)}</b></div><div><span>Mask IoU</span><b>{num(o['mean_iou'])}</b></div>
<div><span>Animals</span><b>{o['gt']:,}</b></div></div>
<p class="fb-muted">On <b>{esc(bench['name'])}</b>, run with flat-bug
<a href="https://github.com/darsa-group/flat-bug/commit/{esc(code['commit'])}"><code>{esc(code['commit'][:12])}</code></a>
· scorer v{esc(r['scorer']['version'])} · {esc(r['run'].get('finished', '')[:10])}</p>
<h2 id="per-dataset">Per dataset</h2>
{dataset_chart(r['datasets'])}
<details class="fb-details"><summary>Per-dataset numbers</summary>{dataset_table(r['datasets'])}</details>
{compare}
<h2 id="run">Benchmark run</h2>
{dl([("Benchmark", f"{esc(bench['name'])} · sha256 <code>{esc(bench['sha256'][:16])}…</code>"),
     ("Code", f"<code>{esc(code['commit'])}</code> {esc(code.get('subject', ''))}"),
     ("Environment", esc(", ".join(f"{k} {v}" for k, v in code.get('environment', {}).items()
                                   if k in ('python', 'torch', 'ultralytics', 'device_name') and v))),
     ("Leak check", esc(r.get('leakage_text'))),
     ("Weights", f"sha256 <code>{esc(r['model']['weights_sha256'][:16])}…</code>")])}"""
    else:
        scores = '<p class="fb-note">Not yet benchmarked.</p>'

    manifest = (f'<details class="fb-details"><summary>Full manifest</summary>'
                f'<pre class="fb-pre fb-pre--long">{esc(m["manifest_text"])}</pre></details>'
                if m.get("manifest_text") else "")
    when = "Trained" if m["status"] == "candidate" else "Released"
    head = dl([("Status", chip(m["status"])), (when, esc(m.get("date"))), ("Size", esc(m.get("size"))),
               ("Architecture", esc(m.get("architecture"))),
               ("Weights", (f'<a href="{esc(m["url"])}">{mb(m.get("bytes"))}</a>' if m.get("url") else mb(m.get("bytes")))
                + (f' · sha256 <code>{esc(m["sha256"][:16])}…</code>' if m.get("sha256") else ""))])
    return f"""# {m['name']}

```{{raw}} html
<p class="fb-back"><a href="../models.html">← All models</a></p>
{f'<p>{esc(m["summary"])}</p>' if m.get('summary') else ''}
{head}
{scores}
<h2 id="training">How it was trained</h2>
{training}
<h2 id="settings">Predictor settings</h2>
{inference}
{manifest}
<script src="../_static/models/compare.js" defer></script>
```
"""


def generate(app):
    src = Path(app.srcdir)
    static = src / "_static" / "models"
    reg = load(static)
    gen = src / "_generated"
    gen.mkdir(exist_ok=True)
    (gen / "models_table.html").write_text(table(reg))
    # The sidebar lists models in registry order (newest first), not alphabetically.
    (gen / "models_toc.md").write_text(
        "```{toctree}\n:hidden:\n\n" + "".join(f"models/{m['name']}\n" for m in reg["models"]) + "```\n")
    pages = src / "models"
    pages.mkdir(exist_ok=True)
    keep = set()
    for m in reg["models"]:
        p = pages / f"{m['name']}.md"
        p.write_text(model_page(m, reg))
        keep.add(p.name)
    for p in pages.glob("*.md"):  # a model removed from the registry loses its page
        if p.name not in keep:
            p.unlink()


def setup(app):
    app.connect("builder-inited", generate)
    return {"version": "1", "parallel_read_safe": True}
