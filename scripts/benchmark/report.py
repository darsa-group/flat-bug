"""The human-readable side of a benchmark run: report.html plus example overlays.

For each dataset the report shows two images - the one the model did worst on, and a typical one
(median F1) - with predictions and ground truth drawn on them:

    green   prediction matched to ground truth (IoU >= 0.5)
    red     prediction with no match (false positive)
    amber   ground truth with no match (missed)

When a baseline results.json is given, every score gets its change against the baseline, and
datasets that moved are flagged. A baseline is only comparable when it was scored on the same
benchmark (sha256) with the same scorer version; otherwise the comparison is refused.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import scoring

# Benchmark images are trusted local files, some of them scans of 100+ megapixels.
Image.MAX_IMAGE_PIXELS = None

GREEN, RED, AMBER = (40, 200, 90), (230, 50, 50), (255, 176, 0)
MAX_SIDE = 1400


def overlay(image_path: Path, gt, pred, dest: Path):
    gh, ph, _ = scoring.match(gt, pred)
    im = Image.open(image_path).convert("RGB")
    s = min(1.0, MAX_SIDE / max(im.size))
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    w = max(2, round(max(im.size) / 500))

    def outline(p, colour):
        xy = [(x * s, y * s) for x, y in np.asarray(p.exterior.coords)]
        d.line(xy + xy[:1], fill=colour, width=w)

    for p, hit in zip(gt, gh):
        if not hit:
            outline(p, AMBER)
    for p, hit in zip(pred, ph):
        outline(p, GREEN if hit else RED)
    im.save(dest, quality=82)


def fmt(x, digits=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def delta(new, old, digits=3):
    if old is None or any(isinstance(v, float) and math.isnan(v) for v in (new, old)):
        return ""
    dv = new - old
    cls = "up" if dv > 0.0005 else ("down" if dv < -0.0005 else "flat")
    return f'<span class="{cls}">{dv:+.{digits}f}</span>'


def leak_text(lk: dict) -> str:
    if not lk.get("checked"):
        return f"not checked: {lk.get('reason', '')}"
    return (f"{lk['train_overlap']} benchmark images in the model's training split, "
            f"{lk['val_overlap']} in its validation split")


def write(out: Path, results: dict, rows: list[dict], polys: dict, bench: Path, baseline: dict | None):
    ex = out / "examples"
    ex.mkdir(exist_ok=True)
    if baseline and (baseline["benchmark"]["sha256"] != results["benchmark"]["sha256"]
                     or baseline["scorer"]["version"] != results["scorer"]["version"]):
        baseline = None
        note = "<p class=warn>Baseline ignored: it was scored on a different benchmark or scorer version.</p>"
    else:
        note = ""

    cards = []
    for d, (gt, pr) in polys.items():
        r = [x for x in rows if x["dataset"] == d and (x["gt"] or x["pred"])]
        if not r:
            continue
        key = [(-1 if math.isnan(x["f1"]) else x["f1"]) for x in r]
        worst = r[int(np.argmin(key))]
        typical = r[int(np.argsort(key)[len(key) // 2])]
        picks = [("worst", worst)] + ([("typical", typical)] if typical is not worst else [])
        figs = []
        for label, x in picks:
            name = f"{d}__{label}.jpg"
            overlay(bench / d / "images" / x["image"], gt[x["image"]], pr.get(x["image"], []), ex / name)
            figs.append(f'<figure><a href="examples/{name}"><img src="examples/{name}" loading="lazy"></a>'
                        f'<figcaption><b>{label}</b> · {html.escape(x["image"])} · F1 {fmt(x["f1"])} · '
                        f'GT {x["gt"]} · pred {x["pred"]}</figcaption></figure>')
        cards.append(f'<section class=card><h3>{html.escape(d)}</h3><div class=figs>{"".join(figs)}</div></section>')

    def row(name, s, b):
        cells = [f"<td class=name>{html.escape(name)}</td>", f"<td>{s['gt']}</td>", f"<td>{s['pred']}</td>"]
        for k in ("precision", "recall", "f1", "mean_iou"):
            cells.append(f"<td>{fmt(s[k])} {delta(s[k], b.get(k)) if b else ''}</td>")
        return "<tr>" + "".join(cells) + "</tr>"

    bds = (baseline or {}).get("datasets", {})
    body = [row("all datasets", results["overall"], (baseline or {}).get("overall"))]
    body += [row(d, s, bds.get(d)) for d, s in sorted(results["datasets"].items(), key=lambda kv: kv[0].lower())]

    m, c, b, r = results["model"], results["code"], results["benchmark"], results["run"]
    env = c["environment"]
    o = results["overall"]
    status = ("" if results["valid"] else
              "<p class=warn>Not a valid benchmark score: "
              + ("only some datasets were run. " if b["datasets_scored"] != "all" else "")
              + (f"{results['missing_predictions']} images have no prediction. " if results["missing_predictions"] else "")
              + (f"{results['leakage']['train_overlap']} benchmark images were in the model's training data."
                 if results["leakage"].get("train_overlap") else "")
              + "</p>")
    vs = (f" vs {html.escape(baseline['model']['name'])} @ {baseline['code']['commit'][:12]}" if baseline else "")
    page = f"""<!doctype html><html lang=en><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(m['name'])} · {c['commit'][:12]} · {html.escape(b['name'])}</title>
<style>
:root{{--bg:#fbfbf9;--fg:#1d2321;--mute:#5f6b66;--line:#dfe3df;--card:#fff;--up:#1d7a46;--down:#b3261e;--warn:#8a5a00}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151917;--fg:#e6ebe8;--mute:#9aa7a1;--line:#2c3430;--card:#1c2220;--up:#5fd08f;--down:#ff8a80;--warn:#f0b54a}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}}
h1{{font-size:1.5rem;margin:0 0 4px}} h2{{font-size:1.1rem;margin:32px 0 8px}} h3{{font-size:1rem;margin:0 0 8px}}
.mute{{color:var(--mute)}} .warn{{color:var(--warn);font-weight:600}}
.kpis{{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}} .kpi b{{display:block;font-size:1.6rem;font-variant-numeric:tabular-nums}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} .wrap{{overflow-x:auto}}
th,td{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td.name{{text-align:left}}
tr:first-child td{{font-weight:600}}
.up{{color:var(--up)}} .down{{color:var(--down)}} .flat{{color:var(--mute)}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:0}} dt{{color:var(--mute)}} dd{{margin:0;overflow-wrap:anywhere}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;margin:12px 0}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
figure{{margin:0}} img{{width:100%;height:auto;display:block;border-radius:4px}} figcaption{{font-size:.85rem;color:var(--mute);margin-top:4px}}
.key span{{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 4px 0 12px;vertical-align:-1px}}
</style>
<h1>{html.escape(m['name'])} <span class=mute>run by</span> {c['commit'][:12]}</h1>
<p class=mute>{html.escape(b['name'])} · scorer v{results['scorer']['version']} (IoU ≥ {results['scorer']['iou_threshold']}, instances ≥ {results['scorer']['min_size_px']:g} px){html.escape(vs)}</p>
{status}{note}
<div class=kpis>
<div class=kpi><span class=mute>F1</span><b>{fmt(o['f1'], 4)}</b></div>
<div class=kpi><span class=mute>Precision</span><b>{fmt(o['precision'], 4)}</b></div>
<div class=kpi><span class=mute>Recall</span><b>{fmt(o['recall'], 4)}</b></div>
<div class=kpi><span class=mute>Mean IoU (matched)</span><b>{fmt(o['mean_iou'])}</b></div>
<div class=kpi><span class=mute>Instances</span><b>{o['gt']:,}</b></div>
</div>
<h2>Per dataset</h2>
<div class=wrap><table><tr><th>Dataset</th><th>GT</th><th>Pred</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mean IoU</th></tr>
{''.join(body)}</table></div>
<h2>Provenance</h2>
<div class=card><dl>
<dt>Model</dt><dd>{html.escape(m['name'])} · weights sha256 {m['weights_sha256'][:16]}… · trained at commit {html.escape(str(m['training_commit']))}</dd>
<dt>Inference config</dt><dd>{html.escape(str(m['inference_config'] or 'flat-bug defaults'))}</dd>
<dt>Code</dt><dd>{c['commit']} · {html.escape(c.get('subject', ''))} · {html.escape(c.get('author_date', ''))}</dd>
<dt>Environment</dt><dd>python {env['python']} · torch {env['torch']} (CUDA {env['cuda']}) · ultralytics {env['ultralytics']} · uv.lock {str(c['uv_lock_sha256'])[:16]}…</dd>
<dt>Hardware</dt><dd>{html.escape(r['host'])} · {html.escape(str(env['device_name'] or 'cpu'))}</dd>
<dt>Benchmark</dt><dd>{html.escape(b['name'])} · sha256 {b['sha256'][:16]}… · {html.escape(b['source']['concept_doi'])}</dd>
<dt>Leak check</dt><dd>{html.escape(leak_text(results['leakage']))}</dd>
<dt>Run</dt><dd>{r['started']} → {r['finished']}</dd>
</dl></div>
<h2>Examples</h2>
<p class="mute key">Worst and typical image per dataset.<span style="background:rgb{GREEN}"></span>matched prediction<span style="background:rgb{RED}"></span>false positive<span style="background:rgb{AMBER}"></span>missed ground truth</p>
{''.join(cards)}
</html>"""
    (out / "report.html").write_text(page)
