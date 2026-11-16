"""Short results page (replaces the long per-episode report): headline, robot table with scenario shifts, encoder
probe table, the scenario figure and three side-by-side videos. Reads the result JSONs, so re-run it as jobs finish.

    .venv/bin/python build_summary.py      # -> report_summary/index.html (publish to the report artifact URL)
"""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

B = Path(__file__).parent
R8 = B / "results_8dof"
OUT = B / "report_summary"
SCN = ["fog", "glare", "lens", "shadow", "clutter", "background", "table"]
SCN_NICE = {"fog": "Fog", "glare": "Glare", "lens": "Wet lens", "shadow": "Shadows", "clutter": "Clutter",
            "background": "New background", "table": "Table­cloth (held out)"}

# (bc tag, label, how the backbone is trained, ours?)
ROBOT = [("pioneer_v8_delta", "PioneerNet v8 (RGB-D, DINOv3 teacher + light scene corruption)", "frozen", True),
         ("pioneer_v6_delta_r2", "PioneerNet v6 (RGB-D, + scene corruption)", "frozen", True),
         ("pioneer_v5_ft", "PioneerNet v5 (RGB-D)", "fine-tuned", True),
         ("pioneer_v5_delta", "PioneerNet v5 (RGB-D)", "frozen", True),
         ("pioneer_v5_rgbonly", "PioneerNet v5, RGB only", "frozen", True),
         ("pioneer_v6_deploy", "PioneerNet v6, real-robot cameras (wrist RGB only)", "frozen", True),
         ("pioneer_v5_deploy", "PioneerNet v5, real-robot cameras (wrist RGB only)", "frozen", True),
         ("dinov3_cnxt_ft", "DINOv3 ConvNeXt-T", "fine-tuned", False),
         ("act_r18_ft_delta", "ResNet-18 (ACT)", "fine-tuned", False),
         ("theia_s16", "Theia-S/16", "frozen", False),
         ("radio_b16", "C-RADIOv2-B/16", "frozen", False),
         ("dinov3_b16", "DINOv3 ViT-B/16", "frozen", False),
         ("fuse_s14_frozen", "DINOv2-S + DeFM-S (RGB-D)", "frozen", False),
         ("defm_s14_depthonly", "DeFM-S (depth only)", "frozen", False),
         ("cradio4_so400m", "C-RADIOv4-SO400M (2026)", "frozen", False),
         ("siglino_70m", "SigLino-70M (2026)", "frozen", False),
         ("blind_proprio", "No camera (joints only)", "", False)]

# (probe file, config, label, ours?)
PROBE = [("probe_pioneer_v6", "pioneernet_v6@ego+wrist_left", "PioneerNet v6 (RGB-D)", True),
         ("probe_pioneer_v5", "pioneernet_v5@ego+wrist_left", "PioneerNet v5 (RGB-D)", True),
         ("probe_v5_rgbonly", "pioneernet_v5:rgb@ego+wrist_left", "PioneerNet v5, RGB only", True),
         ("probe_base", "radio_b16@ego+wrist_left", "C-RADIOv2-B/16", False),
         ("probe_base", "theia_s16@ego+wrist_left", "Theia-S/16", False),
         ("probe_2026_cradio4", "cradio4_so400m@ego+wrist_left", "C-RADIOv4-SO400M (2026)", False),
         ("probe_2026", "siglino_70m@ego+wrist_left", "SigLino-70M (2026)", False),
         ("probe_base", "dinov2_s14@ego+wrist_left,defm_s14@ego+wrist_left", "DINOv2-S + DeFM-S (the teachers)", False),
         ("probe_2025", "dinov3_b16@ego+wrist_left", "DINOv3 ViT-B/16", False),
         ("probe_2025", "siglip2_b16@ego+wrist_left", "SigLIP2 ViT-B/16", False),
         ("probe_base", "r18_imnet@ego+wrist_left", "ResNet-18 ImageNet (ACT)", False)]


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def robot_cells(tag):
    r = _load(R8 / "bc" / f"{tag}.json") or {}
    vals = {}
    for k in ("clean", "light", "color", "camera"):
        x = r.get(k)
        vals[k] = x["success"] * 100 if isinstance(x, dict) and "success" in x else None
        if vals[k] is None and k != "clean":     # shifts run separately with eval_policy.py (policy reruns)
            y = _load(R8 / "bc_exec" / f"{tag}__ensemble_m0.6_{k}.json")
            vals[k] = y["summary"]["success"] * 100 if y else None
    for s in SCN:
        x = _load(R8 / "bc_exec" / f"{tag}__ensemble_m0.6_{s}.json")
        vals[s] = x["summary"]["success"] * 100 if x else None
    return vals


def probe_cells(fname, cfg):
    d = _load(R8 / f"{fname}.json")
    if d is None:
        return None
    for r in (d if isinstance(d, list) else list(d.values())):
        if isinstance(r, dict) and r.get("config") == cfg:
            out = {}
            for c in ("clean", "light", "color", "camera"):
                x = r.get(f"attn/{c}", {}).get("peg_minus_hole_xy_mm")
                out[c] = x.get("mean") if isinstance(x, dict) else x
            return out
    return None


NOT_PLANNED = {("blind_proprio", c) for c in ["light", "color"] + SCN} | \
              {("defm_s14_depthonly", c) for c in ["light", "color"]}   # light and colour leave the depth image unchanged


def cell(v, best=None, lower=False, unit="", planned=True):
    if v is None:
        return "<td class='na'>…</td>" if planned else "<td class='na'>—</td>"
    b = best is not None and abs(v - best) < 1e-6
    return f"<td class='num{' best' if b else ''}'>{v:.0f}{unit}</td>" if not lower else \
        f"<td class='num{' best' if b else ''}'>{v:.1f}</td>"


def main():
    OUT.mkdir(exist_ok=True)
    cols = ["clean", "light", "color", "camera"] + SCN
    rows = [(t, lab, tr, ours, robot_cells(t)) for t, lab, tr, ours in ROBOT]
    rows = [r for r in rows if any(v is not None for v in r[4].values()) or r[0] in
            ("pioneer_v6_delta_r2", "pioneer_v6_deploy", "pioneer_v5_rgbonly", "fuse_s14_frozen", "defm_s14_depthonly", "cradio4_so400m", "siglino_70m", "pioneer_v5_ft")]
    best = {c: max([r[4][c] for r in rows if r[4][c] is not None] or [None]) if
            any(r[4][c] is not None for r in rows) else None for c in cols}
    head = "".join(f"<th>{n}</th>" for n in ["Clean", "Light", "Colour", "Camera moved"] + [SCN_NICE[s] for s in SCN])
    body = []
    for t, lab, tr, ours, v in rows:
        body.append(f"<tr class='{'ours' if ours else ''}'><th scope='row'>{html.escape(lab)}"
                    f"{'<span class=tr>' + tr + '</span>' if tr else ''}</th>"
                    + "".join(cell(v[c], best[c], planned=(t, c) not in NOT_PLANNED) for c in cols) + "</tr>")
    robot_html = f"<thead><tr><th>Vision backbone</th>{head}</tr></thead><tbody>{''.join(body)}</tbody>"

    prow = [(lab, ours, probe_cells(f, c)) for f, c, lab, ours in PROBE]
    pcols = ["clean", "light", "color", "camera"]
    pbest = {c: min([p[c] for _, _, p in prow if p and p.get(c) is not None] or [None]) for c in pcols}
    pbody = []
    for lab, ours, p in prow:
        tds = "".join(cell(p.get(c) if p else None, pbest[c], lower=True) for c in pcols)
        pbody.append(f"<tr class='{'ours' if ours else ''}'><th scope='row'>{html.escape(lab)}</th>{tds}</tr>")
    probe_html = ("<thead><tr><th>Encoder (frozen)</th><th>Clean</th><th>Light</th><th>Colour</th><th>Camera moved</th>"
                  f"</tr></thead><tbody>{''.join(pbody)}</tbody>")

    status = "All runs finished."
    jobs = B / "results" / "jobs"
    pending = [t for t, *_ in ROBOT if any(robot_cells(t)[c] is None and (t, c) not in NOT_PLANNED
                                            for c in ["clean", "light", "color", "camera"] + SCN)]
    if pending:
        status = f"Cells marked … are running or queued ({len(pending)} rows); the page is rebuilt as they finish. — = not tested."
    tpl = (B / "summary_template.html").read_text()
    page = (tpl.replace("{{ROBOT}}", robot_html).replace("{{PROBE}}", probe_html)
               .replace("{{UPDATED}}", dt.datetime.now().strftime("%b %d, %H:%M")).replace("{{STATUS}}", status))
    (OUT / "index.html").write_text(page)
    print("wrote", OUT / "index.html", "| pending rows:", len(pending), "| jobs dir:", jobs.exists())


if __name__ == "__main__":
    main()
