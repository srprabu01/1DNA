"""A single-file HTML readout for one track.

Design brief: this is an instrument panel, not a dashboard. Its one job is to
make the difference between a *measurement* and a *model output* impossible to
miss, because every judgement the engine makes rests on one or the other and
they fail differently. Amber is measured. Violet is inferred. Nothing else in
the page carries colour.

The hero is the track's own loudness trace with the detected sections beneath
it and the three excerpt windows bracketed on top — literally the slice of
audio the encoders saw. If an embedding looks wrong, this is where you find out
that the engine listened to the wrong ten seconds.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

CSS = """
:root {
  --ground: #161a1d;
  --panel: #1c2226;
  --rule: #2b3439;
  --ink: #dae0e4;
  --muted: #8b979e;
  --measured: #e8a13a;
  --inferred: #9a8cff;
  --warn: #e2685f;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 24px 72px;
  background: var(--ground); color: var(--ink);
  font: 400 16px/1.55 "Helvetica Neue", Helvetica, Inter, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1000px; margin: 0 auto; }
header { padding: 48px 0 28px; border-bottom: 1px solid var(--rule); }
h1 { margin: 0; font-size: clamp(28px, 5vw, 46px); font-weight: 700;
     letter-spacing: -0.02em; line-height: 1.05; }
.byline { margin: 10px 0 0; font-size: 19px; color: var(--muted); }
.origin { display: flex; gap: 22px; margin: 22px 0 0; padding: 0; list-style: none;
          font-size: 13px; color: var(--muted); }
.origin li { display: flex; align-items: center; gap: 7px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.dot.m { background: var(--measured); }
.dot.i { background: var(--inferred); }

section { margin: 40px 0 0; }
h2 { font-size: 13px; font-weight: 600; letter-spacing: 0.04em; color: var(--muted);
     margin: 0 0 14px; }
.trace { background: var(--panel); border: 1px solid var(--rule); border-radius: 3px;
         padding: 14px 14px 8px; }
.trace svg { display: block; width: 100%; height: auto; }
.axis { display: flex; justify-content: space-between; font-size: 12px;
        color: var(--muted); font-variant-numeric: tabular-nums; padding-top: 6px; }

.grid { display: grid; gap: 1px; background: var(--rule);
        border: 1px solid var(--rule); border-radius: 3px; overflow: hidden;
        grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); }
.cell { background: var(--panel); padding: 14px 16px 15px; }
.cell .k { font-size: 12.5px; color: var(--muted); }
.cell .v { font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, monospace;
           font-size: 23px; font-variant-numeric: tabular-nums;
           letter-spacing: -0.01em; margin-top: 3px; }
.cell .u { font-size: 13px; color: var(--muted); margin-left: 3px; }
.cell .n { font-size: 12.5px; color: var(--muted); margin-top: 5px; }
.measured .v { color: var(--measured); }
.inferred .v { color: var(--inferred); }

.bars { border: 1px solid var(--rule); border-radius: 3px; background: var(--panel);
        padding: 4px 16px 12px; }
.bar { display: grid; grid-template-columns: 1fr 132px 62px; align-items: center;
       gap: 12px; padding: 9px 0; border-bottom: 1px solid var(--rule); font-size: 14px; }
.bar:last-child { border-bottom: 0; }
.track { height: 5px; background: #263036; border-radius: 3px; overflow: hidden; }
.fill { height: 100%; background: var(--inferred); }
.fill.m { background: var(--measured); }
.num { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px;
       color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }

.verdict { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px;
           margin-bottom: 12px; }
.verdict strong { font-size: 20px; font-weight: 600; }
.verdict.flag strong { color: var(--warn); }
.caveat { font-size: 13.5px; color: var(--muted); max-width: 62ch; margin: 12px 0 0; }
details { margin-top: 12px; }
summary { cursor: pointer; font-size: 13.5px; color: var(--muted); }
summary:focus-visible { outline: 2px solid var(--inferred); outline-offset: 3px; }
pre { overflow-x: auto; font-size: 12.5px; color: var(--muted); background: var(--panel);
      border: 1px solid var(--rule); border-radius: 3px; padding: 14px; }
footer { margin-top: 52px; padding-top: 18px; border-top: 1px solid var(--rule);
         font-size: 13px; color: var(--muted); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def render(track, features: dict, *, tags: list[tuple[str, float]] | None = None,
           slop: dict | None = None, taste: dict[str, float] | None = None,
           excerpts: list[dict] | None = None) -> str:
    env = features.get("_envelope") or []
    sections = features.get("sections") or []
    duration = float(features.get("duration_s", 0.0))
    title = html.escape(track.title or Path(track.path).stem)
    artist = html.escape(" · ".join(x for x in [track.artist, track.album,
                                                str(track.year) if track.year else None] if x))

    body = f"""<div class="wrap">
<header>
  <h1>{title}</h1>
  <p class="byline">{artist or "Unknown artist"}</p>
  <ul class="origin">
    <li><span class="dot m"></span>measured from the signal</li>
    <li><span class="dot i"></span>inferred by a model</li>
  </ul>
</header>

<section>
  <h2>What the engine listened to</h2>
  <div class="trace">
    {_trace_svg(env, sections, excerpts or [], duration)}
    <div class="axis"><span>0:00</span><span>{_mmss(duration / 2)}</span><span>{_mmss(duration)}</span></div>
  </div>
</section>

<section>
  <h2>Rhythm and tonality</h2>
  {_grid(_rhythm_cells(features), "measured")}
</section>

<section>
  <h2>Loudness and spectrum</h2>
  {_grid(_loudness_cells(features), "measured")}
</section>
{_tags_section(tags)}
{_taste_section(taste)}
{_slop_section(slop)}
<section>
  <h2>Everything else</h2>
  <details>
    <summary>Raw feature record</summary>
    <pre>{html.escape(json.dumps({k: v for k, v in features.items()
                                  if not k.startswith("_")}, indent=2))}</pre>
  </details>
</section>

<footer>Generated by deepcut from {html.escape(Path(track.path).name)}.
Measured values are reproducible from the audio; inferred values carry the error
of the model that produced them.</footer>
</div>"""

    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{title} — deepcut</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")


# ---------------------------------------------------------------------------

SECTION_TINTS = ["#3b4a52", "#4a4258", "#2f5049", "#57453a", "#3a3f5c", "#4d3f47"]


def _trace_svg(env: list[float], sections: list[dict], excerpts: list[dict],
               duration: float) -> str:
    """Loudness over time, with the encoded seconds picked out in amber.

    The whole envelope is drawn in slate; the excerpt windows are the same
    envelope redrawn through a clip path in the measured colour. So the hero
    reads as one continuous shape in which the parts the encoders actually
    heard are lit up — which is the single most useful thing to see when a
    recommendation looks wrong.
    """
    W, H = 1000, 196
    lane, gap = 20, 10
    body_h = H - lane - gap
    if not env:
        return (f'<svg viewBox="0 0 {W} {H}"><text x="12" y="30" fill="#8b979e" '
                f'font-size="13">No envelope stored for this track.</text></svg>')

    n = len(env)
    peak = max(env) or 1.0
    mid = body_h / 2
    top, bottom = [], []
    for i, v in enumerate(env):
        x = i * W / max(n - 1, 1)
        h = (v / peak) * (mid - 5)
        top.append(f"{x:.1f},{mid - h:.1f}")
        bottom.append(f"{x:.1f},{mid + h:.1f}")
    poly = " ".join(top + bottom[::-1])

    def span(start: float, length: float) -> tuple[float, float]:
        x0 = float(start) / max(duration, 1e-9) * W
        x1 = (float(start) + float(length)) / max(duration, 1e-9) * W
        return x0, max(x1 - x0, 2.0)

    clips, lit = [], []
    for i, e in enumerate(excerpts):
        x0, w = span(e["start"], e.get("seconds", 10))
        clips.append(f'<clipPath id="ex{i}"><rect x="{x0:.1f}" y="0" '
                     f'width="{w:.1f}" height="{body_h}"/></clipPath>')
        lit.append(f'<polygon points="{poly}" fill="#e8a13a" clip-path="url(#ex{i})"/>'
                   f'<line x1="{x0:.1f}" y1="{body_h - 2}" x2="{x0 + w:.1f}" y2="{body_h - 2}" '
                   f'stroke="#e8a13a" stroke-width="2"/>')

    bands = []
    for s in sections:
        x0, w = span(s["start"], float(s["end"]) - float(s["start"]))
        tint = SECTION_TINTS[int(s.get("label", 0)) % len(SECTION_TINTS)]
        bands.append(f'<rect x="{x0:.1f}" y="{body_h + gap}" width="{w:.1f}" '
                     f'height="{lane}" fill="{tint}" rx="2"/>')

    ticks = []
    minutes = int(duration // 60)
    for m in range(1, minutes + 1):
        x = m * 60 / max(duration, 1e-9) * W
        ticks.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{body_h}" '
                     f'stroke="#2b3439" stroke-width="1"/>')

    return (f'<svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Loudness over time. Amber marks the excerpt windows the '
            f'encoders were given; the strip beneath shows the detected sections.">'
            f'<defs>{"".join(clips)}</defs>'
            f'{"".join(ticks)}'
            f'<polygon points="{poly}" fill="#54646c"/>'
            f'{"".join(lit)}{"".join(bands)}</svg>')


def _grid(cells: list[tuple], kind: str) -> str:
    out = []
    for label, value, unit, note in cells:
        out.append(
            f'<div class="cell {kind}"><div class="k">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(str(value))}'
            f'{f"<span class=u>{html.escape(unit)}</span>" if unit else ""}</div>'
            f'{f"<div class=n>{html.escape(note)}</div>" if note else ""}</div>')
    return f'<div class="grid">{"".join(out)}</div>'


def _rhythm_cells(f: dict) -> list[tuple]:
    bpm_note = "half/double ambiguous" if f.get("bpm_half_double_ambiguous") else \
        f"confidence {f.get('bpm_confidence', 0):.2f}"
    micro = float(f.get("microtiming_ms", 0))
    micro_note = "tight to a grid" if micro < 4 else "played, not quantised" if micro > 12 else ""
    swing = float(f.get("swing_ratio", 0))
    return [
        ("Tempo", f"{f.get('bpm', 0):.1f}", "BPM", bpm_note),
        ("Key", f"{f.get('key', '?')} {f.get('scale', '')}".strip(), "",
         f"Camelot {f.get('camelot', '?')} · confidence {f.get('key_confidence', 0):.2f}"),
        ("Key stability", f"{f.get('key_stability', 0):.2f}", "",
         "modulates" if float(f.get("key_stability", 1)) < 0.75 else "stays put"),
        ("Microtiming", f"{micro:.1f}", "ms", micro_note),
        ("Swing", f"{swing:.2f}", "", "straight" if swing < 0.55 else "swung"),
        ("Onset density", f"{f.get('onset_rate_hz', 0):.1f}", "/s", ""),
        ("Sections", f"{f.get('section_count', 0)}", "",
         f"mean {f.get('mean_section_s', 0):.0f}s"),
        ("Repetition", f"{f.get('repetition_index', 0):.2f}", "",
         "loop-based" if float(f.get("repetition_index", 0)) > 0.6 else "through-composed"),
    ]


def _loudness_cells(f: dict) -> list[tuple]:
    plr = float(f.get("plr_db", 0))
    return [
        ("Integrated loudness", f"{f.get('integrated_lufs', 0):.1f}", "LUFS", ""),
        ("Loudness range", f"{f.get('loudness_range_lu', 0):.1f}", "LU", ""),
        ("True peak", f"{f.get('true_peak_dbtp', 0):.1f}", "dBTP",
         "clipping risk" if float(f.get("true_peak_dbtp", -10)) > -0.3 else ""),
        ("Peak to loudness", f"{plr:.1f}", "dB",
         "heavily limited" if plr < 8 else "dynamic" if plr > 14 else ""),
        ("Spectral centroid", f"{f.get('spectral_centroid_hz', 0):.0f}", "Hz", "brightness"),
        ("Rolloff (99%)", f"{f.get('rolloff99_hz', 0):.0f}", "Hz", "usable bandwidth"),
        ("Harmonic share", f"{f.get('harmonic_ratio', 0):.2f}", "",
         "percussive" if float(f.get("harmonic_ratio", 0.5)) < 0.4 else ""),
        ("Stereo width", f"{f.get('stereo_width', 0):.2f}", "",
         "mono" if float(f.get("stereo_width", 0)) < 0.02 else ""),
    ]


def _tags_section(tags) -> str:
    if not tags:
        return ""
    rows = "".join(
        f'<div class="bar"><span>{html.escape(t)}</span>'
        f'<span class="track"><span class="fill" style="width:{min(w * 100, 100):.0f}%"></span></span>'
        f'<span class="num">{w:.2f}</span></div>' for t, w in tags[:12])
    return ('<section><h2>Description, inferred from the audio</h2>'
            f'<div class="bars">{rows}</div></section>')


def _taste_section(taste) -> str:
    if not taste:
        return ""
    rows = "".join(
        f'<div class="bar"><span>{html.escape(axis)}</span>'
        f'<span class="track"><span class="fill" style="width:{min(max(v, 0) * 100, 100):.0f}%"></span></span>'
        f'<span class="num">{v:.2f}</span></div>' for axis, v in taste.items())
    return ('<section><h2>Your taste probes</h2>'
            f'<div class="bars">{rows}</div></section>')


def _slop_section(slop) -> str:
    if not slop:
        return ""
    score = float(slop.get("score", 0))
    flag = " flag" if score > 0.65 else ""
    contributions = sorted(slop.get("contributions", {}).items(), key=lambda kv: -abs(kv[1]))
    span = max((abs(v) for _, v in contributions), default=1.0) or 1.0
    rows = "".join(
        f'<div class="bar"><span>{html.escape(k.replace("_", " "))}</span>'
        f'<span class="track"><span class="fill m" style="width:{abs(v) / span * 100:.0f}%"></span></span>'
        f'<span class="num">{v:+.2f}</span></div>' for k, v in contributions[:6])
    return (f'<section><h2>Synthetic audio</h2>'
            f'<div class="verdict{flag}"><strong>{html.escape(str(slop.get("verdict", "")))}</strong>'
            f'<span class="num">score {score:.2f} · confidence {float(slop.get("confidence", 0)):.2f}'
            f' · {html.escape(str(slop.get("mode", "")))}</span></div>'
            f'<div class="bars">{rows}</div>'
            f'<p class="caveat">These are the signals behind the score, not proof. '
            f'Sequenced electronic music trips the rhythm and repetition signals honestly; '
            f'the phase and spectral signals are the ones that carry real weight. '
            f'Label a few hundred tracks and refit before trusting the number to filter.</p></section>')


def _mmss(seconds: float) -> str:
    m, s = divmod(int(max(seconds, 0)), 60)
    return f"{m}:{s:02d}"
