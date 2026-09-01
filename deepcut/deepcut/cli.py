"""Command line for deepcut.

    deepcut init
    deepcut ingest ~/Music
    deepcut analyse
    deepcut embed
    deepcut fit
    deepcut similar "kid a"
    deepcut prompt "murky guitar for a rainy Sunday"
    deepcut run --spm 172 --minutes 40 --out run.m3u8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .config import Config, resolve_device
from .retrieve import Constraints, Engine, cadence_query, sequence
from .store import Library


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _progress(prefix: str):
    def report(i: int, n: int, label: str) -> None:
        pct = i / max(n, 1) * 100
        line = f"\r  {prefix} {i}/{n} ({pct:5.1f}%)  {label[:52]:<52}"
        sys.stdout.write(line)
        sys.stdout.flush()
        if i == n:
            sys.stdout.write("\n")
    return report


def _resolve_track(lib: Library, query: str):
    if query.isdigit():
        track = lib.get(int(query))
        if track:
            return track
    matches = lib.find(query, limit=8)
    if not matches:
        sys.exit(f"no track matches {query!r}")
    if len(matches) > 1:
        print("several matches — pick one by id:")
        for t in matches:
            print(f"  {t.id:>6}  {t.label}")
        sys.exit(0)
    return matches[0]


def _encoder(cfg: Config):
    from .embed import SemanticEncoder
    return SemanticEncoder(cfg.embed)


def _print_results(results, engine=None, seed_id=None, explain=False, heads=None):
    if not results:
        print("  nothing matched. Loosen the constraints or lower --min-slop-confidence.")
        return
    for rank, c in enumerate(results, 1):
        marker = "*" if c.origin == "exploration" else " "
        bpm = f"{c.bpm:5.1f}" if c.bpm else "    ?"
        print(f"{marker}{rank:>3}. {c.track.label[:58]:<58} {c.score:+.3f}  "
              f"{bpm} BPM  {c.camelot or '  -':>4}")
        if explain and engine is not None and seed_id is not None:
            from .explain import explain as do_explain
            try:
                print(f"       {do_explain(engine, seed_id, c.track.id, heads)}")
            except Exception as exc:
                print(f"       (no explanation: {exc})")
    if any(c.origin == "exploration" for c in results):
        print("\n  * exploration slot")


def _write_m3u(path: Path, results, lib: Library) -> None:
    lines = ["#EXTM3U"]
    for c in results:
        dur = int(c.track.duration_s or 0)
        lines.append(f"#EXTINF:{dur},{c.track.artist or ''} - {c.track.title or ''}")
        lines.append(str(Path(c.track.path).resolve()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  wrote {path} ({len(results)} tracks)")


def _load_heads(cfg: Config):
    from .taste import TasteHead
    heads = {}
    for axis in TasteHead.list_axes(cfg.paths.models):
        head = TasteHead.load(cfg.paths.models, axis)
        if head:
            heads[axis] = head
    return heads


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init(args, cfg: Config) -> None:
    cfg.paths.ensure()
    sample = Path("deepcut.toml")
    if not sample.exists():
        sample.write_text(
            f'library = "{cfg.paths.root}"\n\n'
            '[excerpts]\ncount = 3\nseconds = 10.0\nstructure_aware = true\n\n'
            '[embed]\nmert_model = "m-a-p/MERT-v1-95M"\n'
            'clap_model = "laion/larger_clap_music"\nbatch_size = 8\nfp16 = true\n\n'
            '[retrieve]\ncandidates = 500\nresults = 25\nmmr_lambda = 0.72\n'
            'epsilon = 0.12\nartist_cap = 2\n')
        print(f"  wrote deepcut.toml")
    print(f"  library at {cfg.paths.root.resolve()}")
    print("  next: deepcut ingest /path/to/music")


def cmd_doctor(args, cfg: Config) -> None:
    import shutil
    print("environment")
    for name in ("ffmpeg", "ffprobe", "fpcalc"):
        found = shutil.which(name)
        note = "" if name != "fpcalc" else "  (optional: AcoustID fingerprinting)"
        print(f"  {name:<12} {'ok  ' + found if found else 'MISSING'}{note}")
    for module, note in (("numpy", ""), ("scipy", ""), ("sklearn", ""),
                         ("librosa", ""), ("pyloudnorm", "  (optional: LUFS)"),
                         ("torch", ""), ("transformers", "")):
        try:
            mod = __import__(module)
            print(f"  {module:<12} ok   {getattr(mod, '__version__', '?')}{note}")
        except ImportError:
            print(f"  {module:<12} MISSING{note}")
    try:
        import torch
        device = resolve_device(cfg.embed.device)
        print(f"  device    {device}")
        if device == "cuda":
            props = torch.cuda.get_device_properties(0)
            print(f"  gpu       {props.name}, {props.total_memory / 2**30:.1f} GiB")
    except ImportError:
        pass
    print(f"  library   {cfg.paths.root.resolve()}")


def cmd_ingest(args, cfg: Config) -> None:
    from .pipeline import ingest
    lib = Library(cfg.paths)
    stats = ingest(cfg, lib, [Path(p) for p in args.paths], online=args.online,
                   progress=_progress("ingest"))
    print("  " + json.dumps(stats))
    lib.close()


def cmd_analyse(args, cfg: Config) -> None:
    from .pipeline import analyse
    lib = Library(cfg.paths)
    tracks = lib.all_tracks() if args.force else lib.pending("analyse")
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"  {len(tracks)} tracks to analyse")
    stats = analyse(cfg, lib, tracks, force=args.force, progress=_progress("analyse"))
    print("  " + json.dumps(stats))
    lib.close()


def cmd_embed(args, cfg: Config) -> None:
    from .pipeline import embed
    lib = Library(cfg.paths)
    tracks = lib.all_tracks() if args.force else lib.pending("embed")
    if args.limit:
        tracks = tracks[: args.limit]
    print(f"  {len(tracks)} tracks to embed on {resolve_device(cfg.embed.device)}")
    stats = embed(cfg, lib, tracks, _encoder(cfg), tag=not args.no_tags,
                  force=args.force, progress=_progress("embed"))
    print("  " + json.dumps(stats))
    print("  next: deepcut fit")
    lib.close()


def cmd_fit(args, cfg: Config) -> None:
    lib = Library(cfg.paths)
    engine = Engine(lib, cfg.retrieve, cfg.paths.models)
    report = engine.fit_corrections(n_components=args.components)
    if not report:
        sys.exit("  no vectors yet. Run `deepcut embed` first.")
    for space, r in report.items():
        print(f"  {space}: {r['tracks']} tracks, dim {r['dim']}, "
              f"removed {r['components_removed']} principal directions")
        print(f"    popularity R^2 {r['popularity_r2_before']:.3f} -> "
              f"{r['popularity_r2_after']:.3f}")
    lib.close()


def cmd_similar(args, cfg: Config) -> None:
    lib = Library(cfg.paths)
    engine = Engine(lib, cfg.retrieve, cfg.paths.models)
    seed = _resolve_track(lib, args.query)
    queries = {}
    for space in ("mert", "clap"):
        v = engine.vector_for_tracks(space, [seed.id])
        if v is not None:
            queries[space] = v
    if not queries:
        sys.exit("  that track has no embedding yet. Run `deepcut embed`.")
    seed_f = lib.features(seed.id) or {}
    constraints = Constraints(
        exclude_track_ids={seed.id},
        max_slop=args.max_slop,
        camelot_seed=seed_f.get("camelot") if args.harmonic else None,
        artist_cap=args.artist_cap,
        unheard_only=args.unheard,
    )
    print(f"  seed: {seed.label}  ({seed_f.get('bpm', 0):.0f} BPM, "
          f"{seed_f.get('camelot', '?')})\n")
    results = engine.search(queries, constraints, k=args.k,
                            weights={"mert": args.mert, "clap": args.clap})
    _print_results(results, engine, seed.id, args.explain, _load_heads(cfg))
    if args.out:
        _write_m3u(Path(args.out), results, lib)
    lib.close()


def cmd_prompt(args, cfg: Config) -> None:
    lib = Library(cfg.paths)
    engine = Engine(lib, cfg.retrieve, cfg.paths.models)
    vec = engine.vector_for_text(args.text, _encoder(cfg))
    constraints = Constraints(max_slop=args.max_slop, artist_cap=args.artist_cap,
                              unheard_only=args.unheard)
    results = engine.search({"clap": vec}, constraints, k=args.k,
                            weights={"clap": 1.0})
    print(f"  prompt: {args.text!r}\n")
    _print_results(results)
    if args.sequence:
        results = sequence(results, energy_arc=args.arc)
        print("\n  sequenced:")
        _print_results(results)
    if args.out:
        _write_m3u(Path(args.out), results, lib)
    lib.close()


def cmd_run(args, cfg: Config) -> None:
    """Cadence-matched sets — the query the dead audio-features endpoint served."""
    lib = Library(cfg.paths)
    engine = Engine(lib, cfg.retrieve, cfg.paths.models)
    lo, hi = cadence_query(args.spm, tolerance=args.tolerance)
    constraints = Constraints(bpm_range=(lo, hi), allow_half_double=args.half_time,
                              max_slop=args.max_slop, artist_cap=args.artist_cap)

    queries: dict = {}
    if args.seed:
        seed = _resolve_track(lib, args.seed)
        for space in ("mert", "clap"):
            v = engine.vector_for_tracks(space, [seed.id])
            if v is not None:
                queries[space] = v
    elif args.prompt:
        queries["clap"] = engine.vector_for_text(args.prompt, _encoder(cfg))
    else:
        history = [t for t, _ in lib.history()][-40:]
        v = engine.vector_for_tracks("mert", history)
        if v is None:
            sys.exit("  give --seed or --prompt, or log some plays first.")
        queries["mert"] = v

    target = args.minutes * 60
    n = max(int(target / 240) + 6, 12)
    results = engine.search(queries, constraints, k=n)
    results = sequence(results, energy_arc=args.arc)

    total = 0.0
    kept = []
    for c in results:
        if total >= target:
            break
        kept.append(c)
        total += float(c.track.duration_s or 0)
    print(f"  {args.spm:.0f} spm ({lo:.0f}-{hi:.0f} BPM"
          f"{', half-time allowed' if args.half_time else ''}), "
          f"{total / 60:.0f} minutes\n")
    _print_results(kept)
    if args.out:
        _write_m3u(Path(args.out), kept, lib)
    lib.close()


def cmd_label(args, cfg: Config) -> None:
    from .taste import suggest_labels
    lib = Library(cfg.paths)
    if args.suggest:
        ids = suggest_labels(lib, args.axis, cfg.paths.models, n=args.suggest)
        if not ids:
            sys.exit("  nothing to suggest. Embed some tracks first.")
        print(f"  label these for '{args.axis}' "
              f"(deepcut label --axis {args.axis} --id N --value 1|0):\n")
        for tid in ids:
            t = lib.get(tid)
            if t:
                print(f"  {tid:>6}  {t.label}")
    elif args.id is not None:
        track = _resolve_track(lib, str(args.id))
        lib.set_label(track.id, args.axis, args.value)
        n = len(lib.labels(args.axis))
        print(f"  {args.axis} = {args.value} for {track.label}  ({n} labels on this axis)")
    else:
        for axis in lib.label_axes():
            print(f"  {axis:<20} {len(lib.labels(axis)):>5} labels")
    lib.close()


def cmd_taste(args, cfg: Config) -> None:
    from .taste import TasteHead, fit
    lib = Library(cfg.paths)
    if args.action == "list":
        axes = TasteHead.list_axes(cfg.paths.models)
        if not axes:
            print("  no taste heads yet. `deepcut label --axis like --suggest 20`")
        for axis in axes:
            head = TasteHead.load(cfg.paths.models, axis)
            if head:
                metric = "AUC" if head.kind == "binary" else "R^2"
                print(f"  {axis:<18} {head.space:<6} {head.n_labels:>4} labels  "
                      f"{metric} {head.cv_score:.3f}")
                if head.layer_profile:
                    prof = "  ".join(f"{k} {v:.2f}" for k, v in head.layer_profile.items())
                    print(f"    depth: {prof}")
    else:
        head = fit(lib, args.axis, space=args.space)
        head.save(cfg.paths.models)
        metric = "AUC" if head.kind == "binary" else "R^2"
        print(f"  fitted '{head.axis}' on {head.n_labels} labels, "
              f"cross-validated {metric} {head.cv_score:.3f}")
        if head.cv_score < 0.65 and head.kind == "binary":
            print("  that is close to chance. Either the axis is not in the audio,")
            print("  or the labels are inconsistent. More labels will not fix the first.")
        if head.layer_profile:
            best = max(head.layer_profile.items(), key=lambda kv: kv[1])
            print(f"  strongest at the {best[0]} depth ({best[1]:.3f}) — "
                  f"this axis is about how the record is "
                  f"{'made' if best[0] == 'production' else 'played' if best[0] == 'playing' else 'written'}")
    lib.close()


def cmd_play(args, cfg: Config) -> None:
    lib = Library(cfg.paths)
    track = _resolve_track(lib, args.query)
    lib.log_play(track.id, args.ms, args.skipped, context=args.context)
    print(f"  logged {'skip' if args.skipped else 'play'} of {track.label}")
    lib.close()


def cmd_eval(args, cfg: Config) -> None:
    from .evaluate import ablate, next_track, sanity
    lib = Library(cfg.paths)
    engine = Engine(lib, cfg.retrieve, cfg.paths.models)

    print("sanity (no history needed)")
    s = sanity(engine)
    if "error" in s:
        sys.exit("  no vectors yet.")
    print(f"  nearest neighbour is same artist   {s['nn_same_artist']:.3f}")
    print(f"  chance rate                        {s['chance_same_artist']:.4f}")
    print(f"  lift                               {s['lift']:.1f}x")
    print(f"  album cohesion                     {s['album_cohesion']:+.3f}")
    if s["lift"] < 3:
        print("  low lift: check the excerpt windows before tuning anything else.")

    if args.ablate:
        print("\nablations")
        for name, rep in ablate(engine, k=args.k):
            print(f"\n  {name}")
            print(rep.table())
    else:
        print("\nnext-track prediction")
        rep = next_track(engine, k=args.k)
        for note in rep.notes:
            print(f"  {note}")
        if rep.n_queries:
            print(rep.table())
    lib.close()


def cmd_report(args, cfg: Config) -> None:
    from .report import render
    lib = Library(cfg.paths)
    track = _resolve_track(lib, args.query)
    features = lib.features(track.id)
    if features is None:
        sys.exit("  not analysed yet. Run `deepcut analyse`.")
    heads = _load_heads(cfg)
    taste = {}
    for axis, head in heads.items():
        v = lib.vectors.vector(lib.db, head.space, track.id)
        if v is not None:
            taste[axis] = float(head.score(v))
    out = Path(args.out) if args.out else cfg.paths.reports / f"{track.id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(track, features, tags=lib.tags(track.id, 12),
                          slop=lib.slop(track.id), taste=taste,
                          excerpts=features.get("_excerpts")), encoding="utf-8")
    print(f"  wrote {out}")
    lib.close()


def cmd_slop(args, cfg: Config) -> None:
    from .slop import fit as fit_slop
    lib = Library(cfg.paths)
    rows = list(csv.DictReader(Path(args.labels).open()))
    examples = []
    for row in rows:
        track = lib.get(int(row["track_id"]))
        if track is None:
            continue
        verdict = lib.slop(track.id)
        if verdict is None:
            continue
        emb = lib.vectors.vector(lib.db, "clap", track.id)
        examples.append((verdict["signals"], emb, int(row["generated"])))
    if len(examples) < 20:
        sys.exit(f"  only {len(examples)} usable rows; need at least 20.")
    meta = fit_slop(examples, cfg.paths.models, use_embedding=not args.signals_only)
    print(f"  fitted on {meta['n']} examples ({meta['positives']} generated), "
          f"cross-validated AUC {meta['cv_auc']:.3f}")
    print("  re-run `deepcut analyse --force` to rescore the library.")
    lib.close()


def cmd_stats(args, cfg: Config) -> None:
    lib = Library(cfg.paths)
    tracks = lib.all_tracks()
    feats = lib.all_features()
    slop = lib.slop_scores()
    plays = lib.play_counts()
    print(f"  tracks        {len(tracks)}")
    print(f"  analysed      {len(feats)}")
    for space in ("mert", "clap"):
        m = lib.vectors.matrix(space)
        print(f"  {space + ' vectors':<13} {0 if m is None else m.shape[0]}"
              f"{'' if m is None else f'  (dim {m.shape[1]})'}")
    print(f"  played        {sum(plays.values())} plays over {len(plays)} tracks")
    if slop:
        flagged = sum(1 for v in slop.values() if v > cfg.retrieve.slop_threshold)
        print(f"  slop-flagged  {flagged} ({flagged / max(len(slop), 1) * 100:.1f}%)")
    if feats:
        import numpy as np
        bpms = [f.get("bpm", 0) for f in feats.values() if f.get("bpm")]
        lufs = [f.get("integrated_lufs", 0) for f in feats.values()]
        if bpms:
            print(f"  median BPM    {np.median(bpms):.0f}")
            print(f"  median LUFS   {np.median(lufs):.1f}")
    lib.close()


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deepcut", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to deepcut.toml")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the library and a config file").set_defaults(fn=cmd_init)
    sub.add_parser("doctor", help="check the environment").set_defaults(fn=cmd_doctor)
    sub.add_parser("stats", help="library summary").set_defaults(fn=cmd_stats)

    q = sub.add_parser("ingest", help="scan files, hash audio, read tags")
    q.add_argument("paths", nargs="+")
    q.add_argument("--online", action="store_true",
                   help="resolve MBIDs and ISRCs via AcoustID (needs an API key)")
    q.set_defaults(fn=cmd_ingest)

    q = sub.add_parser("analyse", aliases=["analyze"], help="measured features and structure")
    q.add_argument("--force", action="store_true")
    q.add_argument("--limit", type=int)
    q.set_defaults(fn=cmd_analyse)

    q = sub.add_parser("embed", help="frozen MERT + CLAP pass")
    q.add_argument("--force", action="store_true")
    q.add_argument("--limit", type=int)
    q.add_argument("--no-tags", action="store_true", help="skip zero-shot tagging")
    q.set_defaults(fn=cmd_embed)

    q = sub.add_parser("fit", help="fit popularity/anisotropy/hubness corrections")
    q.add_argument("--components", type=int, default=2)
    q.set_defaults(fn=cmd_fit)

    q = sub.add_parser("similar", help="tracks like a seed")
    q.add_argument("query", help="track id or search text")
    q.add_argument("-k", type=int, default=25)
    q.add_argument("--mert", type=float, default=0.6)
    q.add_argument("--clap", type=float, default=0.4)
    q.add_argument("--harmonic", action="store_true", help="restrict to compatible keys")
    q.add_argument("--unheard", action="store_true")
    q.add_argument("--max-slop", type=float, default=0.65)
    q.add_argument("--artist-cap", type=int, default=2)
    q.add_argument("--explain", action="store_true")
    q.add_argument("--out", help="write an m3u8 playlist")
    q.set_defaults(fn=cmd_similar)

    q = sub.add_parser("prompt", help="retrieve from a text description")
    q.add_argument("text")
    q.add_argument("-k", type=int, default=25)
    q.add_argument("--unheard", action="store_true")
    q.add_argument("--max-slop", type=float, default=0.65)
    q.add_argument("--artist-cap", type=int, default=2)
    q.add_argument("--sequence", action="store_true")
    q.add_argument("--arc", default="rise", choices=["rise", "fall", "peak", "flat"])
    q.add_argument("--out")
    q.set_defaults(fn=cmd_prompt)

    q = sub.add_parser("run", help="cadence-matched set for running")
    q.add_argument("--spm", type=float, required=True, help="target steps per minute")
    q.add_argument("--minutes", type=float, default=40)
    q.add_argument("--tolerance", type=float, default=0.04)
    q.add_argument("--half-time", action="store_true",
                   help="also accept tracks at half or double the cadence")
    q.add_argument("--seed")
    q.add_argument("--prompt")
    q.add_argument("--arc", default="rise", choices=["rise", "fall", "peak", "flat"])
    q.add_argument("--max-slop", type=float, default=0.65)
    q.add_argument("--artist-cap", type=int, default=2)
    q.add_argument("--out")
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("label", help="record a taste label")
    q.add_argument("--axis", default="like")
    q.add_argument("--id", type=int)
    q.add_argument("--value", type=float, default=1.0)
    q.add_argument("--suggest", type=int, help="propose N tracks worth labelling")
    q.set_defaults(fn=cmd_label)

    q = sub.add_parser("taste", help="fit or list taste probes")
    q.add_argument("action", choices=["fit", "list"])
    q.add_argument("--axis", default="like")
    q.add_argument("--space", default="mert", choices=["mert", "clap"])
    q.set_defaults(fn=cmd_taste)

    q = sub.add_parser("play", help="log a play or a skip")
    q.add_argument("query")
    q.add_argument("--ms", type=int, default=0)
    q.add_argument("--skipped", action="store_true")
    q.add_argument("--context")
    q.set_defaults(fn=cmd_play)

    q = sub.add_parser("eval", help="run the evaluation harness")
    q.add_argument("-k", type=int, default=50)
    q.add_argument("--ablate", action="store_true")
    q.set_defaults(fn=cmd_eval)

    q = sub.add_parser("report", help="write an HTML analysis of one track")
    q.add_argument("query")
    q.add_argument("--out")
    q.set_defaults(fn=cmd_report)

    q = sub.add_parser("slop", help="calibrate the synthetic-audio model")
    q.add_argument("labels", help="CSV with columns track_id,generated")
    q.add_argument("--signals-only", action="store_true")
    q.set_defaults(fn=cmd_slop)
    return p


def main(argv: list[str] | None = None) -> None:
    # Titles carry Tamil/Korean/accented characters; the Windows console/pipe
    # defaults to cp1252 and a bare write of those raises UnicodeEncodeError,
    # which would abort a long run. Force UTF-8 with lossy fallback.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load(args.config)
    try:
        args.fn(args, cfg)
    except KeyboardInterrupt:
        print("\n  interrupted; progress up to the last flush is saved.")
        sys.exit(130)


if __name__ == "__main__":
    main()
