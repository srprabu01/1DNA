"""Populate a deepcut library from the Music-DNA app's cached previews.

The app caches Deezer preview MP3s under ``data/audio_cache/<track_id>.mp3`` and
keeps title/artist/album/plays in ``data/music.db``. Those previews carry no
embedded tags, so we can't read metadata off the files — instead we join each
cached MP3 back to its app row by the numeric filename and hand deepcut the
metadata directly. Play history is imported too, so the popularity debias and
the implicit-taste probe have something to work with.

Idempotent: every track's digest is ``app:<id>``, so re-running updates in place
rather than duplicating. Usage:

    python bootstrap_from_app.py                 # default app + library paths
    python bootstrap_from_app.py --limit 50      # a quick sample
    python bootstrap_from_app.py --no-plays      # skip play-history import
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from deepcut import decode  # noqa: E402
from deepcut.config import Config  # noqa: E402
from deepcut.store import Library  # noqa: E402

APP_ROOT = HERE.parent  # ...\music-analyzer
DEFAULT_APP_DB = APP_ROOT / "data" / "music.db"
DEFAULT_CACHE = APP_ROOT / "data" / "audio_cache"


def _year(release_date: str | None) -> str | None:
    if not release_date:
        return None
    digits = "".join(c for c in str(release_date) if c.isdigit())[:4]
    return digits if len(digits) == 4 else None


def _epoch(played_at: str | None, fallback: float) -> float:
    if not played_at:
        return fallback
    s = str(played_at).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                continue
    return fallback


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app-db", default=str(DEFAULT_APP_DB))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--root", help="deepcut library root (default: ./library)")
    ap.add_argument("--limit", type=int, help="only import the first N previews")
    ap.add_argument("--no-plays", action="store_true", help="skip play-history import")
    ap.add_argument("--write-config", action="store_true", default=True,
                    help="write a deepcut.toml pointing at this library (default on)")
    args = ap.parse_args(argv)

    import sqlite3

    app_db = Path(args.app_db)
    cache = Path(args.cache)
    if not app_db.exists():
        sys.exit(f"app database not found: {app_db}")
    if not cache.exists():
        sys.exit(f"audio cache not found: {cache}")

    cfg = Config.default(args.root) if args.root else Config.load(None)
    cfg.paths.ensure()
    print(f"  deepcut library: {cfg.paths.root}")

    app = sqlite3.connect(str(app_db))
    app.row_factory = sqlite3.Row
    meta = {int(r["id"]): r for r in app.execute(
        "SELECT id, title, artist, album, duration_ms, genre, release_date FROM tracks")}

    previews = sorted(
        (p for p in cache.glob("*.mp3") if p.stem.isdigit()),
        key=lambda p: int(p.stem))
    if args.limit:
        previews = previews[: args.limit]
    print(f"  cached previews: {len(previews)}  (app tracks: {len(meta)})")

    lib = Library(cfg.paths)
    id_map: dict[int, int] = {}   # app id -> deepcut id
    added = skipped = no_meta = 0

    for n, mp3 in enumerate(previews, 1):
        app_id = int(mp3.stem)
        row = meta.get(app_id)
        if row is None:
            no_meta += 1
            continue
        try:
            info = decode.probe(mp3)
            if info.duration_s <= 0 and row["duration_ms"]:
                info.duration_s = float(row["duration_ms"]) / 1000.0
            tags = {
                "title": row["title"],
                "artist": row["artist"],
                "album": row["album"],
                "date": _year(row["release_date"]),
            }
            dc_id = lib.upsert_track(f"app:{app_id}", mp3, info,
                                     {k: v for k, v in tags.items() if v})
            id_map[app_id] = dc_id
            added += 1
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"    skip {mp3.name}: {exc}")
        if n % 250 == 0:
            print(f"    ... {n}/{len(previews)}")

    plays_imported = 0
    if not args.no_plays and id_map:
        # Idempotent: clear any prior imported plays for these tracks first.
        lib.db.executemany("DELETE FROM plays WHERE track_id=?",
                           [(i,) for i in id_map.values()])
        rows = app.execute(
            "SELECT track_id, played_at, source FROM plays ORDER BY played_at").fetchall()
        batch = []
        for i, r in enumerate(rows):
            dc_id = id_map.get(int(r["track_id"]))
            if dc_id is None:
                continue
            batch.append((dc_id, _epoch(r["played_at"], float(i)), 0, 0,
                          r["source"]))
        lib.db.executemany(
            "INSERT INTO plays (track_id, played_at, played_ms, skipped, context)"
            " VALUES (?,?,?,?,?)", batch)
        lib.db.commit()
        plays_imported = len(batch)

    lib.close()
    app.close()

    if args.write_config:
        toml = HERE / "deepcut.toml"
        root_fwd = str(cfg.paths.root).replace("\\", "/")
        toml.write_text(
            f'library = "{root_fwd}"\n\n'
            '# Excerpt windows tuned for 30-second Deezer previews.\n'
            '[excerpts]\ncount = 3\nseconds = 8.0\nstructure_aware = true\n'
            'skip_head_s = 2.0\nskip_tail_s = 2.0\nmin_gap_s = 5.0\n\n'
            '[embed]\nmert_model = "m-a-p/MERT-v1-95M"\n'
            'clap_model = "laion/larger_clap_music"\nbatch_size = 8\nfp16 = true\n\n'
            '[retrieve]\ncandidates = 500\nresults = 25\nmmr_lambda = 0.72\n'
            'epsilon = 0.12\nartist_cap = 2\n', encoding="utf-8")
        print(f"  wrote {toml}")

    print(f"\n  imported {added} tracks "
          f"({no_meta} previews had no app row, {skipped} skipped)")
    print(f"  imported {plays_imported} plays")
    print("\n  next:")
    print("    python -m deepcut.cli analyse         # measured features (CPU)")
    print("    python -m deepcut.cli stats")
    print("    # then, with torch+transformers installed:")
    print("    python -m deepcut.cli embed && python -m deepcut.cli fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
