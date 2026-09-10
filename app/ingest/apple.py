"""Parse an Apple Media Services export (Apple Music listening history).

Request yours at https://privacy.apple.com -> "Get a copy of your data"
-> Apple Media Services, then unzip it. The listening data lives under
``Apple Media Services information/.../Apple Music Activity/``.

Two entry points:
  * ``import_file(content)``    -- one uploaded CSV; auto-detects the format.
  * ``import_directory(path)``  -- the whole unzipped export folder. Uses
        "Play History Daily Tracks" for plays and "Play Activity" for
        album / soundtrack metadata, in a single pass.

Formats handled
---------------
"Apple Music - Play History Daily Tracks.csv"  (authoritative for history)
    ``Track Description`` = ``"Artist - Title"``, plus ``Play Count`` and
    ``Date Played`` (YYYYMMDD). This is the only Apple file that carries the
    track's *artist*, so it drives track + play creation. ``Play Count`` is
    honoured: a row played N times that day becomes N distinct play events.

"Apple Music Play Activity.csv"  (album metadata only)
    Event-level log with ``Song Name`` + ``Album Name`` but **no track artist**
    (its ``Container Artist Name`` is the playlist's artist, usually blank).
    Importing plays from it would label every Apple track "Unknown", so it is
    used *only* to enrich the ``album`` column of tracks matched by title —
    which in turn feeds the soundtrack -> movie step of the normalisation ETL.

"Apple Music - Track Play History.csv" / "... Recently Played Tracks.csv"
    ``Track Description`` only; a light one-play-per-row fallback used when the
    Daily Tracks file is absent.
"""
import csv
import io
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..db import connect, upsert_track, add_play

# A single track played many times in one day is rare; cap defensively so a
# corrupt Play Count can never explode the plays table.
_MAX_PLAYS_PER_DAY = 500


def _pick(row, *names):
    for n in names:
        v = row.get(n)
        if v and v.strip():
            return v.strip()
    return None


def _norm_title(s: str) -> str:
    """Fold a song title to a match key for the album lookup."""
    s = (s or "").strip().lower().replace("’", "'")
    return re.sub(r"\s+", " ", s)


def _split_desc(desc: str):
    """'Artist - Title' -> (artist, title). Splits on the FIRST ' - '."""
    if desc and " - " in desc:
        artist, title = desc.split(" - ", 1)
        return artist.strip() or None, title.strip() or None
    return None, (desc.strip() or None if desc else None)


def _day_timestamps(yyyymmdd: str, n: int):
    """N distinct ISO timestamps within the given day.

    Daily Tracks aggregates a day's plays into one row, but ``plays`` is unique
    on (track_id, played_at, source) -- so N plays need N distinct instants.
    We anchor at local noon and step one second per play (noon+i), which keeps
    every timestamp inside the same calendar day for any realistic N.
    """
    try:
        base = datetime(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]),
                        12, 0, 0, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return []
    return [(base + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(min(max(n, 1), _MAX_PLAYS_PER_DAY))]


def _album_map_from_activity(text: str) -> dict:
    """Build ``{normalised song title -> album}`` from a Play Activity CSV.

    Only titles that map to exactly ONE album are kept (precision over recall):
    a song seen under two different albums is left out rather than guessed.
    """
    from collections import defaultdict

    seen = defaultdict(set)
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        song = _norm_title(row.get("Song Name"))
        album = (row.get("Album Name") or "").strip()
        if song and album:
            seen[song].add(album)
    return {s: next(iter(a)) for s, a in seen.items() if len(a) == 1}


def _is_daily(fields) -> bool:
    return "Track Description" in fields and "Play Count" in fields


def _is_activity(fields) -> bool:
    return "Song Name" in fields and "Play Count" not in fields


# ── core importers ────────────────────────────────────────────────────────

def _import_daily(text: str, album_map: dict | None = None) -> dict:
    """Play History Daily Tracks -> tracks + plays (Play Count honoured)."""
    album_map = album_map or {}
    reader = csv.DictReader(io.StringIO(text))
    tracks_seen = set()
    plays = skipped = albums = 0
    with connect() as con:
        for row in reader:
            desc = _pick(row, "Track Description")
            artist, title = _split_desc(desc) if desc else (None, None)
            when = _pick(row, "Date Played")
            if not title or not when:
                skipped += 1
                continue
            try:
                count = int(float(row.get("Play Count") or 0))
            except ValueError:
                count = 0
            # A row exists because the track was engaged that day; treat a
            # 0/blank Play Count as one listen rather than dropping real history.
            count = max(count, 1)

            album = album_map.get(_norm_title(title))
            tid = upsert_track(con, title=title, artist=artist or "Unknown",
                               album=album)
            if not tid:
                skipped += 1
                continue
            if album and _norm_title(title) not in tracks_seen:
                albums += 1
            tracks_seen.add(_norm_title(title))
            for ts in _day_timestamps(when, count):
                add_play(con, tid, ts, "apple")
                plays += 1
    return {"source": "Play History Daily Tracks", "plays": plays,
            "tracks": len(tracks_seen), "albums_enriched": albums,
            "skipped": skipped}


def _enrich_albums_only(text: str) -> dict:
    """Play Activity uploaded on its own -> fill albums of existing tracks."""
    album_map = _album_map_from_activity(text)
    filled = 0
    with connect() as con:
        for norm, album in album_map.items():
            cur = con.execute(
                "UPDATE tracks SET album=? WHERE album IS NULL "
                "AND lower(trim(title))=?", (album, norm))
            filled += cur.rowcount or 0
    return {"source": "Play Activity", "plays": 0, "albums_enriched": filled,
            "note": "Play Activity has no per-track artist; used only to fill "
                    "album metadata. Import Play History Daily Tracks for plays."}


def _import_generic(text: str) -> dict:
    """Fallback: Track Description only, one play per row."""
    reader = csv.DictReader(io.StringIO(text))
    plays = skipped = 0
    seen = set()
    with connect() as con:
        for row in reader:
            desc = _pick(row, "Track Description", "Track Name", "Song Name", "Title")
            artist, title = _split_desc(desc) if desc and " - " in desc else (
                _pick(row, "Artist Name", "Container Artist Name"), desc)
            when = _pick(row, "Last Played Date", "Date Played",
                         "Event Start Timestamp", "First Event Timestamp",
                         "Last Event Start Timestamp")
            if not title or not when:
                skipped += 1
                continue
            if len(when) == 8 and when.isdigit():
                when = f"{when[:4]}-{when[4:6]}-{when[6:]}T12:00:00Z"
            tid = upsert_track(con, title=title, artist=artist or "Unknown")
            if not tid:
                skipped += 1
                continue
            seen.add(tid)
            add_play(con, tid, when, "apple")
            plays += 1
    return {"source": "generic", "plays": plays, "tracks": len(seen),
            "skipped": skipped}


def import_file(content: bytes) -> dict:
    """Import a single uploaded Apple CSV, dispatching on its columns."""
    text = content.decode("utf-8-sig", errors="replace")
    try:
        header = next(csv.reader(io.StringIO(text)))
    except StopIteration:
        return {"plays": 0, "skipped": 0, "note": "empty file"}
    fields = set(header)
    if _is_daily(fields):
        return _import_daily(text)
    if _is_activity(fields):
        return _enrich_albums_only(text)
    return _import_generic(text)


def import_directory(path: str) -> dict:
    """Import a full unzipped Apple Media Services export folder.

    Prefers Play History Daily Tracks for history and, when present, folds in
    Play Activity album metadata in the same pass. Falls back to whatever
    track-history CSV it can find.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(path)

    def find(name):
        hits = [p for p in root.rglob("*.csv") if p.name.lower() == name.lower()]
        return hits[0] if hits else None

    daily = find("Apple Music - Play History Daily Tracks.csv")
    activity = find("Apple Music Play Activity.csv")

    album_map = {}
    if activity:
        album_map = _album_map_from_activity(
            activity.read_bytes().decode("utf-8-sig", errors="replace"))

    if daily:
        res = _import_daily(
            daily.read_bytes().decode("utf-8-sig", errors="replace"), album_map)
        res["album_map_size"] = len(album_map)
        return res

    # No Daily Tracks -> fall back to the richest history file we have.
    for name in ("Apple Music - Track Play History.csv",
                 "Apple Music - Recently Played Tracks.csv"):
        f = find(name)
        if f:
            return _import_generic(
                f.read_bytes().decode("utf-8-sig", errors="replace"))
    raise FileNotFoundError("no Apple Music history CSV found under " + path)
