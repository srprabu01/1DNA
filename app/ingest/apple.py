"""Parse an Apple Music play-history export.

Request yours at https://privacy.apple.com -> "Get a copy of your data"
-> Apple Media Services. Upload either:
  - "Apple Music - Play History Daily Tracks.csv", or
  - "Apple Music Play Activity.csv"
"""
import csv
import io

from ..db import connect, upsert_track, add_play


def _pick(row, *names):
    for n in names:
        v = row.get(n)
        if v:
            return v.strip()
    return None


def import_file(content: bytes):
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    imported = skipped = 0
    with connect() as con:
        for row in reader:
            desc = _pick(row, "Track Description", "Song Name", "Track Name", "Title")
            artist = _pick(row, "Artist Name", "Container Artist Name", "Track Artist")
            title = desc
            # "Play History Daily Tracks" uses "Artist - Title" in Track Description
            if desc and not artist and " - " in desc:
                artist, title = desc.split(" - ", 1)
            when = _pick(
                row,
                "Date Played", "Event Start Timestamp", "Event Timestamp",
                "Play Date", "Event Received Timestamp",
            )
            if not title or not when:
                skipped += 1
                continue
            # "Date Played" comes as YYYYMMDD
            if len(when) == 8 and when.isdigit():
                when = f"{when[:4]}-{when[4:6]}-{when[6:]}T12:00:00Z"
            dur = _pick(row, "Play Duration Milliseconds", "Media Duration In Milliseconds")
            tid = upsert_track(
                con,
                title=title,
                artist=artist or "Unknown",
                duration_ms=int(float(dur)) if dur else None,
            )
            if tid:
                add_play(con, tid, when, "apple")
                imported += 1
            else:
                skipped += 1
    return {"imported": imported, "skipped": skipped}
