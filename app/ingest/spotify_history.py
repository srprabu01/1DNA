"""Import Spotify streaming-history JSON exports (spotify.com/account/privacy).

Handles both export shapes:
  - Extended streaming history  -> Streaming_History_Audio_*.json
      fields: ts, master_metadata_track_name, master_metadata_album_artist_name,
              master_metadata_album_album_name, ms_played
  - Account-data basic history  -> StreamingHistory_music_*.json
      fields: endTime, artistName, trackName, msPlayed

Each qualifying stream (>= 30s, Spotify's own "counts as a play" threshold)
becomes a play row; COUNT per track = number of times played. Podcast/episode
rows (no track name) are skipped. Re-importing more files is safe — plays are
de-duplicated on (track, timestamp).
"""
import json

from ..db import connect, upsert_track, add_play

MIN_MS = 30000  # Spotify counts a stream once it passes 30 seconds


def import_file(content: bytes, min_ms: int = MIN_MS):
    data = json.loads(content)
    imported = skipped = 0
    with connect() as con:
        for e in data:
            track = e.get("master_metadata_track_name") or e.get("trackName")
            artist = e.get("master_metadata_album_artist_name") or e.get("artistName")
            album = e.get("master_metadata_album_album_name")
            ts = e.get("ts") or e.get("endTime")
            ms = e.get("ms_played")
            if ms is None:
                ms = e.get("msPlayed")
            if not track or not artist or not ts:
                skipped += 1          # podcast episode or malformed row
                continue
            if ms is not None and ms < min_ms:
                skipped += 1          # skipped/previewed, not a real play
                continue
            # basic export uses "2025-05-01 12:34" — normalise to ISO
            if " " in ts and "T" not in ts:
                ts = ts.replace(" ", "T")
                if len(ts) == 16:     # minute precision -> add seconds+Z
                    ts += ":00Z"
            tid = upsert_track(con, title=track, artist=artist, album=album)
            if tid:
                add_play(con, tid, ts, "spotify")
                imported += 1
            else:
                skipped += 1
    return {"imported": imported, "skipped": skipped}
