"""A Spotify-Wrapped-style report: headline stats + narrative for a year or all-time."""
from ..db import connect, rows_to_dicts
from ..analytics import advanced, engine


def generate(year=None):
    where = ""
    params = []
    if year:
        where = "WHERE strftime('%Y', p.played_at) = ?"
        params = [str(year)]

    with connect() as con:
        totals = con.execute(
            f"""SELECT COUNT(*) plays, COUNT(DISTINCT p.track_id) tracks,
                       COUNT(DISTINCT t.artist) artists,
                       SUM(COALESCE(t.duration_ms,210000))/3600000.0 hours
                FROM plays p JOIN tracks t ON t.id=p.track_id {where}""",
            params).fetchone()
        top_tracks = rows_to_dicts(con.execute(
            f"""SELECT t.title, t.artist, COUNT(*) plays
                FROM plays p JOIN tracks t ON t.id=p.track_id {where}
                GROUP BY t.id ORDER BY plays DESC LIMIT 5""", params).fetchall())
        top_artists = rows_to_dicts(con.execute(
            f"""SELECT t.artist name, COUNT(*) plays
                FROM plays p JOIN tracks t ON t.id=p.track_id {where}
                GROUP BY t.artist ORDER BY plays DESC LIMIT 5""", params).fetchall())
        top_genre = con.execute(
            f"""SELECT t.genre, COUNT(*) n FROM plays p JOIN tracks t ON t.id=p.track_id
                {where + (' AND' if where else 'WHERE')} t.genre IS NOT NULL
                GROUP BY t.genre ORDER BY n DESC LIMIT 1""", params).fetchone()
        peak_hour = con.execute(
            f"""SELECT CAST(strftime('%H', played_at) AS INT) h, COUNT(*) n
                FROM plays p {where} GROUP BY h ORDER BY n DESC LIMIT 1""",
            params).fetchone()
        years = rows_to_dicts(con.execute(
            "SELECT DISTINCT strftime('%Y', played_at) y FROM plays ORDER BY y DESC").fetchall())

    profile = engine.audio_profile()
    conc = advanced.concentration()
    disc = advanced.discovery()

    top_artist = top_artists[0]["name"] if top_artists else None
    top_song = f"{top_tracks[0]['title']} — {top_tracks[0]['artist']}" if top_tracks else None
    peak = f"{peak_hour['h']:02d}:00" if peak_hour else None
    top_genre_name = top_genre["genre"] if top_genre else None

    narrative = _narrative(top_artist, top_song, top_genre_name, peak,
                           profile, disc, conc, year)

    return {
        "year": year or "All time",
        "available_years": [y["y"] for y in years if y["y"]],
        "headline": {
            "plays": totals["plays"], "tracks": totals["tracks"],
            "artists": totals["artists"],
            "hours": round(totals["hours"] or 0, 1),
            "minutes": round((totals["hours"] or 0) * 60),
        },
        "top_tracks": top_tracks,
        "top_artists": top_artists,
        "top_genre": top_genre["genre"] if top_genre else None,
        "peak_hour": peak,
        "mood": profile.get("mood_quadrant") if profile.get("available") else None,
        "avg_bpm": profile.get("avg_bpm") if profile.get("available") else None,
        "explorer": disc.get("explorer_label"),
        "concentration": conc.get("label") if conc.get("available") else None,
        "narrative": narrative,
    }


def _narrative(artist, song, genre, peak, profile, disc, conc, year):
    period = f"in {year}" if year else "across all your listening"
    lines = []
    if artist:
        lines.append(f"Your #1 artist {period} was **{artist}**.")
    if song:
        lines.append(f"You kept coming back to **{song}**.")
    if genre:
        lines.append(f"**{genre}** was your defining genre.")
    if profile.get("available"):
        lines.append(f"Your sound sat mostly in the **{profile['mood_quadrant']}** "
                     f"zone, averaging **{profile['avg_bpm']} BPM**.")
    if peak:
        lines.append(f"You listened most around **{peak}**.")
    if disc.get("explorer_label"):
        lines.append(disc["explorer_label"] + ".")
    if conc.get("available"):
        lines.append(conc["label"] + ".")
    return lines
