"""Seed a rich DEMO dataset so the dashboard is fully alive without a live account.

- inserts ~36 well-known tracks across genres/eras/moods
- runs the REAL pipeline (Deezer match -> librosa signal analysis) so audio
  features (BPM, key, Camelot, energy, etc.) are genuine
- synthesizes a realistic multi-month play history with hour/day/seasonal shape
- fetches lyrics for a handful and builds a couple of playlists

Run:  python seed_demo.py
Everything it creates can be wiped from the app (Clear all data) or by deleting
data/music.db.
"""
import random
from datetime import datetime, timedelta

random.seed(7)

TRACKS = [
    # title, artist, genre, release_date, popularity
    ("Blinding Lights", "The Weeknd", "Pop", "2019-11-29", 96),
    ("Levitating", "Dua Lipa", "Pop", "2020-10-01", 90),
    ("As It Was", "Harry Styles", "Pop", "2022-03-31", 93),
    ("Flowers", "Miley Cyrus", "Pop", "2023-01-12", 92),
    ("Anti-Hero", "Taylor Swift", "Pop", "2022-10-21", 91),
    ("HUMBLE.", "Kendrick Lamar", "Hip-Hop", "2017-03-30", 88),
    ("SICKO MODE", "Travis Scott", "Hip-Hop", "2018-08-03", 87),
    ("God's Plan", "Drake", "Hip-Hop", "2018-01-19", 89),
    ("Sunflower", "Post Malone", "Hip-Hop", "2018-10-18", 90),
    ("Bohemian Rhapsody", "Queen", "Rock", "1975-10-31", 85),
    ("Smells Like Teen Spirit", "Nirvana", "Rock", "1991-09-10", 84),
    ("Mr. Brightside", "The Killers", "Rock", "2003-09-29", 86),
    ("Redbone", "Childish Gambino", "R&B", "2016-11-17", 83),
    ("Adorn", "Miguel", "R&B", "2012-06-19", 74),
    ("Titanium", "David Guetta", "Dance/EDM", "2011-12-09", 85),
    ("Wake Me Up", "Avicii", "Dance/EDM", "2013-06-17", 87),
    ("Clarity", "Zedd", "Dance/EDM", "2012-10-05", 80),
    ("Strobe", "deadmau5", "Dance/EDM", "2009-10-06", 68),
    ("Kesariya", "Arijit Singh", "Bollywood", "2022-07-17", 82),
    ("Vaathi Coming", "Anirudh Ravichander", "Tamil", "2020-01-05", 78),
    ("Jai Ho", "A.R. Rahman", "Bollywood", "2008-11-14", 79),
    ("Billie Jean", "Michael Jackson", "Pop", "1982-11-30", 88),
    ("Superstition", "Stevie Wonder", "Soul", "1972-10-24", 78),
    ("Skinny Love", "Bon Iver", "Indie", "2008-02-19", 72),
    ("Holocene", "Bon Iver", "Indie", "2011-06-17", 71),
    ("The Night We Met", "Lord Huron", "Indie", "2015-03-03", 80),
    ("Weightless", "Marconi Union", "Ambient", "2012-10-01", 55),
    ("Experience", "Ludovico Einaudi", "Classical", "2013-11-04", 70),
    ("Despacito", "Luis Fonsi", "Latin", "2017-01-13", 90),
    ("Bailando", "Enrique Iglesias", "Latin", "2014-04-17", 78),
    ("Three Little Birds", "Bob Marley", "Reggae", "1977-06-03", 76),
    ("Enter Sandman", "Metallica", "Metal", "1991-07-29", 82),
    ("Take Five", "Dave Brubeck", "Jazz", "1959-09-29", 68),
    ("Old Town Road", "Lil Nas X", "Country", "2018-12-03", 88),
    ("One Dance", "Drake", "Hip-Hop", "2016-04-05", 89),
    ("bad guy", "Billie Eilish", "Pop", "2019-03-29", 91),
]

# hour-of-day listening weights (evening heavy), weekday weights (weekend heavy)
HOUR_W = [1, 1, 1, 1, 1, 2, 4, 7, 8, 6, 5, 6, 8, 7, 6, 6, 7, 9, 11, 12, 11, 9, 6, 3]
DOW_W = [1.3, 1.0, 1.0, 1.0, 1.1, 1.4, 1.5]  # Mon..Sun
SOURCES = (["spotify"] * 6) + (["ytmusic"] * 3) + (["apple"] * 2)


def seed():
    from app.db import init_db, connect, upsert_track, add_play
    init_db()

    # 1) tracks
    ids = []
    with connect() as con:
        for title, artist, genre, rel, pop in TRACKS:
            tid = upsert_track(con, title=title, artist=artist,
                               release_date=rel, popularity=pop)
            con.execute("UPDATE tracks SET genre=COALESCE(genre,?) WHERE id=?", (genre, tid))
            ids.append(tid)

    # 2) synthesize plays across ~8 months
    start = datetime(2025, 11, 1)
    end = datetime(2026, 7, 1)
    span_days = (end - start).days
    # per-track affinity (some songs are on heavy rotation)
    affinity = [random.random() ** 2 for _ in ids]
    total_target = 2600
    weights_sum = sum(affinity)

    with connect() as con:
        for tid, aff in zip(ids, affinity):
            n = max(3, int(total_target * aff / weights_sum))
            for _ in range(n):
                # seasonal ramp: more listening in recent months
                day = int(span_days * (random.random() ** 0.7))
                d = start + timedelta(days=day)
                hour = random.choices(range(24), weights=HOUR_W)[0]
                dow = d.weekday()
                if random.random() > DOW_W[dow] / 1.5:
                    continue
                ts = d.replace(hour=hour, minute=random.randint(0, 59),
                               second=random.randint(0, 59))
                add_play(con, tid, ts.isoformat() + "Z",
                         random.choice(SOURCES))

    print(f"Seeded {len(ids)} tracks + synthetic plays.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    seed()

    print("Matching to Deezer (real previews)…")
    from app.enrich.deezer import enrich_batch
    while True:
        r = enrich_batch(limit=40)
        print("  ", r)
        if r["remaining"] == 0:
            break

    print("Signal-analyzing previews (real librosa DSP)…")
    from app.audio.features import analyze_batch
    while True:
        r = analyze_batch(limit=8)
        print("  ", r)
        if r["remaining"] == 0:
            break

    print("Fetching lyrics…")
    from app.lyrics import engine as ly
    print("  ", ly.fetch_batch(limit=20))

    print("Building demo playlists…")
    from app.playlist import store
    store.build_smart("top", limit=30)
    store.build_smart("recent", limit=25)
    print("Done.")
