"""Merge duplicate track rows that split a song's play count.

The same song arrives with different (title, artist) metadata depending on the
source — a clean "Vaaste / Dhvani Bhanushali" from the enriched Deezer match, and
a garbled "Nikhil D'Souza | ... / Vaaste Song: Dhvani Bhanushali" from a YouTube
watch-history title where the parser put the song name in the artist field. Since
``tracks`` is UNIQUE(title, artist), these become separate rows and the plays
split, so a song you played ten times can show as 1.

Matching is deliberately high-precision (it is destructive, so a false merge is
worse than a missed one):

* rows are grouped only when one row's *song title* is fully contained in the
  other's combined tokens AND they share an artist token — matching on artist
  alone would collapse a whole discography;
* the contained title must carry a low-document-frequency token, so a recurring
  movie/album name cannot bridge two different songs from the same film;
* compilation/jukebox/mix rows never bridge;
* a cluster is only eligible if a single distinctive token is present in *every*
  member (the song's own identity) and it has at most ``max_cluster`` rows —
  the residual over-merges are all larger movie-soundtrack bundles.
"""

from __future__ import annotations

import re
from collections import defaultdict

_NOISE = {
    "song", "songs", "official", "video", "audio", "lyric", "lyrics", "lyrical",
    "full", "hd", "4k", "8k", "60", "fps", "mv", "visualizer", "visualiser",
    "feat", "ft", "featuring", "from", "movie", "the", "reprise", "cover",
    "version", "ost", "soundtrack", "theme", "title", "track", "presents",
    "presenting", "release", "records", "music", "entertainment", "ver",
    "extended", "promo", "teaser", "tamil", "telugu", "hindi", "with", "and",
    "exclusive", "new", "single", "remix",
}
_COMPILATION = re.compile(
    r"\b(jukebox|mashup|medley|non ?stop|mega ?mix|all songs|full album|"
    r"audio songs|video songs|hits|playlist|\bmix\b|dance cover|reaction)\b", re.I)
_BRACKETS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")


def _toks(s: str) -> set[str]:
    s = (s or "").lower().replace("'", "").replace("’", "")
    s = _BRACKETS.sub(" ", s)
    s = re.sub(r"[^a-z0-9ऀ-ॿ가-힣぀-ヿ]+", " ", s)
    return {t for t in s.split() if len(t) >= 2 and t not in _NOISE}


class _UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _load(con):
    rows = con.execute("""
        SELECT t.id, t.title, t.artist, t.spotify_id, t.deezer_id, t.enriched,
               t.preview_url,
               (SELECT COUNT(*) FROM plays p WHERE p.track_id=t.id) AS plays
        FROM tracks t
    """).fetchall()
    sig = {}
    for r in rows:
        at = _toks(f"{r['title']} {r['artist']}")
        tt = _toks(r["title"])
        art = _toks(r["artist"])
        comp = bool(_COMPILATION.search(f"{r['title']} {r['artist']}"))
        sig[r["id"]] = (at, tt, art, comp)
    return {r["id"]: r for r in rows}, sig


def find_clusters(con, df_max: int = 6, max_cluster: int = 4):
    """Return (eligible, skipped). Each cluster is a dict with canonical + members."""
    meta, sig = _load(con)

    df = defaultdict(int)
    for tid, (at, tt, art, comp) in sig.items():
        if comp:
            continue
        for t in tt:
            df[t] += 1

    def match(a, b):
        ata, tta, arta, ca = sig[a]
        atb, ttb, artb, cb = sig[b]
        if ca or cb:
            return False
        shared = ata & atb
        if len(shared) < 2:          # song title alone is not enough
            return False
        for tt_small, at_big in ((tta, atb), (ttb, ata)):
            if not tt_small or (tt_small - at_big):
                continue
            if not any(df.get(t, 0) <= df_max for t in tt_small):
                continue
            # a shared token beyond the song title = real shared-artist evidence,
            # so "Vaaste" (Dhvani) doesn't absorb "Tere Vaaste" on the word alone
            if not (shared - tt_small):
                continue
            if len(tt_small) >= 2:
                return True
            tok = next(iter(tt_small))
            if len(tok) >= 4:
                return True
        return False

    uf = _UF(list(meta))

    # definite: shared external ids (non-compilation)
    for field in ("deezer_id", "spotify_id"):
        by = defaultdict(list)
        for tid, r in meta.items():
            if r[field] and not sig[tid][3]:
                by[r[field]].append(tid)
        for ids in by.values():
            for i in ids[1:]:
                uf.union(ids[0], i)

    # fuzzy: candidate pairs sharing any token (the song name can sit in the
    # artist field on a mis-parsed row, so index on all tokens, not just title)
    inv = defaultdict(list)
    for tid, (at, tt, art, comp) in sig.items():
        if comp:
            continue
        for t in at:
            inv[t].append(tid)
    seen = set()
    for t, ids in inv.items():
        if len(ids) > 300:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                if match(a, b):
                    uf.union(a, b)

    groups = defaultdict(list)
    for tid in meta:
        groups[uf.find(tid)].append(tid)

    def distinctive(ids):
        common = set.intersection(*(sig[i][0] for i in ids))
        return {t for t in common if df.get(t, 99) <= df_max}

    def canonical(ids):
        return max(ids, key=lambda i: (
            1 if meta[i]["deezer_id"] else 0,
            1 if meta[i]["enriched"] == 1 else 0,
            1 if meta[i]["preview_url"] else 0,
            1 if "|" not in (meta[i]["title"] or "") else 0,
            meta[i]["plays"],
            -len(meta[i]["title"] or ""),
        ))

    def as_dict(ids):
        ids = sorted(ids, key=lambda i: -meta[i]["plays"])
        canon = canonical(ids)
        return {
            "canonical_id": canon,
            "canonical": _row(meta[canon]),
            "members": [_row(meta[i]) for i in ids],
            "total_plays": sum(meta[i]["plays"] for i in ids),
            "key": ", ".join(sorted(distinctive(ids))[:3]),
        }

    eligible, skipped = [], []
    for ids in groups.values():
        if len(ids) < 2:
            continue
        if distinctive(ids) and len(ids) <= max_cluster:
            eligible.append(as_dict(ids))
        else:
            skipped.append(as_dict(ids))
    eligible.sort(key=lambda c: -c["total_plays"])
    skipped.sort(key=lambda c: -len(c["members"]))
    return eligible, skipped


def _row(r):
    return {"id": r["id"], "title": r["title"], "artist": r["artist"],
            "plays": r["plays"], "deezer_id": r["deezer_id"],
            "enriched": r["enriched"]}


# track_id-referencing tables to repoint on merge (besides plays/features/lyrics)
_LINK_TABLES = ("playlist_tracks", "top_items")


def merge_cluster(con, canonical_id: int, dup_ids: list[int]) -> int:
    """Fold every dup row into the canonical row. Returns plays moved."""
    moved = 0
    for dup in dup_ids:
        if dup == canonical_id:
            continue
        before = con.execute(
            "SELECT COUNT(*) FROM plays WHERE track_id=?", (dup,)).fetchone()[0]
        # plays: repoint, dropping (track,time,source) collisions, then clear rest
        con.execute("UPDATE OR IGNORE plays SET track_id=? WHERE track_id=?",
                    (canonical_id, dup))
        con.execute("DELETE FROM plays WHERE track_id=?", (dup,))
        moved += before
        # features / lyrics: canonical keeps its own; else adopt the dup's
        for tbl in ("audio_features", "lyrics"):
            con.execute(f"UPDATE OR IGNORE {tbl} SET track_id=? WHERE track_id=?",
                        (canonical_id, dup))
            con.execute(f"DELETE FROM {tbl} WHERE track_id=?", (dup,))
        for tbl in _LINK_TABLES:
            con.execute(f"UPDATE OR IGNORE {tbl} SET track_id=? WHERE track_id=?",
                        (canonical_id, dup))
            con.execute(f"DELETE FROM {tbl} WHERE track_id=?", (dup,))
        con.execute("DELETE FROM tracks WHERE id=?", (dup,))
    return moved


def apply(con, df_max: int = 6, max_cluster: int = 4, max_passes: int = 6) -> dict:
    """Merge duplicates, repeating until the library converges (merging shifts
    which rows are adjacent, so a couple of passes fully settles it)."""
    clusters_merged = rows_removed = plays_moved = 0
    skipped = []
    for _ in range(max_passes):
        eligible, skipped = find_clusters(con, df_max, max_cluster)
        if not eligible:
            break
        for c in eligible:
            dups = [m["id"] for m in c["members"] if m["id"] != c["canonical_id"]]
            if not dups:
                continue
            plays_moved += merge_cluster(con, c["canonical_id"], dups)
            rows_removed += len(dups)
            clusters_merged += 1
    return {
        "clusters_merged": clusters_merged,
        "rows_removed": rows_removed,
        "plays_consolidated": plays_moved,
        "skipped_for_review": len(skipped),
    }
