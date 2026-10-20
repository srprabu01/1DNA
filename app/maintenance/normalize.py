"""Song-normalization ETL.

Turns messy, multi-language, multi-source track rows into normalized
``(clean_title, artist, movie_album, is_short, is_non_music)`` fields, WITHOUT
mutating the raw ``title``/``artist`` (so it can be re-run and re-inspected).

The rule set was derived from the real library and is documented step-by-step in
``docs/ETL.md``. The guiding principle is **precision over recall**: when a field
cannot be resolved confidently, it is left NULL rather than guessed — a wrong
normalization is worse than a missing one.

Pipeline (see ``normalize()``):
  1  extract      read raw (title, artist)
  2  nonmusic     gate: full sets / mixes / dramas / reactions / promos
  3  shorts       gate: #shorts, /shorts/, dance-trend/reel signals
  4  classify     what does the artist field actually hold? (performer/movie/song/label/handle)
  5  scrub        strip @handles, #hashtags, emoji from a working copy of the title
  6  movie        pull movie/album from From "X" / [OST] / pipe / colon markers
  7  song         extract the song via the branch chosen in step 4
  8  artist       resolve the performing artist (reject labels/handles/movies)
  9  finalize     strip noise tokens, keep real subtitles, tidy whitespace
  10 load         write the normalized columns
"""

from __future__ import annotations

import re
import time
import unicodedata

from ..db import connect

# ── vocabulary ──────────────────────────────────────────────────────────────
NOISE_TOKENS = [
    "official music video", "official video", "music video", "lyric video",
    "lyrical video", "full video song", "video song", "video songs", "full video",
    "audio song", "official audio", "tamil hit song", "tamil movie", "movie songs",
    "m/v", "lyrical", "lyrics", "lyric", "tv size", "hdr", "4k", "8k", "hd", "song",
    "video", "audio", "explicit", "official",
]
_QUALIFIER = re.compile(r"\b(unplugged|instrumental|acoustic)\b", re.I)
_NOISE_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in NOISE_TOKENS) + r")(?!\w)", re.I)
_YEAR_RE = re.compile(r"\((?:19|20)\d{2}\)")
# language tag in (parens) OR [brackets] — so "[Tamil]" is stripped like "(Tamil)"
_LANG_TAG = re.compile(r"[\(\[]\s*(tamil|telugu|hindi|kannada|malayalam|korean|japanese)\s*[\)\]]", re.I)

_NONMUSIC = re.compile(
    r"\b(full set|full concert|dj set|club set|cookout session|mix ?tape|mix playlist"
    r"|party mix|mashup|jukebox|workout|no equipment|reaction|vlog|trailer|gameplay"
    r"|game ?play|podcast|interview|behind the scenes|live[- ]action|documentary"
    r"|unboxing|prank|tutorial|how to|news|full album|audio jukebox"
    # compilations / playlists / listicles are collections, not a single song
    r"|special collections?|birthday special|hits? collection|\[?\s*playlist\s*\]?"
    r"|top\s+\d+|most anticipated|best of\s+(?:19|20)\d{2})\b", re.I)
_DRAMA = re.compile(r"\bep\s?\d+\b.*\b(eng\s?sub|dubbed)\b|【[^】]*drama[^】]*】", re.I)
_PROMO = re.compile(r"\b(auto[- ]?apply|custom resume|job search|ai agent|jobright)\b", re.I)

_SHORTS = re.compile(
    r"(?:^|\W)#?shorts?\b|#ytshorts|/shorts/|#(?:dance(?:challenge)?|trend(?:ing)?|tiktok"
    r"|viral(?:video)?|fyp|reels?)\b|\bdance challenge\b|\bdc[-\s]|\bby @\w+|\bchallenge\b", re.I)
_HASHTAG = re.compile(r"#[^\s#]+")
_HANDLE = re.compile(r"@[\w.]+")
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
                    r"\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F←-⇿⌀-⏿]")

_LABEL = re.compile(
    r"(entertainment|records?|recordz|studios?|media|musiq|vibes|labels?|channel|pictures"
    r"|productions?|hit songs)\s*$|^(1thek|hybe|jyp|sm entertainment|yg\b|stone music"
    r"|starship|t-?series|sun music|ayngaran|yaam records|saregama|think music|sony music"
    r"|divo|zee music|speed records)", re.I)

_KNOWN_BANDS = {"itzy", "bibi", "max", "illenium", "bts", "curv", "mamamoo", "twice", "iu",
                "exo", "nct", "aespa", "ive", "newjeans", "blackpink", "seventeen", "kda", "k/da"}
_HANDLE_MAP = {"arrahman": "A.R. Rahman", "saiabhyankkar": "Sai Abhyankkar", "mcsai": "MC Sai",
               "anirudhofficial": "Anirudh Ravichander", "thisisdsp": "Devi Sri Prasad"}

# (From "X") / [From X] / - From "X"   — "From" MUST follow a bracket or a dash,
# so a bare "from"/"out of" inside a real title ("Far From Home", "Out of Time")
# is never mistaken for a movie marker. Capture stops at a closing quote/bracket
# or end of string.
_FROM = re.compile(
    r"(?:[\(\[]|[-–—])\s*from\s+[\"“‘']?(?P<m>.+?)[\"”’']?\s*(?:[\)\]]|$)", re.I)

# Album names from the Apple export carry the film explicitly, e.g.
#   "Amaran (Original Motion Picture Soundtrack)"  -> movie "Amaran"
#   "Music from the Motion Picture Barbie"          -> movie "Barbie"
# Only these soundtrack/score forms qualify; a plain album (e.g. "Nine Track
# Mind") is left as an album, never promoted to a movie.
_OST_ALBUM = re.compile(
    r"^(?P<m>.+?)\s*[\(\[]\s*(?:the\s+)?(?:original\s+)?"
    r"(?:(?:motion\s+picture|television|tv|web\s+series)\s+)?"
    r"(?:soundtrack|score|ost)\s*[\)\]]\s*$", re.I)
_OST_FROM = re.compile(
    r"^(?:music\s+)?from\s+the\s+(?:motion\s+picture|film|movie)\s+"
    r"[\"'“‘]?(?P<m>.+?)[\"'”’]?\s*$", re.I)
# feat / ft / featuring / with -- the bare word with is matched ONLY inside a
# bracket, so a title that merely contains With (Here With Me, With You,
# With My Tears) is never truncated; unbracketed, only feat/ft/featuring clip.
_FEAT = re.compile(
    r"\s*[\(\[]\s*(?:feat\.?|ft\.?|featuring|with|prod\.?)\b[^)\]]*[\)\]]?"
    r"|\s+(?:feat\.?|ft\.?|featuring)\b[^)\]]*",
    re.I)
_QUOTED_MV = re.compile(r"^(?P<a>.+?)\s*['\"‘“](?P<s>[^'\"’”]+)['\"’”]\s*(?:official\s*)?(?:m/?v|music\s*video)?", re.I)
_VERSION_TAIL = re.compile(
    r"\b(remaster(?:ed)?(?:\s+\d{4})?|tv size|instrumental|acoustic|unplugged|reprise"
    r"|live(?:\s+at\s+.*)?|remix|re-?edit|.*\bmix|(?:japanese|korean|tamil|telugu|hindi)"
    r"(?:\s+version)?|full version|extended version|end title|slowed(?:\s*\+?\s*reverb)?)\b", re.I)


def _fold(s: str) -> str:
    """NFKC-fold styled/mathematical unicode to ASCII for detection only."""
    return unicodedata.normalize("NFKC", s or "")


def _scrub(title: str) -> str:
    t = _HANDLE.sub(" ", title or "")
    t = _HASHTAG.sub(" ", t)
    t = _EMOJI.sub(" ", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def _titlecase(s: str) -> str:
    if s and s.isupper():
        return " ".join(w.capitalize() for w in s.split())
    return s


def _finalize(song: str | None) -> str | None:
    if not song:
        return None
    s = _NOISE_RE.sub(" ", song)
    s = _YEAR_RE.sub(" ", s)
    s = _LANG_TAG.sub(" ", s)
    s = _QUALIFIER.sub(" ", s)
    s = _HASHTAG.sub(" ", _HANDLE.sub(" ", s))
    s = _EMOJI.sub(" ", s)
    s = re.sub(r"[\(\[]\s*[\)\]]", " ", s)   # empty parens/brackets left by strips
    s = s.strip(" \"“”'‘’|-–—_:•[]")
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = _titlecase(s)
    return s or None


def _clean_movie(m: str | None) -> str | None:
    if not m:
        return None
    m = _LANG_TAG.sub("", m).strip(" \"“”'‘’)]").strip()
    m = re.sub(r"\s{2,}", " ", m)
    return m or None


def _movie_from_album(album: str | None) -> str | None:
    """Pull a film name from an Apple soundtrack/score album, else None."""
    if not album:
        return None
    m = _OST_ALBUM.match(album.strip()) or _OST_FROM.match(album.strip())
    return _clean_movie(m.group("m")) if m else None


def _looks_all_caps(s: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", s or "")
    return len(letters) >= 3 and letters.isupper()


_ARTIST_IS_SONG = re.compile(
    r"\b(?:full\s+)?(?:video\s+)?song\s*$|\blyric(?:al)?(?:\s+video)?\s*$"
    r"|\bvideo\s*$|\(extended version\)\s*$", re.I)


_FEAT_SPLIT = re.compile(r"\s*(?:\bft\.?|\bfeat\.?|featuring|\||,|&)\s*", re.I)


def _song_from_garbled_artist(a: str):
    """Some rows put the SONG in the artist field wrapped in video-title noise:
      'DANCE MERI RANI: Guru Randhawa Ft Nora Fatehi'  -> ('DANCE MERI RANI', 'Guru Randhawa')
      'Kusu Kusu Song Ft Nora Fatehi'                  -> ('Kusu Kusu', 'Nora Fatehi')
    Returns (song, performer) or None."""
    if not a:
        return None
    # "Song: performers"  (a real artist name almost never contains a colon)
    if ":" in a:
        pre, post = a.split(":", 1)
        pre, post = pre.strip(), post.strip()
        # "Full Video: <Song>" / "Lyrical: <Song>" — the pre-colon part is pure
        # noise, so the SONG is the post-colon part (not the stripped-to-empty pre).
        if post and not _NOISE_RE.sub("", pre).strip(" -–—|"):
            return post, None
        if 3 <= len(pre) <= 70 and (re.search(r"\b(ft\.?|feat|featuring|&)\b", post, re.I)
                                    or "," in post or "|" in post or " " in post):
            return pre, _FEAT_SPLIT.split(post, 1)[0].strip() or None
    # "<Song> [Video] Song/Lyric ... Ft <performer>"
    m = re.match(r"^(?P<s>.+?)\s+(?:full\s+)?(?:video\s+)?(?:song|lyric(?:al)?)\b(?P<rest>.*)$", a, re.I)
    if m and len(m.group("s")) >= 3:
        pm = re.search(r"\b(?:ft\.?|feat\.?|featuring)\s+(?P<p>[^|,]+)", m.group("rest"), re.I)
        return m.group("s").strip(), (pm.group("p").strip() if pm else None)
    return None


def _is_song_in_artist(a: str) -> bool:
    if not a:
        return False
    if _ARTIST_IS_SONG.search(a):
        return True
    if _looks_all_caps(a) and a.strip().lower() not in _KNOWN_BANDS:
        return True
    # Tamil / Telugu / Devanagari / Hangul script in the artist field
    return bool(re.search(r"[஀-௿ఀ-౿ऀ-ॿ가-힣]", a))


def _extract_song_from_title(wt: str, from_matched: bool = False, artist: str = "") -> str | None:
    """Song from a (handle-scrubbed, movie-removed) working title.

    ``artist`` is the raw artist field; it disambiguates the dash split — an
    ``Artist - Song`` title (Western / K-pop) must keep the TAIL as the song,
    while a ``Song - Subtitle`` title (Tamil) keeps the head.
    """
    wt = wt.strip()
    if not wt:
        return None
    # K-pop quoted MV: Artist "Song" M/V
    q = _QUOTED_MV.match(wt)
    if q and q.group("s"):
        return q.group("s").strip()
    # strip feat/with clauses
    wt = _FEAT.sub(" ", wt).strip()
    # colon prefix: "Movie: Song ..." -> post-colon
    if ":" in wt and "|" in wt and wt.index(":") < wt.index("|"):
        wt = wt.split(":", 1)[1]
    # pipe / // split -> first surviving non-noise segment
    if "|" in wt or "//" in wt:
        segs = [s.strip() for s in re.split(r"\s*\|\s*|\s*//\s*", wt) if s.strip()]
        for seg in segs:
            core = _NOISE_RE.sub("", seg).strip(" -–—_:")
            if core and not _HANDLE.match(core) and len(re.sub(r"[^A-Za-z஀-힣]", "", core)) >= 2:
                return core
        return segs[0] if segs else wt
    # dash tail: split on LAST ' - '; strip version tails, keep head.
    # Skip when a From-clause already delimited the song (its internal dash is real).
    if " - " in wt and not from_matched:
        head, tail = wt.rsplit(" - ", 1)
        if _VERSION_TAIL.search(tail):
            q2 = _VERSION_TAIL.search(tail)
            return f"{head.strip()} ({q2.group(0).strip()})" if q2 else head.strip()
        # "Artist - Song": when the head IS the artist field, the song is the tail.
        a = (artist or "").strip().lower()
        h = head.strip().lower()
        if a and (h == a or (len(h) >= 4 and h in a)):
            return tail.strip()
        # otherwise a poetic subtitle (not a known film) -> keep head, movie empty
        return head.strip()
    return wt


def normalize(title: str, artist: str, album: str | None = None) -> dict:
    raw_t, raw_a = title or "", artist or ""
    blob = _fold(f"{raw_t} {raw_a}")

    # 2 + 3  gates
    is_non_music = bool(_NONMUSIC.search(blob) or _DRAMA.search(blob) or _PROMO.search(blob))
    is_short = bool(_SHORTS.search(blob)) or len(_HASHTAG.findall(raw_t)) >= 2
    if _DRAMA.search(blob):        # drama episodes are long-form, not shorts
        is_short = False

    # 4  what is the artist field?
    a = raw_a.strip()
    art_is_label = bool(_LABEL.search(a))
    art_is_handle = a.startswith("@")
    garbled = None if (art_is_label or art_is_handle) else _song_from_garbled_artist(a)
    # guard: an "Artist - Song" title whose leading segment equals the artist field
    # means the artist is a real performer (even ALL-CAPS / CJK like NCT 127, ROSE),
    # so the artist field must NOT be mistaken for the song.
    _probe = _scrub(raw_t)
    _lead = _probe.split(" - ", 1)[0].strip().lower() if " - " in _probe else ""
    artist_leads_title = bool(a and _lead and (_lead == a.lower()
                              or (len(_lead) >= 4 and _lead in a.lower())))
    # the title is ALREADY a clean song when it carries no pipe / dash / movie
    # marker / video-noise; then the artist field is a real performer, not the
    # song, so the weak ALL-CAPS/CJK heuristic must not fire (LAUV, EGOIST, NCT).
    # a From-marker ("Song (From X)") still leaves a clean song, so it does NOT
    # disqualify; only pipes, dashes and video-noise do.
    title_is_clean_song = not ("|" in _probe or " - " in _probe or _NOISE_RE.search(_probe))
    _strong = garbled is not None or (bool(_ARTIST_IS_SONG.search(a)) and not art_is_label)
    _weak = (_is_song_in_artist(a) and not _ARTIST_IS_SONG.search(a) and not art_is_label)
    art_is_song = _strong or (_weak and not artist_leads_title and not title_is_clean_song)

    # 5  working title (handles/tags/emoji removed)
    wt = _scrub(raw_t)

    # 6  movie / album from explicit markers
    movie = None
    fm = _FROM.search(raw_t)
    if fm:
        movie = _clean_movie(fm.group("m"))
        wt = _FROM.sub(" ", wt).strip()

    # 7 + 8  song + artist by branch
    if art_is_song:
        # SWAPPED-PIPE: the song sits in the artist field, the movie leads the title
        if garbled:
            song, artist_out = garbled
        else:
            song, artist_out = _ARTIST_IS_SONG.sub("", a), None
        if not movie and "|" in wt:
            seg0 = wt.split("|", 1)[0].strip()
            cand = _clean_movie(_YEAR_RE.sub("", _NOISE_RE.sub("", seg0)).strip(" -–—|"))
            # a movie, not a name-list, not the song itself, not a credit phrase
            if (cand and "," not in seg0 and len(seg0.split()) <= 4 and len(cand) > 2
                    and cand.lower() != (song or "").strip().lower()
                    and not re.search(r"\b(singers?|music director|lyrics?|composed|feat\.?|ft\.?)\b",
                                      seg0, re.I)):
                movie = cand
        artist_out = artist_out or _resolve_artist(raw_a, wt, False, False, swapped=True)
    else:
        song = _extract_song_from_title(wt, from_matched=bool(fm), artist=raw_a)
        artist_out = _resolve_artist(raw_a, wt, art_is_label, art_is_handle)

    # 6b  fall back to the album for the film, when the title gave no marker.
    if not movie:
        movie = _movie_from_album(album)

    return {
        # a confirmed non-music row has no song to name -> leave clean_title null
        "clean_title": None if is_non_music else _finalize(song),
        "artist": artist_out,
        "movie_album": movie,
        "is_short": is_short,
        "is_non_music": is_non_music,
    }


def _resolve_artist(raw_a, wt, art_is_label, art_is_handle, swapped=False):
    a = (raw_a or "").strip()
    # reject label / handle / song-in-field, derive from title
    if art_is_label or art_is_handle or swapped or _is_song_in_artist(a):
        # a surviving @handle in the title is usually the composer
        for h in _HANDLE.findall(wt) + _HANDLE.findall(raw_a):
            key = h.lstrip("@").lower()
            if key in _HANDLE_MAP:
                return _HANDLE_MAP[key]
        # K-pop quoted-MV leader
        q = _QUOTED_MV.match(_scrub(wt))
        if q and q.group("a") and not _LABEL.search(q.group("a")):
            lead = q.group("a").strip(" -–—|_")
            if lead:
                return lead
        return None            # precision: leave null rather than guess
    return a or None


# ── ETL runner (Load) ───────────────────────────────────────────────────────
_COLUMNS = [
    ("clean_title", "TEXT"), ("movie_album", "TEXT"),
    ("is_short", "INTEGER"), ("is_non_music", "INTEGER"), ("normalized_at", "TEXT"),
]


def _ensure_columns(con):
    have = {r[1] for r in con.execute("PRAGMA table_info(tracks)")}
    for name, typ in _COLUMNS:
        if name not in have:
            con.execute(f"ALTER TABLE tracks ADD COLUMN {name} {typ}")


def preview(limit: int = 60):
    """Dry-run: normalize a sample and return before/after rows. No writes."""
    with connect() as con:
        rows = con.execute("SELECT id, title, artist, album FROM tracks ORDER BY RANDOM() LIMIT ?",
                           (limit,)).fetchall()
    out = []
    for r in rows:
        n = normalize(r["title"], r["artist"], r["album"])
        out.append({"id": r["id"], "raw_title": r["title"], "raw_artist": r["artist"], **n})
    return out


def run(con=None) -> dict:
    """Full ETL over the library. Writes clean_title / movie_album / is_short /
    is_non_music, keeping raw title & artist intact."""
    own = con is None
    cm = connect() if own else None
    con = cm.__enter__() if own else con
    try:
        _ensure_columns(con)
        rows = con.execute("SELECT id, title, artist, album FROM tracks").fetchall()
        stats = {"total": len(rows), "renamed": 0, "movie_found": 0,
                 "shorts": 0, "non_music": 0}
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        for r in rows:
            n = normalize(r["title"], r["artist"], r["album"])
            con.execute(
                "UPDATE tracks SET clean_title=?, movie_album=?, is_short=?, "
                "is_non_music=?, normalized_at=? WHERE id=?",
                (n["clean_title"], n["movie_album"], int(n["is_short"]),
                 int(n["is_non_music"]), now, r["id"]))
            if n["clean_title"] and n["clean_title"] != (r["title"] or ""):
                stats["renamed"] += 1
            if n["movie_album"]:
                stats["movie_found"] += 1
            stats["shorts"] += int(n["is_short"])
            stats["non_music"] += int(n["is_non_music"])
        if own:
            con.commit()
        return stats
    finally:
        if own:
            cm.__exit__(None, None, None)
