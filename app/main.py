from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .db import init_db, connect, rows_to_dicts
from .ingest import spotify, youtube, apple, ytmusic, youtube_api
from .enrich import deezer
from .analytics import engine as analytics
from .recommend import engine as recommend

app = FastAPI(title="Music DNA Analyzer")
init_db()

STATIC = config.BASE_DIR / "static"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    # No-I/O liveness probe for the container healthcheck (unlike /api/status,
    # which runs analytics queries and can stall during a busy analyze run).
    return {"status": "ok"}


# ---------- status / connections ----------

@app.get("/api/status")
def status():
    return {
        "spotify_connected": spotify.connected(),
        "ytmusic_connected": ytmusic.connected(),
        "youtube_connected": youtube_api.connected(),
        "spotify_oauth_ready": spotify.oauth_configured(),
        "ytmusic_oauth_ready": ytmusic.oauth_configured(),
        "youtube_oauth_ready": youtube_api.oauth_configured(),
        "overview": analytics.overview(),
    }


# ---------- YouTube Data API v3 — official "Sign in with Google" ----------

@app.get("/api/youtube/login")
def youtube_login():
    if not youtube_api.oauth_configured():
        raise HTTPException(400, "Add YTDATA_CLIENT_ID / SECRET to .env first (see README).")
    return RedirectResponse(youtube_api.auth_url())


@app.get("/callback/youtube")
def youtube_callback(code: str = "", error: str = ""):
    if error or not code:
        raise HTTPException(400, f"YouTube sign-in failed: {error or 'no code'}")
    youtube_api.exchange_code(code)
    return RedirectResponse("/?connected=youtube")


@app.post("/api/youtube/sync")
def youtube_data_sync():
    try:
        return youtube_api.sync()
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- Spotify OAuth sign-in ----------

@app.get("/api/spotify/login")
def spotify_login():
    if not spotify.oauth_configured():
        raise HTTPException(400, "Add SPOTIFY_CLIENT_ID / SECRET to .env first (see README).")
    return RedirectResponse(spotify.auth_url())


@app.get("/callback/spotify")
def spotify_callback(code: str = "", error: str = ""):
    if error or not code:
        raise HTTPException(400, f"Spotify sign-in failed: {error or 'no code'}")
    spotify.exchange_code(code)
    return RedirectResponse("/?connected=spotify")


# ---------- YouTube Music OAuth (device-code) sign-in ----------

@app.post("/api/ytmusic/oauth/start")
def ytmusic_oauth_start():
    try:
        return ytmusic.oauth_start()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/ytmusic/oauth/finish")
def ytmusic_oauth_finish():
    try:
        ytmusic.oauth_finish()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"Not approved yet, or sign-in failed: {e}")


@app.post("/api/spotify/cookie")
def spotify_cookie(sp_dc: str = Body(..., embed=True)):
    try:
        spotify.save_cookie(sp_dc)
        return {"ok": True, "method": "cookie"}
    except Exception as e:
        raise HTTPException(400, f"Cookie didn't work: {e}")


@app.post("/api/spotify/bearer")
def spotify_bearer(token: str = Body(..., embed=True)):
    try:
        spotify.save_bearer(token)
        return {"ok": True, "method": "bearer"}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/spotify/sync")
def spotify_sync():
    try:
        return spotify.sync()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/ytmusic/auth")
def ytmusic_auth(headers_raw: str = Body(..., embed=True)):
    try:
        ytmusic.save_auth(headers_raw)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(400, f"Could not read those headers: {e}")


@app.post("/api/ytmusic/sync")
def ytmusic_sync():
    try:
        return ytmusic.sync()
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- imports ----------

@app.post("/api/import/youtube")
async def import_youtube(file: UploadFile = File(...), music_only: bool = True,
                         authoritative: bool = True):
    content = await file.read()
    try:
        # When signed into the YouTube Data API, classify every video by its real
        # YouTube category (authoritative). Otherwise fall back to the heuristic.
        if authoritative and youtube_api.connected():
            return youtube.import_authoritative(content)
        return youtube.import_file(content, music_only=music_only)
    except Exception as e:
        raise HTTPException(400, f"Could not import: {e}")


@app.post("/api/import/apple")
async def import_apple(file: UploadFile = File(...)):
    try:
        return apple.import_file(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")


@app.post("/api/import/spotify")
async def import_spotify_history(file: UploadFile = File(...)):
    from .ingest import spotify_history
    try:
        return spotify_history.import_file(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")


# ---------- processing ----------

@app.post("/api/enrich")
def enrich(limit: int = 40):
    return deezer.enrich_batch(limit=limit)


@app.post("/api/analyze")
def analyze(limit: int = 8, retry_errors: bool = False, parallel: bool = False):
    # parallel=true fans the DSP across all CPU cores (crash-isolated); pass limit=0
    # for "analyze everything". Serial default keeps the incremental UI flow cheap.
    if parallel:
        from .audio.pipeline import analyze_batch_parallel
        return analyze_batch_parallel(retry_errors=retry_errors, limit=limit or None)
    return audio_features().analyze_batch(limit=limit, retry_errors=retry_errors)


def audio_features():
    from .audio import features
    return features


@app.post("/api/lyrics/fetch")
def lyrics_fetch(limit: int = 10):
    from .lyrics import engine as ly
    return ly.fetch_batch(limit=limit)


@app.get("/api/lyrics/overview")
def lyrics_overview():
    from .lyrics import engine as ly
    return ly.overview()


# ---------- analytics ----------

@app.get("/api/stats/overview")
def stats_overview():
    return analytics.overview()


@app.get("/api/stats/top")
def stats_top(kind: str = "tracks", limit: int = 25, days: int | None = None):
    return analytics.top(kind=kind, limit=limit, days=days)


@app.get("/api/stats/patterns")
def stats_patterns():
    return analytics.patterns()


@app.get("/api/stats/genres")
def stats_genres():
    return analytics.genres()


@app.get("/api/stats/audio-profile")
def stats_audio_profile():
    return analytics.audio_profile()


@app.get("/api/track/{track_id}")
def track_detail(track_id: int):
    out = analytics.track_detail(track_id)
    if not out:
        raise HTTPException(404, "track not found")
    return out


@app.get("/api/track/{track_id}/visual")
def track_visual(track_id: int, kind: str = "spectrogram"):
    from .audio import visuals
    try:
        png = visuals.render(track_id, kind=kind)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/api/tracks")
def list_tracks(q: str = "", limit: int = 100):
    # Token search: every word must appear somewhere in title+artist (any order,
    # any field, case-insensitive). Ignores brackets/punctuation so "bibi mv" or
    # "도깨비 ost" find the track even when the words are scattered in the title.
    import re as _re
    tokens = [t for t in _re.split(r"[^\w가-힣぀-ヿ一-鿿]+", q) if t]
    where = "1=1"
    params = []
    for tok in tokens:
        where += " AND (t.title LIKE ? OR t.artist LIKE ?)"
        like = f"%{tok}%"
        params += [like, like]
    params.append(limit)
    with connect() as con:
        rows = con.execute(
            f"""SELECT t.id, t.title, t.artist, t.genre, t.preview_url, t.release_date,
                      t.clean_title, t.movie_album, t.is_short, t.is_non_music,
                      f.bpm, f.key, f.mode, f.camelot, f.energy, f.valence, f.danceability,
                      f.acousticness, f.brightness, f.instrumentalness, f.loudness_db,
                      COALESCE(p.n, 0) plays
               FROM tracks t
               LEFT JOIN audio_features f ON f.track_id = t.id AND f.error IS NULL
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id = t.id
               WHERE {where}
               ORDER BY plays DESC, t.id DESC LIMIT ?""",
            params,
        ).fetchall()
    return rows_to_dicts(rows)


# ---------- advanced macro (Phase 3) ----------

@app.get("/api/stats/mood-over-time")
def stats_mood_over_time():
    from .analytics import advanced
    return advanced.mood_over_time()


@app.get("/api/stats/calendar")
def stats_calendar():
    from .analytics import advanced
    return advanced.calendar()


@app.get("/api/stats/discovery")
def stats_discovery():
    from .analytics import advanced
    return advanced.discovery()


@app.get("/api/stats/concentration")
def stats_concentration():
    from .analytics import advanced
    return advanced.concentration()


@app.get("/api/stats/clusters")
def stats_clusters(k: int = 4):
    from .analytics import advanced
    return advanced.clusters(k=k)


# ---------- playlists (Phase 2) ----------

@app.get("/api/playlists")
def playlists_list():
    from .playlist import store
    return store.list_playlists()


@app.post("/api/playlists/build")
def playlists_build(kind: str = Body("top", embed=True),
                    value: str | None = Body(None, embed=True)):
    from .playlist import store
    pid = store.build_smart(kind=kind, value=value)
    if not pid:
        raise HTTPException(400, "No matching tracks to build a playlist from.")
    return {"playlist_id": pid}


@app.get("/api/playlists/{pid}/analyze")
def playlists_analyze(pid: int):
    from .playlist import analyzer
    out = analyzer.analyze(pid)
    if out is None:
        raise HTTPException(404, "playlist not found")
    return out


@app.delete("/api/playlists/{pid}")
def playlists_delete(pid: int):
    from .playlist import store
    store.delete(pid)
    return {"ok": True}


@app.get("/api/playlists/import/spotify")
def playlists_spotify_list():
    from .playlist import importer
    try:
        return importer.list_spotify()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/playlists/import/spotify")
def playlists_spotify_import(id: str = Body(..., embed=True),
                             name: str = Body(..., embed=True)):
    from .playlist import importer
    try:
        return importer.import_spotify(id, name)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/playlists/import/ytmusic")
def playlists_ytmusic_list():
    from .playlist import importer
    try:
        return importer.list_ytmusic()
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/playlists/import/ytmusic")
def playlists_ytmusic_import(id: str = Body(..., embed=True),
                             name: str = Body(..., embed=True)):
    from .playlist import importer
    try:
        return importer.import_ytmusic(id, name)
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------- deep research ----------

@app.get("/api/research/deep")
def research_deep():
    from .analytics import research
    return research.deep_research()


@app.get("/api/track/{track_id}/dossier")
def track_dossier(track_id: int):
    from .analytics import research
    return research.track_dossier(track_id)


# ---------- wrapped report (Phase 4) ----------

@app.get("/api/wrapped")
def wrapped_report(year: int | None = None):
    from .report import wrapped
    return wrapped.generate(year=year)


# ---------- data management ----------

@app.post("/api/data/clear")
def data_clear():
    with connect() as con:
        for tbl in ("plays", "audio_features", "lyrics", "playlist_tracks",
                    "playlists", "top_items", "recommendations", "tracks"):
            con.execute(f"DELETE FROM {tbl}")
    return {"ok": True}


# ---------- taste / similarity (deepcut-derived upgrades) ----------

@app.post("/api/taste/fit")
def taste_fit():
    from .analytics import taste
    return taste.fit()


@app.get("/api/track/{track_id}/similar")
def track_similar(track_id: int):
    from .analytics import similar
    return similar.similar_to(track_id)


@app.get("/api/space/diagnostics")
def space_diagnostics():
    from .analytics import similar
    return similar.diagnostics()


@app.get("/api/search/semantic")
def search_semantic(q: str = "", k: int = 20):
    from .audio import embed
    return embed.text_search(q, k=k)


@app.post("/api/embed")
def embed_previews(limit: int = 50):
    from .audio import embed
    return embed.embed_batch(limit=limit)


@app.get("/api/embed/status")
def embed_status():
    from .audio import embed
    return embed.status()


# ---------- recommendations ----------

@app.post("/api/recommendations/generate")
def gen_recs():
    return recommend.generate()


@app.get("/api/recommendations")
def get_recs():
    return recommend.latest()


# ---------- duplicate merge (fixes split play counts) ----------

@app.get("/api/dedupe/preview")
def dedupe_preview():
    from .maintenance import dedupe
    with connect() as con:
        eligible, skipped = dedupe.find_clusters(con)
    return {
        "clusters": len(eligible),
        "rows_removed": sum(len(c["members"]) - 1 for c in eligible),
        "plays_affected": sum(c["total_plays"] for c in eligible),
        "skipped": len(skipped),
        "examples": eligible[:40],
    }


@app.post("/api/dedupe/apply")
def dedupe_apply():
    from .maintenance import dedupe
    with connect() as con:
        return dedupe.apply(con)


# ---------- song-normalization ETL (see docs/ETL.md) ----------

@app.get("/api/normalize/preview")
def normalize_preview(limit: int = 60):
    from .maintenance import normalize
    return {"rows": normalize.preview(limit=limit)}


@app.post("/api/normalize/apply")
def normalize_apply():
    from .maintenance import normalize
    return normalize.run()


app.mount("/static", StaticFiles(directory=STATIC), name="static")
