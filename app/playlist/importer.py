"""Import real playlists from your logged-in Spotify / YouTube Music sessions."""
from ..db import connect, upsert_track
from . import store


def list_spotify():
    from ..ingest import spotify
    token = spotify._access_token()
    if not token:
        raise RuntimeError("Spotify not connected.")
    out, offset = [], 0
    while True:
        data = spotify._get("/me/playlists", token, {"limit": 50, "offset": offset})
        for pl in data.get("items", []):
            out.append({"id": pl["id"], "name": pl["name"],
                        "tracks": (pl.get("tracks") or {}).get("total", 0)})
        if not data.get("next"):
            break
        offset += 50
    return out


def import_spotify(playlist_id, name):
    from ..ingest import spotify
    token = spotify._access_token()
    if not token:
        raise RuntimeError("Spotify not connected.")
    track_ids, offset = [], 0
    with connect() as con:
        while True:
            data = spotify._get(f"/playlists/{playlist_id}/tracks", token,
                                {"limit": 100, "offset": offset})
            for it in data.get("items", []):
                tr = it.get("track")
                if tr and tr.get("name"):
                    tid = spotify._store_track(con, tr)
                    if tid:
                        track_ids.append(tid)
            if not data.get("next"):
                break
            offset += 100
    pid = store.create(name, track_ids, source="spotify", external_id=playlist_id)
    return {"playlist_id": pid, "tracks": len(track_ids)}


def list_ytmusic():
    from ..ingest import ytmusic
    yt = ytmusic._client()
    pls = yt.get_library_playlists(limit=100)
    return [{"id": p["playlistId"], "name": p["title"],
             "tracks": p.get("count")} for p in pls if p.get("playlistId")]


def import_ytmusic(playlist_id, name):
    from ..ingest import ytmusic
    yt = ytmusic._client()
    data = yt.get_playlist(playlist_id, limit=1000)
    track_ids = []
    with connect() as con:
        for it in data.get("tracks", []):
            if not it.get("title"):
                continue
            arts = ", ".join(a["name"] for a in (it.get("artists") or []) if a.get("name"))
            tid = upsert_track(con, title=it["title"], artist=arts or "Unknown",
                               album=(it.get("album") or {}).get("name"))
            if tid:
                track_ids.append(tid)
    pid = store.create(name, track_ids, source="ytmusic", external_id=playlist_id)
    return {"playlist_id": pid, "tracks": len(track_ids)}
