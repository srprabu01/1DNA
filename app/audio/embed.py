"""Optional semantic layer — CLAP embeddings of the cached 30s previews.

Adapted from deepcut's embed.py (CLAP tower only, to keep it to one ~600 MB
model). CLAP shares audio and text in ONE latent space, so embedding your
previews unlocks two things the DSP features can't do:

  * text-prompt search — "cavernous reverb, brushed drums, no vocals"
  * genuine sound-alike similarity (used by /similar when embeddings exist)

Requires `torch` + `transformers`. If they're absent, status() reports it and
the endpoints return a clear message instead of crashing — nothing else breaks.
"""
import numpy as np

from ..config import DATA_DIR, CACHE_DIR
from ..db import connect

EMB_DIR = DATA_DIR / "embeddings"
CLAP_MODEL = "laion/larger_clap_music"
_ENC = {}


def available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def _clap():
    if "clap" not in _ENC:
        import torch
        from transformers import ClapModel, ClapProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = ClapProcessor.from_pretrained(CLAP_MODEL)
        model = ClapModel.from_pretrained(CLAP_MODEL).to(dev).eval()
        _ENC["clap"] = (proc, model, dev, torch)
    return _ENC["clap"]


def _l2(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


# ---- storage (one npz: ids + matrix) ----

def clap_matrix():
    p = EMB_DIR / "clap.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["ids"].astype(int), d["X"].astype(np.float32)


def _save(ids, X):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_DIR / "clap.npz", ids=np.asarray(ids, dtype=int),
             X=np.asarray(X, dtype=np.float32))


def status():
    cur = clap_matrix()
    have = 0 if cur is None else len(cur[0])
    with connect() as con:
        pending = con.execute(
            "SELECT COUNT(*) c FROM tracks WHERE preview_url IS NOT NULL").fetchone()["c"]
    return {"available": available(), "model": CLAP_MODEL,
            "embedded": have, "with_preview": pending,
            "note": None if available() else
            "Install PyTorch + transformers to enable semantic search:  "
            "pip install torch transformers librosa"}


def embed_batch(limit: int = 50):
    if not available():
        return {"error": "PyTorch/transformers not installed. "
                         "pip install torch transformers librosa"}
    import librosa
    proc, model, dev, torch = _clap()

    cur = clap_matrix()
    have_ids = set(cur[0].tolist()) if cur else set()
    with connect() as con:
        rows = con.execute(
            """SELECT t.id FROM tracks t
               JOIN audio_features f ON f.track_id = t.id AND f.error IS NULL
               WHERE t.preview_url IS NOT NULL
               ORDER BY (SELECT COUNT(*) FROM plays p WHERE p.track_id=t.id) DESC""").fetchall()
    todo = [r["id"] for r in rows if r["id"] not in have_ids
            and (CACHE_DIR / f"{r['id']}.mp3").exists()][:limit]
    if not todo:
        return {"embedded": 0, "remaining": 0, "total": len(have_ids)}

    new_ids, new_vecs = [], []
    for tid in todo:
        try:
            y, _ = librosa.load(str(CACHE_DIR / f"{tid}.mp3"), sr=48000, mono=True)
            if len(y) < 48000:
                continue
            inp = proc(audios=[y.astype(np.float32)], sampling_rate=48000,
                       return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.inference_mode():
                feat = model.get_audio_features(**inp)
            new_ids.append(tid)
            new_vecs.append(_l2(feat.float().cpu().numpy()[0]))
        except Exception:
            continue

    if new_ids:
        if cur:
            ids = np.concatenate([cur[0], np.array(new_ids)])
            X = np.vstack([cur[1], np.array(new_vecs)])
        else:
            ids, X = np.array(new_ids), np.array(new_vecs)
        _save(ids, X)
    with connect() as con:
        total_prev = con.execute(
            """SELECT COUNT(*) c FROM tracks t JOIN audio_features f
               ON f.track_id=t.id AND f.error IS NULL
               WHERE t.preview_url IS NOT NULL""").fetchone()["c"]
    done = len(have_ids) + len(new_ids)
    return {"embedded": len(new_ids), "total": done, "remaining": max(0, total_prev - done)}


def text_search(query: str, k: int = 20):
    if not query.strip():
        return {"available": True, "results": []}
    if not available():
        return {"error": status()["note"]}
    cur = clap_matrix()
    if cur is None:
        return {"error": "No embeddings yet — run 'Embed previews' first."}
    ids, X = cur
    proc, model, dev, torch = _clap()
    inp = proc(text=[query], return_tensors="pt", padding=True)
    inp = {kk: v.to(dev) for kk, v in inp.items()}
    with torch.inference_mode():
        q = model.get_text_features(**inp)
    qv = _l2(q.float().cpu().numpy()[0])
    sims = X @ qv
    from ..analytics.vectors import track_meta
    top = np.argsort(-sims)[:k]
    meta = track_meta([int(ids[i]) for i in top])
    out = []
    for i in top:
        m = meta.get(int(ids[i]), {})
        out.append({"id": int(ids[i]), "title": m.get("title"), "artist": m.get("artist"),
                    "genre": m.get("genre"), "preview_url": m.get("preview_url"),
                    "match": round(float(sims[i]), 3)})
    return {"available": True, "query": query, "results": out}
