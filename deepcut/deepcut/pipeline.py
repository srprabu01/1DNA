"""The three passes that fill the library: ingest, analyse, embed.

Each is resumable and independent. ``ingest`` records what exists; ``analyse``
measures it on the CPU; ``embed`` runs the frozen encoders on the GPU. A pass
never assumes the one before it finished cleanly — a track missing its measured
features still embeds against fixed excerpt offsets, a track that fails to decode
is counted and skipped rather than killing the run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import decode
from .config import ANALYSIS_SR

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus",
              ".wma", ".aif", ".aiff", ".alac", ".ape", ".wv", ".mp4"}


def _noop(i: int, n: int, label: str) -> None:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _collect_audio(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        candidates = [p] if p.is_file() else (
            sorted(q for q in p.rglob("*") if q.suffix.lower() in AUDIO_EXTS)
            if p.is_dir() else [])
        for q in candidates:
            if q.suffix.lower() not in AUDIO_EXTS:
                continue
            key = str(q.resolve())
            if key not in seen:
                seen.add(key)
                files.append(q)
    return files


def ingest(cfg, lib, paths, online: bool = False, progress=None) -> dict:
    """Hash, probe and tag every audio file under ``paths`` into the library."""
    progress = progress or _noop
    files = _collect_audio([Path(p) for p in paths])
    n = len(files)
    ingested = failed = 0
    identity = _load_identity() if online else None

    for i, fp in enumerate(files, 1):
        try:
            digest = decode.content_digest(fp)
            info = decode.probe(fp)
            tags = decode.read_tags(fp)
            if identity is not None:
                try:
                    tags = {**tags, **identity(fp, info)}
                except Exception:
                    pass
            lib.upsert_track(digest, fp, info, tags)
            ingested += 1
        except Exception:
            failed += 1
        progress(i, n, fp.name)

    return {"scanned": n, "ingested": ingested, "failed": failed,
            "online": bool(identity is not None)}


def _load_identity():
    """AcoustID/MusicBrainz enrichment lives in the optional ``identity`` module.

    It is not part of the standalone reconstruction; ``--online`` degrades to a
    no-op enricher when the module is absent rather than failing ingest."""
    try:
        from . import identity  # type: ignore
    except Exception:
        return None
    fn = getattr(identity, "enrich", None)
    return fn if callable(fn) else None


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

def analyse(cfg, lib, tracks, force: bool = False, progress=None) -> dict:
    """Measured features + structural segmentation + synthetic-audio prior."""
    from . import dsp, slop, structure

    progress = progress or _noop
    n = len(tracks)
    analysed = failed = 0

    for i, track in enumerate(tracks, 1):
        try:
            master, sr = decode.load_master(
                Path(track.path), cfg.paths.cache, digest=track.digest)
            if master.size < sr:  # under a second decoded; nothing to measure
                raise ValueError("decoded audio too short")

            stereo = decode.load_stereo(Path(track.path))
            feats = dsp.analyse(master, sr, stereo=stereo)

            y_a = decode.resample_poly(master, sr, ANALYSIS_SR) if sr != ANALYSIS_SR else master
            sections = structure.segment(y_a, sr=ANALYSIS_SR)
            feats.section_count = len(sections)
            feats.mean_section_s = (
                float(np.mean([s.duration for s in sections])) if sections else 0.0)
            feats.repetition_index = structure.repetition_index(sections)
            feats.sections = [s.to_dict() for s in sections]

            excerpts = structure.choose_excerpts(feats.duration_s, sections, cfg.excerpts)
            fdict = feats.to_dict()
            fdict["_excerpts"] = [float(e.start) for e in excerpts]
            fdict["_excerpt_reasons"] = [e.reason for e in excerpts]
            lib.put_features(track.id, fdict)

            verdict = slop.score(
                feats, y_a, ANALYSIS_SR,
                lossless_source=bool(getattr(track, "lossless", False)),
                source_bitrate=getattr(track, "bitrate", None),
                model_dir=cfg.paths.models,
            )
            lib.put_slop(track.id, verdict)
            lib.mark(track.id, "analyse")
            analysed += 1
        except Exception:
            failed += 1
        progress(i, n, getattr(track, "label", str(track.id)))

    return {"analysed": analysed, "failed": failed}


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed(cfg, lib, tracks, encoder, tag: bool = True,
          force: bool = False, progress=None) -> dict:
    """Frozen MERT + CLAP over each track's chosen excerpts, fused and stored."""
    progress = progress or _noop
    n = len(tracks)
    embedded = failed = 0
    tag_bank = _build_tag_bank(encoder) if tag else None

    # Vectors are staged in memory and only durable after ``flush``. Mark a track
    # embedded strictly AFTER its vector has been flushed, in bounded batches, so
    # a killed run (this environment stops long jobs) re-embeds cleanly instead
    # of marking tracks whose vectors never reached disk.
    pending_mark: list[int] = []

    def _commit_batch() -> None:
        if not pending_mark:
            return
        lib.vectors.flush()
        for tid in pending_mark:
            lib.mark(tid, "embed")
        pending_mark.clear()

    for i, track in enumerate(tracks, 1):
        try:
            feats = lib.features(track.id) or {}
            starts = feats.get("_excerpts") or _fallback_starts(cfg, feats, track)

            master, sr = decode.load_master(
                Path(track.path), cfg.paths.cache, digest=track.digest)
            emb = encoder.encode_track(master, sr, starts, cfg.excerpts.seconds)

            lib.vectors.append(lib.db, "mert", track.id, emb.mert)
            lib.vectors.append(lib.db, "clap", track.id, emb.clap)

            if tag_bank is not None:
                tags = _tag_from_clap(emb.clap, tag_bank)
                if tags:
                    lib.put_tags(track.id, tags, source="clap-zeroshot")

            pending_mark.append(track.id)
            embedded += 1
            if len(pending_mark) >= 128:
                _commit_batch()
        except Exception:
            failed += 1
        progress(i, n, getattr(track, "label", str(track.id)))

    _commit_batch()
    return {"embedded": embedded, "failed": failed}


def _fallback_starts(cfg, feats: dict, track) -> list[float]:
    dur = float(feats.get("duration_s") or getattr(track, "duration_s", None) or 180.0)
    pol = cfg.excerpts
    win = pol.seconds
    return [max(0.0, min(dur * p - win / 2, max(dur - win, 0.0)))
            for p in pol.fallback_positions[: pol.count]]


# -- zero-shot tagging -------------------------------------------------------

_TAG_VOCAB = [
    "acoustic", "electronic", "ambient", "aggressive", "atmospheric", "bass-heavy",
    "bright", "danceable", "dark", "distorted", "dreamy", "energetic", "epic",
    "funky", "groovy", "hypnotic", "lo-fi", "mellow", "melancholic", "minimal",
    "orchestral", "percussive", "psychedelic", "punchy", "raw", "romantic",
    "sad", "soulful", "spacious", "uplifting", "warm", "vocal-driven",
    "instrumental", "acoustic guitar", "synth-heavy", "piano-led", "cinematic",
    "reverb-drenched", "gritty", "smooth",
]


def _build_tag_bank(encoder):
    """CLAP text vectors for the vocabulary, computed once per embed run."""
    try:
        from .retrieve import _prompt_template

        prompts = [_prompt_template(t) for t in _TAG_VOCAB]
        vecs = np.asarray(encoder.clap.encode_text(prompts), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        return vecs, list(_TAG_VOCAB)
    except Exception:
        return None


def _tag_from_clap(clap_vec: np.ndarray, tag_bank, k: int = 8,
                   floor: float = 0.05) -> dict[str, float]:
    vecs, names = tag_bank
    q = np.asarray(clap_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = vecs @ q
    order = np.argsort(-sims)[:k]
    return {names[i]: float(sims[i]) for i in order if sims[i] > floor}
