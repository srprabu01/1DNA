"""Tests for the parts that need neither audio files nor model weights.

The DSP and encoder paths need real input and are covered by `deepcut eval
--ablate` on a real library. Everything else — the wheel arithmetic, pooling,
the corrections, and the whole retrieval path over a synthetic library — runs
here in under a second.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepcut.config import ExcerptPolicy, Paths, RetrieveConfig  # noqa: E402
from deepcut.debias import SpaceCorrection, popularity_leakage  # noqa: E402
from deepcut.dsp import camelot_distance, estimate_key, to_camelot  # noqa: E402
from deepcut.embed import fuse_excerpts, l2norm, pool_time  # noqa: E402
from deepcut.retrieve import Constraints, Engine, cadence_query, sequence  # noqa: E402
from deepcut.store import Library, cosine_search  # noqa: E402
from deepcut.structure import Section, choose_excerpts, repetition_index  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name} {detail}")


# ---------------------------------------------------------------------------

def test_camelot() -> None:
    check("C major is 8B", to_camelot("C", "major") == "8B")
    check("A minor is 8A", to_camelot("A", "minor") == "8A")
    check("relative keys are one step apart", camelot_distance("8B", "8A") == 1)
    check("fifths are one step apart", camelot_distance("8B", "9B") == 1)
    check("tritone is the far side", camelot_distance("1A", "7A") == 6)
    check("wraps around the wheel", camelot_distance("12B", "1B") == 1)
    check("unknown keys are far", camelot_distance("", "8A") == 99)


def test_key_detection() -> None:
    # A synthetic C-major chroma: strong tonic, dominant and mediant.
    chroma = np.full((12, 50), 0.05)
    for pc, weight in ((0, 1.0), (4, 0.7), (7, 0.85), (2, 0.4), (5, 0.4), (9, 0.4), (11, 0.3)):
        chroma[pc] = weight
    key, scale, conf = estimate_key(chroma)
    check("detects C major from a diatonic chroma", (key, scale) == ("C", "major"),
          f"got {key} {scale}")
    check("reports a confidence in range", 0.0 <= conf <= 1.0)


def test_pooling() -> None:
    x = np.random.default_rng(0).normal(size=(40, 16)).astype(np.float32)
    pooled = pool_time(x)
    check("pooling triples the dimension", pooled.shape == (48,), str(pooled.shape))
    check("mean block matches", np.allclose(pooled[:16], x.mean(axis=0), atol=1e-5))
    check("max block matches", np.allclose(pooled[16:32], x.max(axis=0), atol=1e-5))

    # Normalise-then-average: a loud excerpt must not dominate the fusion.
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32) * 50.0
    fused = fuse_excerpts(np.stack([a, b]))
    check("fusion is scale invariant", abs(fused[0] - fused[1]) < 1e-5,
          f"got {fused}")
    check("fusion returns a unit vector", abs(np.linalg.norm(fused) - 1) < 1e-5)


def test_excerpts() -> None:
    policy = ExcerptPolicy()
    picks = choose_excerpts(240.0, [], policy)
    check("falls back to three windows", len(picks) == 3)
    check("fallback windows are ordered", all(a.start <= b.start for a, b in zip(picks, picks[1:])))
    check("fallback skips the intro", picks[0].start >= policy.skip_head_s - 1e-6)

    sections = [Section(0, 30, 0, 0.2), Section(30, 60, 1, 0.9), Section(60, 90, 0, 0.25),
                Section(90, 120, 1, 0.95), Section(120, 150, 2, 0.5)]
    picks = choose_excerpts(150.0, sections, policy)
    check("uses structure when available", any(p.reason.startswith("section") for p in picks))
    check("keeps windows apart", all(b.start - a.start >= policy.min_gap_s
                                     for a, b in zip(picks, picks[1:])))
    check("repetition index sees the repeats",
          0.5 < repetition_index(sections) <= 1.0, f"{repetition_index(sections):.2f}")


def test_corrections() -> None:
    rng = np.random.default_rng(1)
    n, d = 400, 32
    base = rng.normal(size=(n, d)).astype(np.float32)
    pop = rng.integers(0, 200, size=n).astype(np.float32)
    # Plant popularity along one axis, exactly the leak the correction targets.
    direction = np.zeros(d, dtype=np.float32)
    direction[3] = 1.0
    X = l2norm(base + np.outer(np.log1p(pop), direction) * 0.9)

    before = popularity_leakage(X, pop)
    corr = SpaceCorrection.fit(X, pop, n_components=1)
    after = popularity_leakage(corr.apply(X), pop)
    check("popularity is measurable before correction", before > 0.25, f"{before:.3f}")
    check("correction removes most of it", after < before * 0.6, f"{before:.3f} -> {after:.3f}")
    check("corrected rows stay unit length",
          np.allclose(np.linalg.norm(corr.apply(X), axis=1), 1.0, atol=1e-4))
    check("hubness radius is computed", corr.csls_radius is not None
          and corr.csls_radius.shape == (n,))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.npz"
        corr.save(path)
        again = SpaceCorrection.load(path)
        check("correction round-trips", again is not None
              and np.allclose(again.apply(X), corr.apply(X), atol=1e-5))


def test_cosine_search() -> None:
    rng = np.random.default_rng(2)
    M = l2norm(rng.normal(size=(500, 24)).astype(np.float32))
    q = M[17]
    rows, scores = cosine_search(M, q, 5)
    check("finds itself first", rows[0] == 17)
    check("scores descend", all(a >= b for a, b in zip(scores, scores[1:])))


# ---------------------------------------------------------------------------

@dataclass
class FakeInfo:
    duration_s: float
    sample_rate: int = 48000
    channels: int = 2
    codec: str = "flac"
    bit_rate: int | None = 900_000
    lossless: bool = True
    tags: dict | None = None


def build_library(root: Path, n_artists: int = 12, per_artist: int = 8) -> Library:
    """A synthetic library where artists really are clusters in the space."""
    lib = Library(Paths(root=root))
    rng = np.random.default_rng(7)
    dim = 48
    centres = l2norm(rng.normal(size=(n_artists, dim)).astype(np.float32))

    tid = 0
    for a in range(n_artists):
        for t in range(per_artist):
            tid += 1
            tags = {"title": f"Track {t + 1}", "artist": f"Artist {a}",
                    "album": f"Album {a}", "date": "2020"}
            info = FakeInfo(duration_s=180 + (tid % 7) * 15, tags=tags)
            track_id = lib.upsert_track(f"digest-{tid}", Path(f"/music/a{a}/t{t}.flac"),
                                        info, tags)
            vec = l2norm(centres[a] + rng.normal(scale=0.10, size=dim).astype(np.float32))
            lib.vectors.append(lib.db, "mert", track_id, vec)
            lib.vectors.append(lib.db, "clap", track_id,
                               l2norm(centres[a] + rng.normal(scale=0.16, size=dim).astype(np.float32)))
            lib.put_features(track_id, {
                "duration_s": float(info.duration_s),
                "bpm": 90.0 + (a * 7 + t * 3) % 70,
                "camelot": f"{1 + (a + t) % 12}{'A' if t % 2 else 'B'}",
                "integrated_lufs": -16.0 + (t % 5),
                "spectral_centroid_hz": 1500 + a * 60,
                "harmonic_ratio": 0.4 + 0.03 * (t % 5),
                "microtiming_ms": 2.0 + t,
            })
    lib.db.commit()
    lib.vectors.flush()
    return lib


def test_retrieval() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lib"
        lib = build_library(root)
        cfg = RetrieveConfig(candidates=200, results=12, epsilon=0.0,
                             debias_popularity=False)
        engine = Engine(lib, cfg, root / "models")

        seed = lib.all_tracks()[0]
        queries = {s: engine.vector_for_tracks(s, [seed.id]) for s in ("mert", "clap")}
        results = engine.search(queries, Constraints(exclude_track_ids={seed.id},
                                                     min_duration_s=0.0), k=12)
        check("search returns results", len(results) > 0)
        check("seed is excluded", all(c.track.id != seed.id for c in results))
        check("artist cap is respected",
              max(sum(1 for c in results if c.track.artist == a)
                  for a in {c.track.artist for c in results}) <= 2)
        check("same artist ranks first",
              results[0].track.artist == seed.artist,
              f"got {results[0].track.artist} for seed {seed.artist}")

        # MMR at lambda=1 is pure relevance and must be at least as similar.
        engine.cfg = RetrieveConfig(candidates=200, results=12, epsilon=0.0,
                                    mmr_lambda=1.0, artist_cap=99,
                                    debias_popularity=False)
        engine._spaces.clear()
        greedy = engine.search(queries, Constraints(exclude_track_ids={seed.id},
                                                    min_duration_s=0.0, artist_cap=99), k=12)
        engine.cfg = RetrieveConfig(candidates=200, results=12, epsilon=0.0,
                                    mmr_lambda=0.2, artist_cap=99,
                                    debias_popularity=False)
        engine._spaces.clear()
        diverse = engine.search(queries, Constraints(exclude_track_ids={seed.id},
                                                     min_duration_s=0.0, artist_cap=99), k=12)
        from deepcut.evaluate import _intra_list_diversity
        d_greedy = _intra_list_diversity(engine, [c.track.id for c in greedy])
        d_diverse = _intra_list_diversity(engine, [c.track.id for c in diverse])
        check("lower mmr_lambda buys diversity", d_diverse > d_greedy,
              f"{d_greedy:.3f} vs {d_diverse:.3f}")

        # Tempo constraint
        engine.cfg = cfg
        engine._spaces.clear()
        lo, hi = cadence_query(160, tolerance=0.05)
        constrained = engine.search(queries, Constraints(bpm_range=(lo, hi),
                                                         allow_half_double=False,
                                                         min_duration_s=0.0,
                                                         artist_cap=99), k=20)
        check("tempo window is enforced",
              all(lo <= c.bpm <= hi for c in constrained),
              str([round(c.bpm) for c in constrained]))

        ordered = sequence(results, energy_arc="rise")
        check("sequencing preserves the set",
              {c.track.id for c in ordered} == {c.track.id for c in results})
        jumps_before = _mean_jump(results)
        jumps_after = _mean_jump(ordered)
        check("sequencing smooths tempo transitions", jumps_after <= jumps_before + 1e-6,
              f"{jumps_before:.1f} -> {jumps_after:.1f}")

        from deepcut.evaluate import sanity
        s = sanity(engine)
        check("sanity check sees the artist clusters", s["lift"] > 3.0, f"lift {s['lift']}")
        lib.close()


def test_taste_probe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lib"
        lib = build_library(root)
        # Label two artists positive, two negative: a linearly separable axis.
        for track in lib.all_tracks():
            if track.artist in {"Artist 0", "Artist 1"}:
                lib.set_label(track.id, "like", 1.0)
            elif track.artist in {"Artist 5", "Artist 6"}:
                lib.set_label(track.id, "like", 0.0)
        from deepcut.taste import fit
        head = fit(lib, "like", space="mert", use_implicit=False)
        check("probe learns a separable axis", head.cv_score > 0.8, f"AUC {head.cv_score:.3f}")
        check("probe reports its label count", head.n_labels == 32, str(head.n_labels))

        head.save(root / "models")
        from deepcut.taste import TasteHead
        again = TasteHead.load(root / "models", "like")
        check("probe round-trips", again is not None
              and np.allclose(again.weights, head.weights))

        from deepcut.taste import suggest_labels
        picks = suggest_labels(lib, "like", root / "models", n=5)
        labelled = set(lib.labels("like"))
        check("suggestions avoid already-labelled tracks",
              all(p not in labelled for p in picks))
        lib.close()


def _mean_jump(results) -> float:
    bpms = [c.bpm for c in results]
    return float(np.mean([abs(a - b) for a, b in zip(bpms, bpms[1:])])) if len(bpms) > 1 else 0.0


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}")
            fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
