"""Build this before you tune anything.

Every knob in the retrieval path trades one metric against another, so a single
number will always be improvable and always be a lie. The harness reports four
families at once and refuses to summarise them into one score:

* **Accuracy** — leave-one-out next-track prediction over your own history.
* **Coverage** — how much of the library the system is willing to surface.
* **Novelty** — how obscure, on average, its picks are.
* **Diversity** — how unlike each other the picks within one set are.

Plus two unsupervised sanity checks that need no history at all, which is what
you run on day one when there is nothing to evaluate against yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retrieve import Constraints, Engine


@dataclass
class EvalReport:
    n_queries: int = 0
    hit_at_10: float = 0.0
    hit_at_50: float = 0.0
    mrr: float = 0.0
    coverage: float = 0.0
    novelty: float = 0.0
    intra_list_diversity: float = 0.0
    popularity_percentile: float = 0.0
    serendipity: float = 0.0
    notes: list[str] = field(default_factory=list)

    def table(self) -> str:
        rows = [
            ("queries", f"{self.n_queries}"),
            ("hit@10", f"{self.hit_at_10:.3f}"),
            ("hit@50", f"{self.hit_at_50:.3f}"),
            ("MRR", f"{self.mrr:.3f}"),
            ("coverage", f"{self.coverage:.3f}"),
            ("novelty (bits)", f"{self.novelty:.2f}"),
            ("intra-list diversity", f"{self.intra_list_diversity:.3f}"),
            ("popularity percentile", f"{self.popularity_percentile:.3f}"),
            ("serendipity", f"{self.serendipity:.3f}"),
        ]
        width = max(len(r[0]) for r in rows)
        return "\n".join(f"  {k.ljust(width)}  {v}" for k, v in rows)


def next_track(engine: Engine, k: int = 50, max_queries: int = 500,
               session_gap_s: float = 1800.0) -> EvalReport:
    """Given the previous track in a session, can we retrieve the next one?

    Sessions are split on a 30-minute gap. The query is the previous track's
    vector; the target is what actually played next. This is a weak proxy for
    taste — sequence bias, album order, and shuffle all leak into it — but it is
    the only ground truth available locally and it moves when the system gets
    better.
    """
    history = engine.lib.history()
    if len(history) < 20:
        return EvalReport(notes=["not enough play history; log plays with `deepcut play`"])

    pairs: list[tuple[int, int]] = []
    for (a_id, a_t), (b_id, b_t) in zip(history, history[1:]):
        if b_t - a_t <= session_gap_s and a_id != b_id:
            pairs.append((a_id, b_id))
    if not pairs:
        return EvalReport(notes=["no within-session transitions found"])
    if len(pairs) > max_queries:
        idx = np.random.default_rng(0).choice(len(pairs), max_queries, replace=False)
        pairs = [pairs[int(i)] for i in idx]

    plays = engine.lib.play_counts()
    total_plays = max(sum(plays.values()), 1)
    n_tracks = max(len(engine.lib.all_tracks()), 1)

    hits10 = hits50 = 0
    rr: list[float] = []
    surfaced: set[int] = set()
    novelty: list[float] = []
    ild: list[float] = []
    pop_pct: list[float] = []
    serendipitous = 0

    pop_sorted = np.sort(np.array([plays.get(t.id, 0) for t in engine.lib.all_tracks()],
                                  dtype=np.float32))

    for seed, target in pairs:
        queries = {}
        for space in ("mert", "clap"):
            v = engine.vector_for_tracks(space, [seed])
            if v is not None:
                queries[space] = v
        if not queries:
            continue
        results = engine.search(
            queries,
            Constraints(exclude_track_ids={seed}, artist_cap=99, min_duration_s=0.0),
            k=k,
        )
        ids = [c.track.id for c in results]
        surfaced.update(ids)
        if target in ids:
            rank = ids.index(target) + 1
            rr.append(1.0 / rank)
            hits50 += 1
            hits10 += int(rank <= 10)
            if plays.get(target, 0) <= 1:
                serendipitous += 1
        else:
            rr.append(0.0)
        for tid in ids:
            p = (plays.get(tid, 0) + 1) / (total_plays + n_tracks)
            novelty.append(-float(np.log2(p)))
            pop_pct.append(float(np.searchsorted(pop_sorted, plays.get(tid, 0)) / max(len(pop_sorted), 1)))
        ild.append(_intra_list_diversity(engine, ids))

    n = len(rr) or 1
    return EvalReport(
        n_queries=len(rr),
        hit_at_10=hits10 / n,
        hit_at_50=hits50 / n,
        mrr=float(np.mean(rr)) if rr else 0.0,
        coverage=len(surfaced) / n_tracks,
        novelty=float(np.mean(novelty)) if novelty else 0.0,
        intra_list_diversity=float(np.mean(ild)) if ild else 0.0,
        popularity_percentile=float(np.mean(pop_pct)) if pop_pct else 0.0,
        serendipity=serendipitous / n,
    )


def ablate(engine: Engine, k: int = 50) -> list[tuple[str, EvalReport]]:
    """Run the harness with individual mechanisms disabled.

    The point is not to find the best row. It is to see which knob moves which
    metric, so that when you later tighten one you know what you paid for it.
    """
    from dataclasses import replace
    base = engine.cfg
    variants = {
        "full": base,
        "no diversity rerank (mmr=1.0)": replace(base, mmr_lambda=1.0),
        "no exploration (eps=0)": replace(base, epsilon=0.0),
        "no popularity debias": replace(base, debias_popularity=False),
        "MERT only": replace(base, weight_clap=0.0),
        "CLAP only": replace(base, weight_mert=0.0),
    }
    out: list[tuple[str, EvalReport]] = []
    for name, cfg in variants.items():
        engine.cfg = cfg
        engine._spaces.clear()
        out.append((name, next_track(engine, k=k)))
    engine.cfg = base
    engine._spaces.clear()
    return out


# ---------------------------------------------------------------------------
# Unsupervised sanity checks — no listening history required
# ---------------------------------------------------------------------------

def sanity(engine: Engine, space: str = "mert", sample: int = 400) -> dict[str, float]:
    """Does the space encode anything real?

    Two cheap tests with known-correct answers:

    * **Artist consistency** — the nearest neighbour of a track should be by the
      same artist far more often than chance. If it is not, the embeddings, the
      excerpting, or the corrections are broken, and no amount of ranking will
      save it.
    * **Album cohesion** — tracks from one album should be measurably closer to
      each other than to random tracks. Weaker than the artist test but
      catches a space that has collapsed.
    """
    loaded = engine.lib.vectors.ids_and_matrix(engine.lib.db, space)
    if loaded is None:
        return {"error": float("nan")}
    ids, matrix = loaded
    matrix = matrix.astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9

    meta = {t.id: t for t in engine.lib.all_tracks()}
    rng = np.random.default_rng(0)
    n = min(sample, len(ids))
    rows = rng.choice(len(ids), n, replace=False)

    same_artist = 0
    counted = 0
    for r in rows:
        sims = matrix @ matrix[r]
        sims[r] = -np.inf
        nn = int(np.argmax(sims))
        a, b = meta.get(int(ids[r])), meta.get(int(ids[nn]))
        if a and b and a.artist and b.artist:
            counted += 1
            same_artist += int(a.artist.lower() == b.artist.lower())

    artist_counts: dict[str, int] = {}
    for t in meta.values():
        if t.artist:
            artist_counts[t.artist.lower()] = artist_counts.get(t.artist.lower(), 0) + 1
    total = max(sum(artist_counts.values()), 1)
    chance = sum((c / total) ** 2 for c in artist_counts.values())

    within, between = [], []
    by_album: dict[str, list[int]] = {}
    pos = {int(t): i for i, t in enumerate(ids)}
    for t in meta.values():
        if t.album and t.id in pos:
            by_album.setdefault(t.album, []).append(pos[t.id])
    albums = [v for v in by_album.values() if len(v) >= 3][:60]
    for rowset in albums:
        sub = matrix[rowset]
        sims = sub @ sub.T
        iu = np.triu_indices(len(rowset), 1)
        within.append(float(sims[iu].mean()))
        other = matrix[rng.choice(len(ids), min(64, len(ids)), replace=False)]
        between.append(float((sub @ other.T).mean()))

    return {
        "nn_same_artist": round(same_artist / max(counted, 1), 4),
        "chance_same_artist": round(chance, 4),
        "lift": round((same_artist / max(counted, 1)) / max(chance, 1e-6), 2),
        "album_cohesion": round(float(np.mean(within) - np.mean(between)), 4) if within else 0.0,
        "n_sampled": float(n),
    }


def _intra_list_diversity(engine: Engine, ids: list[int], space: str = "mert") -> float:
    if len(ids) < 2:
        return 0.0
    loaded = engine.space(space)
    if loaded is None:
        return 0.0
    all_ids, matrix, _ = loaded
    pos = {int(t): i for i, t in enumerate(all_ids)}
    rows = [pos[t] for t in ids if t in pos]
    if len(rows) < 2:
        return 0.0
    V = matrix[rows]
    sims = V @ V.T
    iu = np.triu_indices(len(rows), 1)
    return float(1.0 - sims[iu].mean())
