"""Full processing pipeline, standalone (survives web-server restarts).

Enrich (Deezer) -> analyze (parallel librosa DSP across all cores) -> lyrics.

The `if __name__ == "__main__"` guard is MANDATORY: the analysis stage spawns a
process pool, and under Windows 'spawn' every child re-imports this module — an
unguarded body would re-run the whole pipeline in each child (a fork bomb).
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")


def main():
    from app.audio.pipeline import analyze_batch_parallel
    from app.db import init_db
    from app.enrich.deezer import enrich_batch
    from app.lyrics import engine as ly

    init_db()  # ensures WAL + schema

    print("=== ENRICH (Deezer match) ===", flush=True)
    while True:
        r = enrich_batch(limit=60)
        print(f"  matched {r['matched']} failed {r['failed']} — {r['remaining']} left", flush=True)
        if r["remaining"] == 0:
            break

    print("=== ANALYZE (parallel librosa, most-played first) ===", flush=True)
    r = analyze_batch_parallel(retry_errors=False)  # one pool, all cores, crash-isolated
    print(f"  analyzed {r['analyzed']}, errors {r['errors']} — {r['remaining']} left", flush=True)

    print("=== LYRICS (top tracks) ===", flush=True)
    for _ in range(8):
        r = ly.fetch_batch(limit=25)
        print(f"  lyrics +{r['analyzed']} ({r['missing']} missing) — {r['remaining']} left", flush=True)
        if r["remaining"] == 0:
            break
    print("=== PIPELINE COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
