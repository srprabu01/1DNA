# deepcut

A local music analysis and retrieval engine. It reads your audio files, measures
them, embeds them with frozen foundation models, and lets you retrieve from that
by example, by sentence, by tempo, or by a model of your own taste that you
train with a couple of hundred clicks.

It talks to no streaming platform, so no platform can switch it off.

```
deepcut ingest ~/Music        # hash, tag, identify
deepcut analyse               # measure: tempo, key, loudness, structure
deepcut embed                 # frozen MERT + CLAP, three excerpts per track
deepcut fit                   # remove popularity, anisotropy, hubs from the space
deepcut prompt "murky guitar for a rainy Sunday"
```

---

## Why it is built this way

Anything keyed on a platform's track IDs and a platform's feature endpoint is
one product decision away from being a museum piece. Every identifier here is
open — a content hash, a Chromaprint fingerprint, an AcoustID, a MusicBrainz ID,
an ISRC — and every feature is either measured from the audio or produced by a
model whose weights are on your disk.

The second decision that shapes everything: **nothing is fine-tuned**. The two
encoders are frozen feature extractors, which means every behaviour on top of
them is a linear model over cached vectors. A taste probe fits in under a
second. You can throw it away and refit it after changing your mind. You never
re-embed the library.

## The six layers

| Layer | What it is | Where |
|---|---|---|
| Identity | content hash, Chromaprint/AcoustID, MBID, ISRC | `identity.py`, `decode.py` |
| Signal | tempo, key→Camelot, LUFS/LRA/true peak, microtiming, spectrum, structure | `dsp.py`, `structure.py` |
| Semantics | frozen MERT-95M and CLAP-music, three excerpts, mean/max/L2 pooling | `embed.py` |
| Tags | zero-shot CLAP vocabulary; optional Essentia Discogs heads | `tags.py` |
| Index | memory-mapped fp16 matrices, exact cosine, corrected geometry | `store.py`, `debias.py` |
| Retrieval | blended recall → MMR → exploration → sequencing | `retrieve.py` |

Each is independently testable and independently replaceable. Swap MERT-330M in
by editing one line of `deepcut.toml`; nothing else changes.

## Quickstart

Requires Python 3.11+, `ffmpeg` on PATH, and a GPU for the embedding pass
(CPU works, slowly).

```bash
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers librosa pyloudnorm

deepcut init
deepcut doctor          # confirms ffmpeg, torch, CUDA, VRAM
deepcut ingest ~/Music
deepcut analyse
deepcut embed
deepcut fit
```

On an 8 GB card, MERT-95M and CLAP together fit comfortably in fp16 and run at a
few hundred tracks per minute once decoding is cached. The `analyse` pass is
CPU-bound and slower per track than the GPU pass — run it overnight the first
time. Both are resumable: re-running picks up where it stopped.

### Then

```bash
deepcut similar "in rainbows" --explain --harmonic
deepcut prompt "cavernous reverb, brushed drums, no vocals" --sequence
deepcut run --spm 172 --minutes 40 --half-time --out run.m3u8
deepcut report 412                      # HTML readout for one track
deepcut eval                            # is any of this working?
```

To build a taste axis:

```bash
deepcut label --axis late-night --suggest 20     # asks about its most uncertain tracks
deepcut label --axis late-night --id 412 --value 1
deepcut taste fit --axis late-night
```

Around 200 labels is where an axis usually becomes useful. The suggester picks
tracks by uncertainty and then spreads them across the space, so you spend your
clicks where they change the model.

---

## Decisions that differ from the conventional recipe

**48 kHz masters, not 24 kHz.** CLAP's front end wants 48 kHz and upsampling a
24 kHz cache throws away the band it uses. MERT's 24 kHz input is a clean 2:1
decimation from the master, so one cached file serves both.

**Excerpts follow the structure, not the clock.** Fixed offsets (25%, 50%, 70%)
are a guess about where the interesting part of a track is. Segmentation finds
the repeated, high-energy sections — the parts a person would hum — and puts the
windows there, preferring one window on a different section type so the fused
vector covers the verse as well as the chorus. Fixed offsets remain the
fallback. This is the single change with the largest effect on retrieval
quality, and `deepcut eval` will show it to you.

**Two spaces, blended at score time, never concatenated.** MERT hears how a
record sounds and is played; CLAP hears how it would be described. Kept
separate, a query can lean on either, and their *disagreement* becomes an
explanation: high MERT with low CLAP means it sounds alike but sits in a
different scene — a cover, a genre crossing, a producer's fingerprint.

**Three corrections before any retrieval.** Popularity leaks into the geometry;
frozen transformer spaces are anisotropic; high-dimensional spaces have hubs
that appear in every neighbour list. `deepcut fit` removes the popularity
direction, strips the top principal components, and applies CSLS scaling. It
prints the popularity R² before and after — if it starts above ~0.15, your
"similar tracks" list was substantially a popularity chart.

**No vector database.** Exact cosine over a memory-mapped fp16 matrix lands in
tens of milliseconds at 100k tracks. That is faster than an HNSW build
amortises, it is exact rather than approximate, and there is no server to keep
alive. The ANN path exists behind the same interface for when a library
genuinely outgrows it.

**The synthetic-audio filter shows its evidence.** A bare score is useless and
slightly dangerous, because the signals that mark a generated track — rigid
grid, loop repetition, narrow dynamics — also mark perfectly good sequenced
electronic music. So the detector returns per-signal contributions, weights the
production-artefact signals (phase incoherence, spectral checkerboarding, an
unexplained HF brick wall) far above the rhythmic ones, and can be refit on your
own labels with `deepcut slop`.

**The eval harness exists before the tuning does.** `deepcut eval` reports
accuracy, coverage, novelty and diversity together and refuses to collapse them
into one number. `--ablate` runs the same harness with individual mechanisms
switched off, so when you tighten one knob you can see what you paid for it.

Day one, with no listening history, run `deepcut eval` anyway: the sanity check
needs no labels. It measures how much more often a track's nearest neighbour is
by the same artist than chance allows. If that lift is under ~3×, something
upstream is broken — usually the excerpt windows — and no amount of ranking will
rescue it.

---

## What it does not do

- **No streaming catalogue.** It works on files you have. Cross-catalogue
  retrieval over Bandcamp or YouTube is a natural extension and is not built.
- **No stem separation.** Worth adding only if you want instrument-specific
  queries; MERT already carries most of what stems would tell you.
- **No fingerprint matching of its own.** AcoustID does that, free and better.
- **No watermark reading.** If a generator embeds one, that is stronger
  evidence than any heuristic in `slop.py` and should take priority.
- **No player.** `--out` writes m3u8; use whatever you already use.

## Honest limits

The synthetic-audio prior is uncalibrated until you label a few hundred tracks.
Treat it as a sort order, not a filter, until then.

Next-track evaluation over your own history is a weak proxy for taste — album
order, shuffle and session bias all leak into it. It moves when the system gets
better, which is what makes it useful, but do not read the absolute numbers as
accuracy.

Key detection is Krumhansl-Schmuckler over chroma, which is solid on tonal
material and unreliable on anything modal, atonal, or heavily processed. The
reported confidence is the margin over the runner-up hypothesis, so low
confidence genuinely means ambiguous.

The zero-shot tagger is below a supervised tagger's accuracy. Its advantage is
that the vocabulary is yours to edit — add a phrase to `tags.py` and it works
immediately.

## Tests

```bash
python tests/test_core.py
```

Covers the Camelot arithmetic, key detection, pooling and fusion, excerpt
selection, the space corrections, and the full retrieval path over a synthetic
library where artists are known clusters — no audio files or model weights
needed. The DSP and encoder paths need real input; `deepcut eval --ablate` on a
real library is their test.

## Layout

```
deepcut/
  config.py      paths, excerpt policy, model choices, retrieval knobs
  decode.py      ffmpeg, content hashing, the 48 kHz master cache
  identity.py    Chromaprint, AcoustID, MusicBrainz
  dsp.py         measured features
  structure.py   segmentation and excerpt selection
  slop.py        synthetic-audio prior, with evidence
  embed.py       frozen MERT + CLAP
  tags.py        zero-shot and supervised tagging
  store.py       SQLite + memory-mapped vectors
  debias.py      popularity, anisotropy, hubness
  retrieve.py    two-stage retrieval and sequencing
  explain.py     per-space, per-layer and per-feature attribution
  taste.py       linear probes and active learning
  evaluate.py    the harness
  report.py      single-track HTML readout
  pipeline.py    ingest / analyse / embed
  cli.py         command line
```
