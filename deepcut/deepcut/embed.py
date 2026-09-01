"""The learned layer: two frozen encoders, never fine-tuned.

Nothing in this file trains. The encoders are treated as fixed feature
extractors, which is what makes the rest of the system cheap: every downstream
behaviour — taste probes, slop calibration, genre heads — is a linear model on
top of vectors you computed once. Re-training a probe takes seconds; you never
re-embed the library.

Two spaces, kept separate on purpose:

* **MERT** hears the music. Masked acoustic modelling over 24 kHz audio, and
  its middle layers carry timbre, articulation, and performance detail.
* **CLAP** hears the *description*. Audio and text share one latent space, so
  the text tower turns a sentence into a query vector with no tagging, no
  labels, and no LLM in the loop.

Blending them at score time (rather than concatenating) means a query can lean
on either: "sounds like this" is MERT-weighted, "feels like a rainy Sunday" is
CLAP-weighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MASTER_SR, MERT_SR, EmbedConfig, resolve_device


@dataclass
class TrackEmbedding:
    mert: np.ndarray                  # fused, L2-normalised: 3 * hidden dims
    clap: np.ndarray                  # fused, L2-normalised: 512
    mert_layers: np.ndarray | None    # (n_layers, hidden) time-mean stack, fp16
    excerpt_starts: list[float]


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for the semantic layer.\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool_time(x: np.ndarray) -> np.ndarray:
    """Mean, max and L2-norm along time, concatenated.

    Mean captures what is present throughout, max captures the one moment that
    matters (a solo, a drop), and the L2 norm captures accumulated energy.
    Together they survive time-shift while keeping information that a bare mean
    throws away. This trick predates the current model generation and is still
    the best three-line summary of a sequence.

    ``x`` is (time, dim). Returns (3 * dim,).
    """
    if x.ndim != 2:
        raise ValueError(f"expected (time, dim), got {x.shape}")
    mean = x.mean(axis=0)
    mx = x.max(axis=0)
    l2 = np.sqrt((x ** 2).sum(axis=0) / max(x.shape[0], 1))
    return np.concatenate([mean, mx, l2]).astype(np.float32)


def l2norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


def fuse_excerpts(vectors: np.ndarray) -> np.ndarray:
    """Average per-excerpt vectors after normalising each one.

    Normalise-then-average, not average-then-normalise: otherwise a loud
    excerpt with a large norm dominates the track's representation.
    """
    if vectors.ndim == 1:
        return l2norm(vectors)
    return l2norm(l2norm(vectors, axis=1).mean(axis=0))


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class MertEncoder:
    """Frozen MERT. Consumes 24 kHz mono, emits per-layer hidden states."""

    def __init__(self, cfg: EmbedConfig):
        torch = _import_torch()
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = torch.float16 if (cfg.fp16 and self.device == "cuda") else torch.float32
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.mert_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            cfg.mert_model, trust_remote_code=True, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        self.sr = MERT_SR
        self._torch = torch

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size) * 3

    def encode(self, excerpts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (pooled per excerpt, per-layer time-means averaged over excerpts)."""
        torch = self._torch
        pooled: list[np.ndarray] = []
        layer_stacks: list[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in range(0, len(excerpts), bs):
            batch = excerpts[i:i + bs]
            inputs = self.processor(batch, sampling_rate=self.sr,
                                    return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device, self.dtype if v.is_floating_point() else None)
                      for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model(**inputs, output_hidden_states=True)
            # (layers, batch, time, dim)
            hidden = torch.stack(out.hidden_states).float().cpu().numpy()
            for b in range(hidden.shape[1]):
                layers = hidden[:, b]                        # (layers, time, dim)
                chosen = [layers[idx] for idx in self.cfg.mert_layers
                          if idx < layers.shape[0]] or [layers[-1]]
                pooled.append(pool_time(np.mean(chosen, axis=0)))
                layer_stacks.append(layers.mean(axis=1))     # (layers, dim)
        stack = np.mean(layer_stacks, axis=0).astype(np.float16) if layer_stacks else None
        return np.stack(pooled), stack


class ClapEncoder:
    """Frozen CLAP. The text tower is the whole point of including it."""

    def __init__(self, cfg: EmbedConfig):
        torch = _import_torch()
        from transformers import ClapModel, ClapProcessor

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = torch.float16 if (cfg.fp16 and self.device == "cuda") else torch.float32
        self.processor = ClapProcessor.from_pretrained(cfg.clap_model)
        self.model = ClapModel.from_pretrained(cfg.clap_model, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        self.sr = 48_000
        self._torch = torch

    def encode_audio(self, excerpts: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        vecs: list[np.ndarray] = []
        bs = self.cfg.batch_size
        for i in range(0, len(excerpts), bs):
            inputs = self.processor(audios=excerpts[i:i + bs], sampling_rate=self.sr,
                                    return_tensors="pt", padding=True)
            inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point()
                          else v.to(self.device)) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = self.model.get_audio_features(**inputs)
            vecs.append(feats.float().cpu().numpy())
        return np.concatenate(vecs)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            feats = self.model.get_text_features(**inputs)
        return l2norm(feats.float().cpu().numpy())


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

class SemanticEncoder:
    """Both towers, loaded lazily so a text-only query never touches MERT."""

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self._mert: MertEncoder | None = None
        self._clap: ClapEncoder | None = None

    @property
    def mert(self) -> MertEncoder:
        if self._mert is None:
            self._mert = MertEncoder(self.cfg)
        return self._mert

    @property
    def clap(self) -> ClapEncoder:
        if self._clap is None:
            self._clap = ClapEncoder(self.cfg)
        return self._clap

    def encode_track(self, master: np.ndarray, sr: int,
                     starts: list[float], seconds: float) -> TrackEmbedding:
        """Cut the chosen excerpts once, feed both towers, fuse."""
        from .decode import resample_poly

        if sr != MASTER_SR:
            master = resample_poly(master, sr, MASTER_SR)
            sr = MASTER_SR
        win = int(seconds * sr)
        cuts48: list[np.ndarray] = []
        for s in starts:
            a = int(max(0, min(s * sr, len(master) - win)))
            seg = master[a:a + win]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            cuts48.append(seg.astype(np.float32))
        cuts24 = [resample_poly(c, sr, MERT_SR) for c in cuts48]

        mert_pooled, layer_stack = self.mert.encode(cuts24)
        clap_vecs = self.clap.encode_audio(cuts48)
        return TrackEmbedding(
            mert=fuse_excerpts(mert_pooled),
            clap=fuse_excerpts(clap_vecs),
            mert_layers=layer_stack if self.cfg.keep_layer_stack else None,
            excerpt_starts=list(starts),
        )
