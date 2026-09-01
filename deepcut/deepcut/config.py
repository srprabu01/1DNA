"""Paths, excerpt policy, model choices, retrieval knobs — the whole config surface.

Everything tunable lives here or in a ``deepcut.toml`` beside the library. The
dataclasses carry working defaults, so ``Config.load(None)`` produces a usable
configuration with no file on disk; a TOML file overrides field by field.

Three sample rates, and the reason they differ:

* ``MASTER_SR`` (48 kHz) is the cached master. CLAP's front end wants 48 kHz and
  upsampling a lower cache throws away the band it uses.
* ``MERT_SR`` (24 kHz) is a clean 2:1 decimation of the master, so one cached
  file feeds both encoders.
* ``ANALYSIS_SR`` (22.05 kHz) is where the measured features run. Tempo, key and
  loudness need nothing above it, and halving the rate roughly halves librosa's
  cost on the CPU-bound analysis pass.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

MASTER_SR = 48_000
MERT_SR = 24_000
ANALYSIS_SR = 22_050


def resolve_device(device: str | None = "auto") -> str:
    """``'auto'`` picks CUDA, then Apple MPS, then CPU. Explicit choices pass through."""
    choice = (device or "auto").strip().lower()
    if choice in ("cuda", "cpu", "mps"):
        return choice
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Sections of the config
# ---------------------------------------------------------------------------

@dataclass
class ExcerptPolicy:
    """Which windows the encoders see. Consumed by ``structure.choose_excerpts``."""

    count: int = 3
    seconds: float = 10.0
    structure_aware: bool = True
    skip_head_s: float = 20.0
    skip_tail_s: float = 15.0
    min_gap_s: float = 8.0
    fallback_positions: tuple[float, ...] = (0.25, 0.50, 0.70)


@dataclass
class EmbedConfig:
    """The frozen-encoder pass. Consumed by ``embed.MertEncoder`` / ``ClapEncoder``."""

    device: str = "auto"
    fp16: bool = True
    batch_size: int = 8
    mert_model: str = "m-a-p/MERT-v1-95M"
    clap_model: str = "laion/larger_clap_music"
    # MERT's middle layers carry timbre/articulation; the last layer alone is
    # closer to the pre-training objective and a worse similarity feature.
    mert_layers: tuple[int, ...] = (5, 6, 7)
    keep_layer_stack: bool = False


@dataclass
class RetrieveConfig:
    """Two-stage retrieval knobs. Consumed by ``retrieve.Engine``."""

    candidates: int = 500
    results: int = 25
    mmr_lambda: float = 0.72
    epsilon: float = 0.12
    artist_cap: int = 2
    weight_mert: float = 0.6
    weight_clap: float = 0.4
    slop_threshold: float = 0.65
    debias_popularity: bool = True


@dataclass
class Paths:
    """Everything on disk hangs off one library root."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    @property
    def db(self) -> Path:
        return self.root / "deepcut.db"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def cache(self) -> Path:
        """48 kHz decoded masters, keyed by content digest."""
        return self.root / "cache"

    def ensure(self) -> "Paths":
        for p in (self.root, self.vectors, self.models, self.reports, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Config:
    paths: Paths
    excerpts: ExcerptPolicy = field(default_factory=ExcerptPolicy)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)

    # -- construction -----------------------------------------------------
    @classmethod
    def default(cls, root: str | Path | None = None) -> "Config":
        return cls(paths=Paths(root=Path(root) if root else _default_root()))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Read ``deepcut.toml`` if one is found; otherwise return defaults.

        Never raises on a missing or malformed file — a broken TOML degrades to
        defaults so ``deepcut init`` can still write a fresh one.
        """
        toml_path = _find_config(path)
        data = _read_toml(toml_path) if toml_path else {}

        root = data.get("library") or os.environ.get("DEEPCUT_HOME") or _default_root()
        cfg = cls(paths=Paths(root=Path(root)))
        if isinstance(data.get("excerpts"), dict):
            _fill(cfg.excerpts, data["excerpts"])
        if isinstance(data.get("embed"), dict):
            _fill(cfg.embed, data["embed"])
        if isinstance(data.get("retrieve"), dict):
            _fill(cfg.retrieve, data["retrieve"])
        return cfg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _default_root() -> Path:
    env = os.environ.get("DEEPCUT_HOME")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "library"


def _find_config(path: str | Path | None) -> Path | None:
    if path:
        p = Path(path).expanduser()
        return p if p.exists() else None
    env = os.environ.get("DEEPCUT_CONFIG")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()
    local = Path.cwd() / "deepcut.toml"
    return local if local.exists() else None


def _read_toml(path: Path) -> dict:
    try:
        import tomllib

        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except Exception:
        # A Windows path written with single backslashes is invalid TOML and
        # trips the strict parser; recover at least the library root by hand.
        return _read_toml_fallback(path)


def _read_toml_fallback(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    m = re.search(r'(?m)^\s*library\s*=\s*"([^"]+)"', text)
    return {"library": m.group(1)} if m else {}


def _fill(obj, values: dict) -> None:
    """Overlay a TOML table onto a dataclass instance, field by field."""
    valid = {f.name: f for f in fields(obj)}
    for key, val in values.items():
        if key not in valid:
            continue
        current = getattr(obj, key)
        if isinstance(current, tuple) and isinstance(val, list):
            val = tuple(val)
        setattr(obj, key, val)
