"""Per-track visualizations rendered server-side to PNG from the cached preview.

Types: 'spectrogram' (mel), 'chroma', 'waveform', 'tempogram'.
Requires the track to have been analyzed (preview mp3 in the cache).
"""
import io

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..config import CACHE_DIR

_BG = "#161b22"
_FG = "#e6edf3"


def _style(ax):
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_color("#2d333b")
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


def render(track_id, kind="spectrogram"):
    import librosa
    import librosa.display

    mp3 = CACHE_DIR / f"{track_id}.mp3"
    if not mp3.exists():
        raise FileNotFoundError("no cached preview — analyze this track first")
    y, sr = librosa.load(str(mp3), sr=22050, mono=True)

    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    fig.patch.set_facecolor(_BG)

    if kind == "waveform":
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#1db954")
        ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    elif kind == "chroma":
        chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=sr)
        img = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time",
                                       sr=sr, ax=ax, cmap="magma")
        ax.set(title="Chromagram (pitch classes over time)")
        fig.colorbar(img, ax=ax).ax.yaxis.set_tick_params(color=_FG)
    elif kind == "selfsim":
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        R = librosa.segment.recurrence_matrix(
            librosa.util.normalize(chroma, axis=0), mode="affinity", sym=True)
        img = librosa.display.specshow(R, x_axis="time", y_axis="time",
                                       sr=sr, ax=ax, cmap="magma")
        ax.set(title="Self-similarity matrix (repeating structure)")
    elif kind == "tempogram":
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
        img = librosa.display.specshow(tg, sr=sr, x_axis="time", y_axis="tempo",
                                       ax=ax, cmap="magma")
        ax.set(title="Tempogram (rhythmic periodicity)")
        fig.colorbar(img, ax=ax).ax.yaxis.set_tick_params(color=_FG)
    else:  # spectrogram (mel)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, x_axis="time", y_axis="mel",
                                       sr=sr, ax=ax, cmap="magma")
        ax.set(title="Mel spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB").ax.yaxis.set_tick_params(color=_FG)

    _style(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
