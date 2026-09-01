"""Camelot wheel helpers for harmonic mixing / key compatibility."""

# key name (sharps) + mode -> Camelot code
_MAJOR = {"C": "8B", "C#": "3B", "D": "10B", "D#": "5B", "E": "12B", "F": "7B",
          "F#": "2B", "G": "9B", "G#": "4B", "A": "11B", "A#": "6B", "B": "1B"}
_MINOR = {"C": "5A", "C#": "12A", "D": "7A", "D#": "2A", "E": "9A", "F": "4A",
          "F#": "11A", "G": "6A", "G#": "1A", "A": "8A", "A#": "3A", "B": "10A"}


def to_camelot(key, mode):
    if not key or not mode:
        return None
    table = _MAJOR if str(mode).lower().startswith("maj") else _MINOR
    return table.get(key)


def _parse(code):
    if not code or len(code) < 2:
        return None
    try:
        return int(code[:-1]), code[-1].upper()
    except ValueError:
        return None


def transition_score(a, b):
    """0..1 harmonic-mix compatibility between two Camelot codes.

    1.00 same key · 0.9 relative major/minor · 0.85 adjacent on wheel (±1) ·
    0.5 energy-boost (+2 same letter) · 0.15 otherwise (clash).
    """
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return None
    na, la = pa
    nb, lb = pb
    if na == nb and la == lb:
        return 1.0
    if na == nb and la != lb:              # relative major/minor
        return 0.9
    diff = min((na - nb) % 12, (nb - na) % 12)
    if la == lb and diff == 1:             # neighbour on the wheel
        return 0.85
    if la == lb and diff == 2:             # energy boost
        return 0.5
    if la == lb and diff == 7:             # dominant-ish
        return 0.4
    return 0.15


def compatible_keys(code):
    """The set of Camelot codes that mix well with `code` (for suggestions)."""
    p = _parse(code)
    if not p:
        return []
    n, l = p
    other = "A" if l == "B" else "B"
    return [f"{n}{l}", f"{n}{other}",
            f"{(n % 12) + 1}{l}", f"{((n - 2) % 12) + 1}{l}"]
