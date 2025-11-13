from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt

from .schemas import Segment


def render_timeline(segments: Iterable[Segment], out_path: str) -> str:
    """Render a simple speaker timeline plot."""

    segs: List[Segment] = [seg if isinstance(seg, Segment) else Segment(**seg) for seg in segments]
    if not segs:
        raise ValueError("No segments provided for visualization.")

    speakers = sorted({seg.speaker for seg in segs})
    speaker_to_idx = {speaker: idx for idx, speaker in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(12, 2 + len(speakers) * 0.6))
    for seg in segs:
        idx = speaker_to_idx[seg.speaker]
        ax.broken_barh([(seg.start, seg.end - seg.start)], (idx - 0.4, 0.8), label=seg.speaker)
        ax.text(seg.start, idx, seg.text[:40], va="center", ha="left", fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_yticks(list(speaker_to_idx.values()))
    ax.set_yticklabels(speakers)
    ax.set_title("Speaker Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)

