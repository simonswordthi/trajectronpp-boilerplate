from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

def save_scenes_pickle(scenes: list[Any], path: str | Path) -> None:
    """Serialize a list of scene-like objects as a pickle file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(scenes, f)


def load_scenes_pickle(path: str | Path) -> list[Any]:
    """Load and return scene-like objects from a pickle file path."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
