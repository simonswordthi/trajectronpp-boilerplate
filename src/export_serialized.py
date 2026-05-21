from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

def save_scenes_pickle(scenes: list[Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(scenes, f)


def load_scenes_pickle(path: str | Path) -> list[Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
