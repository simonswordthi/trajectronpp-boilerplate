from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scene_builder import Scene

def save_scenes_pickle(scenes: list["Scene"] | list[Any], path: str | Path) -> None:
    """Serialize scene-like objects as a pickle file.

    Warning:
        Pickle should only be used for trusted workflows because loading pickle
        data later can execute arbitrary code from malicious payloads.

    Args:
        scenes: Sequence of serializable scene-like objects.
        path: Output pickle file path.

    Raises:
        TypeError: If scenes is not provided as a list.
        OSError: If writing the file fails.
    """
    if not isinstance(scenes, list):
        raise TypeError("scenes must be provided as a list")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(scenes, f)


def load_scenes_pickle(path: str | Path) -> list[Any]:
    """Load scene-like objects from a pickle file.

    Warning:
        Only load pickle files from trusted sources. Pickle deserialization can
        execute arbitrary code for malicious payloads.

    Args:
        path: Input pickle file path.

    Returns:
        Deserialized list of scene-like objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If reading fails.
        pickle.UnpicklingError: If file content is invalid pickle.
    """
    with Path(path).open("rb") as f:
        return pickle.load(f)
