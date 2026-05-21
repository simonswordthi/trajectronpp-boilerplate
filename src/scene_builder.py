from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


STATE_COLUMNS = ["x", "y", "vx", "vy", "ax", "ay"]


@dataclass(slots=True)
class Node:
    node_id: str
    node_type: str
    states: dict[str, list[float]]


@dataclass(slots=True)
class Scene:
    scene_id: str
    t0: int
    target_agent_id: str
    history: np.ndarray
    future: np.ndarray
    neighbors: list[dict[str, Any]]
    nodes: list[Node]


def _valid_states(arr: np.ndarray) -> bool:
    return np.isfinite(arr).all()


def build_sliding_windows(
    feature_df: pd.DataFrame,
    history_len: int,
    future_len: int,
    stride: int = 1,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    total_len = history_len + future_len

    for (scene_id, ped_id), g in feature_df.groupby(["scene_id", "ped_id"], sort=False):
        g = g.sort_values("frame_id").reset_index(drop=True)
        for start in range(0, len(g) - total_len + 1, stride):
            chunk = g.iloc[start : start + total_len]
            history = chunk.iloc[:history_len][STATE_COLUMNS].to_numpy(dtype=float)
            future = chunk.iloc[history_len:][STATE_COLUMNS].to_numpy(dtype=float)
            if not _valid_states(history) or not _valid_states(future):
                continue

            t0 = int(chunk.iloc[history_len - 1]["frame_id"])
            window_end_frame = int(chunk.iloc[-1]["frame_id"])
            visible = feature_df[
                (feature_df["scene_id"] == scene_id)
                & (feature_df["ped_id"] != ped_id)
                & (feature_df["frame_id"] >= t0)
                & (feature_df["frame_id"] <= window_end_frame)
            ]
            neighbors = []
            for n_id, ng in visible.groupby("ped_id", sort=False):
                n_states = ng.sort_values("frame_id")[STATE_COLUMNS].to_numpy(dtype=float)
                if np.isfinite(n_states).all():
                    neighbors.append({"ped_id": str(n_id), "states": n_states})

            windows.append(
                {
                    "scene_id": str(scene_id),
                    "t0": t0,
                    "target_agent_id": str(ped_id),
                    "history": history,
                    "future": future,
                    "neighbors": neighbors,
                }
            )
    return windows


def windows_to_scenes(windows: list[dict[str, Any]]) -> list[Scene]:
    scenes: list[Scene] = []
    for w in windows:
        target_states = np.vstack([w["history"], w["future"]])
        target_node = Node(
            node_id=w["target_agent_id"],
            node_type="PEDESTRIAN",
            states={k: target_states[:, i].tolist() for i, k in enumerate(STATE_COLUMNS)},
        )
        nodes = [target_node]
        for n in w["neighbors"]:
            arr = n["states"]
            nodes.append(
                Node(
                    node_id=n["ped_id"],
                    node_type="PEDESTRIAN",
                    states={k: arr[:, i].tolist() for i, k in enumerate(STATE_COLUMNS)},
                )
            )

        scenes.append(
            Scene(
                scene_id=w["scene_id"],
                t0=w["t0"],
                target_agent_id=w["target_agent_id"],
                history=w["history"],
                future=w["future"],
                neighbors=w["neighbors"],
                nodes=nodes,
            )
        )
    return scenes


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    prep = _load_yaml("configs/preprocessing.yaml")
    inp = Path("data/interim/pie_tracks_features.parquet")
    df = pd.read_parquet(inp)
    windows = build_sliding_windows(
        df,
        history_len=int(prep.get("history_len", 8)),
        future_len=int(prep.get("future_len", 12)),
        stride=int(prep.get("stride", 1)),
    )
    scenes = windows_to_scenes(windows)

    from export_serialized import save_scenes_pickle

    out = Path("data/processed/pie_scenes.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_scenes_pickle(scenes, out)
    print(f"Saved {len(scenes)} scenes to {out}")


if __name__ == "__main__":
    main()
