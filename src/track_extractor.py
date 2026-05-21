from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def extract_tracks(
    annotations: pd.DataFrame,
    min_track_length: int,
    history_len: int,
    future_len: int,
    max_frame_gap: int = 1,
) -> pd.DataFrame:
    required = {"scene_id", "ped_id", "frame_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(annotations.columns))}")

    minimum = max(min_track_length, history_len + future_len)
    grouped = annotations.sort_values(["scene_id", "ped_id", "frame_id"]).groupby(["scene_id", "ped_id"], sort=False)

    rows: list[pd.DataFrame] = []
    for (scene_id, ped_id), g in grouped:
        g = g.copy()
        g["frame_gap"] = g["frame_id"].diff().fillna(1).astype(int)
        g["has_gap"] = g["frame_gap"] > max_frame_gap
        g["track_length"] = len(g)
        g["is_fragmented"] = bool(g["has_gap"].any())

        if len(g) < minimum:
            continue

        if g["has_gap"].any():
            # Keep but mark fragmented; caller can filter if desired.
            pass

        g["scene_id"] = str(scene_id)
        g["ped_id"] = str(ped_id)
        rows.append(g)

    if not rows:
        return pd.DataFrame(columns=list(annotations.columns) + ["frame_gap", "has_gap", "track_length", "is_fragmented"])

    return pd.concat(rows, ignore_index=True)


def tracks_to_dict(track_df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for (scene_id, ped_id), g in track_df.groupby(["scene_id", "ped_id"], sort=False):
        out.setdefault(str(scene_id), {})[str(ped_id)] = g.sort_values("frame_id").reset_index(drop=True)
    return out


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    prep = _load_yaml("configs/preprocessing.yaml")
    inp = Path("data/interim/pie_annotations.parquet")
    df = pd.read_parquet(inp)
    out_df = extract_tracks(
        annotations=df,
        min_track_length=int(prep.get("min_track_length", 20)),
        history_len=int(prep.get("history_len", 8)),
        future_len=int(prep.get("future_len", 12)),
    )
    out = Path("data/interim/pie_tracks.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"Saved {len(out_df)} track rows to {out}")


if __name__ == "__main__":
    main()
