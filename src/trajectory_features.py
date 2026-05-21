from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def compute_bottom_center(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = (out["bbox_x1"] + out["bbox_x2"]) / 2.0
    out["y"] = out["bbox_y2"]
    return out


def add_relative_coordinates(df: pd.DataFrame, history_len: int) -> pd.DataFrame:
    out = df.copy()
    out["x_rel"] = np.nan
    out["y_rel"] = np.nan

    for _, idx in out.groupby(["scene_id", "ped_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("frame_id")
        if len(g) < history_len:
            continue
        anchor_index = g.index[history_len - 1]
        ax = out.at[anchor_index, "x"]
        ay = out.at[anchor_index, "y"]
        out.loc[g.index, "x_rel"] = out.loc[g.index, "x"] - ax
        out.loc[g.index, "y_rel"] = out.loc[g.index, "y"] - ay

    return out


def add_velocity_acceleration(df: pd.DataFrame, frame_dt: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    out[["vx", "vy", "ax", "ay"]] = np.nan

    for _, idx in out.groupby(["scene_id", "ped_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("frame_id")
        vx = g["x"].diff() / frame_dt
        vy = g["y"].diff() / frame_dt
        ax = vx.diff() / frame_dt
        ay = vy.diff() / frame_dt
        out.loc[g.index, "vx"] = vx.to_numpy()
        out.loc[g.index, "vy"] = vy.to_numpy()
        out.loc[g.index, "ax"] = ax.to_numpy()
        out.loc[g.index, "ay"] = ay.to_numpy()

    return out


def smooth_positions(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in ["x", "y"]:
        out[col] = (
            out.groupby(["scene_id", "ped_id"], sort=False)[col]
            .transform(lambda s: s.rolling(window=window, min_periods=1, center=True).mean())
        )
    return out


def build_features(track_df: pd.DataFrame, history_len: int, frame_dt: float = 1.0, smooth: bool = False) -> pd.DataFrame:
    out = compute_bottom_center(track_df)
    if smooth:
        out = smooth_positions(out)
    out = add_relative_coordinates(out, history_len=history_len)
    out = add_velocity_acceleration(out, frame_dt=frame_dt)
    return out


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    prep = _load_yaml("configs/preprocessing.yaml")
    inp = Path("data/interim/pie_tracks.parquet")
    df = pd.read_parquet(inp)
    out_df = build_features(
        df,
        history_len=int(prep.get("history_len", 8)),
        frame_dt=float(prep.get("frame_dt", 1.0)),
        smooth=False,
    )
    out = Path("data/interim/pie_tracks_features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(f"Saved {len(out_df)} rows with features to {out}")


if __name__ == "__main__":
    main()
