from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

REQUIRED_COLUMNS = [
    "scene_id",
    "frame_id",
    "ped_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
]

COLUMN_ALIASES = {
    "video_id": "scene_id",
    "clip_id": "scene_id",
    "frame": "frame_id",
    "pedestrian_id": "ped_id",
    "id": "ped_id",
    "x1": "bbox_x1",
    "y1": "bbox_y1",
    "x2": "bbox_x2",
    "y2": "bbox_y2",
}


@dataclass(slots=True)
class PIEReader:
    """Reader for PIE-style pedestrian annotations driven by YAML config."""
    config_path: str | Path

    def _load_config(self) -> dict:
        with Path(self.config_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _dataset_root(self) -> Path:
        """Return configured PIE dataset root directory."""
        cfg = self._load_config()
        root = cfg.get("pie_dataset_root")
        if not root:
            raise ValueError("Missing 'pie_dataset_root' in config")
        return Path(root)

    def _collect_annotation_files(self, root: Path) -> list[Path]:
        files = list(root.rglob("*.csv")) + list(root.rglob("*.json"))
        if not files:
            raise FileNotFoundError(f"No CSV/JSON annotations found in {root}")
        return sorted(files)

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_json(path)

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
        out = df.rename(columns=rename_map).copy()
        out["source_dataset"] = "PIE"
        return out

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if df.empty:
            raise ValueError("Loaded annotation dataframe is empty")

        if df[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError("NaN values found in required columns")

        invalid_bbox = (df["bbox_x2"] <= df["bbox_x1"]) | (df["bbox_y2"] <= df["bbox_y1"])
        if invalid_bbox.any():
            raise ValueError("Invalid bounding boxes found (x2<=x1 or y2<=y1)")

    def load_annotations(self) -> pd.DataFrame:
        """Load, normalize, validate, and return annotations as a DataFrame."""
        root = self._dataset_root()
        files = self._collect_annotation_files(root)
        frames: Iterable[pd.DataFrame] = (self._normalize_columns(self._read_file(p)) for p in files)
        df = pd.concat(frames, ignore_index=True)

        self._validate(df)

        # Harmonize dtypes
        df["scene_id"] = df["scene_id"].astype(str)
        df["ped_id"] = df["ped_id"].astype(str)
        df["frame_id"] = pd.to_numeric(df["frame_id"]).astype(int)
        for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
            df[c] = pd.to_numeric(df[c]).astype(float)

        return df.sort_values(["scene_id", "ped_id", "frame_id"]).reset_index(drop=True)


def main() -> None:
    reader = PIEReader("configs/paths.yaml")
    df = reader.load_annotations()
    out = Path("data/interim/pie_annotations.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
