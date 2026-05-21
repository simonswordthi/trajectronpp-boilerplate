from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from export_serialized import load_scenes_pickle


def split_scene_ids(
    scene_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    uniq = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = uniq[:n_train]
    val = uniq[n_train : n_train + n_val]
    test = uniq[n_train + n_val :]

    return {"train": train, "val": val, "test": test}


def main() -> None:
    cfg_path = Path("configs/training.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scenes = load_scenes_pickle("data/processed/pie_scenes.pkl")
    scene_ids = [s.scene_id for s in scenes]
    splits = split_scene_ids(scene_ids, seed=int(cfg.get("seed", 42)))

    out = Path("data/processed/splits.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(f"Saved splits to {out}")


if __name__ == "__main__":
    main()
