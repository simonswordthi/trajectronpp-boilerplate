from __future__ import annotations

from pathlib import Path

import numpy as np

from export_serialized import load_scenes_pickle


def ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Compute ADE and FDE for one trajectory pair shaped [T, 2]."""
    # pred/gt: [T, 2] for x,y
    errors = np.linalg.norm(pred - gt, axis=1)
    ade = float(np.mean(errors))
    fde = float(errors[-1])
    return ade, fde


def evaluate_predictions(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    """Aggregate mean ADE/FDE across prediction-target trajectory lists."""
    ades: list[float] = []
    fdes: list[float] = []
    for pred, target in zip(predictions, targets, strict=True):
        ade, fde = ade_fde(pred, target)
        ades.append(ade)
        fdes.append(fde)
    return {"ADE": float(np.mean(ades)) if ades else float("nan"), "FDE": float(np.mean(fdes)) if fdes else float("nan")}


def main() -> None:
    scenes = load_scenes_pickle(Path("data/processed/pie_scenes.pkl"))
    print(f"Loaded {len(scenes)} scenes. Use train_baseline.py for prediction generation.")


if __name__ == "__main__":
    main()
