from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from evaluate_baseline import evaluate_predictions
from export_serialized import load_scenes_pickle


def constant_velocity_predict(history: np.ndarray, future_len: int) -> np.ndarray:
    """Predict future [T,2] by propagating last observed velocity from history."""
    # history: [H, 6], x/y in first two dims, vx/vy in 3rd/4th dims
    last_xy = history[-1, :2]
    vxvy = history[-1, 2:4]
    if not np.isfinite(vxvy).all():
        vxvy = np.array([0.0, 0.0])

    pred = np.zeros((future_len, 2), dtype=float)
    current = last_xy.astype(float)
    for t in range(future_len):
        current = current + vxvy
        pred[t] = current
    return pred


def run_baseline(scenes: list, max_scenes: int | None = None) -> dict[str, float]:
    """Run constant-velocity baseline on scenes and return ADE/FDE metrics."""
    selected = scenes if max_scenes is None else scenes[:max_scenes]
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []

    for s in selected:
        future_len = s.future.shape[0]
        pred = constant_velocity_predict(s.history, future_len=future_len)
        gt = s.future[:, :2]
        preds.append(pred)
        gts.append(gt)

    return evaluate_predictions(preds, gts)


def main() -> None:
    with Path("configs/training.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    scenes = load_scenes_pickle("data/processed/pie_scenes.pkl")
    metrics = run_baseline(scenes, max_scenes=int(cfg.get("max_scenes", 100)))
    print(f"ADE={metrics['ADE']:.4f} FDE={metrics['FDE']:.4f}")


if __name__ == "__main__":
    main()
