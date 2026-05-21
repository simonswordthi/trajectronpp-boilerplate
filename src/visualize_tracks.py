from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from export_serialized import load_scenes_pickle


def plot_scene(scene, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    hist = scene.history
    fut = scene.future
    ax.plot(hist[:, 0], hist[:, 1], "b-o", label="history")
    ax.plot(fut[:, 0], fut[:, 1], "g-o", label="future")

    for n in scene.neighbors:
        arr = n["states"]
        ax.plot(arr[:, 0], arr[:, 1], "k--", alpha=0.3)

    ax.set_title(f"Scene {scene.scene_id} Target {scene.target_agent_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scenes = load_scenes_pickle("data/processed/pie_scenes.pkl")
    out_dir = Path("data/processed/plots")
    for i, scene in enumerate(scenes[:10]):
        plot_scene(scene, out_dir / f"scene_{i:03d}.png")
    print(f"Saved up to 10 plots in {out_dir}")


if __name__ == "__main__":
    main()
