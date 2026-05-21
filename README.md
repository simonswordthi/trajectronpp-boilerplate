# PIE → Trajectron++ Boilerplate

This repository provides a first pedestrian-only baseline pipeline:

1. Load PIE annotations
2. Extract pedestrian tracks
3. Compute trajectory features (`x,y,vx,vy,ax,ay`)
4. Build sliding windows and Scene/Node structures
5. Create reproducible dataset splits
6. Evaluate a simple baseline with ADE/FDE

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

- `configs/paths.yaml`: raw/interim/processed paths
- `configs/preprocessing.yaml`: preprocessing parameters
- `configs/training.yaml`: training/evaluation parameters

## Usage

```bash
pie-reader
track-extractor
build-features
build-scenes
split-dataset
train-baseline
evaluate-baseline
visualize-tracks
```

Note: this baseline intentionally does not use behavioral features during inference.
