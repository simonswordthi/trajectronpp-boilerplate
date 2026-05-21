# PIE → Trajectron++ Boilerplate

Dieses Repository enthält eine erste pedestrian-only Baseline-Pipeline:

1. PIE-Annotationen laden
2. Tracks extrahieren
3. Trajektorien-Features berechnen (`x,y,vx,vy,ax,ay`)
4. Sliding Windows und Scene/Node-Struktur erzeugen
5. Datensplits erstellen
6. Baseline mit einfacher Vorhersage evaluieren (ADE/FDE)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Konfiguration

- `configs/paths.yaml`: Pfade für raw/interim/processed
- `configs/preprocessing.yaml`: Preprocessing-Parameter
- `configs/training.yaml`: Training-/Eval-Parameter

## Nutzung

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

Hinweis: Diese Version nutzt bewusst keine Behavioral-Features in der Inferenz.
