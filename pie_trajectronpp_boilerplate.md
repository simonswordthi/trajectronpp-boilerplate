# PIE zu Trajectron++ Boilerplate-Plan

Ziel dieses Dokuments ist es, ein klares **Boilerplate** für GitHub Copilot zu formulieren, mit dem eine erste Pipeline von **PIE** zu **Trajectron++** implementiert werden kann — bewusst **ohne behavioral features in der Inferenz**. PIE ist explizit für Pedestrian Intention und Trajectory Prediction konzipiert, und Trajectron++ arbeitet mit szenenbasierten Graphen und agentenbezogenen Zustandsgrößen wie Position, Geschwindigkeit und Beschleunigung. [cite:17][cite:105]

## Ziel

Es soll eine lauffähige Baseline entstehen, die:
- PIE-Annotationen einliest,
- Fußgänger-Tracks extrahiert,
- daraus reine Trajektorien ableitet,
- diese in ein Trajectron++-ähnliches Scene/Node-Format überführt,
- und anschließend ein **pedestrian-only** Trajectory-Prediction-Modell trainierbar macht. Trajectron++ modelliert dynamisch interagierende Agenten als Graph und verwendet dafür agentenspezifische Zustandsverläufe. [cite:105][cite:102]

Wichtig: In dieser Phase werden **keine** Features wie Gaze, Pose, Looking, Crossing Intent oder andere behavioral annotations in die Inferenz eingespeist. Sie bleiben höchstens für spätere Analysen erhalten. PIE enthält solche Informationen zwar, aber für die erste Baseline sollten nur Trajektorien genutzt werden. [cite:17][cite:43]

## Architekturidee

Die Implementierung sollte in klar getrennte Module aufgeteilt werden, damit GitHub Copilot jeweils kleine, gut definierte Aufgaben generieren kann.

```text
pie_trajectronpp/
  README.md
  pyproject.toml
  configs/
    paths.yaml
    preprocessing.yaml
    training.yaml
  src/
    pie_reader.py
    track_extractor.py
    trajectory_features.py
    scene_builder.py
    dataset_splitter.py
    export_serialized.py
    train_baseline.py
    evaluate_baseline.py
    visualize_tracks.py
  data/
    raw/
    interim/
    processed/
  notebooks/
  tests/
```

Diese Trennung ist sinnvoll, weil das Trajectron++-Preprocessing selbst typischerweise ebenfalls die Datenvorbereitung von Training und Evaluation separiert. Das offizielle nuScenes-Preprocessing für Trajectron++ zeigt genau diese Logik: Rohdaten werden zunächst in Szenen, Nodes und Zustandsgrößen übersetzt und erst danach für das Modell serialisiert. [cite:102][cite:105]

## Umsetzungsphasen

## Phase 1: PIE einlesen

Zuerst soll ein Reader für PIE geschrieben werden, der die Annotationen strukturiert lädt. Das offizielle PIE-Setup und das PIEPredict-Repository zeigen, dass PIE in Verzeichnisse mit Videos, Annotationen und zugehörigen Metadaten organisiert ist. [cite:99][cite:17]

### Copilot-Auftrag

```md
Erstelle in `src/pie_reader.py` eine Python-Klasse `PIEReader`, die:
- den PIE-Datensatzpfad aus einer YAML-Konfiguration liest,
- alle verfügbaren Annotationen zu Videos, Frames und Pedestrians lädt,
- eine einheitliche pandas-DataFrame-Struktur zurückgibt,
- pro Zeile mindestens `scene_id`, `frame_id`, `ped_id`, `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` enthält,
- und grundlegende Validierungschecks implementiert (fehlende Felder, leere Tracks, ungültige Bounding Boxes).
```

### Erwartetes Zwischenformat

| Spalte | Beschreibung |
|---|---|
| `scene_id` | Eindeutige Clip-/Video-ID |
| `frame_id` | Frameindex im Clip |
| `ped_id` | Eindeutige Fußgänger-ID |
| `bbox_x1,y1,x2,y2` | Originale Bounding Box |
| `source_dataset` | Konstante Markierung `PIE` |

## Phase 2: Tracks extrahieren

Aus den Frame-Annotationen müssen konsistente Pedestrian-Sequenzen gebaut werden. PIE stellt annotierte Fußgänger in Fahrszenen bereit und wird in bestehenden Arbeiten auch für die Bildung von Beobachtungs- und Vorhersagefenstern genutzt. [cite:17][cite:98]

### Copilot-Auftrag

```md
Erstelle in `src/track_extractor.py` Funktionen, die:
- die PIE-Frame-Annotationen nach `scene_id` und `ped_id` gruppieren,
- daraus zeitlich sortierte Tracks erzeugen,
- Track-Längen berechnen,
- kurze oder stark fragmentierte Tracks verwerfen,
- und eine DataFrame- oder Dictionary-Struktur mit vollständigen Agentensequenzen exportieren.
```

### Mindestregeln

- Ein Track muss mindestens Beobachtungs- plus Vorhersagehorizont abdecken.
- IDs dürfen innerhalb eines Tracks nicht springen.
- Lücken müssen markiert werden.
- Noch keine Interpolation in dieser Phase.

## Phase 3: Trajektorien ableiten

Für Trajectron++ brauchst du pro Agent Zustandsgrößen wie Position, Geschwindigkeit und Beschleunigung. Das offizielle Preprocessing für Trajectron++ baut genau diese Zustände aus den Rohtrajektorien auf; bei Pedestrians stehen vor allem Position, Velocity und Acceleration im Vordergrund. [cite:102][cite:105]

Als 2D-Position sollte zunächst der **Bottom-Center** der Bounding Box verwendet werden, weil er die Bodenprojektion eines Fußgängers besser approximiert als die Boxmitte. Die erste PIE-Trajectory-Baseline kann in Bildkoordinaten arbeiten, solange dies explizit als Limitation dokumentiert wird. PIE-basierte Trajektorienarbeiten nutzen ebenfalls relative Koordinaten und abgeleitete Bewegungsgrößen. [cite:98][cite:17]

### Copilot-Auftrag

```md
Erstelle in `src/trajectory_features.py` Funktionen, die für jeden Track:
- aus Bounding Boxes den Bottom-Center berechnen,
- `x`, `y` pro Frame erzeugen,
- relative Koordinaten zum letzten Beobachtungspunkt berechnen,
- `vx`, `vy` als erste Differenz,
- `ax`, `ay` als zweite Differenz,
- und optional eine einfache Glättung gegen Box-Jitter anbieten.

Nutze pandas und numpy. Fehlende Werte sollen sauber markiert werden.
```

## Phase 4: Sliding Windows bauen

Trajectron++ arbeitet auf Zeitfenstern mit Beobachtungs- und Vorhersagehorizont. Für den Start sollte ein fixes Sliding-Window-Schema verwendet werden, zum Beispiel 8 Beobachtungsschritte und 12 Zukunftsschritte; die genaue Fensterlänge kann später an die PIE-Framerate angepasst werden. [cite:105][cite:98]

### Copilot-Auftrag

```md
Erstelle in `src/scene_builder.py` Funktionen, die:
- aus jedem Pedestrian-Track Sliding Windows erzeugen,
- pro Fenster Beobachtungs- und Zukunftssequenzen trennen,
- alle gleichzeitig sichtbaren weiteren Pedestrians als Nachbaragenten sammeln,
- und daraus ein szenenbasiertes Datenobjekt erstellen.
```

### Zielstruktur pro Fenster

```python
{
    "scene_id": str,
    "t0": int,
    "target_agent_id": str,
    "history": np.ndarray,
    "future": np.ndarray,
    "neighbors": list,
}
```

## Phase 5: Scene/Node-Format für Trajectron++

Trajectron++ repräsentiert Szenen als Graphen mit Knoten pro Agent und Zustandsvariablen pro Zeitstempel. Das Vorverarbeitungsbeispiel für nuScenes zeigt, wie pro Node Zustandsfelder serialisiert und in einer Scene gesammelt werden. [cite:102][cite:105]

### Copilot-Auftrag

```md
Erstelle in `src/scene_builder.py` oder `src/export_serialized.py` Klassen und Funktionen, die:
- ein einfaches `Scene`-Objekt definieren,
- ein `Node`-Objekt für `PEDESTRIAN` definieren,
- pro Node die Zustandsfolgen `x`, `y`, `vx`, `vy`, `ax`, `ay` speichern,
- alle Nodes eines Zeitfensters in einer Scene sammeln,
- und die gesamte Struktur als Pickle oder Parquet serialisieren.
```

### Minimaler Node-Type

Nur ein Node-Type wird in der ersten Version benötigt:
- `PEDESTRIAN` [cite:105]

## Phase 6: Datensplits

Train/Validation/Test-Splits sollen auf **Clip-Ebene** oder **Scene-Ebene** erstellt werden, nicht frameweise. Cross-dataset- und Generalisierungsarbeiten zu Pedestrian-Crossing zeigen, dass inkonsistente Splits zu stark verzerrten Ergebnissen führen können. [cite:45]

### Copilot-Auftrag

```md
Erstelle in `src/dataset_splitter.py` Funktionen, die:
- Szenen eindeutig nach Clip-ID gruppieren,
- reproduzierbare Train/Val/Test-Splits mit festem Random Seed erzeugen,
- sicherstellen, dass keine Sequenzen desselben Clips in mehreren Splits liegen,
- und die Splits als JSON oder YAML exportieren.
```

## Phase 7: Baseline-Training

Erst nachdem ein kleines serialisiertes Datenset fertig ist, sollte ein erstes Baseline-Training gestartet werden. Trajectron++ ist auf heterogene Agenten ausgelegt, kann aber auch mit einem einzelnen Node-Type trainiert werden, solange die Zustandsräume konsistent definiert sind. [cite:105]

### Copilot-Auftrag

```md
Erstelle in `src/train_baseline.py` ein Trainingsskript, das:
- die serialisierten PIE-Szenen lädt,
- nur `PEDESTRIAN`-Nodes verwendet,
- eine minimale Trajectron++-Konfiguration einliest,
- einen kleinen Sanity-Check-Durchlauf auf wenigen Szenen ausführt,
- und Metriken wie ADE und FDE ausgibt.
```

## Phase 8: Evaluation und Visualisierung

Eine reine Zahlenbewertung reicht am Anfang nicht. Zusätzlich sollten Ground Truth und vorhergesagte Trajektorien überlagert geplottet werden, damit fehlerhafte Vorverarbeitung oder inkonsistente Koordinaten sofort sichtbar werden. ADE/FDE sind Standardmetriken in Trajectory Prediction. [cite:105]

### Copilot-Auftrag

```md
Erstelle in `src/evaluate_baseline.py` und `src/visualize_tracks.py` Funktionen, die:
- ADE und FDE berechnen,
- Ground Truth und Prediction pro Szene plotten,
- mehrere Beispielsequenzen als PNG speichern,
- und fehlgeschlagene Beispiele automatisch markieren.
```

## Technische Designregeln

Die erste Baseline soll bewusst eng geschnitten sein. Das reduziert das Risiko, dass Preprocessing-Probleme, Verhaltenstags und Modellprobleme gleichzeitig vermischt werden. Das ist methodisch wichtig, weil PIE viele Zusatzannotationen enthält, die später wertvoll sind, aber die erste Trajectory-only-Baseline unnötig verkomplizieren würden. [cite:17][cite:43]

### Regeln

- Verwende zunächst nur Fußgängertracks.
- Keine behavioral features in der Inferenz.
- Keine Gaze-, Pose- oder Intent-Signale im Modellinput.
- Anfangs nur Bildkoordinaten oder einfache relative Koordinaten.
- Später kann die Pipeline um Ego-Signale und Behavior-Features erweitert werden.

## Definition of Done

Der erste Prototyp ist erfolgreich, wenn folgende Punkte erfüllt sind:

- PIE-Annotationen werden reproduzierbar geladen. [cite:17][cite:99]
- Pedestrian-Tracks werden korrekt extrahiert. [cite:17]
- Pro Track werden `x`, `y`, `vx`, `vy`, `ax`, `ay` berechnet. [cite:102][cite:105]
- Sliding Windows für History/Future sind verfügbar. [cite:98][cite:105]
- Szenen sind in einem Trajectron++-ähnlichen Format serialisiert. [cite:102][cite:105]
- Ein pedestrian-only Baseline-Training läuft an.
- ADE/FDE und einfache Visualisierungen werden erzeugt. [cite:105]

## Empfohlene erste Prompts für GitHub Copilot

### Prompt 1 - PIE Reader

```text
Implement a PIEReader class in Python that loads PIE pedestrian annotations into a pandas DataFrame with columns scene_id, frame_id, ped_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, validates missing values, and supports loading from a configurable dataset root path.
```

### Prompt 2 - Track Builder

```text
Implement a track extraction module that groups PIE annotations by scene_id and ped_id, sorts them by frame_id, filters short tracks, and returns consistent pedestrian track sequences for trajectory prediction preprocessing.
```

### Prompt 3 - Trajectory Features

```text
Implement functions to compute bottom-center positions from pedestrian bounding boxes, derive x/y trajectories, and calculate vx/vy and ax/ay using finite differences. Include optional smoothing for noisy bounding boxes.
```

### Prompt 4 - Sliding Windows

```text
Implement a sliding window generator for pedestrian tracks that creates fixed-length history and future trajectory segments and collects neighboring pedestrians visible in the same time window.
```

### Prompt 5 - Scene Export

```text
Implement simple Scene and Node classes for a Trajectron++-style preprocessing pipeline. Each pedestrian node should store x, y, vx, vy, ax, ay sequences. Export scenes to pickle files for later model training.
```

### Prompt 6 - Baseline Training

```text
Implement a baseline training script that loads serialized pedestrian-only PIE scenes, prepares a minimal Trajectron++-compatible dataset, trains a small model, and evaluates ADE and FDE.
```

## Nächster sinnvoller Schritt

Der nächste sinnvolle Schritt ist **nicht** sofort das Training, sondern zuerst ein kleines Testsubset mit 5-10 Clips sauber durch die Reader-, Track- und Feature-Pipeline laufen zu lassen. Erst wenn die Visualisierung dieser Tracks plausibel ist, sollte das Scene-Format finalisiert werden. Eine fehlerhafte Vorverarbeitung ist bei Onboard-Fußgängerdaten deutlich wahrscheinlicher als ein Fehler im Trainingscode. [cite:17][cite:98]
