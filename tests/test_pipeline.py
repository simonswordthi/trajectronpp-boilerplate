import numpy as np
import pandas as pd

from evaluate_baseline import ade_fde
from scene_builder import build_sliding_windows
from track_extractor import extract_tracks
from trajectory_features import build_features


def _synthetic_annotations() -> pd.DataFrame:
    rows = []
    for frame in range(25):
        rows.append(
            {
                "scene_id": "s1",
                "frame_id": frame,
                "ped_id": "p1",
                "bbox_x1": float(frame),
                "bbox_y1": 0.0,
                "bbox_x2": float(frame + 2),
                "bbox_y2": 10.0,
            }
        )
        rows.append(
            {
                "scene_id": "s1",
                "frame_id": frame,
                "ped_id": "p2",
                "bbox_x1": float(frame + 1),
                "bbox_y1": 0.0,
                "bbox_x2": float(frame + 3),
                "bbox_y2": 9.0,
            }
        )
    return pd.DataFrame(rows)


def test_extract_tracks_filters_min_length() -> None:
    df = _synthetic_annotations()
    out = extract_tracks(df, min_track_length=20, history_len=8, future_len=12)
    assert not out.empty
    assert set(out["ped_id"].unique()) == {"p1", "p2"}


def test_feature_and_window_pipeline() -> None:
    df = _synthetic_annotations()
    tracks = extract_tracks(df, min_track_length=20, history_len=8, future_len=12)
    feats = build_features(tracks, history_len=8, frame_dt=1.0, smooth=False)
    windows = build_sliding_windows(feats, history_len=8, future_len=12)
    assert len(windows) > 0
    assert windows[0]["history"].shape == (8, 6)
    assert windows[0]["future"].shape == (12, 6)


def test_ade_fde_zero_for_perfect_prediction() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 1.0]])
    pred = gt.copy()
    ade, fde = ade_fde(pred, gt)
    assert ade == 0.0
    assert fde == 0.0
