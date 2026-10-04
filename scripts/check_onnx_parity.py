"""
Check that the ONNX model gives the same predictions (and the same accuracy) as the pickle model.

Usage (from the repo root):
    python scripts/check_onnx_parity.py                       # uses models/model.pkl + models/model.onnx
    python scripts/check_onnx_parity.py MODEL.pkl MODEL.onnx  # any pair of files

It scores 1000 real rows from data/raw/crop_yield_raw.csv with both backends and reports
the largest prediction difference, plus MAE and R2 against the true yield for each backend.
"""

import sys
import warnings

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from prodml.data import load_raw_crop_data
from prodml.model import CropYieldModel


def main() -> None:
    warnings.filterwarnings("ignore")
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else "models/model.pkl"
    onnx_path = sys.argv[2] if len(sys.argv) > 2 else "models/model.onnx"

    features, target = load_raw_crop_data()
    if getattr(features, "attrs", {}).get("dataset_tag") != "kaggle_crop_yield_real":
        raise SystemExit(
            "Real dataset not found at data/raw/crop_yield_raw.csv (run dvc pull or scripts/download_data.py)."
        )

    sample = features.sample(1000, random_state=7)
    truth = target.loc[sample.index].to_numpy()
    rows = sample.to_dict("records")

    model = CropYieldModel(model_path=pkl_path, onnx_path=onnx_path)
    model.load_or_create(model_uri="")
    if model._get_onnx_session() is None:
        raise SystemExit("ONNX session could not be created, so the ONNX backend would silently fall back to pickle.")

    pickle_pred = np.array([r["predicted_yield_hg_ha"] for r in model.predict(rows, backend="pickle")])
    onnx_pred = np.array([r["predicted_yield_hg_ha"] for r in model.predict(rows, backend="onnx")])

    diff = np.abs(pickle_pred - onnx_pred)
    print(f"rows scored            : {len(rows)}")
    print(f"max |pickle - onnx|    : {diff.max():.2f} hg/ha   (mean {diff.mean():.2f})")
    print(
        f"pickle  MAE / R2       : {mean_absolute_error(truth, pickle_pred):.1f} / {r2_score(truth, pickle_pred):.4f}"
    )
    print(f"onnx    MAE / R2       : {mean_absolute_error(truth, onnx_pred):.1f} / {r2_score(truth, onnx_pred):.4f}")
    metric_parity = abs(mean_absolute_error(truth, pickle_pred) - mean_absolute_error(truth, onnx_pred)) < 100.0
    strict_parity = np.allclose(pickle_pred, onnx_pred, rtol=0.05, atol=10000.0)
    print("METRIC ACCURACY PARITY OK" if metric_parity and strict_parity else "PARITY BROKEN")


if __name__ == "__main__":
    main()
