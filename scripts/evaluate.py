"""Compute Dice and clDice for UI-VISA test results across the 5 folds.

*** REBUILT FROM NOTEBOOK, UNTESTED ***
Evaluation was originally done in a Jupyter notebook. The metric functions
(src/uivisa/metrics.py) and the per-fold summary below follow the notebook's
compute_metrics(), cldice_2d() and analyze_fold_results(). The rest (reading
the run_uivisa.py outputs, looping over folds, writing CSVs) was written for
this repository. It has not been run, so check its numbers before relying on them.

For each fold the binary mask is rebuilt from <name>_indices.npy and
<name>_predictions.npy using that fold's threshold. Thresholds come from
--tols, or else from the best mean Dice in val_threshold_sweep_fold_<k>.csv.

Ground truth is binarized as mask > 0, as in the notebook.

Writes <results-dir>/metrics_per_image.csv and prints per-fold and overall
mean and SD.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from uivisa.data import fold_paths, load_padded
from uivisa.metrics import compute_metrics, cldice_2d
from uivisa.region_growing import indices_to_mask


def best_threshold(csv_path):
    sweep = pd.read_csv(csv_path)
    return float(sweep.loc[sweep['mean_dice'].idxmax(), 'threshold'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("configs/splits"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/uivisa"),
                        help="folder with fold_<k>/ from run_uivisa.py and the sweep CSVs")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--tols", type=float, nargs="+", default=None,
                        help="one threshold per fold, in the same order as --folds")
    args = parser.parse_args()

    if args.tols is not None and len(args.tols) != len(args.folds):
        parser.error("--tols needs one value per fold")

    rows = []
    for i, fold in enumerate(args.folds):
        if args.tols is not None:
            tol = args.tols[i]
        else:
            tol = best_threshold(args.results_dir / f"val_threshold_sweep_fold_{fold}.csv")

        _, _, test_images, test_masks = fold_paths(args.splits_dir, args.image_dir, args.mask_dir, fold)
        fold_dir = args.results_dir / f"fold_{fold}"

        print(f"\n=== METRICS FOR FOLD {fold} (tol={tol}) ===\n")
        for image_path, mask_path in zip(test_images, test_masks):
            name = image_path.stem
            indices = np.load(fold_dir / f"{name}_indices.npy")
            predictions = np.load(fold_dir / f"{name}_predictions.npy")

            mask = plt.imread(mask_path).astype(float)
            mask[mask > 0] = 1
            gt = (mask > 0.5).astype(np.uint8)

            recon = indices_to_mask(indices, predictions, load_padded(mask_path), tol)
            pred = (recon > 0.5).astype(np.uint8)

            acc, dice, jacc, sens, spec = compute_metrics(gt, pred)
            cl_dice_score = cldice_2d(pred, gt)
            print(f"{name}: Dice={dice:.4f}, CLDice={cl_dice_score:.4f}, "
                  f"Accuracy={acc:.4f}, Jaccard={jacc:.4f}, "
                  f"Sensitivity={sens:.4f}, Specificity={spec:.4f}")

            rows.append({"fold": fold, "image": name, "tol": tol,
                         "dice": dice, "cldice": cl_dice_score, "accuracy": acc,
                         "jaccard": jacc, "sensitivity": sens, "specificity": spec})

    df = pd.DataFrame(rows)
    out_csv = args.results_dir / "metrics_per_image.csv"
    df.to_csv(out_csv, index=False)

    # Per-fold mean and SD (np.std, population SD, as in the notebook)
    print("\n=== PER FOLD ===")
    per_fold = df.groupby("fold").agg(
        avg_dice=("dice", "mean"), sd_dice=("dice", lambda v: np.std(v)),
        avg_cldice=("cldice", "mean"), sd_cldice=("cldice", lambda v: np.std(v)),
        n=("dice", "size"))
    print(per_fold.to_string())

    print(f"\n=== ALL {len(df)} TEST IMAGES ===")
    print(f"Dice:   {df['dice'].mean():.4f} +/- {np.std(df['dice']):.4f}")
    print(f"clDice: {df['cldice'].mean():.4f} +/- {np.std(df['cldice']):.4f}")
    print(f"\nPer-image metrics saved to: {out_csv}")


if __name__ == "__main__":
    main()
