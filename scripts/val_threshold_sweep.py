"""Choose the region-growing threshold for one fold on its validation images.

Based on hybrid_val_threshold_sweep.py. Runs UI-VISA on the 20% validation
split for thresholds 0.1, 0.2, ..., 1.0 and writes the mean and SD of Dice
for each threshold. The threshold with the highest mean Dice is the one to
pass to run_uivisa.py.

Changes: paths are arguments; the region-growing loop and model now come from
src/uivisa/ (the code there is the same as in the original script).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import random
import pandas as pd

from sklearn.metrics import f1_score

from uivisa.data import fold_paths, train_val_split, load_padded
from uivisa.models import net
from uivisa.region_growing import grow, scores_to_mask

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fold", type=int, required=True, help="fold number (0-4)")
parser.add_argument("--image-dir", type=Path, required=True)
parser.add_argument("--mask-dir", type=Path, required=True)
parser.add_argument("--splits-dir", type=Path, default=Path("configs/splits"))
parser.add_argument("--seeds-dir", type=Path, default=Path("outputs/seeds"),
                    help="folder containing val_fold_<k>/ from generate_seeds.py --split val")
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="patch CNN state dict (default: outputs/patch_cnn/patch_cnn_fold<k>.pth)")
parser.add_argument("--out-dir", type=Path, default=Path("outputs/uivisa"))
args = parser.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

fold = args.fold
checkpoint = args.checkpoint or Path(f"outputs/patch_cnn/patch_cnn_fold{fold}.pth")

train_images, train_masks, _, _ = fold_paths(args.splits_dir, args.image_dir, args.mask_dir, fold)
train_images, val_images, train_masks, val_masks = train_val_split(train_images, train_masks)

val_image_names = [Path(p).stem for p in val_images]
val_images_padded = [load_padded(p) for p in val_images]
val_masks_padded = [load_padded(p) for p in val_masks]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net()
model = model.float()
model = model.to(device=device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

thresholds = [round(x * 0.1, 1) for x in range(1, 11)]
threshold_dice = {}

args.out_dir.mkdir(parents=True, exist_ok=True)
save_path = args.out_dir / f"val_threshold_sweep_fold_{fold}.csv"

for tol in thresholds:
    print(f"\n================ Threshold {tol} ================")
    dice_scores = []

    for img_idx, (val_image_padded, val_mask_padded) in enumerate(zip(val_images_padded, val_masks_padded)):
        image_name = val_image_names[img_idx]
        print(f"  Image {img_idx + 1}/{len(val_images_padded)}: {image_name}")

        unet_seg = np.load(args.seeds_dir / f"val_fold_{fold}" / f"{image_name}_unet_seg.npy")

        pixel_scores, _ = grow(model, val_image_padded, val_mask_padded, unet_seg, tol, device, verbose=False)
        recon_mask = scores_to_mask(pixel_scores, val_mask_padded, tol)

        gt = (val_mask_padded[40:-40, 40:-40] > 0.5).astype(np.uint8)
        pred_binary = (recon_mask > 0.5).astype(np.uint8)

        dice = f1_score(gt.flatten(), pred_binary.flatten())
        dice_scores.append(dice)
        print(f"    Dice: {dice:.4f}")

    threshold_dice[tol] = dice_scores
    print(f"  Mean dice for tol={tol:.1f}: {np.mean(dice_scores):.4f}")

print("\n================ Threshold Sweep Summary ================")
for tol, scores in threshold_dice.items():
    print(f"  tol={tol:.1f}  mean_dice={np.mean(scores):.4f}  std={np.std(scores):.4f}")

best_tol = max(threshold_dice, key=lambda t: np.mean(threshold_dice[t]))
print(f"\nBest threshold: {best_tol} with mean dice: {np.mean(threshold_dice[best_tol]):.4f}")

summary = pd.DataFrame({
    'threshold': list(threshold_dice.keys()),
    'mean_dice': [np.mean(v) for v in threshold_dice.values()],
    'std_dice':  [np.std(v)  for v in threshold_dice.values()]
})
summary.to_csv(save_path, index=False)
print(f"\nResults saved to: {save_path}")
