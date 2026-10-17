"""Run UI-VISA on the test images of one fold.

Based on hybrid_test.py. For each test image, the U-Net seed mask is loaded,
region growing is run with the patch CNN, and three files are written to
<out-dir>/fold_<k>/:
  <name>_indices.npy      (N, 2) padded pixel coordinates visited
  <name>_predictions.npy  (N,)   averaged patch-CNN probability for each
  <name>_results.png      the final binary mask (averaged probability >= tol)

Changes: paths are arguments; the region-growing loop and model now come from
src/uivisa/ (the code there is the same as in the original script).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from uivisa.data import fold_paths, load_padded
from uivisa.models import net
from uivisa.region_growing import grow, scores_to_mask

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fold", type=int, required=True, help="fold number (0-4)")
parser.add_argument("--tol", type=float, required=True,
                    help="region-growing threshold, chosen with val_threshold_sweep.py")
parser.add_argument("--image-dir", type=Path, required=True)
parser.add_argument("--mask-dir", type=Path, required=True)
parser.add_argument("--splits-dir", type=Path, default=Path("configs/splits"))
parser.add_argument("--seeds-dir", type=Path, default=Path("outputs/seeds"),
                    help="folder containing fold_<k>/ from generate_seeds.py --split test")
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
tol = args.tol
checkpoint = args.checkpoint or Path(f"outputs/patch_cnn/patch_cnn_fold{fold}.pth")

_, _, test_images, test_masks = fold_paths(args.splits_dir, args.image_dir, args.mask_dir, fold)

test_images_padded = [load_padded(p) for p in test_images]
test_masks_padded = [load_padded(p) for p in test_masks]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net()
model = model.float()
model = model.to(device=device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

save_dir = args.out_dir / f"fold_{fold}"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Testing {len(test_images_padded)} images from fold {fold} with tol={tol}")

fold_start = time.time()

for img_idx, (test_image_padded, test_mask_padded) in enumerate(zip(test_images_padded, test_masks_padded)):
    img_start = time.time()
    print(f"\nProcessing image {img_idx + 1}/{len(test_images_padded)}")

    image_name = test_images[img_idx].stem

    unet_seg = np.load(args.seeds_dir / f"fold_{fold}" / f"{image_name}_unet_seg.npy")

    pixel_scores, counter = grow(model, test_image_padded, test_mask_padded, unet_seg, tol, device)

    print(f"Completed after {counter} iterations")

    flat_list_ind = []
    flat_list_pred = []
    for (x, y), scores in pixel_scores.items():
        flat_list_ind.append([x, y])
        flat_list_pred.append(np.mean(scores))

    new_recon_mask = scores_to_mask(pixel_scores, test_mask_padded, tol)

    np.save(save_dir / f"{image_name}_indices.npy", np.array(flat_list_ind))
    np.save(save_dir / f"{image_name}_predictions.npy", np.array(flat_list_pred))

    plt.imshow(new_recon_mask, cmap="gray")
    plt.savefig(save_dir / f"{image_name}_results.png", bbox_inches='tight')
    plt.close()

    print(f"Image {image_name}: Detected {len(flat_list_ind)} pixels in {counter} iterations")
    img_time = time.time() - img_start
    print(f"Image {image_name} took {img_time:.2f} seconds")

fold_time = time.time() - fold_start
print(f"\nTotal time for fold {fold}: {fold_time:.2f} seconds")
print(f"\nCompleted testing all {len(test_images_padded)} images")
print(f"Results saved to: {save_dir}")
