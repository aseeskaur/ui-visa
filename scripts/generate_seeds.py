"""Generate U-Net seed masks (*_unet_seg.npy) for one fold.

*** REBUILT FROM NOTEBOOK, UNTESTED ***
Seed generation was originally done in a Jupyter notebook
(dsa_paper_unet_test.ipynb). This script copies the notebook's TestDataset
and run_inference_for_fold() logic unchanged; only the paths became
arguments and the metric/plotting cells were left out. It has not been run
in this form, so check its output before relying on it.

The image is split into a non-overlapping grid of 128x128 tiles, each tile is
segmented by the U-Net, and the tile predictions are thresholded at 0.5 and
stitched back together.

--split test  writes <out-dir>/fold_<k>/      (used by run_uivisa.py)
--split val   writes <out-dir>/val_fold_<k>/  (used by val_threshold_sweep.py)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from uivisa.data import fold_paths, train_val_split
from uivisa.models import UNet


class TestDataset(Dataset):
    # NOTE (flagged, not changed): assumes 512x512 images, as in the notebook.
    def __init__(self, image_paths, mask_paths, patch_size=128):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size

        self.patches_per_row = 512 // patch_size
        self.patches_per_image = self.patches_per_row ** 2

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, index):
        img_idx = index // self.patches_per_image
        patch_idx = index % self.patches_per_image

        # Load full image
        image = plt.imread(self.image_paths[img_idx]).astype(float)
        mask = plt.imread(self.mask_paths[img_idx]).astype(float)

        image = image / np.max(image)
        mask[mask > 0] = 1

        # Calculate patch position in grid
        row = patch_idx // self.patches_per_row
        col = patch_idx % self.patches_per_row

        # Extract patch at grid position
        top = row * self.patch_size
        left = col * self.patch_size

        image_patch = image[top:top+self.patch_size, left:left+self.patch_size]
        mask_patch = mask[top:top+self.patch_size, left:left+self.patch_size]

        image_tensor = torch.tensor(image_patch.copy(), dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_patch.copy(), dtype=torch.float32).unsqueeze(0)

        return image_tensor, mask_tensor, img_idx, row, col


def run_inference_for_fold(test_images, test_masks, model_path, device, save_dir):
    test_dataset = TestDataset(test_images, test_masks, patch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions_dict = {}
    masks_dict = {}
    image_names_dict = {}

    with torch.no_grad():
        for image, mask, img_idx, row, col in test_loader:
            image = image.to(device)
            pred = torch.sigmoid(model(image.float()))
            final_pred = (pred > 0.5).float()

            final_pred_np = final_pred[0, 0].cpu().numpy()
            mask_np = mask[0, 0].numpy()

            img_idx = img_idx.item()
            row = row.item()
            col = col.item()

            if img_idx not in predictions_dict:
                predictions_dict[img_idx] = np.zeros((512, 512))
                masks_dict[img_idx] = np.zeros((512, 512))
                image_names_dict[img_idx] = Path(test_images[img_idx]).name

            top = row * 128
            left = col * 128
            predictions_dict[img_idx][top:top+128, left:left+128] = final_pred_np
            masks_dict[img_idx][top:top+128, left:left+128] = mask_np

    save_dir.mkdir(parents=True, exist_ok=True)
    for idx in predictions_dict:
        image_name = image_names_dict[idx]
        npy_name = Path(image_name).stem
        np.save(save_dir / f"{npy_name}_unet_seg.npy", predictions_dict[idx])

    print(f"Saved {len(predictions_dict)} seed masks to {save_dir}")
    return predictions_dict, masks_dict, image_names_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True, help="fold number (0-4)")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="test images of the fold, or the 20%% validation split of its training images")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("configs/splits"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="U-Net state dict (default: outputs/unet/unet_fold<k>.pth)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/seeds"))
    args = parser.parse_args()

    checkpoint = args.checkpoint or Path(f"outputs/unet/unet_fold{args.fold}.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_images, train_masks, test_images, test_masks = fold_paths(
        args.splits_dir, args.image_dir, args.mask_dir, args.fold)

    if args.split == "test":
        images, masks = test_images, test_masks
        save_dir = args.out_dir / f"fold_{args.fold}"
    else:
        _, images, _, masks = train_val_split(train_images, train_masks)
        save_dir = args.out_dir / f"val_fold_{args.fold}"

    run_inference_for_fold(images, masks, checkpoint, device, save_dir)


if __name__ == "__main__":
    main()
