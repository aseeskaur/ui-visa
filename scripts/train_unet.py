"""Train the seed U-Net for one cross-validation fold.

Based on dsa_unet_train_full_aug_with_seed.py. Changes: paths are command-line
arguments, and the model classes are imported from src/uivisa/models.py.
Training logic, augmentation and the order of random calls are unchanged.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import random

from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from uivisa.data import fold_paths, train_val_split
from uivisa.models import UNet

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fold", type=int, required=True, help="fold number (0-4)")
parser.add_argument("--epochs", type=int, default=500,
                    help="minimum number of epochs before early stopping can trigger (default: 500; hard cap is 1000)")
parser.add_argument("--image-dir", type=Path, required=True, help="folder with the raw DSA images (PNG)")
parser.add_argument("--mask-dir", type=Path, required=True, help="folder with the ground-truth masks (PNG)")
parser.add_argument("--splits-dir", type=Path, default=Path("configs/splits"))
parser.add_argument("--out-dir", type=Path, default=Path("outputs/unet"),
                    help="where to write the checkpoint and training curve")
args = parser.parse_args()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

fold = args.fold
epochs = args.epochs

train_images, train_masks, test_images, test_masks = fold_paths(
    args.splits_dir, args.image_dir, args.mask_dir, fold)

train_images, val_images, train_masks, val_masks = train_val_split(train_images, train_masks)


def augment(image, mask):
    augmented_images = [image.copy()]
    augmented_masks = [mask.copy()]

    if random.random() > 0.5:
        k = random.randint(0, 3)
        augmented_images.append(np.rot90(image, k).copy())
        augmented_masks.append(np.rot90(mask, k).copy())

    if random.random() > 0.5:
        augmented_images.append(np.fliplr(image).copy())
        augmented_masks.append(np.fliplr(mask).copy())

    if random.random() > 0.5:
        augmented_images.append(np.flipud(image).copy())
        augmented_masks.append(np.flipud(mask).copy())

    if random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        scaled_image = zoom(image, scale, order=1)
        scaled_mask = zoom(mask, scale, order=0)

        h, w = scaled_image.shape
        orig_h, orig_w = image.shape

        start_h = (h - orig_h) // 2
        start_w = (w - orig_w) // 2
        scaled_image = scaled_image[start_h:start_h+orig_h, start_w:start_w+orig_w]
        scaled_mask = scaled_mask[start_h:start_h+orig_h, start_w:start_w+orig_w]

        augmented_images.append(scaled_image.copy())
        augmented_masks.append(scaled_mask.copy())

    return augmented_images, augmented_masks


class DSAPaperDataset(Dataset):
    def __init__(self, image_paths, mask_paths, use_augment=True, patch_size=None, patches_per_image=None):
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.images = []
        self.masks = []

        for img_path, mask_path in zip(image_paths, mask_paths):
            image = plt.imread(img_path).astype(float)
            mask = plt.imread(mask_path).astype(float)
            image = image / np.max(image)
            mask[mask > 0] = 1

            if use_augment:
                aug_images, aug_masks = augment(image, mask)
                self.images.extend(aug_images)
                self.masks.extend(aug_masks)
            else:
                self.images.append(image)
                self.masks.append(mask)

    def __len__(self):
        if self.patch_size and self.patches_per_image:
            return len(self.images) * self.patches_per_image
        return len(self.images)

    def __getitem__(self, index):
        image_index = index // self.patches_per_image if self.patch_size else index

        image = self.images[image_index]
        mask = self.masks[image_index]

        if self.patch_size:
            h, w = image.shape
            ps = self.patch_size

            if h < ps or w < ps:
                pad_h = max(0, ps - h)
                pad_w = max(0, ps - w)
                image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
                mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
                h, w = image.shape

            top = random.randint(0, max(0, h - ps))
            left = random.randint(0, max(0, w - ps))
            image = image[top:top+ps, left:left+ps]
            mask = mask[top:top+ps, left:left+ps]

        image_tensor = torch.tensor(image.copy(), dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0)

        return image_tensor, mask_tensor


train_dataset = DSAPaperDataset(
    image_paths=train_images,
    mask_paths=train_masks,
    use_augment=True,
    patch_size=128,
    patches_per_image=200
)

val_dataset = DSAPaperDataset(
    image_paths=val_images,
    mask_paths=val_masks,
    use_augment=False,
    patch_size=128,
    patches_per_image=200
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")

# Sanity check. Kept from the original script: it draws a random crop, so
# removing it would change the random sequence used during training.
image, mask = train_dataset[0]
print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Mask values: {mask.unique()}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet(in_channels=1, out_channels=1, dropout=0.2)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model_save_dir = args.out_dir
model_save_dir.mkdir(parents=True, exist_ok=True)

curve_save_dir = args.out_dir
curve_save_dir.mkdir(parents=True, exist_ok=True)

best_epoch = 0
patience_level = 25
epochs_no_improve = 0
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"\nStarting training for {epochs} epochs...")

# NOTE (flagged, not changed): `epochs` is a minimum. Early stopping is only
# allowed after `epochs` epochs, and the loop stops at 1000 regardless.
for epoch in range(1000):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"{'='*60}\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_epoch = epoch + 1
        best_model_path = model_save_dir / f'unet_fold{fold}.pth'
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch+1} with val loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs\n")

    if epoch + 1 > epochs and epochs_no_improve >= patience_level:
        print(f"Early stopping at epoch {epoch + 1}")
        break

print(f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"Total epochs trained: {len(train_losses)}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training Progress - Fold {fold}')
plt.legend()
plt.grid(True)
plt.savefig(curve_save_dir / f'unet_training_curve_fold{fold}.png')
print(f"Training curve saved.")
