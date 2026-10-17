"""Train the region-growing patch CNN for one cross-validation fold.

Based on rgc_train_aug_resamp_new_fixing_plots.py. Changes: paths are
command-line arguments, and the model classes are imported from
src/uivisa/models.py. Augmentation, class-balanced patch resampling (each
epoch), training loop and the order of random calls are unchanged.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import random
import json
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from uivisa.data import fold_paths, train_val_split
from uivisa.models import net

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fold", type=int, required=True, help="fold number (0-4)")
parser.add_argument("--image-dir", type=Path, required=True, help="folder with the raw DSA images (PNG)")
parser.add_argument("--mask-dir", type=Path, required=True, help="folder with the ground-truth masks (PNG)")
parser.add_argument("--splits-dir", type=Path, default=Path("configs/splits"))
parser.add_argument("--out-dir", type=Path, default=Path("outputs/patch_cnn"),
                    help="where to write the checkpoint, loss history and loss plot")
args = parser.parse_args()

mpl.rcParams.update({
    'font.size':        16,
    'axes.titlesize':   20,
    'axes.labelsize':   18,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
})

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

fold = args.fold

train_images, train_masks, test_images, test_masks = fold_paths(
    args.splits_dir, args.image_dir, args.mask_dir, fold)

train_images, val_images, train_masks, val_masks = train_val_split(train_images, train_masks)


def augment_full_image(image, mask):
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
    return augmented_images, augmented_masks


train_images_padded = []
train_masks_padded = []
for image_path, mask_path in zip(train_images, train_masks):
    train_image = plt.imread(image_path)
    train_mask = plt.imread(mask_path)
    train_im_nor = train_image / np.max(train_image)
    train_mas_nor = train_mask / np.max(train_mask)
    aug_images, aug_masks = augment_full_image(train_im_nor, train_mas_nor)
    for aug_image, aug_mask in zip(aug_images, aug_masks):
        train_images_padded.append(np.pad(aug_image, 40, mode='constant'))
        train_masks_padded.append(np.pad(aug_mask, 40, mode='constant'))

val_images_padded = []
for image in val_images:
    val_image = plt.imread(image)
    val_im_nor = val_image / np.max(val_image)
    val_image_padded = np.pad(val_im_nor, 40, mode='constant')
    val_images_padded.append(val_image_padded)

val_masks_padded = []
for mask in val_masks:
    val_mask = plt.imread(mask)
    val_mas_nor = val_mask / np.max(val_mask)
    val_mask_padded = np.pad(val_mas_nor, 40, mode='constant')
    val_masks_padded.append(val_mask_padded)


def extract_balance_patches(padded_image, padded_mask, image_name=""):
    sum_categories = {}
    x = padded_image.shape[0]
    y = padded_image.shape[1]
    for i in range(40, x-40):
        for j in range(40, y-40):
            snap_image = padded_image[i-40:i+40, j-40:j+40]
            snap_mask = padded_mask[i-1:i+2, j-1:j+2]
            mask_sum = np.sum(snap_mask)
            if mask_sum in sum_categories:
                sum_categories[mask_sum]['images'].append(snap_image)
                sum_categories[mask_sum]['masks'].append(snap_mask)
            else:
                sum_categories[mask_sum] = {
                    'images': [snap_image],
                    'masks': [snap_mask]
                }
    print(f"\n{image_name} - Category distribution:")
    counts = []
    for category in sorted(sum_categories.keys()):
        count = len(sum_categories[category]['images'])
        print(f"  Category {category}: {count} patches")
        counts.append(count)
    n_min = min(counts)
    print(f"{image_name} - Minimum category count: {n_min}")
    balanced_images = []
    balanced_masks = []
    for category in sum_categories.keys():
        images_for_category = sum_categories[category]['images']
        masks_for_category = sum_categories[category]['masks']
        if len(images_for_category) >= n_min:
            random_indices = random.sample(range(len(images_for_category)), n_min)
            balanced_images.extend([images_for_category[i] for i in random_indices])
            balanced_masks.extend([masks_for_category[i] for i in random_indices])
    return balanced_images, balanced_masks


def process_images(image_data, data_type=""):
    final_images = []
    final_masks = []
    for index, (padded_image, padded_mask) in enumerate(image_data):
        image_name = f"{data_type}_Image_{index+1}" if data_type else f"Image_{index+1}"
        balanced_images, balanced_masks = extract_balance_patches(padded_image, padded_mask, image_name)
        final_images.extend(balanced_images)
        final_masks.extend(balanced_masks)
    return final_images, final_masks


val_data = list(zip(val_images_padded, val_masks_padded))
final_val_images, final_val_masks = process_images(val_data, "Validation")


class dsa_data(Dataset):
    def __init__(self, image_list, mask_list, transforms):
        self.image_list = image_list
        self.mask_list = mask_list
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask = self.mask_list[index].astype(float)
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return (image, mask)


transforms = transforms.ToTensor()

validation_dataset = dsa_data(image_list=final_val_images,
                               mask_list=final_val_masks,
                               transforms=transforms)
validation_loader = DataLoader(validation_dataset, batch_size=50)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net()
model = model.float()
model = model.to(device=device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

save_dir = args.out_dir
save_dir.mkdir(parents=True, exist_ok=True)

results_dir = args.out_dir
results_dir.mkdir(parents=True, exist_ok=True)

# Early stopping variables
max_epochs = 500
patience_level = 25
epochs_no_improve = 0
best_val_loss = float('inf')
best_model_state_dict = None
best_epoch = 0

training_losses = []
validation_losses = []

print(f"Starting training for fold {fold}, max {max_epochs} epochs with patience {patience_level}...")

for epoch in range(max_epochs):
    # Resample balanced patches each epoch
    train_data = list(zip(train_images_padded, train_masks_padded))
    final_train_images, final_train_masks = process_images(train_data, "Training")
    train_dataset = dsa_data(image_list=final_train_images,
                             mask_list=final_train_masks,
                             transforms=transforms)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=50)

    model.train()
    epoch_train_loss = 0.0
    for i, data in enumerate(train_loader):
        images, masks = data
        images = images.to(device=device)
        masks = masks.to(device=device)
        optimizer.zero_grad()
        outputs = model(images.float())
        loss = criterion(outputs, masks.float())
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            images, masks = data
            images = images.to(device=device)
            masks = masks.to(device=device)
            val_outputs = model(images.float())
            val_loss += criterion(val_outputs, masks.float()).item()

    avg_val_loss = val_loss / len(validation_loader)
    validation_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{max_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save loss history every epoch
    loss_history = {'train_loss': training_losses, 'val_loss': validation_losses}
    with open(results_dir / f"patch_cnn_loss_history_fold{fold}.json", 'w') as f:
        json.dump(loss_history, f)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        epochs_no_improve = 0
        # NOTE (flagged, not changed): dict.copy() is shallow, so this
        # dict keeps tracking the live weights. The file saved on the next
        # line is the correct best-epoch checkpoint (the one UI-VISA uses);
        # the full-model file saved after training holds the LAST epoch.
        best_model_state_dict = model.state_dict().copy()
        torch.save(best_model_state_dict, save_dir / f"patch_cnn_fold{fold}.pth")
        print(f"  New best model saved at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epochs")

    if epochs_no_improve >= patience_level:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

if best_model_state_dict:
    model.load_state_dict(best_model_state_dict)
    torch.save(model, save_dir / f"patch_cnn_fold{fold}_full_model.pth")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(training_losses, label='Training Loss')
ax.plot(validation_losses, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'Training and Validation Loss - Fold {fold}')
ax.legend()
plt.tight_layout()
plt.savefig(results_dir / f'patch_cnn_loss_plot_fold{fold}.png', dpi=300)

print('Finished Training')
