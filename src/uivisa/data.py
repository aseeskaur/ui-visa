import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


def read_fold_lists(splits_dir):
    """Read the 5-fold split files in configs/splits/.

    Each file holds the image filenames for one fold concatenated on a single
    line (e.g. "15.png07.png11.png"). Returns (train_names, test_names), both
    dicts mapping fold number -> list of filenames.
    """
    train_image_names = {}
    test_image_names = {}

    for file in Path(splits_dir).glob('*.txt'):
        fold_num = int(file.name.split('_')[-1].split('.')[0])

        with open(file, 'r') as f:
            lines = f.readlines()
            filenames = [filename.strip() + 'png' for filename in lines[0].split('png') if filename.strip()]
            formatted_filenames = [f"{filename}" for filename in filenames]

            if 'train' in file.name:
                train_image_names[fold_num] = formatted_filenames
            elif 'test' in file.name:
                test_image_names[fold_num] = formatted_filenames

    return train_image_names, test_image_names


def fold_paths(splits_dir, image_dir, mask_dir, fold):
    """Return (train_images, train_masks, test_images, test_masks) path lists for one fold."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    train_image_names, test_image_names = read_fold_lists(splits_dir)

    train_images = [image_dir / f for f in train_image_names[fold]]
    train_masks = [mask_dir / f for f in train_image_names[fold]]
    test_images = [image_dir / f for f in test_image_names[fold]]
    test_masks = [mask_dir / f for f in test_image_names[fold]]
    return train_images, train_masks, test_images, test_masks


def train_val_split(train_images, train_masks):
    """The 80/20 train/validation split used by every script (random_state=42)."""
    return train_test_split(train_images, train_masks, test_size=0.2, random_state=42)


def load_normalized(path):
    """Read a PNG and divide by its maximum."""
    image = plt.imread(path)
    return image / np.max(image)


def load_padded(path, pad=40):
    """Read a PNG, divide by its maximum, and zero-pad by `pad` pixels on each side."""
    return np.pad(load_normalized(path), pad, mode='constant')
