"""UI-VISA region growing.

The loop below is copied from hybrid_test.py (the script used for the test
results); hybrid_val_threshold_sweep.py ran the same loop. Only the hardcoded
40-pixel padding was turned into the `pad` argument (default 40).
"""

import numpy as np
import pandas as pd
import torch

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def img_process(next_cents, padded_image, padded_mask):
    image_list, mask_list, val_list, rows, cols = [], [], [], [], []
    for i in next_cents:
        row, col = i[0], i[1]
        val = padded_image[row][col]
        snap_image = padded_image[row-40:row+40, col-40:col+40]
        snap_mask = padded_mask[row-1:row+2, col-1:col+2]
        image_list.append(snap_image)
        mask_list.append(snap_mask)
        val_list.append(val)
        rows.append(row)
        cols.append(col)
    return image_list, mask_list, val_list, rows, cols


class newcelldata(Dataset):
    def __init__(self, image_list, mask_list, rows, cols, transforms):
        self.image_list = image_list
        self.mask_list = mask_list
        self.rows = rows
        self.cols = cols
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index].astype(float)
        mask = self.mask_list[index].astype(float)
        rows = self.rows[index]
        cols = self.cols[index]
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        return (image, mask, rows, cols)


def grow(model, image_padded, mask_padded, seed_mask, tol, device, pad=40, verbose=True):
    """Grow a segmentation from the U-Net seeds.

    image_padded, mask_padded: image and mask, normalized and zero-padded by `pad`.
        The mask is only used to cut matching 3x3 windows; it does not affect the result.
    seed_mask: U-Net binary segmentation (unpadded). Every pixel > 0 is a seed.
    tol: threshold on the averaged patch-CNN probability.

    Returns (pixel_scores, n_iterations). pixel_scores maps padded (row, col)
    to the list of every probability predicted for that pixel.
    """
    transforms_tensor = transforms.ToTensor()

    # Get foreground pixel coordinates from UNet segmentation
    foreground_coords = np.argwhere(seed_mask > 0)

    # Create centroid dataframe from foreground pixels
    centroid_df = pd.DataFrame(foreground_coords, columns=['row', 'col'])
    centroid_df['row'] = centroid_df['row'] + pad
    centroid_df['col'] = centroid_df['col'] + pad

    if verbose:
        print(f"Loaded {len(centroid_df)} foreground pixels from UNet segmentation")

    S = []
    runn_cent = []
    next_S = []
    pixel_scores = {}

    for i in centroid_df.index:
        next_S.append([centroid_df['row'][i], centroid_df['col'][i]])
    # NOTE (flagged, not changed): this appends the whole seed list as a single
    # element of S, so the `i not in S` check below never matches an initial
    # seed and seeds that are re-detected get processed again in iteration 2.
    S.append(next_S)
    counter = 0

    while len(next_S) > 0:
        counter += 1
        if verbose:
            print(f"Iteration {counter}, processing {len(next_S)} centroids")

        image_list, mask_list, val_list, rows, cols = img_process(
            next_S, image_padded, mask_padded)

        dataset = newcelldata(
            image_list=image_list,
            mask_list=mask_list,
            rows=rows,
            cols=cols,
            transforms=transforms_tensor)

        loader = DataLoader(dataset, 50, shuffle=False)

        with torch.no_grad():
            for k, data in enumerate(loader):
                images, masks, rows, cols = data
                images = images.to(device=device)
                pred = model(images.float())
                final_pred = torch.sigmoid(pred)

                for b in range(final_pred.shape[0]):
                    for pred_row in range(final_pred.shape[2]):
                        for pred_col in range(final_pred.shape[3]):
                            row_item = rows[b].item()
                            col_item = cols[b].item()
                            x = row_item - 1 + pred_row
                            y = col_item - 1 + pred_col

                            boundary_size = pad
                            if boundary_size <= x < image_padded.shape[0] - boundary_size and \
                               boundary_size <= y < image_padded.shape[1] - boundary_size:
                                val = final_pred[b, 0, pred_row, pred_col].item()
                                if (x, y) not in pixel_scores:
                                    pixel_scores[(x, y)] = []
                                pixel_scores[(x, y)].append(val)

        for (x, y), scores in pixel_scores.items():
            if np.mean(scores) >= tol:
                if [x, y] not in runn_cent:
                    runn_cent.append([x, y])

        next_S = []
        for i in runn_cent:
            if i not in S:
                next_S.append(i)
        S.extend(next_S)

    return pixel_scores, counter


def scores_to_mask(pixel_scores, like_padded, tol, pad=40):
    """Binary mask (unpadded) of pixels whose averaged score is >= tol."""
    recon_mask = np.zeros_like(like_padded)
    for (x, y), scores in pixel_scores.items():
        if np.mean(scores) >= tol:
            recon_mask[x, y] = 1
    return recon_mask[pad:-pad, pad:-pad]


def indices_to_mask(indices, predictions, like_padded, tol, pad=40):
    """Rebuild the binary mask from the saved *_indices.npy / *_predictions.npy files."""
    recon_mask = np.zeros_like(like_padded)
    for (x, y), avg in zip(indices, predictions):
        if avg >= tol:
            recon_mask[x, y] = 1
    return recon_mask[pad:-pad, pad:-pad]
