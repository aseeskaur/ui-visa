# REBUILT FROM NOTEBOOK, UNTESTED AS A MODULE.
# compute_metrics() and cldice_2d() are copied unchanged from the evaluation
# notebook (dsa_paper_unet_test.ipynb). They were not used from a script
# before this repository was put together.

import numpy as np

from skimage.morphology import skeletonize


def compute_metrics(gt, pred):
    """Pixel-wise metrics for binary arrays. Returns (accuracy, dice, jaccard, sensitivity, specificity)."""
    gt = gt.flatten()
    pred = pred.flatten()

    TP = np.sum((gt == 1) & (pred == 1))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))
    TN = np.sum((gt == 0) & (pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    jaccard = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)

    return accuracy, dice, jaccard, sensitivity, specificity


def cldice_2d(pred: np.ndarray, gt: np.ndarray):
    """clDice (centerline Dice) for 2D binary masks."""
    pred = (pred > 0).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)

    skel_pred = skeletonize(pred).astype(np.uint8)
    skel_gt   = skeletonize(gt).astype(np.uint8)

    tprec = (np.logical_and(skel_pred == 1, gt == 1).sum()) / (skel_pred.sum() + 1e-8)
    trec = (np.logical_and(skel_gt == 1, pred == 1).sum()) / (skel_gt.sum() + 1e-8)

    return float((2 * tprec * trec) / (tprec + trec + 1e-8))
