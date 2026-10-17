# UI-VISA: U-Net Initialized Vascular Image Segmentation Architecture

Code for the paper:

> Asees Kaur, Suzanne S. Sindi, Erica M. Rutter.
> *UI-VISA: U-Net Initialized Vascular Image Segmentation Architecture.*
> arXiv:2609.01598, 2026. https://arxiv.org/abs/2609.01598

UI-VISA segments blood vessels in 2D digital subtraction angiography (DSA) images in two stages:

1. **Seed selection.** A U-Net segments the image. Every foreground pixel of the U-Net output is used as a seed.
2. **Region growing.** A small CNN takes an 80×80 patch centered on a pixel and predicts vessel probabilities for that pixel's 3×3 neighborhood. Starting from the seeds, each pixel's predicted probabilities are averaged over every patch that covered it. Pixels whose average reaches a threshold are added to the segmentation and become new patch centers. This repeats until no new pixels are added. The segmentation only grows outward from pixels already in it, which keeps vessel connectivity intact.

The threshold is chosen separately for each fold on validation images. Results in the paper use 5-fold cross validation on 26 images and report Dice and clDice.

## Repository layout

```
configs/splits/   5-fold train/test image lists used in the paper
src/uivisa/       models, data loading, region growing, metrics
scripts/          training, seed generation, threshold selection, testing, evaluation
slurm/            example SLURM job scripts (one array task per fold)
```

## Installation

Tested with Python 3.12 and the package versions in `requirements.txt`.

```bash
git clone https://github.com/aseeskaur/ui-visa.git
cd ui-visa
conda create -n uivisa python=3.12
conda activate uivisa
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install `torch` and `torchvision` first, following https://pytorch.org, and then install the rest of `requirements.txt`.

All commands below are run from the repository root.

## Data

The DSA images and vessel annotations come from:

> Zhang M, Zhang C, Wu X, Cao X, Young GS, Chen H, Xu X. A neural network approach to segment brain blood vessels in digital subtraction angiography. *Computer Methods and Programs in Biomedicine* 185 (2020): 105159. https://doi.org/10.1016/j.cmpb.2019.105159

The dataset is not publicly available and is not included in this repository. It can be requested from the corresponding author of that paper.

The code expects this layout. Images and masks share filenames, and the names must match those in `configs/splits/`:

```
data/
├── raw/    01.png ... 26.png    DSA images (single channel)
└── gt/     01.png ... 26.png    binary vessel masks
```

`scripts/generate_seeds.py` assumes 512×512 images.

## Pretrained weights

Pretrained weights are not currently provided. Results can be reproduced by training with the provided 5-fold splits in `configs/splits/`.

## Pipeline

Every step runs one fold at a time (`--fold 0` to `--fold 4`). Outputs go to `outputs/` by default. Each script's `--help` lists its options. To run all folds as a SLURM job array, use the matching script in `slurm/`, after editing the partition, environment name and data paths.

| Step | Script | Output |
|---|---|---|
| 1. Train the U-Net | `scripts/train_unet.py` | `outputs/unet/unet_fold<k>.pth` |
| 2. Train the patch CNN | `scripts/train_patch_cnn.py` | `outputs/patch_cnn/patch_cnn_fold<k>.pth` |
| 3. Generate U-Net seeds | `scripts/generate_seeds.py` | `outputs/seeds/{fold_<k>,val_fold_<k>}/*_unet_seg.npy` |
| 4. Choose the threshold on validation images | `scripts/val_threshold_sweep.py` | `outputs/uivisa/val_threshold_sweep_fold_<k>.csv` |
| 5. Run UI-VISA on test images | `scripts/run_uivisa.py` | `outputs/uivisa/fold_<k>/` |
| 6. Compute Dice and clDice | `scripts/evaluate.py` | `outputs/uivisa/metrics_per_image.csv` |

Steps 1 and 2 are independent and can run at the same time.

Within each fold, the training images are split 80/20 into training and validation sets with a fixed seed (`random_state=42`). The validation set is used for early stopping in steps 1 and 2 and for choosing the threshold in step 4. Test images are used only in steps 5 and 6.

### Training

```bash
# U-Net (early stopping with patience 25; can only stop after --epochs epochs, at most 1000 epochs)
python scripts/train_unet.py --fold 0 --epochs 500 --image-dir data/raw --mask-dir data/gt

# Patch CNN (class-balanced patches resampled every epoch; early stopping with patience 25, at most 500 epochs)
python scripts/train_patch_cnn.py --fold 0 --image-dir data/raw --mask-dir data/gt
```

### Seeds and threshold selection

```bash
python scripts/generate_seeds.py --fold 0 --split val  --image-dir data/raw --mask-dir data/gt
python scripts/generate_seeds.py --fold 0 --split test --image-dir data/raw --mask-dir data/gt

# Tries thresholds 0.1, 0.2, ..., 1.0 and prints the one with the highest mean validation Dice
python scripts/val_threshold_sweep.py --fold 0 --image-dir data/raw --mask-dir data/gt
```

### Testing

```bash
python scripts/run_uivisa.py --fold 0 --tol 0.5 --image-dir data/raw --mask-dir data/gt
```

Use the threshold that `val_threshold_sweep.py` picked for that fold as `--tol`. Region growing is slow: an image can take a long time, and the threshold sweep repeats it 10 times for each validation image. The example SLURM scripts request 48 hours per fold for these two steps.

### Evaluation

```bash
python scripts/evaluate.py --image-dir data/raw --mask-dir data/gt
```

By default each fold's threshold is read from its sweep CSV. To set thresholds by hand, pass `--tols t0 t1 t2 t3 t4`. The script prints Dice and clDice for each image, mean and SD per fold, and mean and SD over all test images.

### Reproducing the cross-validation results

1. Put the data in `data/raw` and `data/gt` as described above.
2. For folds 0–4, run steps 1–5:
   ```bash
   mkdir -p logs
   sbatch slurm/train_unet.sub
   sbatch slurm/train_patch_cnn.sub
   # after both training jobs finish:
   sbatch slurm/generate_seeds.sub
   # after seeds are written:
   sbatch slurm/val_threshold_sweep.sub
   # after the sweeps finish:
   sbatch slurm/run_uivisa.sub
   ```
3. Run `python scripts/evaluate.py --image-dir data/raw --mask-dir data/gt` once all five folds are done.

Training uses fixed random seeds, but GPU training is not bit-for-bit deterministic. Retrained models, and the numbers they produce, may differ slightly from those in the paper.

## Note on rebuilt scripts

Seed generation and Dice/clDice evaluation were originally done in Jupyter notebooks. `scripts/generate_seeds.py`, `scripts/evaluate.py` and `src/uivisa/metrics.py` were rebuilt from those notebooks for this repository. The model inference, thresholding and metric code were copied from the notebooks without changes. These three files have not been run in their current form; check their output before relying on it.

## Citation

If you use this code, please cite:

```bibtex
@misc{kaur2026uivisa,
  title={{UI-VISA}: U-Net Initialized Vascular Image Segmentation Architecture},
  author={Kaur, Asees and Sindi, Suzanne S. and Rutter, Erica M.},
  year={2026},
  eprint={2609.01598},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2609.01598}
}
```

If you use the DSA data, please also cite:

```bibtex
@article{zhang2020dsa,
  title={A neural network approach to segment brain blood vessels in digital subtraction angiography},
  author={Zhang, Min and Zhang, Chen and Wu, Xian and Cao, Xinhua and Young, Geoffrey S and Chen, Huai and Xu, Xiaoyin},
  journal={Computer Methods and Programs in Biomedicine},
  volume={185},
  pages={105159},
  year={2020},
  doi={10.1016/j.cmpb.2019.105159}
}
```

## License

MIT. See `LICENSE`.
