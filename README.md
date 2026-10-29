# EDM on CIFAR-10: conditioning, CFG, and FID

Checkpoint code for DSC 180 (D32). The CIFAR-10 notebook generates images with NVIDIA's pretrained EDM models, scores them, and trains a smaller EDM network on 100 and 1,000 images. The MNIST guidance lab is in `dsc180-d32-mnist-guidance-lab/` for the later MNIST CFG network.

Lab handout: `dsc180-d32-cifar10-edm-lab/README.md`.

## Setup

```bash
conda create -n edm_cifar python=3.11 -y
conda activate edm_cifar
conda install -y -c pytorch -c conda-forge pytorch numpy matplotlib pandas pyarrow pillow jupyter ipykernel
cd dsc180-d32-cifar10-edm-lab
jupyter lab cifar10_edm_lab.ipynb
```

Use the `edm_cifar` kernel and run the cells from the top. On CUDA the first cell sets `QUICK = False` (1,000 images per `w` in part 5, 20,000 updates in part 6). On a laptop `QUICK = True`. Use a DSMLP or Colab GPU for the checkpoint numbers. Written answers go in `notes.md`.

## Data downloads

The downloads cell creates `dsc180-d32-cifar10-edm-lab/downloads/` (about 720 MB, checksummed, resumable). Figures and snapshots go in `outputs/`. `.gitignore` leaves both out, and it also leaves out the MNIST weights and helper modules, which those notebooks download or write on first run.

| File                                            | What it is                                                      |
| ----------------------------------------------- | --------------------------------------------------------------- |
| `edm-<commit>/`                                 | EDM's code at commit `008a4e5316c8e3bfe61a62f874bddba254295afb` |
| `edm-cifar10-32x32-uncond-vp.pkl`               | Pretrained unconditional CIFAR-10 network                       |
| `edm-cifar10-32x32-cond-vp.pkl`                 | Pretrained conditional CIFAR-10 network                         |
| `cifar10-32x32.npz`                             | EDM's FID reference statistics for the 50,000 training images   |
| `inception-2015-12-05.pkl`                      | Inception-v3 network used for FID                               |
| `cifar10_resnet56-187c023a.pt`                  | Pretrained CIFAR-10 classifier for label accuracy               |
| `cifar10-train.parquet`, `cifar10-test.parquet` | CIFAR-10 from Hugging Face (`uoft-cs/cifar10`)                  |

If `downloads/cifar-10-python.tar.gz` is already present, the notebook uses that archive instead of the parquet files. To use a separate clone of [NVlabs/edm](https://github.com/NVlabs/edm), set `EDM_DIR` to that folder before starting the notebook.

## Figures and tables

Run the notebook from the top through the section named below. Later sections depend on the functions defined above them. Part 5 reuses files already in `outputs/`, so a rerun does not sample those images again.

| Result                                        | Notebook section                       | What produces it                                                                                                                                            |
| --------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unconditional sample grid                     | 1. Unconditional samples               | `edm_sample` on `uncond`, seed 0, then `show_grid`                                                                                                          |
| Unconditional trajectory                      | 1. Inside the sampler                  | `edm_sample(..., record=True)` on `uncond`, seed 0, then `show_trajectory`                                                                                  |
| Conditional sample grid                       | 2. Conditional samples                 | `edm_sample` on `cond`, one row per class, seed 1, then `show_grid`                                                                                         |
| CFG grid toward cat                           | 3. CFG with two networks               | `edm_sample` with `D_cfg` for `w` in 0, 1, 1.5, 2, 3, 4, class cat, seed 2, then `show_grid`                                                                |
| Conditional and CFG trajectories              | 3. Your turn                           | The empty cell: `edm_sample(..., record=True)` for `w = 1` and `w = 4` toward one class, same seed, then `show_trajectory`                                  |
| FID in two dimensions                         | 4. FID in two dimensions               | The three-panel scatter of `cloud` draws, with the FID printed in each title                                                                                |
| Accuracy, FID, and FID_class table            | 5. The CFG sweep                       | The `generate` loop over `W`. Printed in the cell and saved to `outputs/cfg_sweep_n<N_EVAL>.json`. Images for each `w` are `outputs/cfg_w<w>_n<N_EVAL>.npz` |
| Accuracy, FID, FID_class, and class histogram | 5. The CFG sweep                       | The four-panel `plt.subplots` cell immediately after the table                                                                                              |
| Training loss and FID across snapshots        | 6. Loss and FID across snapshots       | `train` on 100 and 1,000 images, then the two-panel plot. Snapshots are `outputs/subset<n>_update<step>.pt`                                                 |
| Samples at each snapshot                      | 6. Loss and FID across snapshots       | `show_grid` on `run["progress"]` in that same cell                                                                                                          |
| Samples next to the nearest training image    | 6. Is it memorization?                 | `nearest` and `show_grid` for each training-set size                                                                                                        |
| CFG grids from the trained network            | 6. The effect of CFG with your network | `own_samples` at `w` in 0, 1, 2, 4, then `show_grid`                                                                                                        |

`N_EVAL` is 200 while `QUICK` is true and 1,000 on CUDA. The checkpoint table is the 1,000-image run. Part 6 on CUDA trains for 20,000 updates and scores 1,000 unconditional samples at each snapshot. On a laptop those are 500 updates and 100 samples.

The encouraged full training run is separate. The last code cells of part 6 write CIFAR-10 as a zip of PNGs and print a `torchrun … train.py` command. Run that command in a terminal on a DSMLP GPU. After snapshots exist, set `RUN_DIR` and rerun the following cell for the loss curve and FID across snapshots.
