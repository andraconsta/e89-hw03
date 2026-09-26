# Session Summary: CSCI E-89 HW03, Problem 3

**Student:** Andra Constandache
**Tool:** Claude Code
**Repository:** https://github.com/andraconsta/e89-hw03 (folder `prob3/`)

## Goal

Build the FashionMNIST image classifier from "Building an Image Classifier
with PyTorch" (Geron, *Hands-On Machine Learning*, ch. 10) as five step
scripts, then assemble them into two notebooks, execute them and export
them to HTML.

## What was built

| Step | File | Content |
|---|---|---|
| 1 | `load_data.py` | FashionMNIST with `ToImage` and `ToDtype(float32, scale=True)`, 55,000/5,000 split with seed 42, DataLoaders (batch 32, re-seeded) |
| 2 | `model.py` | Device selection, `ImageClassifier` MLP (784, 300, 100, 10), `CrossEntropyLoss` |
| 3 | `train.py` | `evaluate_tm`, `train2`, SGD lr 0.1, torchmetrics accuracy, 20 epochs; saves `fashion_mnist_mlp.pt` and `history.json` |
| 4 | `plot_accuracy.py` | Training accuracy per epoch, saved as `training_accuracy.png` |
| 5 | `predict.py` | Loads the weights and predicts 3 validation images |
| 6 | `build_notebook.py` | Builds `e89_Constandache_Andra_HW03_Prob3_scripts.ipynb` from the 5 scripts |
| 7 | `e89_Constandache_Andra_HW03_Prob3.ipynb` | Readable notebook with explanations, inline plot and prediction images |
| 8 | `*.html` | HTML export of both executed notebooks |

## Key results

- **Dataset sizes:** 55,000 train, 5,000 valid, 10,000 test. Sample shape
  `[1, 28, 28]` float32, first sample "Ankle boot". Batch `[32, 1, 28, 28]`.
- **Model:** 266,610 parameters. Output `[32, 10]`. Untrained loss 2.3156,
  close to ln(10) = 2.30.
- **Training (scripts and readable notebook, identical runs):**

| Metric | Epoch 1 | Epoch 20 |
|---|---|---|
| Train loss | 0.6060 | 0.1876 |
| Train accuracy | 0.7814 | 0.9286 |
| Valid accuracy | 0.8424 | 0.8788 |

  Valid accuracy peaks at 0.8886 (epoch 16) and levels off after about
  epoch 10 while train accuracy keeps rising: mild overfitting.
- **Predictions (first 3 validation images, all correct):**

| Image | Predicted | Confidence | True |
|---|---|---|---|
| 1 | Sneaker | 85.9% | Sneaker |
| 2 | Coat | 99.3% | Coat |
| 3 | Pullover | 59.2% | Pullover |

- **Scripts notebook:** final train accuracy 0.9286, valid accuracy 0.8906.
  All 3 predictions correct (Sneaker 99.7%, Coat 99.6%, Pullover 72.8%).

## Problems and fixes

1. **Missing packages in the cloud container.** torch was not installed and
   `download.pytorch.org` was blocked by the proxy. Fix: installed torch
   2.14.0, torchvision 0.29.0 and torchmetrics 1.9.0 from PyPI.
2. **Import inside the script body.** The first `load_data.py` imported
   `DataLoader` mid-file. Fix: moved the import to the top.
3. **Headless backend in the plot script.** `matplotlib.use("Agg")` would
   stop the plot from showing inline in the notebook. Fix: removed it;
   `savefig` works without it.
4. **Notebook execution too slow.** Both notebooks were first executed in
   parallel. They competed for 4 CPU cores and were still training after
   21 minutes (about 5.5 minutes each alone), close to the 30 minute cell
   timeout. Fix: stopped both runs and executed them one after the other.
5. **Scripts notebook results differ slightly.** In that notebook the
   `__main__` checks of `load_data.py` and `model.py` both draw a batch from
   the shuffled training loader, which advances the random state before
   training. The shuffle order changes, so the final numbers differ a little
   (valid 0.8906 vs 0.8788). This is expected and does not indicate an
   error.

## Environment

The session ran in a Linux cloud container (CPU only, no GPU), not on the
local Windows machine. The code selects `cuda` or `mps` automatically when
available.
