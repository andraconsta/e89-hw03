# Dialog Summary: CSCI E-89 HW03, Problem 1

**Student:** Andra Constandache
**Tool:** Claude Code (desktop app, Code tab)
**Repository:** https://github.com/andraconsta/e89-hw03

## Goal

Use Claude Code to build, step by step, the FashionMNIST image classifier from
the "Building an Image Classifier with PyTorch" section of Geron's chapter 10
notebook (`10_neural_nets_with_pytorch.ipynb`), add a plot of the training
accuracy, and keep every generated script in the GitHub repository.

## Setup

- **Connecting to the repo:** I asked Claude to connect to
  `andraconsta/e89-hw03`. It first set up git in the wrong local folder. After
  I pointed it to `C:\Users\const\OneDrive\Desktop\CSCI E-89\WK3`, it undid the
  stray setup, turned WK3 into a git repository, and added
  `https://github.com/andraconsta/e89-hw03.git` as `origin`.
- **`.gitignore`:** keeps the downloaded dataset, the course materials, earlier
  HW03 drafts, and training outputs (`*.pt`, `history.json`) out of the repo.
- **Environment:** Anaconda base Python with torch 2.14.0 (CPU) and torchvision
  0.29.0. Claude installed `torchmetrics` 1.9.0 when the training step needed
  it.

## Step 1: `load_data.py` (loading and splitting the data)

- **First request:** load FashionMNIST with `torchvision.transforms.v2`
  (`ToImage` and `ToDtype(float32, scale=True)`), split the training set
  55,000/5,000 into train and validation with `torch.manual_seed(42)`, and
  build DataLoaders with batch size 32.
- **Revision 1:** I asked for comments explaining each step, and Claude added
  them.
- **Revision 2:** I asked Claude to follow the Geron section. It compared the
  script with the notebook and aligned it:
  - It now uses the notebook's names (`toTensor`, `train_and_valid_data`,
    `train_data`, `valid_data`, `test_data`).
  - It sets the seed a second time before building the DataLoaders, so the
    shuffle order is reproducible.
  - It inspects one sample, the way Geron does.
- **Result:** 55,000 train, 5,000 valid and 10,000 test images. Each batch is
  `[32, 1, 28, 28]` float32. The first sample is an "Ankle boot".

## Step 2: `model.py` (the classifier)

- **The network:** Geron's `ImageClassifier`, built as Flatten, Linear
  784→300, ReLU, Linear 300→100, ReLU, Linear 100→10, with `CrossEntropyLoss`
  as the loss. It picks a device (cuda, mps or cpu); here it's the CPU.
- **Sanity check:** 266,610 parameters, and the output for one batch is
  `[32, 10]`. The untrained loss is 2.3156, close to ln(10) ≈ 2.30, which is
  what random guessing over 10 classes gives.

## Step 3: `train.py` (training)

- **Code:** Geron's `evaluate_tm` and `train2` functions, training with SGD at
  learning rate 0.1 for 20 epochs and tracking `torchmetrics` multiclass
  accuracy.
- **Saved outputs:** the trained weights go to `fashion_mnist_mlp.pt` and the
  per-epoch history to `history.json`, so later steps don't have to retrain.
- **Result:** after about 4 min 45 s on CPU, training accuracy went from 0.781
  to 0.928 and training loss from 0.606 to 0.188.
- **Validation:** validation accuracy peaked at 0.888 in epoch 17 and ended at
  0.878. It levels off after about epoch 11 while training accuracy keeps
  rising, so the model is starting to overfit slightly.

## Step 4: `plot_accuracy.py` (the plot the assignment asks for)

- **What it does:** reads `history.json`, plots training accuracy for epochs 1
  to 20, labels the final value, and saves `training_accuracy.png`.
- **The curve:** accuracy rises sharply from epoch 1 to 2, then more slowly,
  and passes 0.90 at epoch 9.

## Step 5: `predict.py` (predictions)

- **What it does:** loads the saved weights and predicts the first 3
  validation images the way Geron does (`model.eval()`, `no_grad`, `argmax`).
  It shows the class names, the confidence (from softmax) and the true labels.
- **Result:** all 3 are correct: Sneaker (95.9%), Coat (99.7%) and Pullover
  (83.2%).

## Step 6: Commit and push

- **Pushed:** all scripts, `training_accuracy.png` and `.gitignore` are on
  `main` in the repository, with one commit per step or revision.
- **Order to run them:** `load_data.py`, `model.py`, `train.py`,
  `plot_accuracy.py`, `predict.py`.

## Step 7: Summary and notebook

- **Summary:** I asked Claude to summarize the dialog and save it as
  `summary.md` (this file), then pushed it to the repository.
- **Notebook:** I downloaded the repository as a ZIP and unzipped it. Claude
  then wrote `build_notebook.py`, which assembles the scripts into
  `e89_Constandache_Andra_HW03_Prob1.ipynb`: one labeled code cell per script,
  in executable order, each preceded by a markdown cell naming the script.
