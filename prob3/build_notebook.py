"""Step 6: assemble the 5 scripts into one notebook.

CSCI E-89 Deep Learning, Assignment 03, Problem 3 (Andra Constandache).
Writes e89_Constandache_Andra_HW03_Prob3_scripts.ipynb: a title cell, then
for each script a "Cell N" markdown cell and a code cell with its code.
Imports between the scripts are commented out, because in a notebook every
name defined by an earlier cell is already available.
"""

import re

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = "e89_Constandache_Andra_HW03_Prob3_scripts.ipynb"

# The scripts in execution order, with a short description of each.
SCRIPTS = [
    ("load_data.py", "Load FashionMNIST with `transforms.v2`, split it "
     "55,000/5,000 into train/valid with seed 42, and build DataLoaders "
     "(batch size 32)."),
    ("model.py", "Pick the device and define the `ImageClassifier` MLP "
     "(784 -> 300 -> 100 -> 10) with `CrossEntropyLoss`."),
    ("train.py", "Define `evaluate_tm` and `train2`, train for 20 epochs with "
     "SGD (lr 0.1) and torchmetrics accuracy, save the weights and history."),
    ("plot_accuracy.py", "Plot the training accuracy per epoch and save "
     "`training_accuracy.png`."),
    ("predict.py", "Load the saved weights and predict 3 validation images."),
]
MODULES = {name[:-3] for name, _ in SCRIPTS}

# Matches "from load_data import ..." / "import model" for our own scripts.
LOCAL_IMPORT = re.compile(
    r"^(\s*)(from\s+(\w+)\s+import\s+.*|import\s+(\w+).*)$")


def comment_local_imports(code):
    """Comment out imports of the other scripts in this folder."""
    lines = []
    for line in code.splitlines():
        m = LOCAL_IMPORT.match(line)
        if m and (m.group(3) in MODULES or m.group(4) in MODULES):
            line = f"{m.group(1)}# {m.group(2)}  (defined in an earlier cell)"
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


# Title cell: name, assignment, problem, and a table mapping cells to scripts.
table = "\n".join(f"| Cell {i} | `{name}` | {desc} |"
                  for i, (name, desc) in enumerate(SCRIPTS, start=1))
title = (
    "# CSCI E-89 Deep Learning: Assignment 03, Problem 3\n\n"
    "**Andra Constandache**\n\n"
    "FashionMNIST image classifier in PyTorch, following \"Building an Image "
    "Classifier with PyTorch\" (Geron, *Hands-On Machine Learning*, ch. 10). "
    "Each code cell below is one of the step scripts in this folder, in "
    "execution order. Imports of earlier scripts are commented out, so the "
    "notebook runs top to bottom on its own.\n\n"
    "| Cell | Script | Purpose |\n|---|---|---|\n" + table + "\n")

cells = [new_markdown_cell(title)]
for i, (name, desc) in enumerate(SCRIPTS, start=1):
    with open(name) as f:
        code = f.read()
    cells.append(new_markdown_cell(f"## Cell {i}: `{name}`\n\n{desc}"))
    cells.append(new_code_cell(comment_local_imports(code)))

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
nbformat.write(nb, OUTPUT)
print(f"Wrote {OUTPUT} ({len(cells)} cells)")
