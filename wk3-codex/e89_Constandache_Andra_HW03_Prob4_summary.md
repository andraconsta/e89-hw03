# CSCI E-89 Assignment 03: Problem 4 Session Summary

- **Student:** Andra Constandache
- **Tool:** OpenAI Codex
- **Repository:** https://github.com/andraconsta/e89-hw03/tree/main/wk3-codex
- **Goal:** Rebuild Problem 4 from the Geron PyTorch template in an isolated workspace, then publish the scripts, executed notebook, HTML export, and this summary in the existing assignment repository.
- **Interpreter:** `C:\Users\const\venvs\csci-e89-gpu\Scripts\python.exe` (CUDA-enabled PyTorch; NVIDIA GeForce RTX 3050 Ti Laptop GPU).

## Work completed

1. **Setup:** Built the solution in a separate workspace containing the Geron notebook and dataset. Confirmed torch, torchvision, torchmetrics, Jupyter, and CUDA with the selected interpreter. After the initial isolated-repository path proved unsuitable, placed the final deliverables in the existing repository at `wk3-codex`.
2. **Data loading:** Wrote `data.py` from the Geron template section with torchvision transforms v2, seeded 55,000/5,000 split, batch-size-32 loaders, and a data inspection main block. Observed 55,000 train, 5,000 validation, 10,000 test and batch shape `[32, 1, 28, 28]`.
3. **Model:** Wrote `classifier.py` with the template 784-300-100-10 MLP, ReLU, and cuda/mps/cpu fallback. Observed 266,610 trainable parameters, logits `[32, 10]`, and initial cross-entropy loss 2.3156.
4. **Training:** Wrote `training.py` using Geron's `evaluate_tm` and `train2`, SGD lr=0.1, multiclass accuracy, and 20 epochs. The executed notebook finished at train accuracy 0.9284 and validation accuracy 0.8734.
5. **Plot:** Wrote `accuracy_plot.py` to plot all 20 training-accuracy values with axis labels, title, grid, legend, and final-value annotation; it saves `accuracy_plot.png`.
6. **Prediction:** Wrote `inference.py` to load the trained weights and display the first three validation predictions with confidence and truth. The executed notebook predicted Sneaker (85.9%), Coat (99.6%), and Pullover (80.9%); all three were correct.
7. **Review:** Compared the scripts with the template for split, seeds, batch size, architecture, SGD/lr=0.1, epoch count, and accuracy metric. These match. Run order: `data.py` → `classifier.py` → `training.py` → `accuracy_plot.py` → `inference.py`.
8. **Errors/corrections:** The command-line plot display used a non-interactive backend; the figure saved successfully, and the notebook displays it inline. An initial push to an isolated private repository returned `Repository not found`; the final deliverables were pushed to the existing `andraconsta/e89-hw03` repository under `wk3-codex`.
9. **Notebook/HTML:** Built and executed the requested notebook in the CUDA environment, then exported it to HTML. The five code cells have sequential execution counts 1–5 and no errors; the notebook shows all 20 epoch lines, the accuracy plot, and the three predictions with their figure.

## Key results

- Split: 55,000 / 5,000 / 10,000; batch size 32.
- Parameters: 266,610; initial loss: 2.3156.
- Final training accuracy: 0.9284; final validation accuracy: 0.8734.
- Predictions: Sneaker, Coat, Pullover; all three correct.
- Training is stochastic; the executed notebook output is the authoritative run record.
