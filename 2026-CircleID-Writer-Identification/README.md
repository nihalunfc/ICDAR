# ICDAR 2026 - CircleID: Writer Identification

This project is part of the **ICDAR 2026 CircleID Competition**, which focuses on identifying the writer of a hand-drawn circle from a scanned image. 

## Project Overview
While drawing a circle seems simple, the subtle motor behaviors, pressure, and stroke characteristics of the writer provide unique cues for identification. This competition specifically addresses **Writer Identification** with an open-set constraint: identifying known writers and flagging unknown writers.

### The Challenge
* **Writer Identification:** Predict the correct `writer_id` for known individuals.
* **Open-Set Recognition:** The test set contains writers not present in the training data. These must be predicted as `-1`.
* **Evaluation Metric:** Top-1 Accuracy on the full test set (known + unknown).

## Repository Structure
The project is organized as follows:

- `data/`: Contains competition CSV files (`train.csv`, `test.csv`).
- `notebooks/`: Jupyter/Kaggle notebooks for exploratory data analysis (EDA) and model training.
- `src/`: Python scripts for modularized code, including data loaders and model architectures.
- `models/`: Directory to save trained model weights (`.pth` or `.h5`).
- `submissions/`: Stores generated `submission.csv` files.

> **Note:** The image dataset consists of 40,000+ PNG files. Due to size constraints, these images are stored locally and are excluded from this repository via `.gitignore`.

## Methodology
The current approach utilizes a Deep Learning pipeline:
1.  **Backbone:** CNN-based architectures (e.g., ResNet, EfficientNet) to capture fine-grained ink and stroke features.
2.  **Training:** Supervised learning on known writers from the training set.
3.  **Inference Logic:** To handle unknown writers, a confidence threshold is applied to the final Softmax layer. If the maximum confidence is below a set threshold, the writer is classified as `-1`.
4.  **Augmentations:** Given the nature of circles, random rotations and subtle scaling are used to improve model robustness.

## Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install torch torchvision pandas scikit-learn tqdm
    ```
2.  **Data Placement:** Ensure competition metadata files are placed in the `data/` folder.
3.  **Run Training:** Scripts within `src/` or notebooks in `notebooks/` can be used to train the model and generate predictions.

## References
- Competition Page: [ICDAR 2026 - CircleID: Writer Identification](https://www.kaggle.com/competitions/icdar-2026-circleid-writer-identification)
- Citation: *Thomas Gorges. ICDAR 2026 - CircleID: Writer Identification. Kaggle, 2026.*
