# EEG-Emotion-MLP

> Predicting Arousal and Valence from Single-Channel EEG using a Multilayer Perceptron

---

## Overview

This repository provides a PyTorch implementation of a multilayer perceptron (MLP) for predicting two key affective dimensions—**arousal** and **valence**—from hand-crafted EEG features. The project is inspired by the DEAP dataset and a VR-based emotion recall experiment.

---

## Features

- Input: 29-dimensional EEG feature vectors (customizable)
- Model: 2-layer MLP with BatchNorm, ReLU, and Dropout
- Tasks: Predict continuous arousal and valence values
- Training and validation scripts included
- Evaluation metrics: MSE, MAE, Correlation Coefficient
- Easy to adapt to other datasets

---

## File Structure

.
├── model.py # MLP network definition
├── train.py # Training and validation workflow
├── data/ # Example EEG feature files (CSV)
├── utils.py # Helper functions (metrics, etc.)
├── requirements.txt # Python dependencies
├── README.md
└── LICENSE


---

## Data Format

Prepare your EEG data as a CSV file with the following columns:
- `id`: Subject ID
- `t_0, t_1, ..., t_28`: 29 EEG feature columns
- `Valence`: Self-reported valence score
- `Arousal`: Self-reported arousal score

Example:

| id | t_0 | ... | t_28 | Valence | Arousal |
|----|-----|-----|------|---------|---------|
|  1 | ... | ... | ...  |  6.0    |  7.0    |

---

## Getting Started

1. **Clone this repository**
    ```bash
    git clone https://github.com/yourusername/EEG-Emotion-MLP.git
    cd EEG-Emotion-MLP
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare your EEG data**
    - Format as described above and place it in the `data/` folder.

4. **Train the model**
    ```bash
    python train.py
    ```

5. **Evaluate performance**
    - The training script will output validation metrics: MSE, MAE, and correlation coefficient (CC).

---

## Model Architecture

- **Input:** 29-dimensional EEG features
- **Hidden Layer 1:** Linear(29, 128) + BatchNorm1d + ReLU + Dropout(0.2)
- **Hidden Layer 2:** Linear(128, 128) + ReLU + Dropout(0.2)
- **Output:** Linear(128, 1) (regression for either valence or arousal)

---

## Results

| Metric | Typical Value (Valence) | Typical Value (Arousal) |
|--------|------------------------|------------------------|
| MSE    | 0.19                   | 0.22                   |
| MAE    | 0.33                   | 0.35                   |
| CC     | 0.60–0.68              | 0.56–0.65              |

*Note: Results may vary depending on dataset and experiment settings.*

---

## Citation

If you use this repository for your research, please cite:

@misc{he2025eegemotionmlp,
author = {Gabriel He},
title = {EEG-Emotion-MLP: Predicting Arousal and Valence from Single-Channel EEG using an MLP},
year = {2025},
url = {https://github.com/yourusername/EEG-Emotion-MLP}
}


---

## Acknowledgements

- Inspired by the DEAP dataset and GUC 2025 course project.
- Thanks to all contributors and instructors.

---

## License

MIT License

---
