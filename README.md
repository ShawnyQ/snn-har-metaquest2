# 🧠 Spiking Neural Networks for Human Activity Recognition using Meta Quest 2

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![snntorch](https://img.shields.io/badge/snntorch-0.6.0-orange)](https://snntorch.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

This project implements a full spike-based learning pipeline for classifying gym exercises using **motion data collected from the Meta Quest 2’s embedded IMU sensors**. The model uses a biologically inspired **Spiking Neural Network (SNN)** with Leaky Integrate-and-Fire (LIF) neurons to classify **bicep curls**, **bench press**, and **shoulder press** movements.

This work was completed as part of my **Master’s Thesis** on low-power, neuromorphic Human Activity Recognition (HAR) systems.

---

## 🚀 Project Highlights

- ✅ Raw motion data captured from Meta Quest 2 (accelerometer-based)
- 🧹 Preprocessing: outlier removal, signal segmentation, temporal alignment
- ⚡ Spike Encoding: Rate Encoding → Binary spike train (116 time steps)
- 🧠 SNN Architecture:
  - 3 hidden layers with LIF neurons
  - Custom triangle surrogate gradient for training
  - Dropout + BatchNorm for regularization
- 🔁 5-Fold **Leave-Two-Repetitions-Out Cross-Validation**

---

## 📊 Results Summary

| Metric       | Mean     | Std Dev  |
|--------------|----------|----------|
| **Accuracy** | 93.33%   | ±8.16%   |
| **Precision**| 95.56%   | ±5.44%   |
| **Recall**   | 93.33%   | ±8.16%   |
| **F1 Score** | 92.89%   | ±8.71%   |

- ✅ 3 out of 5 folds achieved **100% classification accuracy**
- ⚠️ Minor misclassifications occurred only in the **Shoulder Press** class
- 💡 Model generalizes well across **unseen repetitions**

---

## 🧠 SNN Model Overview

```text
Input: [batch, 116 time steps, 6 spike features]
 → Linear → BatchNorm → LIF → Dropout
 → Linear → BatchNorm → LIF → Dropout
 → Linear → BatchNorm → LIF → Dropout
 → Linear → LIF
 → Mean over time → Softmax
