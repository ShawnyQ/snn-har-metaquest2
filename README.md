# 🧠 Spiking Neural Networks for Human Activity Recognition using Meta Quest 2

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![snntorch](https://img.shields.io/badge/snntorch-0.6.0-orange)](https://snntorch.readthedocs.io)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

This repository contains the full code and research implementation of my **Master’s Thesis**: a biologically inspired approach to **Human Activity Recognition (HAR)** using **Spiking Neural Networks (SNNs)** trained on motion data from the **Meta Quest 2** headset.

The system classifies **repetitive strength exercises** such as **bicep curls**, **bench press**, and **shoulder press**, using only raw accelerometer signals and spike-encoded inputs.

---

## 🎓 Project Overview

This project explores the use of SNNs to perform motion classification by leveraging the temporal structure of movement data. The Meta Quest 2’s built-in sensors capture acceleration signals during exercise reps, which are then spike-encoded and passed through a custom-built Leaky Integrate-and-Fire (LIF) neural architecture.

The goal is to evaluate whether **energy-efficient SNNs** can serve as a viable alternative to conventional deep learning models in **AR/VR fitness and health tracking applications**.

---

## 🧠 Why Spiking Neural Networks?

Spiking Neural Networks differ from traditional ANNs by simulating biological neurons that only fire when their membrane potential exceeds a threshold. This enables:

- ⚡ **Energy-efficient computation**  
- ⏱️ **Temporal pattern recognition**  
- 📱 **Edge deployment on neuromorphic or wearable devices**

---

## ❓ Problem Statement

Can SNNs accurately classify three types of strength-based exercises — **bicep curls**, **bench press**, and **shoulder press** — using only accelerometer signals from the Meta Quest 2?

---

## 🔍 Methodology

### 📦 Data Collection
- 3 exercises × 10 repetitions each = 30 total segments  
- Captured using Meta Quest 2’s IMU at ~72 Hz  

### 🧹 Preprocessing
- Removal of head position signals  
- Outlier trimming on first/last 100 samples  
- Resampling to uniform time base  
- Range calculation per axis  
- Windowing into 116 time-step segments  

### ⚡ Spike Encoding
- **Rate Encoding** based on positional range  
- Converted into 0/1 spike trains over time  

### 🧠 SNN Model Architecture
- 4 fully connected layers  
- Leaky Integrate-and-Fire (LIF) neurons  
- Custom **Triangle surrogate gradient**  
- Dropout + BatchNorm regularization  
- Temporal mean pooling over spike outputs  

### 🔁 Evaluation Strategy
- **Leave-Two-Repetitions-Out 5-Fold Cross-Validation**  
- Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix  

---

## 📊 Results Summary

| Metric       | Mean     | Std Dev  |
|--------------|----------|----------|
| **Accuracy** | 93.33%   | ±8.16%   |
| **Precision**| 95.56%   | ±5.44%   |
| **Recall**   | 93.33%   | ±8.16%   |
| **F1 Score** | 92.89%   | ±8.71%   |

- ✅ Perfect accuracy in 3 out of 5 folds  
- ⚠️ Minor misclassifications in **Shoulder Press** during final 2 folds  
- 🎯 Model generalized well to **unseen repetitions**  

---

## 🧪 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/snn-har-metaquest2.git
   cd snn-har-metaquest2
2. **Install dependencies (Requires Python 3.10+)**
   ```bash
   pip install -r requirements.txt
3. **Run the full training and evaluation pipeline**
   ```bash
   python run_crossval.py
4. **Alternatively, explore the project in Jupyter Notebook**
   ```bash
   notebooks/snn_har_pipeline.ipynb
snn-har-metaquest2/
 ┣ data/            # Raw & processed sensor data
 ┣ notebooks/       # Full Jupyter pipeline
 ┣ src/             # Custom SNN architecture & training code
 ┣ results/         # Metrics, logs, confusion matrices
 ┣ run_crossval.py  # Training & evaluation entry point
 ┣ requirements.txt # Environment dependencies
 ┗ README.md

 ---

### 📄 License
- This project is licensed under the MIT License.
- You are free to use, modify, and distribute it with proper attribution.
