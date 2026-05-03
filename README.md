# Deep Learning-Based Receiver for Low-Resolution Massive MIMO

## Overview
This repository contains the Python codes and research implementation for my Bachelor's thesis, which explores the application of Deep Learning (DL) for non-coherent signal detection in Massive Multiple-Input Multiple-Output (MaMIMO) uplink systems. 

The primary challenge addressed in this project is reliable signal detection under severe hardware limitations—specifically, when the receiver frontend is restricted to low-resolution (1-bit) quantized data. By replacing traditional analytical models with neural networks, this implementation demonstrates competitive Symbol Error Rate (SER) performance, particularly in multi-user environments with high interference.

## Neural Network Architecture & Approach
* **Per-Symbol Parallel Prediction:** Instead of predicting entire symbol blocks jointly (which scales poorly), the architecture processes each symbol in parallel using dedicated output layers (Softmax activation), drastically reducing computational complexity.
* **Feature Extraction:** Extracts spatial correlation matrices from the quantized received signals to capture spatial dependencies across $100$ receive antennas.
* **Multi-User Interference Mitigation:** The model implicitly learns to mitigate interference across multiple users ($N_u = 3$) without requiring explicit interference cancellation algorithms.

## Quantization & Data Strategies
The implementation explores how different data training strategies affect neural network convergence under hardware constraints:
* **Quantization Models:** Supports strict 1-bit quantization (sign function) and a tunable hyperbolic tangent (tanh) quantization that acts as a smoother approximation during training.
* **Data Generation:** Evaluates the trade-offs between offline pre-generated datasets (which favor memorization and stability) versus on-the-fly data generation during training (which favors generalization).

## Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Scientific Computing:** NumPy, SciPy (for MATLAB `.mat` compatibility)
* **Optimization:** Adam Optimizer with Categorical Cross-Entropy Loss

## Results
The Deep Learning Multiple-Symbol Differential Detector (DL-MSDD) successfully achieves competitive, and often superior, SER performance compared to conventional Decision-Feedback Differential Detection (DFDD), especially in lower SNR regimes and multi-user environments.
