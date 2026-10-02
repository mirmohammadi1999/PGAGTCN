# PG-AGTCN: Physics-Guided Adaptive Gated TCN for Data-Efficient Nonlinear Self-Interference Cancellation

This repository contains the official implementation of **PG-AGTCN**, a lightweight and data-efficient deep learning framework for nonlinear self-interference cancellation (SIC) in in-band full-duplex wireless systems.

PG-AGTCN combines a physics-guided sparse polynomial model with an **Adaptive Gated Temporal Convolutional Network (TCN)** to capture nonlinear memory effects while significantly reducing the required training data and model complexity.

## 🏛️ Model Architecture

PG-AGTCN follows a hybrid physics-guided and data-driven design. The framework first identifies the dominant nonlinear components using **Lasso-based sparse polynomial extraction** followed by **OLS debiasing**. Taking the original transmitted signal as input, a lightweight Adaptive Gated TCN is then trained to predict and compensate for the remaining nonlinear residual. A DC-aware loss function is additionally used to improve the suppression of DC offset and carrier leakage (driving the DC-bin power down to -99.57 dBm, virtually reaching the noise floor).

## 🏆 Key Results

PG-AGTCN is evaluated on two hardware-measured full-duplex datasets under different nonlinear operating conditions.

| **Method**              | **Dataset 1 (10 dBm)** | **Dataset 2 (32 dBm)** |
| ----------------------- | ---------------------- | ---------------------- |
| Sparse Polynomial Only  | 43.90 dB               | 30.60 dB               |
| Sparse + TCN (No Gate)  | 43.73 dB               | 35.69 dB               |
| **PG-AGTCN (Proposed)** | **45.00 dB**           | **38.30 dB**           |

Under a strict training constraint of only **5,000 samples**, the proposed model achieves:

- **45.0 dB** SIC on Dataset 1
- **38.3 dB** SIC on Dataset 2
- Only **1,779** trainable parameters
- Approximately **~3.4k FLOPs/sample** (inference complexity)

## ⚠️ Data-Efficient Nonlinear SIC

A key objective of PG-AGTCN is achieving high nonlinear cancellation performance with limited training data. The model uses a maximum of **5,000 samples**, divided into:

- 80% Training
- 10% Validation
- 10% Testing

Despite the limited amount of training data, PG-AGTCN maintains strong SIC performance, particularly under the severe nonlinear conditions of Dataset 2. Notably:
- It maintains a highly competitive 32.6 dB of cancellation with as few as 2,000 total samples, a regime where standard deep learning models typically collapse.

## 🚀 Quick Start

The complete evaluation can be reproduced directly in **Google Colab**. Simply open a new [Google Colab](https://colab.research.google.com/) notebook, copy the following code into **one single cell**, and run it:

```bash
# 1. Clone the repository
!git clone https://github.com/mirmohammadi1999/PGAGTCN.git

# 2. Navigate to the directory
%cd PGAGTCN

# 3. Run the evaluation pipeline
!python PGAGTCN_Cancellation.py
```

> **Note:** The complete evaluation pipeline, including physical parameter extraction, adaptive training, and PSD plotting, is automatically executed by the main script.

## 📝 Citation

If you find this repository or our proposed **PG-AGTCN** architecture useful in your research, please consider citing our paper:

> **Physics-Guided Adaptive Gated TCN for Data-Efficient Nonlinear Self-Interference Cancellation**
>
> *(Full citation details will be provided upon publication.)*

## 📜 License

This project is released under the **MIT License**.
