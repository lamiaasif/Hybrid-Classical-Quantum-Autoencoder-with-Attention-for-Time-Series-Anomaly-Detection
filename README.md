# Hybrid Classical-Quantum Autoencoder for Time Series Anomaly Detection

**Authors:** Fatima Farhan · Lamia Asif · Najaah Noor — FAST-NUCES Karachi

---

## Results

| Method | AUC-ROC | F1 | Parameters |
|---|---|---|---|
| Classical LSTM-AE | 0.2566 | 0.3871 | 52,828 |
| QAE (Frehner & Stockinger) | 0.7140 | 0.7654 | 32 |
| **Hybrid Attn-QAE (ours)** | **0.8192** | **0.8095** | **72** |

735× fewer parameters than the classical baseline with a 14.8% AUC improvement over the base QAE.

---

## What's in the notebook

`code.ipynb` is a single self-contained Colab notebook that runs the full pipeline end-to-end:

**1. Setup & Data Generation**
- Installs PennyLane, PyTorch, scikit-learn
- Generates a synthetic ECG-like dataset (3,000 normal + 400 anomalous, L=140 time steps)
- Four anomaly subtypes: missing R-peak, irregular rhythm, flatline segment, noise burst
- Min-max normalization and train/test split

**2. Classical LSTM-AE Baseline**
- 2-layer LSTM encoder-decoder (52,828 params)
- Trained on 2,400 normal samples for 30 epochs
- Anomaly score = reconstruction MSE

**3. Base QAE Replication**
- Replicates Frehner & Stockinger (arXiv:2410.04154)
- 8-qubit HEA with RY encoding + linear CNOT chain (32 params)
- SWAP-test fidelity as anomaly score
- Trained on 150 samples (simulation constraint)

**4. Hybrid Attention-QAE (proposed)**
- Dual-axis encoding: RY + RZ + IsingZZ gates
- Alternating CZ/CNOT ansatz across 3 layers (72 params)
- Ensemble score: 0.6 × SWAP fidelity + 0.4 × latent qubit entropy
- Same 150-sample training budget as the base QAE

**5. Evaluation & Plots**
- ROC curves for all three methods
- Side-by-side metric bar charts (AUC, F1, Precision, Recall)
- Parameter count comparison (log scale)
- Training loss convergence curves
- Confusion matrices
- Score distribution histogram with Youden threshold
- PCA of latent space (QAE base vs. Hybrid)

---

## Quickstart

Open in Google Colab and run all cells — everything is self-contained.

```
Runtime → Run all
```

Or locally:

```bash
pip install pennylane torch scikit-learn matplotlib seaborn
jupyter notebook quantum_project.ipynb
```

**Expected runtimes on Colab T4:**

| Model | Time |
|---|---|
| Classical LSTM-AE | ~7 s |
| QAE Base | ~109 s |
| Hybrid Attn-QAE | ~174 s |

The bottleneck is classical simulation of quantum circuits — real hardware would be orders of magnitude faster.

---

## File structure

```
├── quantum_project.ipynb   # full implementation
└── paper.pdf               # written report
```

---

## Dependencies

```
pennylane
pennylane-lightning
torch
numpy
scikit-learn
matplotlib
seaborn
```

---

## Reference

Base QAE this work extends:
> R. Frehner and K. Stockinger, "Applying quantum autoencoders for time series anomaly detection," arXiv:2410.04154, 2024.
