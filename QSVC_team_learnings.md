# Quantum Kernels for Tumor Detection — Team Learnings

**Project:** Can a quantum circuit spot a tumor? A hybrid quantum-classical classifier (Qiskit `QSVC`) on the Wisconsin Diagnostic Breast Cancer dataset, benchmarked fairly against strong classical models.

**Files:**
- `qsvc_breast_cancer.py`: runs everything (about 2.5 min on a laptop). Writes 12 figures and `results.json` to `outputs/`.
- `QSVC_breast_cancer_explained.md`: detailed write-up and judge Q&A.
- This file: a summary of what we learned.

---

## 1. TL;DR

- **What we built:** each patient's 30 cell measurements are compressed to 8 numbers. Those 8 numbers are encoded into an 8-qubit quantum state. The overlap between two patients' states is used as their similarity score, and a standard SVM classifies on those scores.
- **Result:** the quantum kernel is **competitive but not better**.
  - On the test set it catches **52 of 53 cancers** (recall 0.981, F1 0.954).
  - A classical RBF SVM on the *same 8 numbers* catches 50 of 53. But across repeated tests the gap is not statistically significant.
  - The best classical models using all 30 measurements still win (poly-3 SVM: F1 0.990).
- **Surprises:**
  - The best quantum circuit used **no entanglement**, so a classical computer can compute it easily.
  - On real hardware it would need **about 30,000 measurements per patient pair** to stay accurate.
- **Main value:** a rigorous, fair method for testing whether quantum similarity helps. The answer here was measured, not assumed.

---

## 2. The problem in one paragraph

An SVM classifies a new patient by comparing them with past patients and taking a weighted vote of the most similar borderline cases. The function that scores "how similar are two patients?" is called the **kernel**. A classical kernel uses a formula on the raw numbers. A **quantum kernel** turns each patient into a quantum state and uses the **overlap of two states** as the similarity. `QSVC` is scikit-learn's `SVC` with the kernel swapped for the quantum one. The `.fit`, `.predict` and `.score` calls are unchanged.

---

## 3. End-to-end architecture

```
┌──────────────── CLASSICAL ────────────────┐   ┌──── QUANTUM ────┐   ┌────── CLASSICAL ──────┐
30 measurements → standardize → PCA(8) → angles → circuit → state → similarity matrix → SVM → diagnosis
                                         ×λ                         (kernel)          (QSVC)
```

| Layer | What happens | Runs on |
|---|---|---|
| Data | 569 patients, 30 features; 426 train / 143 test (stratified, seed 42) | CPU |
| Classical encoder | `StandardScaler` → `PCA(8)` → `MinMaxScaler(0, π)` → × bandwidth λ. Fit on training data only. | CPU |
| **Quantum kernel** | 8 angles → 8-qubit circuit → quantum state; similarity of 2 patients = \|⟨ψ(a)\|ψ(b)⟩\|² | Simulated QPU |
| Classical learner | The SVM picks support vectors (41 borderline patients) and weights from the kernel matrix | CPU |
| Harness | Model search, fair baselines, significance tests, diagnostics, plots | CPU |

**Only the quantum kernel layer changes when moving from simulator to real hardware.**

---

## 4. The quantum kernel, layer by layer

Based on the `qiskit-machine-learning` 0.9.1 source (`FidelityStatevectorKernel`).

```
L7  CONSUMER        QSVC: SVC.kernel = quantum_kernel.evaluate → libsvm optimizer
L6  INTERFACE       evaluate(x, y): check 8 features == 8 circuit parameters
L5  ASSEMBLY        loop over all pairs → matrix K; repair to a valid (PSD) matrix if shots are used
L4  FIDELITY        k = |⟨ψ(a)|ψ(b)⟩|²  (exact)  or  Binomial(shots, k)/shots  (noisy)
L3  EXECUTION       Statevector(circuit): 2⁸ = 256 amplitudes; cached once per patient
L2  BINDING         plug this patient's 8 angles into the circuit template
L1  FEATURE MAP     circuit template: Z map (H + phase rotations) or ZZ map (+ CNOT entanglement)
L0  INPUT           8 PCA scores → angles in [0, π] × bandwidth λ
```

**Selected circuit (Z map, reps = 2):** for each of the 8 qubits, apply `H → P(2·x) → H → P(2·x)`. Depth 4, zero two-qubit gates.

**Tuning knobs and what they control:**

| Knob | Chosen | Why it matters |
|---|---|---|
| Bandwidth λ | 0.2 | Too large: every pair of states looks unrelated (all k ≈ 0). Too small: every pair looks identical (all k ≈ 1). This is the most important knob. |
| Feature map | Z (tried Z, ZZ, Y+ZZ) | Sets the "shape" of similarity. Entangling maps capture feature interactions but are noisier. |
| Reps | 2 (tried 1, 2) | Circuit depth; more expressive but more noise |
| Shots | exact (sweep: 32 … 32,768) | Number of measurements per entry on hardware; too few gives a noisy kernel |
| SVM C | 100 | Penalty for misclassifying training patients; too high overfits, too low underfits |

**Simulator vs hardware:** a simulator computes each patient's state once (569 simulations) and gets overlaps exactly. Hardware uses **compute-uncompute**: run U(a) followed by the inverse of U(b), then measure. The fraction of shots that return all zeros *is* the similarity. That's about 90k circuits for the training matrix alone, each with many shots.

**Why the Z map is "quantum but classically easy":** with no entanglement, each qubit evolves independently. The similarity is then a product of 8 single-qubit overlaps, which a laptop computes cheaply. Only entangling maps can produce similarity that is hard to compute classically.

---

## 5. Feature engineering: PCA and "why 8"

### How PCA was computed (on the 426 training patients only)

1. **Standardize** each of the 30 measurements to mean 0, variance 1. Otherwise `mean area` (std 345) would dominate `mean smoothness` (std 0.013). After this step, total variance = 30.
2. **Covariance matrix** (30×30) captures which measurements move together. For example, radius and perimeter have correlation 1.000.
3. **Eigen-decomposition** gives 30 directions (the principal components), each with an **eigenvalue** equal to the variance along that direction.
4. **Explained variance ratio** = eigenvalue ÷ sum of all eigenvalues. For example, PC1 = 13.20 / 30.07 = **43.9%**.
5. **Keep the top 8** and project each patient onto them. Each score is a weighted sum of all 30 z-scores.

This was verified by hand with numpy against scikit-learn: identical ratios and identical directions.

### The 8 features are not 8 original measurements

Each component is a mix of **all 30** measurements:

| Qubit | PC | Variance | Mostly measures | Separates classes? |
|---|---|---|---|---|
| 0 | PC1 | 43.9% | Shape irregularity: concave points, concavity, compactness | **Strongly (1.64)** |
| 1 | PC2 | 19.5% | Boundary roughness: fractal dimension | Weakly (0.37) |
| 2 | PC3 | 9.4% | Variability across the cell: texture, smoothness, symmetry, radius *errors* | Weakly (0.34) |
| 3 | PC4 | 6.8% | Texture | Weakly (0.26) |
| 4 | PC5 | 5.6% | Smoothness against concavity error | Weakly (0.25) |
| 5 | PC6 | 4.0% | Symmetry against smoothness | Barely (0.06) |
| 6 | PC7 | 2.2% | Fractal dimension against concave-points error | Barely (0.16) |
| 7 | PC8 | 1.6% | Smoothness error against texture error | Barely (0.08) |

"Separates classes" = gap between the malignant and benign averages ÷ the component's spread.

### Why 8 qubits

8 was a **design choice made upfront**, not picked by cross-validation. The qubit sweep checked it afterwards.

| Qubits | 2 | 4 | 6 | **8** | 10 | 12 |
|---|---|---|---|---|---|---|
| Cumulative variance kept | 63.4% | 79.6% | 89.2% | **93.0%** | 95.4% | ~97% |
| QSVC CV F1 | 0.932 | 0.957 | **0.978** | 0.972 | 0.974 | 0.968 |
| RBF SVM, same features, CV F1 | 0.936 | 0.960 | 0.971 | 0.971 | 0.975 | 0.968 |

- **Variance:** after 8 components, each extra one adds about 1% or less.
- **Accuracy:** it levels off between 6 and 10 qubits. 6 scored marginally higher, but the gap is within noise (section 6).
- **Cost:** simulation cost doubles with every qubit (30 qubits ≈ 16 GB *per patient state*). More qubits also worsen kernel concentration and need more shots.

**Caveat:** PCA ignores the diagnosis. "Most variance" is not the same as "most predictive": PC6–PC8 rank in the top 8 but barely separate the classes.

---

## 6. Statistics: fold-to-fold noise

**Cross-validation:** split the training set into 5 folds (~86 patients, ~32 malignant each). Train on 4, score on the 5th, rotate, then average.

**Fold-to-fold noise** is how much the score moves just because of *which patients landed in which fold*. With only 32 cancers per fold, **one patient changes F1 by about 0.015**.

**Real example (6 vs 8 qubits):**

| Fold | 1 | 2 | 3 | 4 | 5 | Mean |
|---|---|---|---|---|---|---|
| 6 qubits | 0.968 | 0.969 | 0.984 | **0.984** | 0.984 | 0.978 |
| 8 qubits | 0.968 | 0.969 | 0.984 | **0.954** | 0.984 | 0.972 |

- The entire gap comes from **one fold**, which is about one or two patients.
- Reshuffling the folds 10 times moves each model's mean between 0.961 and 0.978, which is about **3× the gap**.
- 6 beats 8 in only **6 of 10** reshuffles, close to a coin flip.

**Why it matters:**
- A difference smaller than the noise is not a finding.
- That's why we use **5×5 repeated CV with the Nadeau–Bengio corrected t-test** and **Wilcoxon tests** before claiming anything.
- The test set is noisier still: 53 cancers, so one patient ≈ 0.01 F1. **Treat test differences under about 0.02 as ties.**

---

## 7. Results

### Held-out test set (143 patients, 53 malignant)

| Model | Features | Accuracy | F1 (malignant) | Recall (malignant) | AUC | Missed cancers |
|---|---|---|---|---|---|---|
| SVC poly-3 | 30 | **0.993** | **0.990** | 0.981 | **0.999** | 1 |
| SVC linear | 30 | 0.986 | 0.981 | 0.981 | 0.997 | 1 |
| Logistic regression | 30 | 0.986 | 0.981 | 0.981 | 0.998 | 1 |
| SVC RBF (default) | 30 | 0.979 | 0.972 | 0.981 | 0.997 | 1 |
| SVC RBF (tuned) | 30 | 0.965 | 0.954 | 0.981 | 0.997 | 1 |
| **QSVC, Z map** | **8 qubits** | 0.965 | 0.954 | 0.981 | 0.996 | **1** |
| Hybrid (½ quantum + ½ RBF) | 8 | 0.965 | 0.954 | 0.981 | 0.996 | 1 |
| Random forest | 30 | 0.965 | 0.952 | 0.943 | 0.995 | 3 |
| QSVC, ZZ map (entangled) | 8 qubits | 0.958 | 0.945 | 0.981 | 0.994 | 1 |
| SVC RBF, same 8 features | 8 | 0.958 | 0.943 | 0.943 | 0.995 | **3** |

Classical models: five use standard settings, and two RBF SVMs are tuned (on 30 features and on the same 8 features). All tuned SVMs use the same grid-search folds and `class_weight="balanced"`.

### Statistical comparisons (training set, repeated CV)

| Comparison | Mean ΔF1 | p-value | Verdict |
|---|---|---|---|
| QSVC vs RBF, same 8 features | +0.0003 | 0.98 | Tie |
| QSVC vs tuned RBF, 30 features | +0.0072 | 0.45 | Not significant |
| Hybrid vs tuned RBF, 30 features | +0.0085 | 0.40 | Not significant |
| Learning curve, n = 320 (QSVC vs same-feature RBF) | +0.0077 | **0.01** (17/20 wins) | Small but consistent |
| Learning curve, n = 20 | −0.0062 | 0.65 | Quantum slightly worse with tiny data |

### Diagnostics

| Diagnostic | Finding | Meaning |
|---|---|---|
| Kernel concentration | At λ = 1, the spread of kernel values halves from 2 to 12 qubits | Untuned quantum kernels get worse as qubits are added; the bandwidth must be tuned |
| Eigen-spectrum | Quantum kernel is richer than RBF on the same features (effective dimension 1.8 vs 1.2) | Plausibly explains its edge on the same 8 features |
| Kernel-target alignment | Quantum 0.44, RBF same features 0.47, RBF 30 features 0.57 | PCA throws away some label-relevant signal |
| Geometric difference g | 2.4–9.5 (maximum ≈ 20.6) | A classical kernel can largely mimic the quantum one, so there's little room for advantage |
| Shot noise | F1: 0.79 at 32 shots → 0.92 at 8k → 0.94 at 32k | Hardware needs a very large measurement budget |

---

## 8. Key learnings

1. **Fair benchmarking is the whole game.** Always compare the quantum model against a classical model given **the same features** and tuned with **the same effort**. Otherwise any "advantage" is an artifact of the setup.
2. **The bandwidth is not optional.** Quantum kernels suffer *exponential concentration*: without scaling the input angles, every patient looks equally different, and it gets worse as qubits are added.
3. **Entanglement isn't automatically better.** On nearly linearly separable data, the simple product-state (Z) map won. Entangling maps add interactions the labels don't need.
4. **A non-entangled quantum kernel is classically simulable.** Any advantage claim needs entangling circuits *and* a large geometric difference.
5. **Measure the noise before claiming a win.** Fold-to-fold noise here (~0.01 F1) is as large as most gaps we observed.
6. **Shots are the hardware bottleneck.** Small-bandwidth kernels have entries close together, so they need tens of thousands of shots to resolve.
7. **Dataset choice matters.** Wisconsin is almost linearly separable, so simple classical models reach about 99%. That's a hard place for any exotic kernel to show value.
8. **Hybrid design keeps the quantum part swappable.** Only the kernel layer changes between simulator and hardware. Encoder, SVM and evaluation stay the same.
9. **Compute the kernel once, slice it many times.** Similarity between two patients doesn't depend on the CV fold. Precomputing the full matrix made a 144-configuration search take seconds.

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| **Qubit** | Quantum bit. Here, one "dial" that holds one feature as a rotation angle. |
| **Feature map** | The circuit recipe that turns a patient's numbers into a quantum state |
| **Statevector** | The full quantum state: 2ⁿ complex numbers (256 for 8 qubits) |
| **Fidelity / overlap** | How alike two quantum states are, from 0 to 1. Used as the similarity score. |
| **Kernel matrix** | Table of similarities between every pair of patients |
| **Entanglement** | Qubits linked by two-qubit gates (CNOT), so features interact |
| **Shots** | Repeated runs and measurements on hardware, used to estimate a probability |
| **Bandwidth λ** | Scale on the input angles; controls how spread out patients' states are |
| **Support vectors** | The borderline training patients whose "votes" define the decision boundary |
| **PCA** | Compresses correlated measurements into a few uncorrelated summary directions |
| **Explained variance** | Share of the total variation that a component captures |
| **Cross-validation fold** | One slice of the training data, held out for scoring |
| **Fold-to-fold noise** | Score variation caused only by how patients were split into folds |
| **Geometric difference** | How different the quantum kernel's geometry is from a classical one; large means room for advantage |
| **Kernel concentration** | All kernel values bunching together, which makes patients indistinguishable |

---

## 10. Next steps

1. **Trainable quantum kernel:** `TrainableFidelityQuantumKernel` + `QuantumKernelTrainer` to learn a scaling per feature.
2. **Projected quantum kernels** (Huang et al. 2021): more robust to concentration.
3. **Real IBM hardware run** on a small subset (e.g. 80 patients) with error mitigation, compared against our shot-noise curve.
4. **Reduce shot cost:** Nyström approximation, classical shadows, or kernel rows for support vectors only.
5. **Label-aware feature selection** instead of PCA (keep `worst concave points`, `worst area`, …); more interpretable and may need fewer qubits.
6. **Nested cross-validation** to remove the small hyperparameter-tuning bias.
7. **Harder datasets** with richer non-linear structure (genomics, imaging) where quantum kernels have more room to help.
8. **Screening threshold:** pick an operating point targeting 100% malignant recall and report the false-positive cost.

---

## 11. How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install qiskit qiskit-machine-learning scikit-learn scipy matplotlib pylatexenc
python qsvc_breast_cancer.py
```

Tested with Python 3.14, qiskit 2.5.2, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, matplotlib 3.11.2. Fixed seed 42; all numbers above are reproducible from `outputs/results.json`.

---

## 12. References

- Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature (2019)
- Huang et al., *Power of data in quantum machine learning*, Nature Communications (2021)
- Thanasilp et al., *Exponential concentration in quantum kernel methods* (2022/2024)
- Shaydulin & Wild, *Importance of kernel bandwidth in quantum machine learning*, Phys. Rev. A (2022)
- Nadeau & Bengio, *Inference for the generalization error*, Machine Learning (2003)
- Qiskit Machine Learning documentation: `QSVC`, `FidelityStatevectorKernel`, `FidelityQuantumKernel`
- UCI ML Repository: Breast Cancer Wisconsin (Diagnostic)
