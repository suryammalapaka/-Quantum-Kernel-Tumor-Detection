# Can a Quantum Circuit Spot a Tumor? — Project Explanation

Companion notes for `qsvc_breast_cancer.py`. Use this to prepare for judges' questions. All numbers below come from the script's actual run (`outputs/results.json`, seed 42).

---

## 1. One-paragraph summary (elevator pitch)

We built a hybrid quantum-classical tumor classifier on the Wisconsin Diagnostic Breast Cancer dataset (569 patients, 30 measurements). A classical step compresses each patient to 8 numbers (PCA), a quantum circuit encodes them into an 8-qubit state, and the overlap between two patients' quantum states becomes the similarity score (kernel) for Qiskit's `QSVC`. We compared it against seven classical models: five with standard settings on all 30 features (linear, polynomial and RBF SVMs, logistic regression, random forest) and two tuned RBF SVMs, one on all 30 features and one given *the exact same 8 features*. **Result:** the quantum kernel is competitive. On the held-out test set it catches 52 of 53 cancers (recall 0.981, F1 0.954). On the same compressed features, the classical RBF SVM misses 3 cancers to its 1, and the quantum kernel beats it consistently in the learning curve. However, it does **not** beat the best classical model on all 30 features (poly-3 SVM, F1 0.990), and most of its gains are not statistically significant. The best quantum kernel also turned out to use no entanglement. Our contribution is a rigorous, honest test of where quantum similarity helps and where it doesn't, plus diagnostics that explain why.

---

## 2. Pipeline: how it works

```
30 raw features ──► StandardScaler ──► PCA(8) ──► MinMax to [0, π] ──► × bandwidth λ
                         (classical pre-processing, fit on training set only)
                                              │
                                              ▼
                     8-qubit feature-map circuit U(x)  →  |ψ(x)⟩ = U(x)|0⟩
                                              │
                                              ▼
            Quantum kernel  k(x, x') = |⟨ψ(x)|ψ(x')⟩|²   (FidelityStatevectorKernel)
                                              │
                                              ▼
                 QSVC (scikit-learn SVC with the quantum kernel swapped in)
```

| Stage | What | Why |
|---|---|---|
| StandardScaler + PCA(8) | 30 → 8 features, keeps ~93% of the variance (fig 01) | One qubit per feature. 8 qubits are cheap to simulate and realistic for today's hardware. |
| MinMax to [0, π] | Turns features into rotation angles | Feature maps take angles. A fixed range keeps the encoding well-defined. |
| **Bandwidth λ** | Multiplies every angle by λ ∈ {0.05 … 1.0} | **This matters most.** Without it, quantum kernels suffer *exponential concentration*: every pair of patients looks almost equally (dis)similar and the SVM learns nothing. See section 4. |
| Feature maps tried | `Z` (no entanglement), `ZZ` (linear CNOT-phase entanglement), `Y+ZZ` (Pauli map), reps ∈ {1, 2} | Tests directly whether entanglement adds useful structure. |
| Kernel | `FidelityStatevectorKernel`: exact state overlaps on a simulator; also supports shot noise | Fast (under 1 s for a 569×569 kernel), so a full hyperparameter search is affordable. |
| Classifier | `qiskit_machine_learning.algorithms.QSVC`, `class_weight="balanced"` | As the challenge asks: the same `.fit`/`.predict` API as scikit-learn, with only the kernel changed. |

**Implementation trick:** the kernel value for a pair of patients doesn't depend on which cross-validation fold they're in. So the script computes each quantum kernel matrix **once** over all patients and runs every CV fold and every C value by slicing that matrix (`SVC(kernel="precomputed")`). This makes the full grid of 6 maps × 6 bandwidths × 4 C values feasible in seconds. The final reported models are fit with the real `QSVC` API. A precomputed SVM and QSVC give identical results for the same kernel.

**Leakage control:** the scaler, PCA, and all hyperparameters are fit or selected on the 426 training patients only. The 143 test patients are touched once, at the end.

---

## 3. Results

### 3.1 Held-out test set (143 patients; 53 malignant)

| Model | Features | Accuracy | F1 (malignant) | Recall (malignant) | ROC AUC | Missed cancers |
|---|---|---|---|---|---|---|
| SVC poly-3 | 30 | **0.993** | **0.990** | 0.981 | **0.999** | 1 |
| SVC linear | 30 | 0.986 | 0.981 | 0.981 | 0.997 | 1 |
| Logistic regression | 30 | 0.986 | 0.981 | 0.981 | 0.998 | 1 |
| SVC RBF (default γ) | 30 | 0.979 | 0.972 | 0.981 | 0.997 | 1 |
| SVC RBF (tuned) | 30 | 0.965 | 0.954 | 0.981 | 0.997 | 1 |
| **QSVC, Z map (quantum)** | **8 (qubits)** | 0.965 | 0.954 | **0.981** | 0.996 | **1** |
| Hybrid kernel (0.5·quantum + 0.5·RBF) | 8 | 0.965 | 0.954 | 0.981 | 0.996 | 1 |
| Random forest | 30 | 0.965 | 0.952 | 0.943 | 0.995 | 3 |
| QSVC, ZZ map (entangled) | 8 | 0.958 | 0.945 | 0.981 | 0.994 | 1 |
| SVC RBF, **same 8 features** | 8 | 0.958 | 0.943 | 0.943 | 0.995 | **3** |

**Caveat:** with 143 test patients, one misclassified malignant case moves F1 by about 0.01. Treat test-set differences below about 0.02 as noise. That's why the script also runs the statistical tests below.

### 3.2 Where the quantum subroutine makes a measurable difference

1. **Against a classical kernel on identical information (the fair fight).** With the same 8 PCA features, QSVC catches 52/53 cancers vs 50/53 for a tuned RBF SVM. That is 2 fewer missed tumors, +0.011 F1 and +0.038 recall. In repeated CV, though, the two are tied (mean ΔF1 +0.0003, p = 0.98).
2. **Data efficiency (fig 07).** Over 20 random training subsets per size, QSVC beats the same-feature RBF SVM in mean F1 at **every training size from 40 to 426 patients**. At n = 320 it wins 17 of 20 subsets (Wilcoxon p = 0.010). The gains are small (+0.006 to +0.008 F1) and not significant at n ≤ 160. At n = 20 the quantum kernel is slightly *worse*.
3. **Against the full 30-feature tuned RBF.** With only 8 qubits, QSVC matches it on the test set (same F1, same recall). In 5×5 repeated CV it is ahead in mean F1 (+0.0072, 15 wins / 5 ties / 5 losses), but not significantly (p = 0.45).
4. **Hybrid kernel.** CV chose a 50/50 mix of quantum and RBF as the best kernel (CV F1 0.975, the highest of any SVM tuned this way). It is still not significantly better than classical (p = 0.40).

### 3.3 Where it does NOT make a difference (state this upfront; judges respect it)

- **Simple classical models win on all 30 features.** Poly-3 SVM, linear SVM, and logistic regression reach F1 0.98–0.99. The dataset is close to linearly separable, so there's little hidden structure for any exotic kernel to find.
- **Entanglement didn't help.** The best quantum kernel is the `Z` map: one-qubit rotations only, zero two-qubit gates (fig 02). A kernel like that can be computed efficiently on a classical computer: it equals a product of cosines. The best entangled map (`ZZ`, reps=1) scored lower in CV (0.955 vs 0.972) and on the test set (0.945 vs 0.954).
- **Shot noise is the hardware bottleneck (fig 10).** Real quantum hardware estimates each kernel entry from a finite number of measurements ("shots"). Test F1 drops to 0.79 at 32 shots and 0.92 at 8,192 shots, and only reaches about 0.94 at 32,768 shots. At these small bandwidths the kernel values sit close together, so sampling noise hides the differences.

---

## 4. Why the diagnostics matter (researcher-level talking points)

| Diagnostic (fig) | What we found | What it means |
|---|---|---|
| **Exponential concentration** (05a) | At bandwidth 1.0, the spread of off-diagonal kernel values halves from 2 to 12 qubits (0.28 → 0.14). At 0.4 it stays flat (~0.22). At 0.1 the spread is tiny (~0.02) because every entry sits near 1. | Matches Thanasilp et al. 2022 and Shaydulin & Wild 2022. Too large a bandwidth makes all patients look different, and the problem worsens as qubits grow. Too small makes them all look the same. Tuning the bandwidth is required, not optional. The tiny spread at small bandwidth is also why shot noise hurts so much (section 3.3). |
| **Eigen-spectrum** (05b) | The quantum kernel's spectrum decays more slowly than the same-feature RBF's (effective dimension 1.8 vs 1.2). | The quantum kernel is more expressive than RBF on the same 8 numbers, which plausibly explains its edge in the fair fight. |
| **Kernel-target alignment** (05c) | Quantum 0.443, RBF same-features 0.467, RBF 30-features 0.569 | The 30-feature classical kernel aligns best with the labels. PCA to 8 features throws away a little useful signal. |
| **Geometric difference g(K_C‖K_Q)** (05d), Huang et al. 2021 "Power of data" | 2.4 vs 30-feature RBF, 6.3 vs same-feature RBF, 9.5 vs linear (maximum possible ≈ √N ≈ 20.6) | g measures how far the quantum kernel's geometry is from a classical one. Values well below √N mean a classical kernel can approximate the quantum one. Huang et al. show a prediction advantage is *possible* only when g is large, and then only if the labels actually use that extra structure. Here the potential is modest, which matches the results. |
| **Qubit sweep** (06) | QSVC ≥ same-feature RBF at 6 qubits (0.978 vs 0.971, the best CV score in the project) and at 8 qubits. They tie at 12, and QSVC is slightly behind at 2, 4, and 10. | There's no monotonic quantum advantage. 6–8 qubits is the sweet spot for this dataset. |

---

## 5. Why this is a strong solution (for "why should we pick you?")

1. **Fair comparison.** Most quantum ML demos compare a tuned quantum model against an untuned classical one, or give them different information. We tuned the quantum and RBF kernels with the same grid search and folds, gave the classical RBF *the exact same features*, and also included the strongest 30-feature models.
2. **Clinically relevant metric.** F1 and recall for *malignant* are the primary metrics, and the confusion matrices count missed cancers directly. A false negative is the expensive mistake.
3. **Statistical rigor.** 5×5 repeated CV with the Nadeau–Bengio corrected t-test (it accounts for overlapping training sets), Wilcoxon tests on the learning curve, and test-set caveats stated explicitly.
4. **Explains why, not just what.** The concentration, spectrum, alignment, and geometric-difference diagnostics connect our results to current quantum ML theory.
5. **Hardware realism.** The shot-noise study gives a concrete measurement budget needed before real-device runs.
6. **Hybrid by design.** The classical steps (scaling, PCA, SVM optimization) do what they're good at. The quantum circuit does only one job: computing similarity. The quantum+RBF kernel mix is a practical deployment pattern.
7. **Reproducible.** One script, fixed seeds, about 2.5 minutes on a laptop, every number saved to `results.json`.

---

## 6. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install qiskit qiskit-machine-learning scikit-learn scipy matplotlib pylatexenc
python qsvc_breast_cancer.py
```

Tested with Python 3.14, qiskit 2.5.2, qiskit-machine-learning 0.9.1, scikit-learn 1.9.1, matplotlib 3.11.2. Outputs go to `./outputs/`:

| File | Content |
|---|---|
| `01_dataset_overview.png` | Class balance, PCA variance curve, PC1–PC2 scatter |
| `02_feature_map_circuit.png` / `02b_…` | Selected (Z) and best entangling (ZZ) circuits |
| `03_qsvc_model_selection.png` | CV F1 heatmaps: feature map × reps × bandwidth × C |
| `04_kernel_matrices.png` | Quantum vs RBF kernel matrices, patients sorted by diagnosis |
| `05_kernel_diagnostics.png` | Concentration, spectrum, alignment, geometric difference |
| `06_qubit_sweep.png` | CV F1 vs number of qubits (quantum vs classical) |
| `07_learning_curve.png` | Test F1 vs training-set size |
| `08_test_leaderboard.png` | All models, four metrics |
| `09_roc_confusion.png` | ROC curves; confusion matrices for QSVC vs same-feature RBF |
| `10_shot_noise.png` | QSVC test F1 vs measurement shots |
| `11_paired_significance.png` | Per-fold paired differences and p-values |
| `results.json` | Every number above |

---

## 7. Likely judge questions — short answers

**Q: Is this a quantum advantage?**
No, and we don't claim one. On this dataset the quantum kernel matches strong classical kernels and is slightly better than a classical kernel given identical inputs, but the differences mostly aren't significant. The winning map has no entanglement, so it's classically simulable. The value of the project is the rigorous method for answering "does quantum similarity capture structure classical kernels miss?" For this dataset the measured answer is: a little, at best.

**Q: Why PCA to 8 qubits instead of 30 qubits?**
Simulating 30 qubits exactly takes about 16 GB per state vector. Hardware noise and kernel concentration also get worse with width. The qubit sweep shows accuracy levels off at 6–10 qubits, and 8 components keep about 93% of the variance.

**Q: What are the 8 features? Which of the 30 measurements did you pick?**
None were picked; none were dropped. The 8 inputs are the first 8 principal components. Each is a weighted mix of **all 30** standardized measurements, chosen to capture as much of the variation between patients as possible. Measured on the training set (separation = gap between the malignant and benign averages, divided by the component's spread):

| Qubit | Component | Variance kept | What it mostly measures (largest weights) | Separates classes? |
|---|---|---|---|---|
| 0 | PC1 | 43.9% | Shape irregularity: concave points, concavity, compactness (mean and worst) | **Strongly (1.64)** |
| 1 | PC2 | 19.5% | Boundary roughness: fractal dimension (mean, error, worst), compactness error | Weakly (0.37) |
| 2 | PC3 | 9.4% | Variability across the cell: texture, smoothness, symmetry and radius *errors* | Weakly (0.34) |
| 3 | PC4 | 6.8% | Texture: worst and mean texture | Weakly (0.26) |
| 4 | PC5 | 5.6% | Smoothness (mean and worst) against concavity error | Weakly (0.25) |
| 5 | PC6 | 4.0% | Symmetry against smoothness | Barely (0.06) |
| 6 | PC7 | 2.2% | Fractal dimension against concave-points error | Barely (0.16) |
| 7 | PC8 | 1.6% | Smoothness error against texture error | Barely (0.08) |

Together they keep 93% of the variance. PC1 alone carries most of the diagnostic signal: irregular, concave cell nuclei mean malignancy, which matches the clinical intuition. One caveat: PCA ignores the labels, so "most variance" is not the same as "most predictive". That explains why the 30-feature kernel aligns better with the diagnoses (section 4).

**Q: Why exactly 8 qubits?**
It was a design choice made upfront, not something cross-validation picked. The reasons: 8 components keep 93% of the variance; an 8-qubit exact simulation is fast enough for a full hyperparameter search; and 8-qubit circuits of this depth are realistic on today's hardware. The qubit sweep (fig 06) checked the choice afterwards: performance levels off between 6 and 10 qubits. 6 qubits actually scored slightly higher in CV (0.978 vs 0.972), but that gap is within fold-to-fold noise, so we kept 8.

**Q: What does the bandwidth do?**
It scales the rotation angles. Large angles push every patient's state toward being almost perpendicular to every other's, so kernel entries all approach the same small value and the SVM can't tell patients apart (exponential concentration). Smaller angles keep similar patients close. Figure 05a shows the effect.

**Q: Why was the non-entangled Z map chosen?**
CV picked it on merit. This dataset is close to linearly separable, so a smooth, product-of-cosines similarity is enough. Entangling maps add feature interactions the labels don't need, which adds variance. This is a real finding, not a bug.

**Q: Why is the tuned 30-feature RBF worse on the test set than the default one?**
Tuning maximizes CV F1, and the tuned model was best in CV (0.971 vs 0.955). On 143 test patients, a difference of two or three cases is within noise. That is exactly why we report repeated CV and p-values alongside the test set.

**Q: How would this run on real hardware?**
Swap `FidelityStatevectorKernel` for `FidelityQuantumKernel` with a `ComputeUncompute` fidelity on an IBM Runtime `Sampler`. A full 426×426 training kernel is about 90k circuits, each needing roughly 30k+ shots for stable accuracy (fig 10). That's expensive, so see next steps for ways to reduce it.

**Q: Isn't the hyperparameter selection biased?**
Slightly, and equally for every model: all SVMs were tuned with the same CV folds, and the repeated-CV significance test reuses training data that informed tuning. The held-out test set is untouched. Nested CV is listed as a next step.

**Q: Why class_weight="balanced"?**
There are 212 malignant vs 357 benign patients. Balancing class weights raises the cost of missing a cancer, which suits a screening tool.

---

## 8. Next steps

1. **Trainable quantum kernels.** Use `TrainableFidelityQuantumKernel` + `QuantumKernelTrainer` to optimize per-feature scaling parameters for kernel-target alignment. Learned bandwidths could beat a single global λ.
2. **Projected quantum kernels** (Huang et al. 2021). Compare single-qubit reduced density matrices instead of full-state fidelity. They resist concentration and tend to have larger geometric difference.
3. **Real hardware run.** Use a small training subset (e.g. 80 patients, where the learning curve shows QSVC holds its own) on IBM hardware. Add error mitigation (twirling, ZNE) and compare to the shot-noise curve here.
4. **Cut shot cost.** Use Nyström / landmark approximation (only m ≪ N kernel columns), classical shadows, or compute only support-vector rows at prediction time.
5. **Nested cross-validation** to remove the small tuning bias from the significance tests.
6. **Harder datasets.** This one is nearly linearly separable, so it's a tough place for quantum kernels to shine. Rerun the same pipeline on data with richer non-linear or group structure (genomics, high-dimensional imaging features) where geometric difference is larger.
7. **Decision-threshold tuning for screening.** Pick an operating point on the ROC curve that targets 100% malignant recall and report the resulting false-positive rate.
8. **Better encodings.** Try feature selection instead of PCA (keep interpretable features like `worst concave points`), and data re-uploading circuits.

---

## 9. References

- Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567 (2019): ZZ feature map and quantum kernel SVM.
- Huang et al., *Power of data in quantum machine learning*, Nature Communications 12 (2021): geometric difference, projected kernels.
- Thanasilp et al., *Exponential concentration in quantum kernel methods* (2022/2024).
- Shaydulin & Wild, *Importance of kernel bandwidth in quantum machine learning*, Phys. Rev. A 106 (2022).
- Nadeau & Bengio, *Inference for the generalization error*, Machine Learning 52 (2003): corrected resampled t-test.
- Qiskit Machine Learning docs: `QSVC`, `FidelityStatevectorKernel`, `FidelityQuantumKernel`.
- Wolberg, Street, Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*, UCI ML Repository.
