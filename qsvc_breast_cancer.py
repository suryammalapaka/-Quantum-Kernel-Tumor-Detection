"""
Can a quantum circuit spot a tumor?
Quantum kernel SVM (Qiskit QSVC) vs strong classical baselines on Wisconsin Diagnostic Breast Cancer.

Dataset: 569 samples, 30 features; labels 0 = malignant, 1 = benign.
Positive class for F1/recall is malignant (0): a missed cancer is the costly error.

Requirements: qiskit>=2.1, qiskit-machine-learning>=0.9, scikit-learn, scipy, matplotlib, pylatexenc
Run:          python qsvc_breast_cancer.py      (figures + results.json are written to ./outputs)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import pauli_feature_map, z_feature_map, zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.utils import algorithm_globals
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------- config
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
CV_REPEATS = 5
MALIGNANT = 0

N_QUBITS = 8
FEATURE_MAPS = ("Z", "ZZ", "Y+ZZ")
REPS_GRID = (1, 2)
BANDWIDTHS = (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)
C_GRID = (0.1, 1.0, 10.0, 100.0)
RBF_GAMMAS_ENCODED = tuple(np.logspace(-3, 1, 9))
RBF_GAMMAS_FULL = tuple(np.logspace(-4, 0, 9))
HYBRID_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
QUBIT_SWEEP = (2, 4, 6, 8, 10, 12)
TRAIN_SIZES = (20, 40, 80, 160, 320)
LC_SEEDS = 20
SHOTS_GRID = (32, 128, 512, 2048, 8192, 32768)
SHOT_SEEDS = 3
GEO_DIFF_REG = 1e-2

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# ----------------------------------------------------------------------------- style
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
QUANTUM = "#2a78d6"
CLASSICAL = "#eb6834"
HYBRID = "#1baf7a"
REFERENCE = "#52514e"
MALIGNANT_C = "#6250d6"
BENIGN_C = "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
ORDINAL_BLUES = ("#86b6ef", "#2a78d6", "#104281")


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "font.size": 9,
            "text.color": INK,
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved outputs/{name}")


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, malignant_score) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_malignant": f1_score(y_true, y_pred, pos_label=MALIGNANT),
        "recall_malignant": recall_score(y_true, y_pred, pos_label=MALIGNANT),
        "precision_malignant": precision_score(y_true, y_pred, pos_label=MALIGNANT),
        "roc_auc": roc_auc_score((y_true == MALIGNANT).astype(int), malignant_score),
    }


def format_metrics(m: dict) -> str:
    return (
        f"acc={m['accuracy']:.3f}  F1(m)={m['f1_malignant']:.3f}  "
        f"recall(m)={m['recall_malignant']:.3f}  AUC={m['roc_auc']:.3f}"
    )


def cv_scoring_params() -> dict:
    return {
        "accuracy": "accuracy",
        "f1_malignant": make_scorer(f1_score, pos_label=MALIGNANT),
        "recall_malignant": make_scorer(recall_score, pos_label=MALIGNANT),
    }


def malignant_score(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return -model.decision_function(X)
    return model.predict_proba(X)[:, MALIGNANT]


# ----------------------------------------------------------------------------- precomputed-kernel SVM helpers
def svc_precomputed(C: float) -> SVC:
    return SVC(kernel="precomputed", C=C, class_weight="balanced")


def fit_eval_precomputed(K, y, train_idx, test_idx, C) -> tuple[np.ndarray, np.ndarray]:
    clf = svc_precomputed(C).fit(K[np.ix_(train_idx, train_idx)], y[train_idx])
    K_te = K[np.ix_(test_idx, train_idx)]
    return clf.predict(K_te), -clf.decision_function(K_te)


def cv_f1(K, y, splits, C) -> np.ndarray:
    scores = []
    for tr, va in splits:
        pred, _ = fit_eval_precomputed(K, y, tr, va, C)
        scores.append(f1_score(y[va], pred, pos_label=MALIGNANT))
    return np.array(scores)


def tune_C(K_train, y_train, splits) -> tuple[float, float, dict]:
    per_c = {C: cv_f1(K_train, y_train, splits, C).mean() for C in C_GRID}
    best_C = max(per_c, key=per_c.get)
    return best_C, per_c[best_C], per_c


# ----------------------------------------------------------------------------- quantum encoding
def make_encoder(n_qubits: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=RANDOM_STATE)),
            ("angle", MinMaxScaler(feature_range=(0.0, np.pi))),
        ]
    )


def build_feature_map(name: str, n_qubits: int, reps: int) -> QuantumCircuit:
    if name == "Z":
        return z_feature_map(n_qubits, reps=reps)
    if name == "ZZ":
        return zz_feature_map(n_qubits, reps=reps, entanglement="linear")
    if name == "Y+ZZ":
        return pauli_feature_map(n_qubits, reps=reps, entanglement="linear", paulis=["Y", "ZZ"])
    raise ValueError(name)


def quantum_gram(fm: str, n_qubits: int, reps: int, X, shots: int | None = None) -> np.ndarray:
    kernel = FidelityStatevectorKernel(
        feature_map=build_feature_map(fm, n_qubits, reps), shots=shots
    )
    return kernel.evaluate(X)


# ----------------------------------------------------------------------------- kernel diagnostics
def centered_alignment(K, y) -> float:
    n = len(y)
    yy = np.where(y == MALIGNANT, 1.0, -1.0)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Y = np.outer(yy, yy)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def psd_sqrt(K) -> np.ndarray:
    w, v = np.linalg.eigh(K)
    return (v * np.sqrt(np.clip(w, 0, None))) @ v.T


def geometric_difference(K_classical, K_quantum, reg=GEO_DIFF_REG) -> float:
    n = len(K_classical)
    Kc = n * K_classical / np.trace(K_classical)
    Kq = n * K_quantum / np.trace(K_quantum)
    sq = psd_sqrt(Kq)
    M = sq @ np.linalg.inv(Kc + reg * np.eye(n)) @ sq
    return float(np.sqrt(np.max(np.linalg.eigvalsh(M))))


def off_diagonal(K) -> np.ndarray:
    return K[~np.eye(len(K), dtype=bool)]


def normalized_spectrum(K) -> np.ndarray:
    w = np.clip(np.linalg.eigvalsh(K)[::-1], 1e-12, None)
    return w / w.sum()


def effective_dimension(K) -> float:
    p = normalized_spectrum(K)
    return float(np.exp(-np.sum(p * np.log(p))))


def corrected_paired_ttest(diffs, n_train, n_test) -> tuple[float, float]:
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


# ----------------------------------------------------------------------------- plots
def plot_dataset_overview(X_train, y_train) -> None:
    scaled = StandardScaler().fit_transform(X_train)
    pca = PCA(random_state=RANDOM_STATE).fit(scaled)
    proj = pca.transform(scaled)[:, :2]
    cum = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    counts = [(y_train == MALIGNANT).sum(), (y_train != MALIGNANT).sum()]
    bars = axes[0].bar(["Malignant", "Benign"], counts, color=[MALIGNANT_C, BENIGN_C], width=0.55)
    axes[0].bar_label(bars, color=INK_2, fontsize=9, padding=3)
    axes[0].set_title("Training-set class balance")
    axes[0].set_ylabel("patients")
    axes[0].grid(axis="x", visible=False)

    axes[1].plot(np.arange(1, len(cum) + 1), cum, color=QUANTUM, marker="o", markersize=4)
    axes[1].axvline(N_QUBITS, color=INK_3, linestyle="--", linewidth=1)
    axes[1].annotate(
        f"{N_QUBITS} qubits keep {cum[N_QUBITS - 1]:.0%} of variance",
        (N_QUBITS, cum[N_QUBITS - 1]),
        xytext=(N_QUBITS + 2, cum[N_QUBITS - 1] - 0.15),
        color=INK_2,
        arrowprops={"arrowstyle": "-", "color": INK_3},
    )
    axes[1].set_title("PCA compression: 30 features -> n qubits")
    axes[1].set_xlabel("principal components (= qubits)")
    axes[1].set_ylabel("cumulative explained variance")

    for label, color, name in ((1, BENIGN_C, "Benign"), (MALIGNANT, MALIGNANT_C, "Malignant")):
        m = y_train == label
        axes[2].scatter(proj[m, 0], proj[m, 1], s=14, color=color, label=name,
                        edgecolor=SURFACE, linewidth=0.5, alpha=0.9)
    axes[2].set_title("First two principal components")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "01_dataset_overview.png")


def plot_circuit(fm: str, reps: int, label: str, filename: str) -> None:
    circuit = build_feature_map(fm, N_QUBITS, reps).decompose()
    try:
        fig = circuit.draw(output="mpl", fold=40, style="clifford")
        fig.suptitle(
            f"{label}: {fm}, {N_QUBITS} qubits, reps={reps}  "
            f"(depth {circuit.depth()}, {circuit.num_nonlocal_gates()} two-qubit gates)",
            x=0.02, ha="left", fontsize=11, color=INK,
        )
        save(fig, f"{filename}.png")
    except Exception as exc:
        (OUT_DIR / f"{filename}.txt").write_text(str(circuit.draw(output="text")))
        print(f"  mpl circuit drawer unavailable ({exc}); wrote text diagram instead")


def plot_model_selection(grid: dict, best: dict) -> None:
    fig, axes = plt.subplots(len(REPS_GRID), len(FEATURE_MAPS), figsize=(13, 3.2 * len(REPS_GRID)),
                             squeeze=False)
    values = np.array([v for d in grid.values() for v in d.values()])
    vmin, vmax = values.min(), values.max()
    for i, reps in enumerate(REPS_GRID):
        for j, fm in enumerate(FEATURE_MAPS):
            ax = axes[i, j]
            M = np.array([[grid[(fm, reps)][(bw, C)] for C in C_GRID] for bw in BANDWIDTHS])
            ax.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
            for (r, c), v in np.ndenumerate(M):
                is_best = (fm, reps, BANDWIDTHS[r], C_GRID[c]) == (best["fm"], best["reps"], best["bw"], best["C"])
                ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if v > (vmin + vmax) / 2 else INK,
                        fontweight="bold" if is_best else "normal")
                if is_best:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=CLASSICAL, linewidth=2))
            ax.set_xticks(range(len(C_GRID)), [f"{C:g}" for C in C_GRID])
            ax.set_yticks(range(len(BANDWIDTHS)), [f"{b:g}" for b in BANDWIDTHS])
            ax.set_xlabel("SVM C")
            ax.set_ylabel("bandwidth")
            ax.set_title(f"{fm} map, reps={reps}")
            ax.grid(False)
    fig.suptitle("QSVC model selection: 5-fold CV F1(malignant) on training set (best boxed)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_qsvc_model_selection.png")


def plot_kernel_matrices(kernels: dict, y_train) -> None:
    order = np.argsort(y_train, kind="stable")
    n_mal = int((y_train == MALIGNANT).sum())
    fig, axes = plt.subplots(1, len(kernels), figsize=(4.6 * len(kernels), 4.4))
    for ax, (name, K) in zip(axes, kernels.items()):
        im = ax.imshow(K[np.ix_(order, order)], cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(n_mal - 0.5, color=CLASSICAL, linewidth=1)
        ax.axvline(n_mal - 0.5, color=CLASSICAL, linewidth=1)
        ax.set_title(name)
        ax.set_xticks([n_mal / 2, n_mal + (len(order) - n_mal) / 2], ["malignant", "benign"])
        ax.set_yticks([n_mal / 2, n_mal + (len(order) - n_mal) / 2], ["malignant", "benign"], rotation=90, va="center")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="similarity k(x, x')")
    fig.suptitle("Training kernel matrices, patients sorted by diagnosis", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, "04_kernel_matrices.png")


def plot_kernel_diagnostics(concentration: dict, spectra: dict, alignment: dict, geo: dict, n_train: int) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    ax = axes[0]
    for color, bw in zip(ORDINAL_BLUES, (1.0, 0.4, 0.1)):
        xs = sorted(concentration)
        ax.plot(xs, [concentration[n][bw] for n in xs], color=color, marker="o", label=f"bandwidth {bw:g}")
    ax.set_yscale("log")
    ax.set_xlabel("qubits")
    ax.set_ylabel("std of off-diagonal kernel values")
    ax.set_title("Exponential concentration vs bandwidth")
    ax.legend()

    ax = axes[1]
    colors = {"Quantum (QSVC)": QUANTUM, "RBF, same features": CLASSICAL, "RBF, 30 features": REFERENCE}
    for name, spec in spectra.items():
        ax.plot(np.arange(1, 61), spec[:60], color=colors[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("normalized eigenvalue")
    ax.set_title("Kernel eigen-spectrum (expressivity)")
    ax.legend()

    ax = axes[2]
    names = list(alignment)
    ax.barh(names, [alignment[n] for n in names], color=[colors[n] for n in names], height=0.5)
    for i, n in enumerate(names):
        ax.text(alignment[n], i, f" {alignment[n]:.3f}", va="center", color=INK_2)
    ax.set_xlabel("centered kernel-target alignment")
    ax.set_title("How well each kernel matches the labels")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(alignment.values()) * 1.25)

    ax = axes[3]
    names = list(geo)
    ax.barh(names, [geo[n] for n in names], color=INK_3, height=0.5)
    for i, n in enumerate(names):
        ax.text(geo[n], i, f" {geo[n]:.2f}", va="center", color=INK_2)
    ax.set_xlabel(f"g(K_C || K_Q)   (max possible ~ sqrt(N) = {np.sqrt(n_train):.1f})")
    ax.set_title("Geometric difference: room for advantage")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(geo.values()) * 1.3)
    fig.tight_layout()
    save(fig, "05_kernel_diagnostics.png")


def plot_qubit_sweep(sweep: dict, full_rbf_cv: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = sorted(sweep)
    q = [sweep[n]["quantum"] for n in xs]
    c = [sweep[n]["classical"] for n in xs]
    ax.plot(xs, q, color=QUANTUM, marker="o", label="QSVC (quantum kernel)")
    ax.plot(xs, c, color=CLASSICAL, marker="s", label="RBF SVC, same n features")
    ax.axhline(full_rbf_cv, color=REFERENCE, linestyle="--", linewidth=1.2, label="RBF SVC, all 30 features")
    ax.text(xs[-1], q[-1], "  quantum", color=INK_2, va="center")
    ax.text(xs[-1], c[-1], "  classical", color=INK_2, va="center")
    ax.set_xticks(xs)
    ax.set_xlabel("qubits = PCA components")
    ax.set_ylabel("5-fold CV F1 (malignant)")
    ax.set_title("Accuracy vs circuit width (each point tuned independently)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "06_qubit_sweep.png")


def plot_learning_curve(lc: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    styles = {
        "QSVC (quantum kernel)": (QUANTUM, "o"),
        "RBF SVC, same features": (CLASSICAL, "s"),
        "RBF SVC, 30 features": (REFERENCE, "^"),
    }
    for name, (color, marker) in styles.items():
        sizes = sorted(lc[name])
        mean = np.array([np.mean(lc[name][s]) for s in sizes])
        std = np.array([np.std(lc[name][s]) for s in sizes])
        ax.plot(sizes, mean, color=color, marker=marker, label=name)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(sorted(lc["QSVC (quantum kernel)"]), [str(s) for s in sorted(lc["QSVC (quantum kernel)"])])
    ax.minorticks_off()
    ax.set_xlabel("training patients (log scale)")
    ax.set_ylabel("held-out test F1 (malignant)")
    ax.set_title(f"Data efficiency: mean ± 1 sd over {LC_SEEDS} random subsets")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "07_learning_curve.png")


def entity_color(name: str) -> str:
    if name.startswith("QSVC"):
        return QUANTUM
    if name.startswith("Hybrid"):
        return HYBRID
    if "same" in name:
        return CLASSICAL
    return INK_3


def plot_leaderboard(test_results: dict) -> None:
    names = sorted(test_results, key=lambda n: test_results[n]["f1_malignant"])
    metrics = (("accuracy", "Accuracy"), ("f1_malignant", "F1 (malignant)"),
               ("recall_malignant", "Recall (malignant)"), ("roc_auc", "ROC AUC"))
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 0.42 * len(names) + 1.6), sharey=True)
    for ax, (key, title) in zip(axes, metrics):
        vals = [test_results[n][key] for n in names]
        colors = [entity_color(n) for n in names]
        ax.hlines(range(len(names)), min(vals) - 0.01, vals, color=GRID, linewidth=1)
        ax.scatter(vals, range(len(names)), color=colors, s=60, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for i, v in enumerate(vals):
            ax.text(v, i + 0.28, f"{v:.3f}", ha="center", fontsize=7, color=INK_2)
        ax.set_title(title)
        ax.set_xlim(min(vals) - 0.01, 1.003)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(names)), names)
    fig.suptitle("Held-out test set (143 patients): every model, same split", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_test_leaderboard.png")


def plot_roc_confusion(y_test, curves: dict, preds: dict) -> None:
    fig, axes = plt.subplots(1, 1 + len(preds), figsize=(5 * (1 + len(preds)), 4.4))
    ax = axes[0]
    y_bin = (y_test == MALIGNANT).astype(int)
    for name, score in curves.items():
        fpr, tpr, _ = roc_curve(y_bin, score)
        ax.plot(fpr, tpr, color=entity_color(name), linestyle="--" if name.startswith("Hybrid") else "-",
                label=f"{name} (AUC {roc_auc_score(y_bin, score):.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (malignant caught)")
    ax.set_title("ROC on held-out test set")
    ax.legend(loc="lower right", fontsize=7)

    for ax, (name, pred) in zip(axes[1:], preds.items()):
        cm = confusion_matrix(y_test, pred, labels=[MALIGNANT, 1])
        ax.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.max())
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                    color=SURFACE if v > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["malignant", "benign"])
        ax.set_yticks([0, 1], ["malignant", "benign"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(f"{name}\nmissed cancers: {cm[0, 1]}", fontsize=10)
        ax.grid(False)
    fig.tight_layout()
    save(fig, "09_roc_confusion.png")


def plot_shot_noise(shot_results: dict, exact_f1: float, classical_f1: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    shots = sorted(shot_results)
    mean = np.array([np.mean(shot_results[s]) for s in shots])
    std = np.array([np.std(shot_results[s]) for s in shots])
    ax.errorbar(shots, mean, yerr=std, color=QUANTUM, marker="o", capsize=3, label="QSVC with finite shots")
    ax.axhline(exact_f1, color=QUANTUM, linestyle=":", linewidth=1.2, label="QSVC exact (statevector)")
    ax.axhline(classical_f1, color=CLASSICAL, linestyle="--", linewidth=1.2, label="RBF SVC, same features")
    ax.set_xscale("log")
    ax.set_xlabel("shots per kernel entry (log scale)")
    ax.set_ylabel("held-out test F1 (malignant)")
    ax.set_title("Hardware readiness: robustness to measurement shot noise")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "10_shot_noise.png")


def plot_significance(diffs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(diffs) + 1.4))
    rng = np.random.default_rng(RANDOM_STATE)
    for i, (name, d) in enumerate(diffs.items()):
        jitter = rng.uniform(-0.12, 0.12, len(d["diffs"]))
        ax.scatter(d["diffs"], i + jitter, s=14, color=INK_3, alpha=0.8, edgecolor="none")
        ax.scatter([np.mean(d["diffs"])], [i], s=90, color=QUANTUM if "Hybrid" not in name else HYBRID,
                   zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(max(d["diffs"]) + 0.004, i,
                f"mean {np.mean(d['diffs']):+.4f}, p={d['p']:.3f}", va="center", color=INK_2, fontsize=8)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(diffs)), list(diffs))
    ax.invert_yaxis()
    ax.set_xlabel("F1(malignant) difference per fold  (> 0 favours the left-hand model)")
    ax.set_title(f"Paired {CV_REPEATS}x{CV_FOLDS} repeated CV, corrected resampled t-test")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.25)
    fig.tight_layout()
    save(fig, "11_paired_significance.png")


# ----------------------------------------------------------------------------- experiment
def classical_baselines() -> dict:
    return {
        "SVC linear (30 feat)": Pipeline(
            [("scale", StandardScaler()), ("clf", SVC(kernel="linear", random_state=RANDOM_STATE))]
        ),
        "SVC poly-3 (30 feat)": Pipeline(
            [("scale", StandardScaler()),
             ("clf", SVC(kernel="poly", degree=3, coef0=1, random_state=RANDOM_STATE))]
        ),
        "SVC RBF default (30 feat)": Pipeline(
            [("scale", StandardScaler()), ("clf", SVC(kernel="rbf", gamma="scale", random_state=RANDOM_STATE))]
        ),
        "Logistic regression (30 feat)": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]
        ),
        "Random forest (30 feat)": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        ),
    }


def tune_rbf(X_all, y_train, train_idx, splits, gammas) -> dict:
    best = {"cv_f1": -1.0}
    for g in gammas:
        K = rbf_kernel(X_all, gamma=g)
        C, score, _ = tune_C(K[np.ix_(train_idx, train_idx)], y_train, splits)
        if score > best["cv_f1"]:
            best = {"gamma": float(g), "C": C, "cv_f1": score, "K": K}
    return best


def main() -> None:
    t_start = time.time()
    set_style()
    OUT_DIR.mkdir(exist_ok=True)
    algorithm_globals.random_seed = RANDOM_STATE

    data = load_breast_cancer()
    X, y = data.data, data.target
    print("Wisconsin Diagnostic Breast Cancer (sklearn.load_breast_cancer)")
    print(f"  samples={X.shape[0]}, features={X.shape[1]}, malignant={(y == MALIGNANT).sum()}, benign={(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    tr = np.arange(len(y_train))
    te = np.arange(len(y_train), len(y_all))
    splits = list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train))
    print(f"  train/test = {len(tr)}/{len(te)} (stratified)\n")

    print("[1/9] Dataset overview")
    plot_dataset_overview(X_train, y_train)

    print("\n[2/9] Classical baselines on all 30 features")
    test_results, test_scores, test_preds = {}, {}, {}
    cv_strat = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, model in classical_baselines().items():
        cv = cross_validate(model, X_train, y_train, cv=cv_strat, scoring=cv_scoring_params(), n_jobs=1)
        model.fit(X_train, y_train)
        pred, score = model.predict(X_test), malignant_score(model, X_test)
        test_results[name] = compute_metrics(y_test, pred, score)
        test_scores[name], test_preds[name] = score, pred
        print(f"  {name:32s} CV F1(m)={cv['test_f1_malignant'].mean():.3f} | test {format_metrics(test_results[name])}")

    scaler = StandardScaler().fit(X_train)
    X_all_std = scaler.transform(X_all)
    full_rbf = tune_rbf(X_all_std, y_train, tr, splits, RBF_GAMMAS_FULL)
    pred, score = fit_eval_precomputed(full_rbf["K"], y_all, tr, te, full_rbf["C"])
    name_full = "SVC RBF tuned (30 feat)"
    test_results[name_full] = compute_metrics(y_test, pred, score)
    test_scores[name_full], test_preds[name_full] = score, pred
    print(f"  {name_full:32s} CV F1(m)={full_rbf['cv_f1']:.3f} | test {format_metrics(test_results[name_full])}"
          f"  (gamma={full_rbf['gamma']:.4g}, C={full_rbf['C']:g})")

    print(f"\n[3/9] Quantum encoding: StandardScaler -> PCA({N_QUBITS}) -> angles in [0, pi] -> x bandwidth")
    encoder = make_encoder(N_QUBITS).fit(X_train)
    Z_all = encoder.transform(X_all)

    print("\n[4/9] QSVC model selection (feature map x reps x bandwidth x C), 5-fold CV on training set")
    grid, best_q, best_ent = {}, {"cv_f1": -1.0}, {"cv_f1": -1.0}
    for fm in FEATURE_MAPS:
        for reps in REPS_GRID:
            grid[(fm, reps)] = {}
            for bw in BANDWIDTHS:
                K = quantum_gram(fm, N_QUBITS, reps, Z_all * bw)
                _, _, per_c = tune_C(K[np.ix_(tr, tr)], y_train, splits)
                for C, s in per_c.items():
                    grid[(fm, reps)][(bw, C)] = s
                    if s > best_q["cv_f1"]:
                        best_q = {"fm": fm, "reps": reps, "bw": bw, "C": C, "cv_f1": s, "K": K}
                    if fm != "Z" and s > best_ent["cv_f1"]:
                        best_ent = {"fm": fm, "reps": reps, "bw": bw, "C": C, "cv_f1": s, "K": K}
            top = max(grid[(fm, reps)].items(), key=lambda kv: kv[1])
            print(f"  {fm:5s} reps={reps}: best CV F1(m)={top[1]:.3f} at bandwidth={top[0][0]:g}, C={top[0][1]:g}")
    print(f"  -> selected {best_q['fm']} reps={best_q['reps']} bandwidth={best_q['bw']:g} C={best_q['C']:g}"
          f" CV F1(m)={best_q['cv_f1']:.3f}")
    print(f"  -> best entangling map: {best_ent['fm']} reps={best_ent['reps']} bandwidth={best_ent['bw']:g}"
          f" C={best_ent['C']:g} CV F1(m)={best_ent['cv_f1']:.3f}")
    plot_circuit(best_q["fm"], best_q["reps"], "Selected quantum feature map", "02_feature_map_circuit")
    plot_circuit(best_ent["fm"], best_ent["reps"], "Best entangling feature map", "02b_entangling_feature_map_circuit")
    plot_model_selection(grid, best_q)

    print("\n[5/9] Fair head-to-head: classical RBF on the identical encoded features + hybrid kernel")
    matched = tune_rbf(Z_all, y_train, tr, splits, RBF_GAMMAS_ENCODED)
    print(f"  RBF same features: CV F1(m)={matched['cv_f1']:.3f} (gamma={matched['gamma']:.4g}, C={matched['C']:g})")

    best_h = {"cv_f1": -1.0}
    for w in HYBRID_WEIGHTS:
        K = w * best_q["K"] + (1 - w) * matched["K"]
        C, s, _ = tune_C(K[np.ix_(tr, tr)], y_train, splits)
        print(f"  hybrid w_quantum={w:.2f}: CV F1(m)={s:.3f} (C={C:g})")
        if s > best_h["cv_f1"]:
            best_h = {"w": w, "C": C, "cv_f1": s, "K": K}

    names_qsvc = []
    for cfg in (best_q, best_ent):
        qkernel = FidelityStatevectorKernel(feature_map=build_feature_map(cfg["fm"], N_QUBITS, cfg["reps"]))
        qsvc = QSVC(quantum_kernel=qkernel, C=cfg["C"], class_weight="balanced")
        t0 = time.time()
        qsvc.fit(Z_all[tr] * cfg["bw"], y_train)
        q_pred = qsvc.predict(Z_all[te] * cfg["bw"])
        q_score = -qsvc.decision_function(Z_all[te] * cfg["bw"])
        q_time = time.time() - t0
        name = f"QSVC {cfg['fm']} ({N_QUBITS} qubits)"
        test_results[name] = compute_metrics(y_test, q_pred, q_score)
        test_scores[name], test_preds[name] = q_score, q_pred
        names_qsvc.append(name)
        print(f"  {name:32s} test {format_metrics(test_results[name])}  (QSVC fit+predict {q_time:.1f}s,"
              f" {len(qsvc.support_)} support vectors)")
    name_q, name_ent = names_qsvc

    name_m = f"SVC RBF same {N_QUBITS} feat"
    pred, score = fit_eval_precomputed(matched["K"], y_all, tr, te, matched["C"])
    test_results[name_m] = compute_metrics(y_test, pred, score)
    test_scores[name_m], test_preds[name_m] = score, pred
    print(f"  {name_m:32s} test {format_metrics(test_results[name_m])}")

    name_h = f"Hybrid kernel (w_q={best_h['w']:.2f})"
    pred, score = fit_eval_precomputed(best_h["K"], y_all, tr, te, best_h["C"])
    test_results[name_h] = compute_metrics(y_test, pred, score)
    test_scores[name_h], test_preds[name_h] = score, pred
    print(f"  {name_h:32s} test {format_metrics(test_results[name_h])}")

    print(f"\n[6/9] Paired significance: {CV_REPEATS}x{CV_FOLDS} repeated CV on training set")
    rep_splits = list(RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS,
                                              random_state=RANDOM_STATE).split(X_train, y_train))
    rep = {
        "quantum": cv_f1(best_q["K"][np.ix_(tr, tr)], y_train, rep_splits, best_q["C"]),
        "matched": cv_f1(matched["K"][np.ix_(tr, tr)], y_train, rep_splits, matched["C"]),
        "full": cv_f1(full_rbf["K"][np.ix_(tr, tr)], y_train, rep_splits, full_rbf["C"]),
        "hybrid": cv_f1(best_h["K"][np.ix_(tr, tr)], y_train, rep_splits, best_h["C"]),
    }
    n_val = len(tr) // CV_FOLDS
    comparisons = {
        "QSVC vs RBF (same features)": rep["quantum"] - rep["matched"],
        "QSVC vs RBF (30 features)": rep["quantum"] - rep["full"],
        "Hybrid vs RBF (30 features)": rep["hybrid"] - rep["full"],
    }
    significance = {}
    for label, d in comparisons.items():
        t, p = corrected_paired_ttest(d, len(tr) - n_val, n_val)
        significance[label] = {"diffs": d, "mean": float(d.mean()), "t": t, "p": p,
                               "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}
        print(f"  {label:30s} mean dF1={d.mean():+.4f}  t={t:+.2f}  p={p:.3f}  "
              f"W/T/L={significance[label]['wins']}/{significance[label]['ties']}/{significance[label]['losses']}")
    plot_significance(significance)

    print("\n[7/9] Qubit sweep + kernel diagnostics")
    sweep, concentration = {}, {}
    for n in QUBIT_SWEEP:
        Zn = make_encoder(n).fit(X_train).transform(X_all)
        best_n, concentration[n] = -1.0, {}
        for bw in BANDWIDTHS:
            K = quantum_gram(best_q["fm"], n, best_q["reps"], Zn * bw)
            concentration[n][bw] = float(np.std(off_diagonal(K[np.ix_(tr, tr)])))
            best_n = max(best_n, tune_C(K[np.ix_(tr, tr)], y_train, splits)[1])
        classical_n = tune_rbf(Zn, y_train, tr, splits, RBF_GAMMAS_ENCODED)["cv_f1"]
        sweep[n] = {"quantum": best_n, "classical": classical_n}
        print(f"  {n:2d} qubits: QSVC CV F1(m)={best_n:.3f}   RBF same features={classical_n:.3f}")
    plot_qubit_sweep(sweep, full_rbf["cv_f1"])

    Kq_tr = best_q["K"][np.ix_(tr, tr)]
    Km_tr = matched["K"][np.ix_(tr, tr)]
    Kf_tr = full_rbf["K"][np.ix_(tr, tr)]
    Klin_tr = linear_kernel(X_all_std[tr]) + 1.0
    kernel_views = {"Quantum (QSVC)": Kq_tr, "RBF, same features": Km_tr, "RBF, 30 features": Kf_tr}
    plot_kernel_matrices(kernel_views, y_train)
    alignment = {k: centered_alignment(K, y_train) for k, K in kernel_views.items()}
    spectra = {k: normalized_spectrum(K) for k, K in kernel_views.items()}
    eff_dim = {k: effective_dimension(K) for k, K in kernel_views.items()}
    geo = {
        "vs RBF, same features": geometric_difference(Km_tr, Kq_tr),
        "vs RBF, 30 features": geometric_difference(Kf_tr, Kq_tr),
        "vs linear, 30 features": geometric_difference(Klin_tr, Kq_tr),
    }
    for k in kernel_views:
        print(f"  {k:20s} alignment={alignment[k]:.3f}  effective dimension={eff_dim[k]:.1f}")
    for k, v in geo.items():
        print(f"  geometric difference g(K_C||K_Q) {k:24s} = {v:.2f}  (sqrt(N)={np.sqrt(len(tr)):.1f})")
    plot_kernel_diagnostics(concentration, spectra, alignment, geo, len(tr))

    print(f"\n[8/9] Learning curve: test F1(m) vs training size ({LC_SEEDS} seeds)")
    lc_kernels = {
        "QSVC (quantum kernel)": (best_q["K"], best_q["C"]),
        "RBF SVC, same features": (matched["K"], matched["C"]),
        "RBF SVC, 30 features": (full_rbf["K"], full_rbf["C"]),
    }
    lc = {name: {} for name in lc_kernels}
    for m in (*TRAIN_SIZES, len(tr)):
        for name in lc_kernels:
            lc[name][m] = []
        for seed in range(LC_SEEDS if m < len(tr) else 1):
            sub = tr if m == len(tr) else train_test_split(tr, train_size=m, stratify=y_train, random_state=seed)[0]
            for name, (K, C) in lc_kernels.items():
                pred, _ = fit_eval_precomputed(K, y_all, sub, te, C)
                lc[name][m].append(f1_score(y_test, pred, pos_label=MALIGNANT))
        print(f"  n={m:3d}  " + "  ".join(f"{k}={np.mean(v[m]):.3f}" for k, v in lc.items()))
    plot_learning_curve(lc)
    lc_tests = {}
    for m in TRAIN_SIZES:
        d = np.array(lc["QSVC (quantum kernel)"][m]) - np.array(lc["RBF SVC, same features"][m])
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        lc_tests[m] = {"mean_dF1": float(d.mean()), "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "p": p}
        print(f"  QSVC - RBF same features at n={m:3d}: dF1={d.mean():+.4f}  W/L={lc_tests[m]['wins']}/"
              f"{lc_tests[m]['losses']}  Wilcoxon p={p:.3f}")

    print("\n[9/9] Shot-noise robustness (finite-measurement emulation)")
    shot_results = {}
    for shots in SHOTS_GRID:
        shot_results[shots] = []
        for seed in range(SHOT_SEEDS):
            algorithm_globals.random_seed = RANDOM_STATE + seed
            K = quantum_gram(best_q["fm"], N_QUBITS, best_q["reps"], Z_all * best_q["bw"], shots=shots)
            pred, _ = fit_eval_precomputed(K, y_all, tr, te, best_q["C"])
            shot_results[shots].append(f1_score(y_test, pred, pos_label=MALIGNANT))
        print(f"  shots={shots:6d}: test F1(m)={np.mean(shot_results[shots]):.3f} ± {np.std(shot_results[shots]):.3f}")
    plot_shot_noise(shot_results, test_results[name_q]["f1_malignant"], test_results[name_m]["f1_malignant"])

    plot_leaderboard(test_results)
    plot_roc_confusion(
        y_test,
        {k: test_scores[k] for k in (name_full, name_m, name_q, name_h)},
        {k: test_preds[k] for k in (name_q, name_m)},
    )

    best_classical = max((k for k in test_results if not k.startswith(("QSVC", "Hybrid"))),
                         key=lambda k: test_results[k]["f1_malignant"])
    verdict = {
        "qsvc_vs_matched_rbf_test_dF1": test_results[name_q]["f1_malignant"] - test_results[name_m]["f1_malignant"],
        "qsvc_vs_best_classical_test_dF1": test_results[name_q]["f1_malignant"] - test_results[best_classical]["f1_malignant"],
        "best_classical_model": best_classical,
        "hybrid_vs_best_classical_test_dF1": test_results[name_h]["f1_malignant"] - test_results[best_classical]["f1_malignant"],
        "entangled_vs_product_qsvc_test_dF1": test_results[name_ent]["f1_malignant"] - test_results[name_q]["f1_malignant"],
        "learning_curve_qsvc_vs_matched_rbf": lc_tests,
        "missed_cancers": {k: int(confusion_matrix(y_test, test_preds[k], labels=[MALIGNANT, 1])[0, 1])
                           for k in (name_q, name_ent, name_h, name_m, best_classical)},
    }

    results = {
        "config": {"n_qubits": N_QUBITS, "feature_maps": FEATURE_MAPS, "reps_grid": REPS_GRID,
                   "bandwidths": BANDWIDTHS, "C_grid": C_GRID, "random_state": RANDOM_STATE},
        "selected_quantum": {k: v for k, v in best_q.items() if k != "K"},
        "selected_entangling_quantum": {k: v for k, v in best_ent.items() if k != "K"},
        "selected_matched_rbf": {k: v for k, v in matched.items() if k != "K"},
        "selected_full_rbf": {k: v for k, v in full_rbf.items() if k != "K"},
        "selected_hybrid": {k: v for k, v in best_h.items() if k != "K"},
        "test_results": test_results,
        "repeated_cv_mean_f1": {k: float(v.mean()) for k, v in rep.items()},
        "significance": {k: {kk: vv for kk, vv in v.items() if kk != "diffs"} for k, v in significance.items()},
        "qubit_sweep": sweep,
        "kernel_alignment": alignment,
        "effective_dimension": eff_dim,
        "geometric_difference": geo,
        "learning_curve_mean_f1": {k: {str(m): float(np.mean(v)) for m, v in d.items()} for k, d in lc.items()},
        "shot_noise_mean_f1": {str(s): float(np.mean(v)) for s, v in shot_results.items()},
        "verdict": verdict,
        "runtime_seconds": time.time() - t_start,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))

    print("\n================ RESULTS (held-out test set, positive class = malignant) ================")
    for name in sorted(test_results, key=lambda k: -test_results[k]["f1_malignant"]):
        print(f"  {name:32s} {format_metrics(test_results[name])}")
    print("\n================ VERDICT ================")
    sig_m = significance["QSVC vs RBF (same features)"]
    rows = [
        (f"QSVC vs RBF, identical {N_QUBITS} features",
         f"dF1 {verdict['qsvc_vs_matched_rbf_test_dF1']:+.3f} test | {sig_m['mean']:+.4f} repeated CV (p={sig_m['p']:.3f})"),
        (f"QSVC vs best classical ({best_classical})", f"dF1 {verdict['qsvc_vs_best_classical_test_dF1']:+.3f} test"),
        ("Hybrid vs best classical", f"dF1 {verdict['hybrid_vs_best_classical_test_dF1']:+.3f} test"),
        ("Entangling vs product-state QSVC", f"dF1 {verdict['entangled_vs_product_qsvc_test_dF1']:+.3f} test"),
        *[(f"Learning curve n={m}", f"QSVC - RBF same features {v['mean_dF1']:+.4f} (W/L {v['wins']}/{v['losses']}, p={v['p']:.3f})")
          for m, v in lc_tests.items()],
        ("Missed cancers on test set", str(verdict["missed_cancers"])),
    ]
    width = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"  {label:<{width}} : {value}")
    print(f"\n  Total runtime {time.time() - t_start:.0f}s. Figures + results.json in {OUT_DIR}")


if __name__ == "__main__":
    main()
