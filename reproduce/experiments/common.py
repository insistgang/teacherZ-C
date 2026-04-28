import csv
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path


SEED = 20260428
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "reproduce" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DOCS_FIGURES_DIR = REPO_ROOT / "docs" / "assets" / "repro"


def ensure_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def has_module(name):
    return importlib.util.find_spec(name) is not None


def require_modules(*names):
    missing = []
    for name in names:
        if not has_module(name):
            missing.append(name)
            continue
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
    return missing


def timer():
    start = time.perf_counter()
    return lambda: round(time.perf_counter() - start, 4)


def dice_score(truth, pred):
    truth = truth.astype(bool)
    pred = pred.astype(bool)
    denom = truth.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2 * (truth & pred).sum() / denom)


def iou_score(truth, pred):
    truth = truth.astype(bool)
    pred = pred.astype(bool)
    union = (truth | pred).sum()
    if union == 0:
        return 1.0
    return float((truth & pred).sum() / union)


def psnr(truth, pred):
    import numpy as np

    mse = float(np.mean((truth - pred) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20 * math.log10(1.0 / math.sqrt(mse)))


def snr(truth, pred):
    import numpy as np

    noise = np.linalg.norm(truth - pred)
    signal = np.linalg.norm(truth)
    if noise <= 1e-12:
        return 99.0
    return float(20 * np.log10(signal / noise))


def clustering_accuracy(truth, labels):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    truth = np.asarray(truth).ravel()
    labels = np.asarray(labels).ravel()
    true_vals = np.unique(truth)
    label_vals = np.unique(labels)
    matrix = np.zeros((len(true_vals), len(label_vals)), dtype=int)
    for i, t in enumerate(true_vals):
        for j, l in enumerate(label_vals):
            matrix[i, j] = np.sum((truth == t) & (labels == l))
    row, col = linear_sum_assignment(-matrix)
    return float(matrix[row, col].sum() / truth.size)


def simple_kmeans(features, n_clusters, seed=SEED, n_iter=30):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < n_clusters:
        raise ValueError("n_clusters larger than sample count")
    centers = x[rng.choice(len(x), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(n_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            if np.any(labels == k):
                centers[k] = x[labels == k].mean(axis=0)
    return labels


def save_figure(fig, filename):
    ensure_dirs()
    docs_path = DOCS_FIGURES_DIR / filename
    result_path = FIGURES_DIR / filename
    fig.savefig(docs_path, dpi=130, bbox_inches="tight")
    shutil.copyfile(docs_path, result_path)
    return f"assets/repro/{filename}"


def completed(priority, paper_id, experiment_id, reproduction_level, metrics, figure_files, runtime_seconds, notes="", extra=None):
    return {
        "priority": priority,
        "id": paper_id,
        "experiment_id": experiment_id,
        "reproductionLevel": reproduction_level,
        "status": "completed",
        "runtime_seconds": runtime_seconds,
        "metrics": metrics,
        "resultFiles": figure_files,
        "notes": notes,
        **(extra or {})
    }


def skipped(priority, paper_id, experiment_id, reproduction_level, reason):
    return {
        "priority": priority,
        "id": paper_id,
        "experiment_id": experiment_id,
        "reproductionLevel": reproduction_level,
        "status": "skipped",
        "runtime_seconds": 0.0,
        "metrics": {},
        "resultFiles": [],
        "skipped_reason": reason,
        "notes": "Dependency or runtime guard skipped this experiment."
    }


def write_results(results):
    ensure_dirs()
    json_path = RESULTS_DIR / "repro_results.json"
    csv_path = RESULTS_DIR / "repro_results.csv"
    json_payload = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    json_path.write_bytes(json_payload.encode("utf-8"))

    metric_keys = sorted({key for row in results for key in row.get("metrics", {}).keys()})
    fieldnames = [
        "priority",
        "id",
        "experiment_id",
        "reproductionLevel",
        "status",
        "runtime_seconds",
        "skipped_reason",
        "notes",
        "resultFiles"
    ] + metric_keys
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in results:
            flat = {key: row.get(key, "") for key in fieldnames}
            flat["resultFiles"] = ";".join(row.get("resultFiles", []))
            flat["skipped_reason"] = row.get("skipped_reason", "")
            for key, value in row.get("metrics", {}).items():
                flat[key] = value
            writer.writerow(flat)
    docs_json = DOCS_FIGURES_DIR / "repro_results.json"
    docs_json.write_bytes(json_payload.encode("utf-8"))
    return json_path, csv_path
