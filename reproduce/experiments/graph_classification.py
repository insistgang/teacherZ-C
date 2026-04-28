from common import SEED, completed, require_modules, save_figure, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy")
    if missing:
        return [
            {"priority": p, "id": i, "experiment_id": "graph_classification", "reproductionLevel": "partial", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}
            for p, i in [(9, "two-stage-classification"), (10, "efficient-variational-classification")]
        ]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    elapsed = timer()
    rng = np.random.default_rng(SEED + 910)
    n = 360
    t = rng.uniform(0, np.pi, n // 2)
    moon1 = np.c_[np.cos(t), np.sin(t)]
    moon2 = np.c_[1 - np.cos(t), 0.48 - np.sin(t)]
    x = np.vstack([moon1, moon2]) + rng.normal(0, 0.18, (n, 2))
    y = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    labeled = np.zeros_like(y, dtype=bool)
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        labeled[rng.choice(idx, size=10, replace=False)] = True

    centroids = np.vstack([x[labeled & (y == cls)].mean(axis=0) for cls in [0, 1]])
    warm = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    initial_acc = float((warm == y).mean())

    tree = cKDTree(x)
    _, neigh = tree.query(x, k=13)
    graph = np.zeros((len(x), len(x)), dtype=float)
    for row, cols in enumerate(neigh):
        graph[row, cols] = 1.0
    graph = np.maximum(graph, graph.T)
    degree = np.maximum(graph.sum(axis=1), 1.0)
    labels = warm.copy()
    probs = np.zeros((len(y), 2), dtype=float)
    probs[np.arange(len(y)), labels] = 1.0
    probs[labeled] = 0
    probs[labeled, y[labeled]] = 1.0
    acc_trace = [initial_acc]
    for _ in range(18):
        probs = 0.72 * probs + 0.28 * (graph @ probs) / degree[:, None]
        probs[labeled] = 0
        probs[labeled, y[labeled]] = 1.0
        labels = probs.argmax(axis=1)
        acc_trace.append(float((labels == y).mean()))
    smooth_acc = acc_trace[-1]

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    axes[0].scatter(x[:, 0], x[:, 1], c=warm, s=12, cmap="coolwarm")
    axes[0].scatter(x[labeled, 0], x[labeled, 1], c="black", s=18, marker="x")
    axes[0].set_title("warm init", fontsize=8)
    axes[1].scatter(x[:, 0], x[:, 1], c=labels, s=12, cmap="coolwarm")
    axes[1].set_title("graph smoothing", fontsize=8)
    axes[2].plot(acc_trace, marker="o")
    axes[2].set_title("accuracy trace", fontsize=8)
    for ax in axes[:2]:
        ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "graph_classification_before_after.png")
    plt.close(fig)

    metrics = {
        "initial_accuracy": round(initial_acc, 4),
        "smoothed_accuracy": round(smooth_acc, 4),
        "accuracy_gain": round(smooth_acc - initial_acc, 4),
        "iterations": len(acc_trace) - 1
    }
    runtime = elapsed()
    return [
        completed(9, "two-stage-classification", "graph_classification", "partial", metrics, [fig_file], runtime, "Toy graph classification: centroid warm initialization, kNN graph smoothing, argmax projection."),
        completed(10, "efficient-variational-classification", "graph_classification", "partial", metrics, [fig_file], runtime, "Toy repeated graph smoothing: demonstrates independent label-function update idea without full graph TV primal-dual solver.")
    ]
