from common import SEED, clustering_accuracy, completed, require_modules, save_figure, simple_kmeans, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy")
    if missing:
        return [{"priority": 4, "id": "segmentation-restoration", "experiment_id": "segmentation_restoration", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    elapsed = timer()
    rng = np.random.default_rng(SEED + 4)
    n = 96
    yy, xx = np.mgrid[:n, :n]
    truth = np.zeros((n, n), dtype=int)
    truth[(xx - 34) ** 2 + (yy - 45) ** 2 < 18 ** 2] = 1
    truth[(xx - 65) ** 2 + (yy - 50) ** 2 < 16 ** 2] = 2
    levels = np.array([0.22, 0.52, 0.78])
    clean = levels[truth]
    degraded = gaussian_filter(clean, sigma=2.0) + rng.normal(0, 0.11, clean.shape)
    missing = rng.random(clean.shape) < 0.12
    degraded[missing] = 0.0
    direct = simple_kmeans(degraded.reshape(-1, 1), 3, seed=SEED).reshape(n, n)

    g = degraded.copy()
    labels = direct.copy()
    for _ in range(8):
        restored = gaussian_filter(g, sigma=1.1)
        g[missing] = restored[missing]
        g = 0.55 * g + 0.45 * restored
        labels = simple_kmeans(g.reshape(-1, 1), 3, seed=SEED).reshape(n, n)

    joint_acc = clustering_accuracy(truth, labels)
    direct_acc = clustering_accuracy(truth, direct)
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    for ax, arr, title in [
        (axes[0], degraded, "degraded"),
        (axes[1], truth, "truth"),
        (axes[2], direct, "direct K-means"),
        (axes[3], labels, "joint toy"),
    ]:
        ax.imshow(arr, cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "segmentation_restoration_toy.png")
    plt.close(fig)

    return [completed(4, "segmentation-restoration", "segmentation_restoration", "toy", {
        "direct_accuracy": round(direct_acc, 4),
        "joint_toy_accuracy": round(joint_acc, 4),
        "accuracy_gain": round(joint_acc - direct_acc, 4),
        "alternating_iterations": 8
    }, [fig_file], elapsed(), "Toy alternating restoration-segmentation over g, class means and labels; not full variational AM proof reproduction.")]
