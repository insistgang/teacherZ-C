from common import SEED, clustering_accuracy, completed, require_modules, save_figure, simple_kmeans, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy")
    if missing:
        return [{"priority": 7, "id": "slat-color", "experiment_id": "slat_color", "reproductionLevel": "partial", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    elapsed = timer()
    rng = np.random.default_rng(SEED + 7)
    n = 96
    truth = np.zeros((n, n), dtype=int)
    truth[:48, 48:] = 1
    truth[48:, :48] = 2
    truth[48:, 48:] = 3
    colors = np.array([
        [0.82, 0.25, 0.22],
        [0.70, 0.36, 0.20],
        [0.25, 0.55, 0.78],
        [0.20, 0.62, 0.42],
    ])
    rgb = colors[truth]
    degraded = np.clip(rgb + rng.normal(0, 0.15, rgb.shape), 0, 1)
    degraded[30:65, 40:58] = degraded[30:65, 40:58] * 0.55
    smooth = np.stack([gaussian_filter(degraded[..., c], sigma=1.1) for c in range(3)], axis=-1)
    luminance = 0.2126 * smooth[..., 0] + 0.7152 * smooth[..., 1] + 0.0722 * smooth[..., 2]
    rg = smooth[..., 0] - smooth[..., 1]
    yb = 0.5 * (smooth[..., 0] + smooth[..., 1]) - smooth[..., 2]
    lab = np.stack([luminance, rg, yb], axis=-1)

    rgb_labels = simple_kmeans(smooth.reshape(-1, 3), 4, seed=SEED).reshape(n, n)
    rgblab_features = np.concatenate([smooth, lab], axis=-1)
    rgblab_labels = simple_kmeans(rgblab_features.reshape(-1, 6), 4, seed=SEED).reshape(n, n)
    rgb_acc = clustering_accuracy(truth, rgb_labels)
    rgblab_acc = clustering_accuracy(truth, rgblab_labels)

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    for ax, arr, title in [
        (axes[0], degraded, "degraded RGB"),
        (axes[1], truth, "truth"),
        (axes[2], rgb_labels, "RGB only"),
        (axes[3], rgblab_labels, "RGB + Lab"),
    ]:
        ax.imshow(arr, cmap=None if arr.ndim == 3 else "viridis")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "slat_rgb_vs_rgblab.png")
    plt.close(fig)

    return [completed(7, "slat-color", "slat_color", "partial", {
        "rgb_only_accuracy": round(rgb_acc, 4),
        "rgb_lab_accuracy": round(rgblab_acc, 4),
        "accuracy_gain": round(rgblab_acc - rgb_acc, 4)
    }, [fig_file], elapsed(), "Toy SLaT: channel smoothing, RGB plus Lab-like luminance/chroma lifting, K-means on synthetic degraded color image. Current toy shows only a small metric gain; a better synthetic color case is needed to highlight Lab lifting.")]
