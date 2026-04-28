from common import SEED, clustering_accuracy, completed, require_modules, save_figure, simple_kmeans, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy")
    if missing:
        return [
            {"priority": p, "id": i, "experiment_id": "sat_rof_trof", "reproductionLevel": "partial", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}
            for p, i in [(1, "sat-overview"), (2, "pcms-rof-linkage"), (3, "iterated-rof")]
        ]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    elapsed = timer()
    rng = np.random.default_rng(SEED)
    n = 96
    yy, xx = np.mgrid[:n, :n]

    truth2 = ((xx - 48) ** 2 + (yy - 48) ** 2 < 26 ** 2).astype(int)
    image2 = truth2 * 0.75 + (1 - truth2) * 0.22 + rng.normal(0, 0.16, (n, n))
    smooth2 = gaussian_filter(image2, sigma=1.1)
    baseline2 = image2 > 0.48
    sat2 = smooth2 > 0.48

    truth4 = np.zeros((n, n), dtype=int)
    truth4[(yy < 48) & (xx >= 48)] = 1
    truth4[(yy >= 48) & (xx < 48)] = 2
    truth4[(yy >= 48) & (xx >= 48)] = 3
    levels = np.array([0.26, 0.36, 0.47, 0.58])
    image4 = levels[truth4] + rng.normal(0, 0.07, (n, n))
    smooth4 = gaussian_filter(image4, sigma=1.0)

    km_raw = simple_kmeans(image4.reshape(-1, 1), 4, seed=SEED).reshape(n, n)
    km_sat = simple_kmeans(smooth4.reshape(-1, 1), 4, seed=SEED).reshape(n, n)

    thresholds = np.quantile(smooth4, [0.25, 0.5, 0.75])
    threshold_trace = [thresholds.copy()]
    trof_labels = np.digitize(smooth4, thresholds)
    for _ in range(12):
        means = []
        for k in range(4):
            vals = smooth4[trof_labels == k]
            means.append(float(vals.mean()) if vals.size else float(levels[k]))
        new_thresholds = np.array([(means[i] + means[i + 1]) / 2 for i in range(3)])
        threshold_trace.append(new_thresholds.copy())
        if np.max(np.abs(new_thresholds - thresholds)) < 1e-4:
            thresholds = new_thresholds
            break
        thresholds = new_thresholds
        trof_labels = np.digitize(smooth4, thresholds)

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.8))
    for ax, arr, title in [
        (axes[0, 0], truth2, "truth 2-phase"),
        (axes[0, 1], image2, "degraded"),
        (axes[0, 2], baseline2, "direct threshold"),
        (axes[0, 3], sat2, "TV + threshold"),
        (axes[1, 0], truth4, "truth 4-phase"),
        (axes[1, 1], image4, "close gray noisy"),
        (axes[1, 2], km_raw, "raw K-means"),
        (axes[1, 3], trof_labels, "T-ROF toy"),
    ]:
        ax.imshow(arr, cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    sat_file = save_figure(fig, "sat_demo.png")
    plt.close(fig)

    trace = np.array(threshold_trace)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i in range(trace.shape[1]):
        ax.plot(trace[:, i], marker="o", label=f"tau{i + 1}")
    ax.set_title("T-ROF threshold updates")
    ax.set_xlabel("iteration")
    ax.set_ylabel("threshold")
    ax.legend(fontsize=8)
    trof_file = save_figure(fig, "trof_thresholds.png")
    plt.close(fig)

    runtime = elapsed()
    acc_raw = clustering_accuracy(truth4, km_raw)
    acc_sat = clustering_accuracy(truth4, km_sat)
    acc_trof = clustering_accuracy(truth4, trof_labels)
    dice_direct = float((2 * ((truth2 == 1) & baseline2).sum()) / ((truth2 == 1).sum() + baseline2.sum()))
    dice_sat = float((2 * ((truth2 == 1) & sat2).sum()) / ((truth2 == 1).sum() + sat2.sum()))

    return [
        completed(1, "sat-overview", "sat_rof_trof", "toy-to-partial", {
            "direct_accuracy": round(acc_raw, 4),
            "sat_accuracy": round(acc_sat, 4),
            "accuracy_gain": round(acc_sat - acc_raw, 4)
        }, [sat_file], runtime, "Gaussian smoothing is used as a lightweight proxy for convex ROF/TV smoothing on a synthetic toy image."),
        completed(2, "pcms-rof-linkage", "sat_rof_trof", "toy-to-partial", {
            "direct_dice": round(dice_direct, 4),
            "rof_threshold_dice": round(dice_sat, 4),
            "pcms_like_energy": round(float(np.abs(np.gradient(sat2.astype(float))[0]).sum() + np.abs(np.gradient(sat2.astype(float))[1]).sum()), 4)
        }, [sat_file], runtime, "This synthetic toy demonstrates thresholding after proxy smoothing, but does not solve the exact ROF model or prove Theorem 3.6."),
        completed(3, "iterated-rof", "sat_rof_trof", "toy-to-partial", {
            "raw_kmeans_accuracy": round(acc_raw, 4),
            "trof_accuracy": round(acc_trof, 4),
            "threshold_iterations": len(threshold_trace) - 1
        }, [sat_file, trof_file], runtime, "This synthetic toy implements the threshold update tau_i = 1/2(m_{i-1}+m_i) after proxy smoothing; strict T-ROF should solve ROF once.")
    ]
