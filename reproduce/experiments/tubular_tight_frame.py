from common import SEED, completed, dice_score, iou_score, require_modules, save_figure, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy", "skimage")
    if missing:
        return [
            {"priority": p, "id": i, "experiment_id": "tubular_tight_frame", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}
            for p, i in [(5, "framelet-tubular"), (6, "tight-frame-vessel")]
        ]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from skimage.draw import line
    from skimage.morphology import dilation, disk

    elapsed = timer()
    rng = np.random.default_rng(SEED + 56)
    n = 112
    mask = np.zeros((n, n), dtype=bool)
    segments = [(12, 18, 96, 86), (18, 84, 78, 32), (45, 8, 52, 105), (70, 25, 102, 55)]
    for r0, c0, r1, c1 in segments:
        rr, cc = line(r0, c0, r1, c1)
        mask[rr, cc] = True
    mask = dilation(mask, disk(3))
    image = mask.astype(float) * 0.75 + 0.18 + rng.normal(0, 0.13, (n, n))
    image = np.clip(image, 0, 1)

    current = image.copy()
    lambda_sizes = []
    alpha, beta = 0.38, 0.62
    for _ in range(12):
        uncertain = (current > alpha) & (current < beta)
        lambda_sizes.append(int(uncertain.sum()))
        if uncertain.sum() == 0:
            break
        smooth = gaussian_filter(current, sigma=1.0)
        current[uncertain] = smooth[uncertain]
        pred = current > 0.5
        current[pred & (current >= beta)] = 1.0
        current[(~pred) & (current <= alpha)] = 0.0
        alpha += 0.008
        beta -= 0.008
        if beta <= alpha:
            break
    pred = current > 0.5
    dice = dice_score(mask, pred)
    iou = iou_score(mask, pred)

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    for ax, arr, title in [
        (axes[0], image, "noisy tube"),
        (axes[1], mask, "truth"),
        (axes[2], pred, "toy output"),
        (axes[3], lambda_sizes, "Lambda size"),
    ]:
        if title == "Lambda size":
            ax.plot(arr, marker="o")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("iter")
        else:
            ax.imshow(arr, cmap="gray")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "tubular_lambda_shrinkage.png")
    plt.close(fig)

    metrics = {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "lambda_initial": lambda_sizes[0],
        "lambda_final": lambda_sizes[-1],
        "iterations": len(lambda_sizes)
    }
    runtime = elapsed()
    return [
        completed(5, "framelet-tubular", "tubular_tight_frame", "toy", metrics, [fig_file], runtime, "Approximate toy reproduction: Gaussian smoothing stands in for framelet smoothing inside uncertain boundary interval. Dice is measured on a simple synthetic 2D vessel toy; it does not represent real 2D/3D MRA paper-level performance."),
        completed(6, "tight-frame-vessel", "tubular_tight_frame", "toy", metrics, [fig_file], runtime, "Approximate toy reproduction: Lambda boundary set shrinkage and finite convergence pattern on synthetic 2D vessel network. Dice is measured on a simple synthetic 2D vessel toy; it does not represent real 2D/3D MRA paper-level performance.")
    ]
