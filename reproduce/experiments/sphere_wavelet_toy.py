from common import SEED, completed, dice_score, require_modules, save_figure, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy")
    if missing:
        return [{"priority": 8, "id": "sphere-wavelet", "experiment_id": "sphere_wavelet_toy", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    elapsed = timer()
    rng = np.random.default_rng(SEED + 8)
    h, w = 72, 144
    lat = np.linspace(-np.pi / 2, np.pi / 2, h)[:, None]
    lon = np.linspace(-np.pi, np.pi, w)[None, :]
    truth = (np.abs(lat - 0.35 * np.sin(2 * lon)) < 0.11) | (np.abs(lat + 0.2) < 0.06)
    image = truth.astype(float) * 0.72 + 0.22 + rng.normal(0, 0.12, (h, w))
    smooth = gaussian_filter(image, sigma=(1.0, 1.6), mode=("reflect", "wrap"))
    grad_lat, grad_lon = np.gradient(smooth)
    spherical_grad = np.sqrt(grad_lat ** 2 + (grad_lon / np.maximum(np.cos(lat), 0.2)) ** 2)
    pred = (smooth > 0.47) | (spherical_grad > np.quantile(spherical_grad, 0.93))
    dice = dice_score(truth, pred)

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.5))
    for ax, arr, title in [
        (axes[0], image, "equirectangular toy"),
        (axes[1], truth, "truth bands"),
        (axes[2], spherical_grad, "approx sphere grad"),
        (axes[3], pred, "segmentation"),
    ]:
        ax.imshow(arr, cmap="viridis", aspect="auto")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "sphere_wavelet_toy.png")
    plt.close(fig)

    return [completed(8, "sphere-wavelet", "sphere_wavelet_toy", "toy", {
        "dice": round(dice, 4),
        "gradient_threshold_quantile": 0.93
    }, [fig_file], elapsed(), "Approximate sphere toy: equirectangular smoothing plus spherical-gradient correction; no S2LET/SSHT/SO3 stack.")]
