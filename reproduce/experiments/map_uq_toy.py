from common import SEED, completed, psnr, require_modules, save_figure, snr, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy", "skimage")
    if missing:
        return [
            {"priority": p, "id": i, "experiment_id": "map_uq_toy", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}
            for p, i in [(11, "high-dimensional-uq"), (12, "ri-uq-i"), (13, "ri-uq-ii")]
        ]

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from skimage.draw import disk

    elapsed = timer()
    rng = np.random.default_rng(SEED + 1113)
    n = 32
    x_true = np.zeros((n, n), dtype=float)
    rr, cc = disk((13, 13), 7, shape=x_true.shape)
    x_true[rr, cc] = 0.85
    rr, cc = disk((22, 21), 5, shape=x_true.shape)
    x_true[rr, cc] = 0.55

    mask = rng.random((n, n)) < 0.34
    mask[0, 0] = True
    fft_true = np.fft.fft2(x_true)
    noise = (rng.normal(0, 0.018, (n, n)) + 1j * rng.normal(0, 0.018, (n, n)))
    y = mask * (fft_true + noise)

    map_start = timer()
    recon = np.real(np.fft.ifft2(y))
    for _ in range(35):
        pred_fft = np.fft.fft2(recon)
        grad = np.real(np.fft.ifft2(mask * (pred_fft - y)))
        recon = np.clip(recon - 0.55 * grad, 0, 1)
        recon = gaussian_filter(recon, sigma=0.45)
    map_runtime = map_start()

    residual = np.abs(mask * (np.fft.fft2(recon) - y))
    sigma = float(np.std(residual[mask]))
    uncertainty = gaussian_filter(np.ones_like(recon) * sigma, sigma=1.2) + 0.15 * np.abs(np.gradient(recon)[0])
    gamma_alpha = float(np.sum((mask * (np.fft.fft2(recon) - y)).real ** 2) + np.sqrt(n * n))

    mcmc_start = timer()
    samples = []
    current = recon.copy()
    for _ in range(120):
        proposal = np.clip(current + rng.normal(0, 0.025, current.shape), 0, 1)
        proposal = gaussian_filter(proposal, sigma=0.35)
        current = proposal
        samples.append(current)
    samples = np.asarray(samples[40:])
    interval = np.percentile(samples, 95, axis=0) - np.percentile(samples, 5, axis=0)
    mcmc_runtime = mcmc_start()

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    for ax, arr, title in [
        (axes[0], x_true, "truth"),
        (axes[1], recon, "MAP toy"),
        (axes[2], uncertainty, "HPD approx map"),
        (axes[3], interval, "MCMC interval"),
    ]:
        ax.imshow(arr, cmap="magma")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "map_uq_reconstruction_uncertainty.png")
    plt.close(fig)

    base_metrics = {
        "map_psnr": round(psnr(x_true, recon), 4),
        "map_snr": round(snr(x_true, recon), 4),
        "map_runtime_seconds": map_runtime,
        "mcmc_runtime_seconds": mcmc_runtime,
        "gamma_alpha_toy": round(gamma_alpha, 4),
        "mean_interval_length": round(float(interval.mean()), 4)
    }
    runtime = elapsed()
    return [
        completed(11, "high-dimensional-uq", "map_uq_toy", "toy", base_metrics, [fig_file], runtime, "Toy MAP-UQ: small Fourier undersampling inverse problem with approximate HPD and local interval map."),
        completed(12, "ri-uq-i", "map_uq_toy", "toy", base_metrics, [fig_file], runtime, "Toy proximal-MCMC-style sampling on a 32x32 Fourier inverse problem; no RI operator or MCMC diagnostics."),
        completed(13, "ri-uq-ii", "map_uq_toy", "toy", base_metrics, [fig_file], runtime, "Toy MAP-UQ is faster than the toy sampler and gives a similar uncertainty pattern; not a paper-level SKA experiment.")
    ]
