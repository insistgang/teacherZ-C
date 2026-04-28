from common import SEED, completed, psnr, require_modules, save_figure, snr, timer


def run():
    missing = require_modules("numpy", "matplotlib", "skimage")
    if missing:
        return [{"priority": 14, "id": "online-ri", "experiment_id": "online_ri_toy", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}]

    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.draw import disk

    elapsed = timer()
    rng = np.random.default_rng(SEED + 14)
    n = 40
    truth = np.zeros((n, n), dtype=float)
    rr, cc = disk((17, 18), 8, shape=truth.shape)
    truth[rr, cc] = 0.9
    rr, cc = disk((27, 27), 5, shape=truth.shape)
    truth[rr, cc] = 0.45
    full_fft = np.fft.fft2(truth)
    all_mask = rng.random((n, n)) < 0.38
    all_mask[0, 0] = True
    y = all_mask * (full_fft + rng.normal(0, 0.012, full_fft.shape))

    offline = np.clip(np.real(np.fft.ifft2(y)), 0, 1)
    coords = np.argwhere(all_mask)
    rng.shuffle(coords)
    blocks = np.array_split(coords, 6)
    online_y = np.zeros_like(y)
    online_mask = np.zeros_like(all_mask)
    quality_trace = []
    peak_online = 0
    recon = np.zeros_like(truth)
    for block in blocks:
        peak_online = max(peak_online, len(block))
        for r, c in block:
            online_mask[r, c] = True
            online_y[r, c] = y[r, c]
        recon = np.clip(np.real(np.fft.ifft2(online_y)), 0, 1)
        quality_trace.append(psnr(truth, recon))

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    for ax, arr, title in [
        (axes[0], truth, "truth"),
        (axes[1], offline, "offline"),
        (axes[2], recon, "online final"),
        (axes[3], quality_trace, "online PSNR"),
    ]:
        if title == "online PSNR":
            ax.plot(arr, marker="o")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("block")
        else:
            ax.imshow(arr, cmap="magma")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "online_ri_storage_quality.png")
    plt.close(fig)

    return [completed(14, "online-ri", "online_ri_toy", "toy", {
        "offline_psnr": round(psnr(truth, offline), 4),
        "online_psnr": round(psnr(truth, recon), 4),
        "offline_snr": round(snr(truth, offline), 4),
        "online_snr": round(snr(truth, recon), 4),
        "peak_stored_measurements_offline": int(all_mask.sum()),
        "peak_stored_measurements_online": int(peak_online)
    }, [fig_file], elapsed(), "Toy online RI: split Fourier measurements into blocks, assimilate each block, then discard it conceptually.")]
