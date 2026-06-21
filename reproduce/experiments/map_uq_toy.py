"""Real l1-wavelet MAP reconstruction + local credible intervals.

Shared runner for the three RI-UQ papers (priority 11/12/13):
  #11 high-dimensional-uq, #12 ri-uq-i, #13 ri-uq-ii.

This is a faithful, runnable implementation of the paper method
(Cai-Pereyra-McEwen) on a NON-paper test image:

  1. Undersampled Fourier inverse problem y = A x + n, with a
     variable-density (low-frequency biased) ~10% sampling mask and
     i.i.d. complex Gaussian noise at SNR = 30 dB (paper convention).
  2. l1-wavelet (Daubechies-8) ANALYSIS-prior MAP estimate solved by
     FISTA / forward-backward splitting with a closed-form soft-threshold
     prox (Psi orthonormal => Psi^T Psi = I).
  3. Real HPD threshold gamma'_alpha = f(x*) + g(x*) + sqrt(16 log(3/alpha)) sqrt(N) + N
     (Cai et al. Eq.(6) / Eq.(19), alpha = 0.01 => 99% credible level).
  4. Real LOCAL CREDIBLE INTERVALS: for each superpixel, bisection search
     for the min/max constant intensity xi keeping the perturbed image
     inside the HPD region C'_alpha (Eq.(7-9) / Eq.(26-28)); report the
     mean interval width.

reproductionLevel stays "toy" and a fidelityWarning is attached: this is
the real method on a standard test image, NOT the paper's M31 / BrainWeb /
RI data, and there is no true radio-interferometric (NUFFT) operator, no
auto-mu, and no Px-MALA / MYULA sampler baseline.
"""

from common import SEED, completed, psnr, require_modules, save_figure, snr, timer


def run():
    missing = require_modules("numpy", "matplotlib", "scipy", "skimage", "pywt")
    if missing:
        return [
            {"priority": p, "id": i, "experiment_id": "map_uq_toy", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}
            for p, i in [(11, "high-dimensional-uq"), (12, "ri-uq-i"), (13, "ri-uq-ii")]
        ]

    import math

    import numpy as np
    import matplotlib.pyplot as plt
    import pywt
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    elapsed = timer()
    rng = np.random.default_rng(SEED + 1113)

    # --- ground truth: standard Shepp-Logan phantom (NOT the paper data) ---
    n = 64
    N = n * n
    img = shepp_logan_phantom()
    img = resize(img, (n, n), anti_aliasing=True)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    # --- forward operator A: undersampled, low-frequency-biased Fourier ---
    # (variable-density profile, in the spirit of RI u-v coverage; ~10% samples)
    fr = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(fr, fr, indexing="ij")
    radius = np.sqrt(fx ** 2 + fy ** 2)
    prob = np.exp(-(radius / 0.18) ** 2)
    prob = prob / prob.max()
    mask = rng.random((n, n)) < prob
    mask[0, 0] = True  # always keep the DC component
    sampling_rate = float(mask.mean())

    def A(x):
        return mask * np.fft.fft2(x) / np.sqrt(N)

    def At(yv):
        return np.real(np.fft.ifft2(mask * yv)) * np.sqrt(N)

    # --- noisy measurements at SNR = 30 dB (paper convention) ---
    ft = A(img)
    snr_db = 30.0
    sig_power = float(np.linalg.norm(ft[mask]) / np.sqrt(mask.sum()))
    sigma = sig_power * 10 ** (-snr_db / 20)
    cplx_noise = (rng.normal(0, sigma, (n, n)) + 1j * rng.normal(0, sigma, (n, n))) / np.sqrt(2)
    y = ft + mask * cplx_noise

    # --- Daubechies-8 orthonormal wavelet dictionary Psi (analysis prior) ---
    wavelet = "db8"
    level = 2
    mode = "periodization"
    _coeffs = pywt.wavedec2(img, wavelet, level=level, mode=mode)
    _, slices = pywt.coeffs_to_array(_coeffs)

    def psi_t(x):  # analysis: Psi^T x  (wavelet decomposition)
        coeffs = pywt.wavedec2(x, wavelet, level=level, mode=mode)
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr

    def psi(arr):  # synthesis: Psi a  (wavelet reconstruction)
        coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedec2")
        return pywt.waverec2(coeffs, wavelet, mode=mode)

    mu = 3.0 * sigma  # l1 regularisation parameter

    def f_prior(x):  # f(x) = mu ||Psi^T x||_1
        return float(mu * np.sum(np.abs(psi_t(x))))

    def g_data(x):  # g(x) = ||y - A x||^2 / (2 sigma^2)
        return float(np.sum(np.abs(y - A(x)) ** 2) / (2 * sigma ** 2))

    def grad_g(x):  # nabla g(x) = A^H (A x - y) / sigma^2
        return At(A(x) - y) / sigma ** 2

    def soft(z, thr):
        return np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)

    def prox_f(x, step):  # analysis prox, closed form since Psi^T Psi = I (Eq.29)
        u = psi_t(x)
        return x + psi(soft(u, step * mu) - u)

    # Lipschitz constant of grad_g: ||A^H A|| / sigma^2 = 1 / sigma^2 (A subsampled unitary)
    lip = 1.0 / sigma ** 2
    step = 1.0 / lip

    # --- MAP via FISTA / forward-backward splitting ---
    map_start = timer()
    x = At(y)            # dirty image initialisation
    z = x.copy()
    t_k = 1.0
    for _ in range(250):
        x_prev = x
        x = prox_f(z - step * grad_g(z), step)
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
        z = x + ((t_k - 1.0) / t_next) * (x - x_prev)
        t_k = t_next
    x_map = np.clip(x, 0.0, None)
    map_runtime = map_start()

    dirty = np.clip(At(y), 0.0, None)

    # --- real HPD threshold gamma'_alpha (Eq.6 / Eq.19) ---
    alpha = 0.01
    tau_alpha = math.sqrt(16.0 * math.log(3.0 / alpha))
    fg_map = f_prior(x_map) + g_data(x_map)
    gamma_alpha = fg_map + tau_alpha * math.sqrt(N) + N

    # --- real local credible intervals via superpixel bisection (Eq.7-9 / Eq.26-28) ---
    lci_start = timer()
    grid = 8  # 8x8 superpixels -> 64x64 image
    xi_lo = np.zeros_like(x_map)
    xi_hi = np.zeros_like(x_map)
    hi_bound = float(x_map.max() * 2.0 + 1.0)

    def obj_region(region, xi):
        xr = x_map.copy()
        xr[region] = xi
        return f_prior(xr) + g_data(xr)

    for bi in range(0, n, grid):
        for bj in range(0, n, grid):
            region = np.zeros((n, n), dtype=bool)
            region[bi:bi + grid, bj:bj + grid] = True
            base = float(x_map[region].mean())
            # upper bound xi_+ : largest constant intensity still inside C'_alpha
            if obj_region(region, hi_bound) <= gamma_alpha:
                xi_plus = hi_bound
            else:
                a, b = base, hi_bound
                for _ in range(30):
                    m = 0.5 * (a + b)
                    if obj_region(region, m) <= gamma_alpha:
                        a = m
                    else:
                        b = m
                xi_plus = a
            # lower bound xi_- : smallest constant intensity still inside C'_alpha
            if obj_region(region, 0.0) <= gamma_alpha:
                xi_minus = 0.0
            else:
                a, b = 0.0, base
                for _ in range(30):
                    m = 0.5 * (a + b)
                    if obj_region(region, m) <= gamma_alpha:
                        b = m
                    else:
                        a = m
                xi_minus = b
            xi_lo[region] = xi_minus
            xi_hi[region] = xi_plus
    interval = xi_hi - xi_lo
    lci_runtime = lci_start()

    # --- figure: truth / dirty / MAP / local credible interval width ---
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    for ax, arr, title in [
        (axes[0], img, "truth (Shepp-Logan)"),
        (axes[1], dirty, "dirty A^H y"),
        (axes[2], x_map, "l1-wavelet MAP"),
        (axes[3], interval, "LCI width (alpha=0.01)"),
    ]:
        ax.imshow(arr, cmap="magma")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig_file = save_figure(fig, "map_uq_reconstruction_uncertainty.png")
    plt.close(fig)

    base_metrics = {
        "map_psnr": round(psnr(img, x_map), 4),
        "map_snr": round(snr(img, x_map), 4),
        "dirty_snr": round(snr(img, dirty), 4),
        "snr_gain_over_dirty_db": round(snr(img, x_map) - snr(img, dirty), 4),
        "sampling_rate": round(sampling_rate, 4),
        "noise_sigma": round(float(sigma), 6),
        "map_runtime_seconds": map_runtime,
        "lci_runtime_seconds": lci_runtime,
        "gamma_alpha_hpd": round(gamma_alpha, 4),
        "mean_interval_length": round(float(interval.mean()), 4),
    }
    runtime = elapsed()

    fidelity = (
        "Real l1-wavelet (Daubechies-8) MAP via FISTA + real local credible "
        "intervals via HPD-threshold bisection, on a standard Shepp-Logan "
        "phantom under ~10% low-frequency-biased Fourier undersampling (SNR=30dB). "
        "NOT the paper's M31/BrainWeb/RI data; no true radio-interferometric "
        "(NUFFT) operator, no auto-mu estimation, and no Px-MALA/MYULA sampler. "
        "Metrics and the O(10^5) speedup claim are not comparable to the paper."
    )

    return [
        completed(
            11, "high-dimensional-uq", "map_uq_toy", "toy", base_metrics, [fig_file], runtime,
            "Real l1-wavelet MAP (FISTA forward-backward, db8 analysis prior) on an "
            "undersampled Fourier inverse problem, with the paper's HPD threshold "
            "gamma'_alpha=f+g+sqrt(16 log(3/alpha))sqrt(N)+N and local credible intervals "
            "by superpixel bisection. MAP beats the dirty-image baseline by the reported "
            "SNR gain. Uses a synthetic toy test image (Shepp-Logan phantom), not the paper's M31/BrainWeb data; no "
            "auto-mu, no SARA dictionary, no Px-MALA.",
            extra={"fidelityWarning": fidelity},
        ),
        completed(
            12, "ri-uq-i", "map_uq_toy", "toy", base_metrics, [fig_file], runtime,
            "Real l1-wavelet (db8) MAP via forward-backward splitting on a synthetic toy ~10% "
            "undersampled Fourier problem, plus real HPD credible region threshold and "
            "local credible intervals. This implements the optimisation/UQ machinery but "
            "NOT the paper's proximal-MCMC samplers (MYULA/Px-MALA), nor a true RI "
            "measurement operator, nor M31/Cygnus A/W28/3C288 data. Runtime is not "
            "comparable to the paper's large-scale results.",
            extra={"fidelityWarning": fidelity},
        ),
        completed(
            13, "ri-uq-ii", "map_uq_toy", "toy", base_metrics, [fig_file], runtime,
            "Real MAP-based UQ: db8 l1-wavelet MAP (forward-backward, Algorithm 1 style), "
            "real concentration-inequality HPD threshold (Eq.19) and local credible "
            "intervals by bisection (Eq.26-28), on a synthetic toy test image. No true RI "
            "operator/NUFFT, no Px-MALA baseline, no hypothesis testing, and not the "
            "paper's data; the O(10^5) MAP-vs-MCMC speedup is not reproduced here.",
            extra={"fidelityWarning": fidelity},
        ),
    ]
