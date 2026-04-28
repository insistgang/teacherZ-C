from common import SEED, completed, require_modules, save_figure, timer


def run():
    missing = require_modules("numpy", "matplotlib")
    if missing:
        return [{"priority": 15, "id": "proximal-nested-sampling", "experiment_id": "nested_sampling_toy", "reproductionLevel": "toy", "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "resultFiles": [], "skipped_reason": f"Missing modules: {', '.join(missing)}"}]

    import math
    import numpy as np
    import matplotlib.pyplot as plt

    elapsed = timer()
    rng = np.random.default_rng(SEED + 15)
    dim = 2
    prior_width = 12.0
    n_live = 80
    n_iter = 180
    live = rng.uniform(-prior_width / 2, prior_width / 2, size=(n_live, dim))

    def loglike(x):
        return -0.5 * np.sum(x * x, axis=-1)

    log_l = loglike(live)
    weights = []
    loglikes = []
    for i in range(n_iter):
        worst = int(np.argmin(log_l))
        log_x_prev = -i / n_live
        log_x_new = -(i + 1) / n_live
        weight = math.exp(log_x_prev) - math.exp(log_x_new)
        weights.append(weight)
        loglikes.append(float(log_l[worst]))
        threshold = log_l[worst]
        for _ in range(4000):
            cand = rng.uniform(-prior_width / 2, prior_width / 2, size=dim)
            ll = float(loglike(cand))
            if ll > threshold:
                live[worst] = cand
                log_l[worst] = ll
                break

    weights = np.asarray(weights)
    loglikes = np.asarray(loglikes)
    evidence = float(np.sum(weights * np.exp(loglikes)))
    analytic = float((2 * math.pi) / (prior_width ** 2))
    log_est = math.log(max(evidence, 1e-300))
    log_ref = math.log(analytic)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    cumulative = np.cumsum(weights * np.exp(loglikes))
    ax.plot(np.log(np.maximum(cumulative, 1e-300)), label="estimated log evidence")
    ax.axhline(log_ref, color="black", linestyle="--", linewidth=1, label="analytic")
    ax.set_title("Nested sampling toy evidence")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)
    fig_file = save_figure(fig, "nested_sampling_evidence_trace.png")
    plt.close(fig)

    return [completed(15, "proximal-nested-sampling", "nested_sampling_toy", "toy", {
        "estimated_log_evidence": round(log_est, 4),
        "reference_log_evidence": round(log_ref, 4),
        "absolute_log_error": round(abs(log_est - log_ref), 4),
        "live_points": n_live,
        "iterations": n_iter
    }, [fig_file], elapsed(), "Toy nested sampling on a 2D Gaussian likelihood under a uniform prior; not proximal constrained MCMC. Completed with large error; use as nested sampling mechanism demo only.", {
        "resultQuality": "rough illustrative",
        "warning": "large evidence error; toy only"
    })]
