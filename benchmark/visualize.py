import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import numpy as np
from typing import List, Dict, Optional
from benchmark import BenchmarkResult


def plot_results(
    results: List[BenchmarkResult], output_path: str = "benchmark_results.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = list(set(r.category for r in results))
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    cat_colors = {cat: colors[i] for i, cat in enumerate(categories)}

    ax1 = axes[0, 0]
    names = list(set(r.name for r in results))
    x = np.arange(len(names))

    time_means = []
    time_stds = []
    bar_colors = []

    for name in names:
        name_results = [r for r in results if r.name == name]
        times = [r.time_ms for r in name_results]
        time_means.append(np.mean(times))
        time_stds.append(np.std(times))
        bar_colors.append(cat_colors[name_results[0].category])

    bars = ax1.bar(x, time_means, yerr=time_stds, capsize=3, color=bar_colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Execution Time Comparison")
    ax1.set_yscale("log")

    ax2 = axes[0, 1]
    memory_means = []
    memory_stds = []

    for name in names:
        name_results = [r for r in results if r.name == name]
        memories = [r.memory_mb for r in name_results]
        memory_means.append(np.mean(memories))
        memory_stds.append(np.std(memories))

    bars = ax2.bar(x, memory_means, yerr=memory_stds, capsize=3, color=bar_colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Memory Usage Comparison")

    ax3 = axes[1, 0]
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_names = list(set(r.name for r in cat_results))
        accuracies = [
            np.mean([r.accuracy for r in cat_results if r.name == n]) for n in cat_names
        ]
        ax3.barh(cat_names, accuracies, color=cat_colors[cat], alpha=0.7, label=cat)

    ax3.set_xlabel("Accuracy/Score")
    ax3.set_title("Accuracy Comparison by Category")
    ax3.legend(loc="lower right", fontsize=8)

    ax4 = axes[1, 1]
    sizes = list(set(r.input_size for r in results))
    sizes.sort(key=lambda x: x[0] * x[1] if len(x) > 1 else x[0])

    for name in names[:5]:
        name_results = [r for r in results if r.name == name]
        size_time = []
        size_labels = []
        for size in sizes:
            size_results = [r for r in name_results if r.input_size == size]
            if size_results:
                size_time.append(np.mean([r.time_ms for r in size_results]))
                size_labels.append(
                    f"{size[0]}x{size[1]}" if len(size) > 1 else str(size[0])
                )

        if size_time:
            ax4.plot(size_labels, size_time, "o-", label=name, markersize=4)

    ax4.set_xlabel("Input Size")
    ax4.set_ylabel("Time (ms)")
    ax4.set_title("Scaling Performance")
    ax4.legend(loc="upper left", fontsize=7)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_category_comparison(
    results: List[BenchmarkResult], category: str, output_path: str
):
    cat_results = [r for r in results if r.category == category]

    if not cat_results:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    names = list(set(r.name for r in cat_results))
    x = np.arange(len(names))

    ax1 = axes[0]
    times = [np.mean([r.time_ms for r in cat_results if r.name == n]) for n in names]
    time_stds = [np.std([r.time_ms for r in cat_results if r.name == n]) for n in names]
    ax1.bar(x, times, yerr=time_stds, capsize=3, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"{category} - Execution Time")

    ax2 = axes[1]
    memories = [
        np.mean([r.memory_mb for r in cat_results if r.name == n]) for n in names
    ]
    memory_stds = [
        np.std([r.memory_mb for r in cat_results if r.name == n]) for n in names
    ]
    ax2.bar(x, memories, yerr=memory_stds, capsize=3, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title(f"{category} - Memory Usage")

    ax3 = axes[2]
    accuracies = [
        np.mean([r.accuracy for r in cat_results if r.name == n]) for n in names
    ]
    ax3.bar(x, accuracies, color="seagreen")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.set_ylabel("Accuracy")
    ax3.set_title(f"{category} - Accuracy")

    plt.suptitle(f"{category} Benchmark Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_performance_heatmap(results: List[BenchmarkResult], output_path: str):
    names = list(set(r.name for r in results))
    sizes = list(set(r.input_size for r in results))
    sizes.sort(key=lambda x: x[0] * x[1] if len(x) > 1 else x[0])

    time_matrix = np.zeros((len(names), len(sizes)))

    for i, name in enumerate(names):
        for j, size in enumerate(sizes):
            matching = [r for r in results if r.name == name and r.input_size == size]
            if matching:
                time_matrix[i, j] = np.mean([r.time_ms for r in matching])
            else:
                time_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(time_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(sizes)))
    ax.set_yticks(np.arange(len(names)))

    size_labels = [f"{s[0]}x{s[1]}" if len(s) > 1 else str(s[0]) for s in sizes]
    ax.set_xticklabels(size_labels, rotation=45, ha="right")
    ax.set_yticklabels(names)

    ax.set_xlabel("Input Size")
    ax.set_ylabel("Algorithm")
    ax.set_title("Execution Time Heatmap (ms)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Time (ms)")

    for i in range(len(names)):
        for j in range(len(sizes)):
            if not np.isnan(time_matrix[i, j]):
                text = ax.text(
                    j,
                    i,
                    f"{time_matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_html_report(results: List[BenchmarkResult], output_path: str):
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
        table { border-collapse: collapse; width: 100%; background: white; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        tr:hover { background-color: #ddd; }
        .summary { background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .chart { background: white; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>Benchmark Test Report</h1>
"""

    categories = set(r.category for r in results)

    html += """
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {}</p>
        <p><strong>Categories:</strong> {}</p>
        <p><strong>Algorithms:</strong> {}</p>
    </div>
""".format(
        len(results),
        ", ".join(sorted(categories)),
        ", ".join(sorted(set(r.name for r in results))),
    )

    for category in sorted(categories):
        cat_results = [r for r in results if r.category == category]

        html += f"""
    <div class="chart">
        <h2>{category}</h2>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Input Size</th>
                <th>Time (ms)</th>
                <th>Memory (MB)</th>
                <th>Accuracy</th>
            </tr>
"""

        for r in cat_results:
            size_str = (
                f"{r.input_size[0]}x{r.input_size[1]}"
                if len(r.input_size) > 1
                else str(r.input_size[0])
            )
            html += f"""
            <tr>
                <td>{r.name}</td>
                <td>{size_str}</td>
                <td>{r.time_ms:.2f}</td>
                <td>{r.memory_mb:.2f}</td>
                <td>{r.accuracy:.4f}</td>
            </tr>
"""

        html += """
        </table>
    </div>
"""

    html += """
    <div class="chart">
        <h2>Performance Charts</h2>
        <img src="benchmark_results.png" alt="Benchmark Results">
    </div>
    
    <div class="chart">
        <h2>Performance Heatmap</h2>
        <img src="benchmark_heatmap.png" alt="Performance Heatmap">
    </div>
"""

    html += """
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
