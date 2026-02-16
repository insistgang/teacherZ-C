import os
import sys
from benchmark import BenchmarkSuite, BenchmarkResult
from denoising import ROFDenoising, BM3DDenoising, DnCNNBenchmark
from segmentation import (
    SLATSegmentation,
    UNetBenchmark,
    DeepLabBenchmark,
    GraphCutSegmentation,
)
from pointcloud import (
    VarifoldBenchmark,
    PointNetBenchmark,
    DGCBenchmark,
    TuckerDecompositionBenchmark,
)
from visualize import (
    plot_results,
    plot_category_comparison,
    plot_performance_heatmap,
    generate_html_report,
)


def create_denoising_config():
    return {
        "sizes": [(128, 128), (256, 256), (512, 512)],
        "params": [
            {"lambda": 0.1, "max_iter": 50, "noise_std": 0.1},
            {"lambda": 0.2, "max_iter": 50, "noise_std": 0.1},
        ],
        "metric": "PSNR",
    }


def create_segmentation_config():
    return {
        "sizes": [(128, 128), (256, 256)],
        "params": [
            {"threshold": 0.5, "max_iter": 30},
            {"threshold": 0.3, "max_iter": 30},
        ],
        "metric": "mIoU",
    }


def create_pointcloud_config():
    return {
        "sizes": [(500,), (1000,), (2000,)],
        "params": [
            {"sigma": 1.0, "n_features": 32, "n_classes": 5},
            {"sigma": 0.5, "n_features": 32, "n_classes": 5},
        ],
        "metric": "Accuracy",
    }


def create_tucker_config():
    return {
        "sizes": [(32, 32, 32), (64, 64, 64)],
        "params": [
            {"ranks": (16, 16, 16)},
            {"ranks": (8, 8, 8)},
        ],
        "metric": "Reconstruction",
    }


def run_full_benchmark(output_dir: str = "."):
    print("=" * 60)
    print("Algorithm Performance Benchmark Suite")
    print("=" * 60)

    suite = BenchmarkSuite()

    print("\n[1/4] Registering Denoising Benchmarks...")
    for method in ["gradient_descent", "chambolle", "primal_dual"]:
        suite.register(ROFDenoising(method=method))
    suite.register(BM3DDenoising())
    suite.register(DnCNNBenchmark())

    print("[2/4] Registering Segmentation Benchmarks...")
    suite.register(SLATSegmentation())
    suite.register(UNetBenchmark())
    suite.register(DeepLabBenchmark())
    suite.register(GraphCutSegmentation())

    print("[3/4] Registering Point Cloud Benchmarks...")
    suite.register(VarifoldBenchmark())
    suite.register(PointNetBenchmark())
    suite.register(DGCBenchmark())
    suite.register(TuckerDecompositionBenchmark())

    print("[4/4] Running all benchmarks...\n")

    config = {
        "ROF_gradient_descent": create_denoising_config(),
        "ROF_chambolle": create_denoising_config(),
        "ROF_primal_dual": create_denoising_config(),
        "BM3D": {
            "sizes": [(128, 128), (256, 256)],
            "params": [{"block_size": 8, "sigma": 0.1}],
            "metric": "PSNR",
        },
        "DnCNN": {
            "sizes": [(128, 128), (256, 256)],
            "params": [{"num_layers": 3}],
            "metric": "PSNR",
        },
        "SLaT": create_segmentation_config(),
        "U-Net": {
            "sizes": [(128, 128), (256, 256)],
            "params": [{"depth": 2}],
            "metric": "mIoU",
        },
        "DeepLab": {
            "sizes": [(128, 128), (256, 256)],
            "params": [{"output_stride": 4}],
            "metric": "mIoU",
        },
        "GraphCut": create_segmentation_config(),
        "Varifold": create_pointcloud_config(),
        "PointNet": create_pointcloud_config(),
        "DGCNN": {
            "sizes": [(500,), (1000,)],
            "params": [{"k": 10, "n_features": 32}],
            "metric": "Accuracy",
        },
        "Tucker": create_tucker_config(),
    }

    suite.run_with_config(config, verbose=True)

    print("\n" + "=" * 60)
    print("Generating Reports...")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "benchmark_report.md")
    suite.save_results(report_path, format="markdown")
    print(f"[OK] Markdown report saved to: {report_path}")

    json_path = os.path.join(output_dir, "benchmark_results.json")
    suite.save_results(json_path, format="json")
    print(f"[OK] JSON results saved to: {json_path}")

    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    suite.save_results(csv_path, format="csv")
    print(f"[OK] CSV results saved to: {csv_path}")

    chart_path = os.path.join(output_dir, "benchmark_results.png")
    try:
        plot_results(suite.results, chart_path)
        print(f"[OK] Performance charts saved to: {chart_path}")
    except Exception as e:
        print(f"[WARN] Failed to generate charts: {e}")

    heatmap_path = os.path.join(output_dir, "benchmark_heatmap.png")
    try:
        plot_performance_heatmap(suite.results, heatmap_path)
        print(f"[OK] Heatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"[WARN] Failed to generate heatmap: {e}")

    categories = set(r.category for r in suite.results)
    for category in categories:
        cat_file = category.replace("/", "_")
        cat_path = os.path.join(output_dir, f"benchmark_{cat_file}.png")
        try:
            plot_category_comparison(suite.results, category, cat_path)
            print(f"[OK] Category chart saved to: {cat_path}")
        except Exception as e:
            print(f"[WARN] Failed to generate {category} chart: {e}")

    html_path = os.path.join(output_dir, "benchmark_report.html")
    try:
        generate_html_report(suite.results, html_path)
        print(f"[OK] HTML report saved to: {html_path}")
    except Exception as e:
        print(f"[WARN] Failed to generate HTML report: {e}")

    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    stats = suite.get_statistics()
    print(f"Total tests: {stats['total_tests']}")
    print(
        f"Time (ms): mean={stats['time']['mean']:.2f}, std={stats['time']['std']:.2f}"
    )
    print(
        f"Memory (MB): mean={stats['memory']['mean']:.2f}, std={stats['memory']['std']:.2f}"
    )
    if "accuracy" in stats:
        print(
            f"Accuracy: mean={stats['accuracy']['mean']:.4f}, std={stats['accuracy']['std']:.4f}"
        )

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    return suite


def run_quick_benchmark():
    print("Running Quick Benchmark (small sizes)...\n")

    suite = BenchmarkSuite()

    suite.register(ROFDenoising(method="gradient_descent"))
    suite.register(SLATSegmentation())
    suite.register(VarifoldBenchmark())

    config = {
        "ROF_gradient_descent": {
            "sizes": [(64, 64), (128, 128)],
            "params": [{"lambda": 0.1, "max_iter": 20}],
            "metric": "PSNR",
        },
        "SLaT": {
            "sizes": [(64, 64), (128, 128)],
            "params": [{"threshold": 0.5, "max_iter": 10}],
            "metric": "mIoU",
        },
        "Varifold": {
            "sizes": [(100,), (200,)],
            "params": [{"sigma": 1.0, "n_features": 16}],
            "metric": "Accuracy",
        },
    }

    suite.run_with_config(config, verbose=True)

    print("\n" + suite.generate_report())

    return suite


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Algorithm Benchmark Suite")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with small sizes"
    )
    parser.add_argument(
        "--output", "-o", default=".", help="Output directory for reports"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_benchmark()
    else:
        run_full_benchmark(args.output)
