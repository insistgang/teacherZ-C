import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Optional
import json
import tracemalloc
from abc import ABC, abstractmethod


@dataclass
class BenchmarkResult:
    name: str
    category: str
    input_size: tuple
    time_ms: float
    memory_mb: float
    accuracy: float
    metric_name: str
    params: dict

    def to_dict(self) -> dict:
        return asdict(self)


class BaseBenchmark(ABC):
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.results: List[BenchmarkResult] = []

    @abstractmethod
    def setup(self, size: tuple, **kwargs):
        pass

    @abstractmethod
    def run(self, **kwargs) -> tuple:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def measure(
        self, size: tuple, params: dict, metric_name: str = "accuracy"
    ) -> BenchmarkResult:
        self.setup(size, **params)

        tracemalloc.start()
        start_time = time.perf_counter()

        output, accuracy = self.run(**params)

        elapsed = (time.perf_counter() - start_time) * 1000
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.cleanup()

        return BenchmarkResult(
            name=self.name,
            category=self.category,
            input_size=size,
            time_ms=elapsed,
            memory_mb=peak / (1024 * 1024),
            accuracy=accuracy,
            metric_name=metric_name,
            params=params,
        )


class BenchmarkSuite:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.benchmarks: List[BaseBenchmark] = []

    def register(self, benchmark: BaseBenchmark):
        self.benchmarks.append(benchmark)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def run_all(
        self, sizes: List[tuple], params_list: List[dict], verbose: bool = True
    ):
        for benchmark in self.benchmarks:
            if verbose:
                print(f"Running {benchmark.name}...")
            for size in sizes:
                for params in params_list:
                    result = benchmark.measure(size, params)
                    self.add_result(result)
                    if verbose:
                        print(
                            f"  Size {size}: {result.time_ms:.2f}ms, {result.memory_mb:.2f}MB"
                        )

    def run_with_config(self, config: dict, verbose: bool = True):
        for benchmark in self.benchmarks:
            bench_config = config.get(benchmark.name, {})
            sizes = bench_config.get("sizes", [(256, 256)])
            params_list = bench_config.get("params", [{}])

            if verbose:
                print(f"Running {benchmark.name}...")

            for size in sizes:
                for params in params_list:
                    try:
                        result = benchmark.measure(
                            size,
                            params,
                            metric_name=bench_config.get("metric", "accuracy"),
                        )
                        self.add_result(result)
                        if verbose:
                            print(f"  Size {size}: {result.time_ms:.2f}ms")
                    except Exception as e:
                        if verbose:
                            print(f"  Size {size}: Error - {e}")

    def get_results_by_category(self, category: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.category == category]

    def get_results_by_name(self, name: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.name == name]

    def generate_report(self) -> str:
        report = "# 基准测试报告\n\n"
        report += f"总测试数: {len(self.results)}\n\n"

        categories = set(r.category for r in self.results)

        for category in sorted(categories):
            report += f"## {category}\n\n"
            cat_results = self.get_results_by_category(category)

            report += "| 算法 | 输入大小 | 时间(ms) | 内存(MB) | 指标值 |\n"
            report += "|------|----------|----------|----------|--------|\n"

            for r in cat_results:
                size_str = (
                    f"{r.input_size[0]}x{r.input_size[1]}"
                    if len(r.input_size) == 2
                    else str(r.input_size)
                )
                report += f"| {r.name} | {size_str} | {r.time_ms:.2f} | {r.memory_mb:.2f} | {r.accuracy:.4f} |\n"

            report += "\n"

        return report

    def generate_markdown_table(self, group_by: str = "category") -> str:
        if group_by == "category":
            return self.generate_report()

        report = "# 基准测试结果\n\n"
        report += "| 算法 | 类别 | 输入大小 | 时间(ms) | 内存(MB) | 指标值 |\n"
        report += "|------|------|----------|----------|----------|--------|\n"

        for r in self.results:
            size_str = (
                f"{r.input_size[0]}x{r.input_size[1]}"
                if len(r.input_size) == 2
                else str(r.input_size)
            )
            report += f"| {r.name} | {r.category} | {size_str} | {r.time_ms:.2f} | {r.memory_mb:.2f} | {r.accuracy:.4f} |\n"

        return report

    def save_results(self, filepath: str, format: str = "json"):
        if format == "json":
            data = [r.to_dict() for r in self.results]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "csv":
            import csv

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "name",
                        "category",
                        "input_size",
                        "time_ms",
                        "memory_mb",
                        "accuracy",
                        "metric_name",
                        "params",
                    ],
                )
                writer.writeheader()
                for r in self.results:
                    row = r.to_dict()
                    row["input_size"] = str(row["input_size"])
                    row["params"] = str(row["params"])
                    writer.writerow(row)
        elif format == "markdown":
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.generate_report())

    def get_statistics(self) -> Dict:
        if not self.results:
            return {}

        times = [r.time_ms for r in self.results]
        memories = [r.memory_mb for r in self.results]
        accuracies = [r.accuracy for r in self.results if r.accuracy > 0]

        stats = {
            "total_tests": len(self.results),
            "time": {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
            },
            "memory": {
                "mean": np.mean(memories),
                "std": np.std(memories),
                "min": np.min(memories),
                "max": np.max(memories),
            },
        }

        if accuracies:
            stats["accuracy"] = {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
            }

        return stats
