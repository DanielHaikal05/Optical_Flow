#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TRAINING_METRICS = ROOT / "training_metrics.json"
EVAL_METRICS = ROOT / "validation_metrics.json"
PLOTS_DIR = ROOT / "plots"


def load_metrics(path: Path) -> dict[str, dict[str, list[float]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with path.open("r", encoding="utf-8") as file:
        text = file.read()

    decoder = json.JSONDecoder()
    objects = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        data, index = decoder.raw_decode(text, index)
        objects.append(data)

    if len(objects) == 1:
        metrics = objects[0]
        if "loss" in metrics or "pee" in metrics:
            return metrics
        return {"loss": metrics}
    if len(objects) == 2:
        return {"loss": objects[0], "pee": objects[1]}
    raise ValueError(f"Expected one metrics object or loss/pee objects in {path}")


def normalize_series(metric_values: object, default_label: str) -> dict[str, list[float]]:
    if isinstance(metric_values, list):
        return {default_label: metric_values}
    if isinstance(metric_values, dict):
        normalized = {}
        for label, values in metric_values.items():
            if isinstance(values, list):
                normalized[label] = values
        return normalized
    return {}


def plot_metric(
    metric_values: object,
    title: str,
    ylabel: str,
    output: Path,
    default_label: str,
) -> None:
    series = normalize_series(metric_values, default_label)
    if not series:
        print(f"Skipped {output}: no data")
        return

    plt.figure(figsize=(10, 5))
    for label, values in series.items():
        plt.plot(values, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if series:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


def metric_label(metric_name: str) -> str:
    labels = {
        "lr": "Learning Rate",
        "learning_rate": "Learning Rate",
        "pee": "PEE",
    }
    return labels.get(metric_name, metric_name.replace("_", " ").title())


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    training_metrics = load_metrics(TRAINING_METRICS)
    eval_metrics = load_metrics(EVAL_METRICS)

    for metric_name in sorted(set(training_metrics) | set(eval_metrics)):
        ylabel = metric_label(metric_name)
        plot_metric(
            training_metrics.get(metric_name, {}),
            f"NFlowNet Training {ylabel}",
            ylabel,
            PLOTS_DIR / f"training_{metric_name}_plot.png",
            "training",
        )
        plot_metric(
            eval_metrics.get(metric_name, {}),
            f"NFlowNet Evaluation {ylabel}",
            ylabel,
            PLOTS_DIR / f"evaluation_{metric_name}_plot.png",
            "evaluation",
        )
    plt.show()


if __name__ == "__main__":
    main()
