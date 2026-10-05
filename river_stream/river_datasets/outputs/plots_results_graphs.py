import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    rankdata,
    studentized_range,
)


BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR

FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


METHOD_FILES = {
    "FEDD_cos": "fedd_cosine.npy",
    "FEDD_pear": "fedd_pearson.npy",
    "ELM_ECDD": "elm_ecdd.npy",
}

METHOD_ORDER = [
    "FEDD_cos",
    "FEDD_pear",
    "ELM_ECDD",
]


def load_results(path):
    """Load detector results."""

    return list(
        np.load(
            path,
            allow_pickle=True
        )
    )


def load_metadata(path):
    """Load dataset ground-truth metadata."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_detected_drifts(results):
    """Extract timestamps where drift=True."""

    detected = []

    for result in results:

        if not isinstance(result, dict):
            try:
                result = result.item()
            except Exception:
                continue

        if result.get("drift", False):
            detected.append(
                int(result["t"])
            )

    return sorted(detected)


def evaluate_detections(
    detected_drifts,
    true_drifts,
    series_length,
):
    """
    Calculate the metrics used in the FEDD paper.

    The first detection after each true drift and before the
    next drift is considered a correct detection.

    Remaining detections are false alarms.

    If a drift is missed, all observations belonging to the
    new concept are counted as detection delay.
    """

    detected_drifts = sorted(
        int(t)
        for t in detected_drifts
    )

    true_drifts = sorted(
        int(t)
        for t in true_drifts
    )

    used_detections = set()

    false_alarms = 0
    missed_drifts = 0
    total_delay = 0

    for index, true_drift in enumerate(
        true_drifts
    ):

        if index < len(true_drifts) - 1:
            concept_end = (
                true_drifts[index + 1]
            )
        else:
            concept_end = series_length

        candidates = [
            detection
            for detection in detected_drifts
            if (
                true_drift <= detection < concept_end
                and detection not in used_detections
            )
        ]

        if candidates:

            detection = candidates[0]

            used_detections.add(
                detection
            )

            total_delay += (
                detection - true_drift
            )

        else:

            missed_drifts += 1

            total_delay += (
                concept_end - true_drift
            )

    false_alarms = sum(
        detection not in used_detections
        for detection in detected_drifts
    )

    miss_detection_rate = (
        100.0
        * missed_drifts
        / len(true_drifts)
    )

    return {
        "false_alarms": false_alarms,
        "detection_delay": total_delay,
        "miss_detection_rate": miss_detection_rate,
    }


def collect_metrics():
    """Read all FriedmanDrift experiment results."""

    rows = []

    for drift_kind in [
        "abrupt",
        "gradual"
    ]:

        drift_dir = (
            RESULTS_DIR
            / drift_kind
        )

        if not drift_dir.exists():
            continue

        dataset_dirs = sorted(
            path
            for path in drift_dir.iterdir()
            if path.is_dir()
        )

        for dataset_dir in dataset_dirs:

            metadata_path = (
                dataset_dir
                / "metadata.json"
            )

            if not metadata_path.exists():
                continue

            metadata = load_metadata(
                metadata_path
            )

            true_drifts = metadata[
                "true_drifts"
            ]

            series_length = metadata.get(
                "n_samples",
                12000
            )

            for method, filename in METHOD_FILES.items():

                result_path = (
                    dataset_dir
                    / filename
                )

                if not result_path.exists():
                    continue

                results = load_results(
                    result_path
                )

                detected = (
                    extract_detected_drifts(
                        results
                    )
                )

                metrics = evaluate_detections(
                    detected,
                    true_drifts,
                    series_length,
                )

                rows.append({
                    "dataset":
                        dataset_dir.name,

                    "drift_kind":
                        drift_kind,

                    "method":
                        method,

                    "false_alarms":
                        metrics["false_alarms"],

                    "detection_delay":
                        metrics["detection_delay"],

                    "miss_detection_rate":
                        metrics["miss_detection_rate"],
                })

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        raise RuntimeError(
            f"No experiment results found in {RESULTS_DIR}"
        )

    return dataframe


def metric_table(
    dataframe,
    metric
):
    """Build paired dataset x method table."""

    table = dataframe.pivot_table(
        index=[
            "drift_kind",
            "dataset"
        ],
        columns="method",
        values=metric,
        aggfunc="first",
    )

    table = table[
        METHOD_ORDER
    ]

    return table.dropna()


def average_ranks(table):
    """Calculate average method ranks."""

    ranks = []

    for row in table.to_numpy():

        ranks.append(
            rankdata(
                row,
                method="average"
            )
        )

    return np.mean(
        ranks,
        axis=0
    )


def critical_difference(
    n_datasets,
    n_methods,
    alpha=0.05,
):
    """Calculate the Nemenyi critical difference."""

    q_alpha = (
        studentized_range.ppf(
            1 - alpha,
            n_methods,
            np.inf,
        )
        / np.sqrt(2)
    )

    return q_alpha * np.sqrt(
        n_methods
        * (n_methods + 1)
        / (6 * n_datasets)
    )


def draw_rank_panel(
    axis,
    dataframe,
    metric,
    title,
):
    """Draw one Figure 3 average-rank panel."""

    table = metric_table(
        dataframe,
        metric
    )

    ranks = average_ranks(
        table
    )

    cd = critical_difference(
        len(table),
        len(METHOD_ORDER),
    )

    samples = [
        table[method].to_numpy()
        for method in METHOD_ORDER
    ]

    statistic, p_value = (
        friedmanchisquare(
            *samples
        )
    )

    print()
    print(title)
    print(
        f"Friedman statistic: {statistic:.6f}"
    )
    print(
        f"p-value: {p_value:.6g}"
    )
    print(
        f"Critical Difference: {cd:.4f}"
    )

    for method, rank in zip(
        METHOD_ORDER,
        ranks
    ):
        print(
            f"{method}: {rank:.4f}"
        )

    x = np.arange(
        len(METHOD_ORDER)
    )

    axis.errorbar(
        x,
        ranks,
        yerr=cd / 2,
        fmt="o",
        capsize=5,
        markersize=4,
        linewidth=0.9,
        elinewidth=0.9,
    )

    axis.set_title(
        title,
        fontsize=10,
    )

    axis.set_ylabel(
        "Average Ranks",
        fontsize=8,
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        METHOD_ORDER,
        fontsize=7,
    )

    axis.set_ylim(
        1,
        3.5
    )

    axis.set_yticks(
        np.arange(
            1,
            3.6,
            0.5
        )
    )

    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )

    axis.tick_params(
        axis="y",
        labelsize=7,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def generate_figure_3(
    dataframe
):
    """Generate Figure 3."""

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(5.0, 6.2),
    )

    draw_rank_panel(
        axes[0],
        dataframe,
        "false_alarms",
        "Number of False Alarms",
    )

    draw_rank_panel(
        axes[1],
        dataframe,
        "detection_delay",
        "Drift Detection Delay",
    )

    draw_rank_panel(
        axes[2],
        dataframe,
        "miss_detection_rate",
        "Miss-detection Rates",
    )

    for axis, label in zip(
        axes,
        ["(a)", "(b)", "(c)"]
    ):

        axis.text(
            0.5,
            -0.28,
            label,
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )

    figure.subplots_adjust(
        hspace=0.75
    )

    output_path = (
        FIGURES_DIR
        / "figure_3_friedman.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print()
    print(
        "Figure 3 saved:",
        output_path
    )


def boxplot_values(
    dataframe,
    drift_kind,
    metric,
):
    """Return one array per method."""

    subset = dataframe[
        dataframe["drift_kind"]
        == drift_kind
    ]

    return [
        subset[
            subset["method"]
            == method
        ][metric].to_numpy()
        for method in METHOD_ORDER
    ]


def draw_boxplot(
    axis,
    dataframe,
    drift_kind,
    metric,
    title,
    ylabel,
):
    """Draw one Figure 4 boxplot."""

    values = boxplot_values(
        dataframe,
        drift_kind,
        metric,
    )

    axis.boxplot(
        values,
        tick_labels=METHOD_ORDER,
        widths=0.55,
        showfliers=True,

        medianprops={
            "linewidth": 0.9
        },

        boxprops={
            "linewidth": 0.7
        },

        whiskerprops={
            "linewidth": 0.7
        },

        capprops={
            "linewidth": 0.7
        },

        flierprops={
            "marker": "+",
            "markersize": 4,
        },
    )

    axis.set_title(
        title,
        fontsize=8,
    )

    axis.set_ylabel(
        ylabel,
        fontsize=7,
    )

    axis.tick_params(
        axis="x",
        labelsize=6,
    )

    axis.tick_params(
        axis="y",
        labelsize=6,
    )

    axis.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.7,
    )

    for spine in axis.spines.values():
        spine.set_linewidth(0.7)


def generate_figure_4(
    dataframe
):
    """Generate Figure 4."""

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(8.0, 6.5),
    )

    draw_boxplot(
        axes[0, 0],
        dataframe,
        "abrupt",
        "false_alarms",
        "Number of False Alarms: Abrupt Drifts",
        "False alarms",
    )

    draw_boxplot(
        axes[0, 1],
        dataframe,
        "gradual",
        "false_alarms",
        "Number of False Alarms: Gradual Drifts",
        "False alarms",
    )

    draw_boxplot(
        axes[1, 0],
        dataframe,
        "abrupt",
        "detection_delay",
        "Drift Detection Delay: Abrupt Drifts",
        "Delay (instances)",
    )

    draw_boxplot(
        axes[1, 1],
        dataframe,
        "gradual",
        "detection_delay",
        "Drift Detection Delay: Gradual Drifts",
        "Delay (instances)",
    )

    draw_boxplot(
        axes[2, 0],
        dataframe,
        "abrupt",
        "miss_detection_rate",
        "Miss-detection Rates: Abrupt Drifts",
        "Rate (%)",
    )

    draw_boxplot(
        axes[2, 1],
        dataframe,
        "gradual",
        "miss_detection_rate",
        "Miss-detection Rates: Gradual Drifts",
        "Rate (%)",
    )

    axes[2, 0].set_ylim(
        -5,
        105
    )

    axes[2, 1].set_ylim(
        -5,
        105
    )

    axes[2, 0].set_yticks(
        [0, 20, 40, 60, 80, 100]
    )

    axes[2, 1].set_yticks(
        [0, 20, 40, 60, 80, 100]
    )

    for row, label in enumerate(
        ["(a)", "(b)", "(c)"]
    ):

        axes[row, 0].text(
            1.05,
            -0.27,
            label,
            transform=axes[row, 0].transAxes,
            ha="center",
            fontsize=9,
        )

    figure.subplots_adjust(
        hspace=0.55,
        wspace=0.27,
    )

    output_path = (
        FIGURES_DIR
        / "figure_4_friedman.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        "Figure 4 saved:",
        output_path
    )


def main():
    """Calculate metrics and generate Figures 3 and 4."""

    print(
        "Reading FriedmanDrift experiment results..."
    )

    dataframe = collect_metrics()

    metrics_path = (
        RESULTS_DIR
        / "friedman_metrics.csv"
    )

    dataframe.to_csv(
        metrics_path,
        index=False
    )

    print()
    print(
        dataframe.groupby(
            [
                "drift_kind",
                "method"
            ]
        )[
            [
                "false_alarms",
                "detection_delay",
                "miss_detection_rate",
            ]
        ].mean()
    )

    generate_figure_3(
        dataframe
    )

    generate_figure_4(
        dataframe
    )

    print()
    print(
        "Metrics saved:",
        metrics_path
    )


if __name__ == "__main__":
    main()