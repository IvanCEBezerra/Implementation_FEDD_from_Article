from pathlib import Path

import numpy as np

from river_stream.river_datasets.river_exp_func import (
    select_river_datasets,
    run_fedd_experiment,
    run_baseline_experiment,
    save_results,
    save_metadata,
)


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "river_data"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
)


def run_experiment():

    datasets = select_river_datasets(
        DATA_DIR
    )

    print(
        "Datasets found:",
        len(datasets)
    )

    print(
        "Total executions:",
        len(datasets) * 3
    )

    for index, dataset in enumerate(
        datasets,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(datasets)}] "
            f"{dataset['name']}"
        )

        print(
            "Drift kind:",
            dataset["drift_kind"]
        )

        print(
            "True drifts:",
            dataset["true_drifts"]
        )

        vector = np.load(
            dataset["path"]
        )

        print(
            "Running FEDD cosine..."
        )

        fedd_cosine_results = (
            run_fedd_experiment(
                vector,
                distance="cosine"
            )
        )

        print(
            "Running FEDD Pearson..."
        )

        fedd_pearson_results = (
            run_fedd_experiment(
                vector,
                distance="pearson"
            )
        )

        print(
            "Running ELM-ECDD..."
        )

        baseline_results = (
            run_baseline_experiment(
                vector
            )
        )

        dataset_output = (
            OUTPUT_DIR
            / dataset["drift_kind"]
            / dataset["name"]
        )

        save_results(
            fedd_cosine_results,
            dataset_output
            / "fedd_cosine.npy"
        )

        save_results(
            fedd_pearson_results,
            dataset_output
            / "fedd_pearson.npy"
        )

        save_results(
            baseline_results,
            dataset_output
            / "elm_ecdd.npy"
        )

        save_metadata(
            dataset,
            dataset_output
            / "metadata.json"
        )

        print(
            "Saved:",
            dataset_output
        )

    print()
    print("Experiment finished.")


if __name__ == "__main__":
    run_experiment()