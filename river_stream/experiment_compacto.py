from pathlib import Path
import numpy as np

from river_stream.experiment_func import save_results, select_linear_datasets
from river_stream.experiment_func import run_fedd_experiment, run_baseline_experiment

OUTPUT_DIR = Path("outputs_experiment")

print("Iniciando Experimento...")


datasets = select_linear_datasets()

for i, dataset in enumerate(datasets, start=1):

    print(
        f"[{i}/{len(datasets)}] "
        f"{dataset['drift_type']} / {dataset['name']}"
    )

    vector = np.load(dataset["path"])

    # FEDD Cosine
    fedd_cosine_results = run_fedd_experiment(
        vector,
        distance="cosine"
    )

    # FEDD Pearson
    fedd_pearson_results = run_fedd_experiment(
        vector,
        distance="pearson"
    )

    # ELM-ECDD
    baseline_results = run_baseline_experiment(
        vector
    )

    dataset_output = (
        OUTPUT_DIR
        / dataset["drift_type"]
        / dataset["name"]
    )

    save_results(
        fedd_cosine_results,
        dataset_output / "fedd_cosine.npy"
    )

    save_results(
        fedd_pearson_results,
        dataset_output / "fedd_pearson.npy"
    )

    save_results(
        baseline_results,
        dataset_output / "elm_ecdd.npy"
    )