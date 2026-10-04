import numpy as np
from fedd.drift_detection.detector import FEDDDetector
from fedd.drift_detection.elm_ecdd import ELM_ECDD_Detector
from river_stream.vector_stream import vector_stream
from pathlib import Path


def select_linear_datasets(data_dir):
    data_dir = Path(data_dir)

    selected = []

    for drift_type in ["abrupt", "gradual"]:
        folder = data_dir / drift_type

        datasets = sorted(folder.glob("linear_*_1.npy"))

        for dataset in datasets:
            selected.append({
                "path": dataset,
                "drift_type": drift_type,
                "name": dataset.stem
            })

    return selected

from pathlib import Path


def select_linear_datasets():
    base_path = Path("data/generated/linear")
    datasets = []

    for drift_type in ["abrupt", "gradual"]:
        for series in ["linear_1", "linear_2", "linear_3"]:
            for sample in range(1, 6):

                path = (
                    base_path
                    / drift_type
                    / f"{series}_{sample}.npy"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        f"Dataset não encontrado: {path}"
                    )

                datasets.append({
                    "name": f"{series}_{sample}",
                    "series": series,
                    "sample": sample,
                    "drift_type": drift_type,
                    "path": path
                })

    return datasets

def save_results(results, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        np.array(results, dtype=object),
        allow_pickle=True
    )

def run_fedd_experiment(vector, distance="cosine"):
    detector = FEDDDetector(distance=distance)  # Initialize the FEDDD detector
    datastream = vector_stream(vector)  # Create a stream from the input vector

    results = []  # List to store the results of the detector
    for observation in datastream:
        value = observation[0][0]  # Extract the value from the observation
        result = detector.update(value)  # Update the detector with each value
        results.append(result)

    return results

def run_baseline_experiment(vector):
    detector = ELM_ECDD_Detector()
    datastream = vector_stream(vector)

    results = []
    for observation in datastream:
        value = observation[0][0]
        result = detector.update(value)
        results.append(result)

    return results


vector = np.load(
    "data/generated/linear/abrupt/linear_1_1.npy"
)

baseline_results = run_baseline_experiment(vector)

print("Total de entradas:", len(vector))
print("Total de resultados:", len(baseline_results))

print("\nDrifts ELM-ECDD:")

for result in baseline_results:
    if result["drift"]:
        print(
            "t =", result["t"],
            "| error =", result["error"],
            "| retrained =", result["retrained"]
        )