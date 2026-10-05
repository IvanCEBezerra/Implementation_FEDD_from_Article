"""Geração de séries com FriedmanDrift (River) e salvamento em ./river_data.

Tipos de drift usados:
  - "lea": Local Expanding Abrupt   -> 3 posições de drift
  - "gsg": Global and Slow Gradual  -> 2 posições + transition_window

A série univariada para o FEDD é o alvo `y`. Os atributos `X` também são
salvos, caso você queira usá-los depois.

Uso:
    python generate_friedman_drift.py
"""

import json
from itertools import islice
from pathlib import Path

import numpy as np
from river.datasets import synth

OUTPUT_DIR = Path("river_stream\\river_datasets\\river_data")

# Valores padrão (inspirados no artigo: séries com 12.000 pontos)
N_SAMPLES = 12_000
LEA_POSITIONS = (3000, 6000, 9000)
GSG_POSITIONS = (4000, 8000)
GSG_TRANSITION_WINDOW = 400


# --------------------------------------------------------------------------
# Geração
# --------------------------------------------------------------------------
def _stream_to_arrays(dataset, n_samples):
    """Consome `n_samples` pontos do stream e devolve (X, y) como arrays numpy."""
    feature_names = None
    rows, targets = [], []

    for x, y in islice(dataset, n_samples):
        if feature_names is None:
            feature_names = list(x.keys())
        rows.append([x[name] for name in feature_names])
        targets.append(y)

    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def generate_lea(seed=42, n_samples=N_SAMPLES, positions=LEA_POSITIONS):
    """Gera uma série FriedmanDrift com drift local expansivo abrupto (lea)."""
    dataset = synth.FriedmanDrift(
        drift_type="lea",
        position=tuple(positions),
        seed=seed,
    )
    X, y = _stream_to_arrays(dataset, n_samples)
    meta = {
        "generator": "FriedmanDrift",
        "drift_type": "lea",
        "drift_kind": "abrupt",
        "seed": seed,
        "n_samples": n_samples,
        "positions": list(positions),
    }
    return X, y, meta


def generate_gsg(
    seed=42,
    n_samples=N_SAMPLES,
    positions=GSG_POSITIONS,
    transition_window=GSG_TRANSITION_WINDOW,
):
    """Gera uma série FriedmanDrift com drift global gradual lento (gsg)."""
    dataset = synth.FriedmanDrift(
        drift_type="gsg",
        position=tuple(positions),
        transition_window=transition_window,
        seed=seed,
    )
    X, y = _stream_to_arrays(dataset, n_samples)
    meta = {
        "generator": "FriedmanDrift",
        "drift_type": "gsg",
        "drift_kind": "gradual",
        "seed": seed,
        "n_samples": n_samples,
        "positions": list(positions),
        "transition_window": transition_window,
    }
    return X, y, meta


# --------------------------------------------------------------------------
# Salvamento
# --------------------------------------------------------------------------
def save_series(name, X, y, meta, output_dir=OUTPUT_DIR):
    """Salva uma série na pasta `output_dir`.

    Arquivos criados:
      <name>.npy       -> série univariada (y), pronta para o FEDD
      <name>_X.npy     -> atributos
      <name>.json      -> metadados (posições dos drifts = verdade de base)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{name}.npy", y)
    np.save(output_dir / f"{name}_X.npy", X)
    with open(output_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return output_dir / f"{name}.npy"


def generate_and_save(drift_type, seed, output_dir=OUTPUT_DIR, **kwargs):
    """Gera uma série do tipo `drift_type` ("lea" ou "gsg") e salva em disco."""
    generators = {"lea": generate_lea, "gsg": generate_gsg}
    if drift_type not in generators:
        raise ValueError(f"drift_type deve ser 'lea' ou 'gsg', recebido: {drift_type!r}")

    X, y, meta = generators[drift_type](seed=seed, **kwargs)
    name = f"friedman_{drift_type}_seed{seed}"
    return save_series(name, X, y, meta, output_dir)


def generate_many(n_series=5, base_seed=0, output_dir=OUTPUT_DIR):
    """Gera `n_series` exemplares de cada tipo (lea e gsg), com seeds diferentes."""
    saved = []
    for i in range(n_series):
        seed = base_seed + i
        saved.append(generate_and_save("lea", seed, output_dir))
        saved.append(generate_and_save("gsg", seed, output_dir))
    return saved


if __name__ == "__main__":
    paths = generate_many(n_series=3, base_seed=0)
    for p in paths:
        print("salvo:", p)