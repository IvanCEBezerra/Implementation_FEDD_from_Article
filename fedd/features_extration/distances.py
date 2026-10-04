import numpy as np


def cosine_distance(vec_a, vec_b):
    """
    Computes the cosine distance between two feature vectors.

    dist_cos(A, B) =
        1 - (<A, B> / (||A|| * ||B||))
    """

    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)

    dot_product = np.dot(vec_a, vec_b)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = (
        dot_product
        / (norm_a * norm_b)
    )

    # Proteção contra pequenos erros numéricos,
    # por exemplo 1.0000000000000002.
    similarity = np.clip(
        similarity,
        -1.0,
        1.0
    )

    distance = 1.0 - similarity

    return float(distance)


def pearson_distance(vec_a, vec_b):
    """
    Computes the Pearson correlation distance
    between two feature vectors.

    dist_pear(A, B) =
        1 - Corr(A, B)
    """

    vec_a = np.asarray(vec_a, dtype=float)
    vec_b = np.asarray(vec_b, dtype=float)

    if len(vec_a) < 2 or len(vec_b) < 2:
        return 0.0

    correlation = np.corrcoef(
        vec_a,
        vec_b
    )[0, 1]

    if np.isnan(correlation):
        return 0.0

    # Proteção contra pequenos erros numéricos.
    correlation = np.clip(
        correlation,
        -1.0,
        1.0
    )

    distance = 1.0 - correlation

    return float(distance)