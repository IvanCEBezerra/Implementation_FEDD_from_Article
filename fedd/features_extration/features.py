import numpy as np

from scipy.stats import skew, kurtosis
from scipy.special import digamma
from scipy.spatial import cKDTree


# ============================================================
# Turning points
# ============================================================

def calculate_turning_points(window):
    """
    Calcula a taxa de turning points da janela.

    Um ponto é considerado turning point quando é:
        - maior que os dois vizinhos; ou
        - menor que os dois vizinhos.

    Esta feature é calculada na série original,
    não na série diferenciada.
    """

    window = np.asarray(window, dtype=float)

    if len(window) < 3:
        return 0.0

    previous = window[:-2]
    current = window[1:-1]
    next_value = window[2:]

    local_maximum = (
        (current > previous)
        & (current > next_value)
    )

    local_minimum = (
        (current < previous)
        & (current < next_value)
    )

    turning_points = (
        local_maximum
        | local_minimum
    )

    return np.mean(turning_points)


# ============================================================
# Autocorrelation
# ============================================================

def calculate_autocorrelation(diff_window, max_lag=5):
    """
    Calcula as autocorrelações dos lags 1 até max_lag.

    Implementação direta com NumPy, sem statsmodels.

    A feature é calculada sobre a série diferenciada.
    """

    x = np.asarray(diff_window, dtype=float)

    if len(x) <= max_lag:
        return np.zeros(max_lag)

    x_centered = x - np.mean(x)

    denominator = np.sum(
        x_centered ** 2
    )

    if denominator == 0:
        return np.zeros(max_lag)

    values = []

    for lag in range(1, max_lag + 1):

        numerator = np.sum(
            x_centered[lag:]
            * x_centered[:-lag]
        )

        autocorrelation = (
            numerator / denominator
        )

        values.append(
            autocorrelation
        )

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# Partial autocorrelation
# ============================================================

def calculate_partial_autocorrelation(
    diff_window,
    max_lag=5
):
    """
    Calcula as autocorrelações parciais dos lags
    1 até max_lag.

    A PACF é calculada através das equações de
    Yule-Walker.

    Isso substitui:

        statsmodels.tsa.stattools.pacf(
            ...,
            method="yw"
        )

    sem exigir statsmodels.

    A feature é calculada sobre a série diferenciada.
    """

    x = np.asarray(
        diff_window,
        dtype=float
    )

    if len(x) <= max_lag:
        return np.zeros(max_lag)

    # Centralização
    x = x - np.mean(x)

    n = len(x)

    # --------------------------------------------------------
    # Autocovariâncias
    # --------------------------------------------------------

    autocovariances = np.empty(
        max_lag + 1,
        dtype=float
    )

    for lag in range(max_lag + 1):

        autocovariances[lag] = (
            np.dot(
                x[lag:],
                x[:n - lag]
            )
            / n
        )

    # Série constante
    if autocovariances[0] == 0:
        return np.zeros(max_lag)

    values = []

    # --------------------------------------------------------
    # Yule-Walker para cada ordem
    # --------------------------------------------------------

    for lag in range(1, max_lag + 1):

        indices = np.abs(
            np.subtract.outer(
                np.arange(lag),
                np.arange(lag)
            )
        )

        R = autocovariances[indices]

        r = autocovariances[
            1:lag + 1
        ]

        try:

            coefficients = np.linalg.solve(
                R,
                r
            )

            # O último coeficiente AR corresponde
            # à PACF do lag atual.
            pacf_value = coefficients[-1]

        except np.linalg.LinAlgError:

            pacf_value = 0.0

        values.append(
            pacf_value
        )

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# Bicorrelation
# ============================================================

def calculate_bicorrelation(
    diff_window,
    max_lag=3
):
    """
    Calcula a bicorrelação / three-point autocorrelation
    para os lags 1 até max_lag.

    Para cada lag k:

        E[x_t * x_(t-k) * x_(t-2k)]

    A feature é calculada sobre a série diferenciada.

    O artigo descreve esta feature como bicorrelation /
    three-point autocorrelation, mas não apresenta a
    fórmula completa no texto devido à limitação de espaço.
    """

    diff_window = np.asarray(
        diff_window,
        dtype=float
    )

    values = []

    for lag in range(1, max_lag + 1):

        if len(diff_window) <= 2 * lag:
            values.append(0.0)
            continue

        x_t = diff_window[
            2 * lag:
        ]

        x_t_lag = diff_window[
            lag:-lag
        ]

        x_t_2lag = diff_window[
            :-2 * lag
        ]

        bicorrelation = np.mean(
            x_t
            * x_t_lag
            * x_t_2lag
        )

        values.append(
            bicorrelation
        )

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# Continuous mutual information
# ============================================================

def _continuous_mutual_information(
    x,
    y,
    n_neighbors=3
):
    """
    Estima a informação mútua entre duas variáveis
    contínuas através de um estimador baseado em
    k-nearest-neighbors.

    Essa implementação evita a dependência do
    scikit-learn.

    Parameters
    ----------
    x : array-like
        Primeira variável.

    y : array-like
        Segunda variável.

    n_neighbors : int
        Número de vizinhos utilizado pelo estimador.

    Returns
    -------
    float
        Informação mútua estimada.
    """

    x = np.asarray(
        x,
        dtype=float
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=float
    ).reshape(-1)

    if len(x) != len(y):
        raise ValueError(
            "x e y precisam possuir o mesmo tamanho."
        )

    n_samples = len(x)

    if n_samples <= n_neighbors:
        return 0.0

    # Variável constante não contém informação
    # para esse cálculo.
    if (
        np.all(x == x[0])
        or np.all(y == y[0])
    ):
        return 0.0

    # --------------------------------------------------------
    # Pequeno ruído para desempatar observações repetidas
    # --------------------------------------------------------

    # Seed fixa para garantir reprodutibilidade.
    rng = np.random.default_rng(42)

    scale_x = max(
        1.0,
        np.mean(np.abs(x))
    )

    scale_y = max(
        1.0,
        np.mean(np.abs(y))
    )

    x = (
        x
        + 1e-10
        * scale_x
        * rng.standard_normal(n_samples)
    )

    y = (
        y
        + 1e-10
        * scale_y
        * rng.standard_normal(n_samples)
    )

    # --------------------------------------------------------
    # Espaço conjunto (x, y)
    # --------------------------------------------------------

    xy = np.column_stack(
        (x, y)
    )

    joint_tree = cKDTree(
        xy
    )

    # p = infinito -> distância de Chebyshev
    distances, _ = joint_tree.query(
        xy,
        k=n_neighbors + 1,
        p=np.inf
    )

    # A primeira distância é sempre zero
    # porque corresponde ao próprio ponto.
    epsilon = distances[
        :,
        n_neighbors
    ]

    # Queremos contar pontos estritamente dentro
    # da vizinhança.
    epsilon = np.nextafter(
        epsilon,
        0
    )

    # --------------------------------------------------------
    # Árvores marginais
    # --------------------------------------------------------

    x_tree = cKDTree(
        x.reshape(-1, 1)
    )

    y_tree = cKDTree(
        y.reshape(-1, 1)
    )

    nx = np.empty(
        n_samples,
        dtype=int
    )

    ny = np.empty(
        n_samples,
        dtype=int
    )

    for i in range(n_samples):

        nx[i] = (
            len(
                x_tree.query_ball_point(
                    [x[i]],
                    epsilon[i],
                    p=np.inf
                )
            )
            - 1
        )

        ny[i] = (
            len(
                y_tree.query_ball_point(
                    [y[i]],
                    epsilon[i],
                    p=np.inf
                )
            )
            - 1
        )

    # --------------------------------------------------------
    # Estimador k-NN da informação mútua
    # --------------------------------------------------------

    mutual_information = (
        digamma(n_neighbors)
        + digamma(n_samples)
        - np.mean(
            digamma(nx + 1)
            + digamma(ny + 1)
        )
    )

    # MI teoricamente >= 0.
    # Pequenos valores negativos podem aparecer
    # devido ao erro do estimador.
    return max(
        0.0,
        float(mutual_information)
    )


# ============================================================
# Mutual information features
# ============================================================

def calculate_mutual_information(
    diff_window,
    max_lag=3
):
    """
    Calcula a informação mútua entre:

        x_t

    e

        x_(t-k)

    para:

        k = 1, ..., max_lag

    A feature é calculada sobre a série diferenciada.

    Utiliza um estimador k-NN para variáveis contínuas,
    sem depender de scikit-learn.
    """

    diff_window = np.asarray(
        diff_window,
        dtype=float
    )

    values = []

    for lag in range(1, max_lag + 1):

        if len(diff_window) <= lag:
            values.append(0.0)
            continue

        x_lagged = diff_window[
            :-lag
        ]

        x_current = diff_window[
            lag:
        ]

        # Variáveis constantes
        if (
            np.all(
                x_lagged == x_lagged[0]
            )
            or np.all(
                x_current == x_current[0]
            )
        ):
            values.append(0.0)
            continue

        try:

            mi = _continuous_mutual_information(
                x_lagged,
                x_current,
                n_neighbors=3
            )

        except (
            ValueError,
            FloatingPointError
        ):

            mi = 0.0

        values.append(
            mi
        )

    return np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# Nonlinear features
# ============================================================

def extract_nonlinear_features(
    diff_window
):
    """
    Extrai as duas categorias de features não lineares:

        - bicorrelation:       3 features
        - mutual information: 3 features

    Total: 6 features.
    """

    bicorrelation = (
        calculate_bicorrelation(
            diff_window,
            max_lag=3
        )
    )

    mutual_information = (
        calculate_mutual_information(
            diff_window,
            max_lag=3
        )
    )

    return (
        bicorrelation,
        mutual_information
    )


# ============================================================
# Main feature extraction
# ============================================================

def extract_features(window):
    """
    Extrai o vetor completo de características
    utilizado pelo FEDD.

    Estrutura:

        1 - turning points rate
        1 - variance
        1 - skewness
        1 - kurtosis
        5 - autocorrelation
        5 - partial autocorrelation
        3 - bicorrelation
        3 - mutual information

    Total = 20 features.

    Todas as features, exceto turning points rate,
    são calculadas sobre a série diferenciada.
    """

    window = np.asarray(
        window,
        dtype=float
    )

    if len(window) < 3:
        raise ValueError(
            "A janela precisa possuir pelo menos 3 pontos."
        )

    # --------------------------------------------------------
    # Turning points
    # --------------------------------------------------------

    turning_points = (
        calculate_turning_points(
            window
        )
    )

    # --------------------------------------------------------
    # Differencing
    # --------------------------------------------------------

    diff_window = np.diff(
        window
    )

    # --------------------------------------------------------
    # Linear features
    # --------------------------------------------------------

    autocorrelation = (
        calculate_autocorrelation(
            diff_window,
            max_lag=5
        )
    )

    partial_autocorrelation = (
        calculate_partial_autocorrelation(
            diff_window,
            max_lag=5
        )
    )

    variance = np.var(
        diff_window
    )

    skewness = skew(
        diff_window
    )

    kurt = kurtosis(
        diff_window
    )

    # --------------------------------------------------------
    # Nonlinear features
    # --------------------------------------------------------

    (
        bicorrelation,
        mutual_information
    ) = extract_nonlinear_features(
        diff_window
    )

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = np.concatenate([
        np.array([
            turning_points,
            variance,
            skewness,
            kurt
        ]),

        autocorrelation,

        partial_autocorrelation,

        bicorrelation,

        mutual_information
    ])

    # --------------------------------------------------------
    # Segurança numérica
    # --------------------------------------------------------

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return features