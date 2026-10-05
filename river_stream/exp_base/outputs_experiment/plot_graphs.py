from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, rankdata
from scipy.stats import studentized_range


# ============================================================
# PATHS
# ============================================================

# plot_graphs.py está dentro de:
#
# river_stream/
# └── outputs_experiment/
#     ├── abrupt/
#     ├── gradual/
#     ├── output/
#     └── plot_graphs.py
#
BASE_DIR = Path(__file__).resolve().parent

ABRUPT_DIR = BASE_DIR / "abrupt"
GRADUAL_DIR = BASE_DIR / "gradual"

# Somente as figuras finais serão salvas aqui.
FIGURES_DIR = BASE_DIR / "output"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

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


# Séries lineares utilizadas no experimento.
#
# Cada série possui 12000 observações e três concept drifts.
TRUE_DRIFTS = [3000, 6000, 9000]

SERIES_LENGTH = 12000


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results(path):
    """
    Carrega os resultados salvos pelo experiment.py.
    """

    data = np.load(
        path,
        allow_pickle=True
    )

    return list(data)


def extract_detected_drifts(results):
    """
    Retorna apenas os instantes onde drift=True.
    """

    detected_drifts = []

    for result in results:

        # Alguns elementos do ndarray podem continuar como
        # np.object_; item() converte para dict quando necessário.
        if hasattr(result, "item") and not isinstance(result, dict):
            try:
                result = result.item()
            except ValueError:
                pass

        if not isinstance(result, dict):
            continue

        if result.get("drift", False):
            detected_drifts.append(
                int(result["t"])
            )

    return sorted(detected_drifts)


# ============================================================
# METRICS
# ============================================================

def evaluate_detections(
    detected_drifts,
    true_drifts=TRUE_DRIFTS,
    series_length=SERIES_LENGTH,
):
    """
    Calcula as três métricas usadas no artigo:

    1. Número de falsos alarmes
    2. Drift detection delay
    3. Miss-detection rate

    Cada detecção pode corresponder a apenas um drift real.

    Para cada drift real:
        - a primeira detecção após o drift e antes do próximo
          drift é considerada correta;
        - as demais detecções são falsos alarmes.

    Se um drift não for detectado:
        - todo o tamanho do conceito seguinte conta como delay,
          conforme definido no artigo.
    """

    detected_drifts = sorted(
        int(t) for t in detected_drifts
    )

    used_detections = set()

    false_alarms = 0
    missed_drifts = 0
    total_delay = 0

    delays = []

    # --------------------------------------------------------
    # Avalia cada drift conhecido
    # --------------------------------------------------------

    for i, true_drift in enumerate(true_drifts):

        if i < len(true_drifts) - 1:
            concept_end = true_drifts[i + 1]
        else:
            concept_end = series_length

        # Detecções que ocorreram dentro do conceito iniciado
        # por este drift.
        candidates = [
            detection
            for detection in detected_drifts
            if (
                true_drift <= detection < concept_end
                and detection not in used_detections
            )
        ]

        if candidates:

            # Primeira detecção válida após o drift.
            detection = candidates[0]

            used_detections.add(detection)

            delay = detection - true_drift

            delays.append(delay)
            total_delay += delay

        else:

            # Miss-detection.
            missed_drifts += 1

            # O artigo considera todas as observações
            # pertencentes ao novo conceito como atraso.
            miss_delay = concept_end - true_drift

            delays.append(miss_delay)
            total_delay += miss_delay

    # --------------------------------------------------------
    # Tudo que não foi usado como detecção de um drift real
    # é falso alarme.
    # --------------------------------------------------------

    for detection in detected_drifts:

        if detection not in used_detections:
            false_alarms += 1

    miss_detection_rate = (
        100.0
        * missed_drifts
        / len(true_drifts)
    )

    return {
        "false_alarms": false_alarms,
        "detection_delay": total_delay,
        "miss_detection_rate": miss_detection_rate,
        "missed_drifts": missed_drifts,
        "delays": delays,
    }


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_dataset_directories(root):
    """
    Procura recursivamente diretórios que contenham pelo menos
    um dos arquivos produzidos pelo experimento.

    Isso deixa o código robusto mesmo que haja uma camada
    adicional de pastas dentro de abrupt/ ou gradual/.
    """

    if not root.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {root}"
        )

    directories = set()

    filenames = set(METHOD_FILES.values())

    for path in root.rglob("*.npy"):

        if path.name in filenames:
            directories.add(path.parent)

    return sorted(directories)


# ============================================================
# COLLECT METRICS
# ============================================================

def collect_metrics():
    rows = []

    drift_directories = {
        "abrupt": ABRUPT_DIR,
        "gradual": GRADUAL_DIR,
    }

    for drift_type, root in drift_directories.items():

        dataset_directories = (
            find_dataset_directories(root)
        )

        print(
            f"{drift_type}: "
            f"{len(dataset_directories)} datasets encontrados"
        )

        for dataset_dir in dataset_directories:

            dataset_name = dataset_dir.name

            for method, filename in METHOD_FILES.items():

                result_path = dataset_dir / filename

                if not result_path.exists():

                    print(
                        "[WARNING] Arquivo ausente:",
                        result_path
                    )

                    continue

                results = load_results(result_path)

                detections = extract_detected_drifts(
                    results
                )

                metrics = evaluate_detections(
                    detections
                )

                rows.append(
                    {
                        "dataset": dataset_name,
                        "drift_type": drift_type,
                        "method": method,

                        "false_alarms":
                            metrics["false_alarms"],

                        "detection_delay":
                            metrics["detection_delay"],

                        "miss_detection_rate":
                            metrics["miss_detection_rate"],
                    }
                )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        raise RuntimeError(
            "\nNenhum resultado foi encontrado.\n\n"
            f"Esperado em:\n"
            f"  {ABRUPT_DIR}\n"
            f"  {GRADUAL_DIR}\n"
        )

    return dataframe


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def build_metric_table(dataframe, metric):
    """
    Linhas: datasets
    Colunas: métodos

    Cada linha representa uma comparação pareada entre os
    métodos sobre o mesmo dataset.
    """

    table = dataframe.pivot_table(
        index=["drift_type", "dataset"],
        columns="method",
        values=metric,
        aggfunc="first",
    )

    available_methods = [
        method
        for method in METHOD_ORDER
        if method in table.columns
    ]

    table = table[available_methods]

    # Friedman só pode comparar datasets para os quais todos os
    # métodos possuem resultado.
    table = table.dropna()

    return table


def calculate_average_ranks(table):
    """
    Menor valor da métrica = melhor rank.

    Rank 1 é o melhor método.
    """

    ranks = []

    for _, row in table.iterrows():

        row_ranks = rankdata(
            row.values,
            method="average"
        )

        ranks.append(row_ranks)

    ranks = np.asarray(ranks)

    return ranks.mean(axis=0)


def calculate_cd(
    number_datasets,
    number_methods,
    alpha=0.05,
):
    """
    Critical Difference do teste de Nemenyi.

        CD = q_alpha *
             sqrt(k(k+1)/(6N))
    """

    q_alpha = (
        studentized_range.ppf(
            1.0 - alpha,
            number_methods,
            np.inf,
        )
        / np.sqrt(2.0)
    )

    return q_alpha * np.sqrt(
        number_methods
        * (number_methods + 1)
        / (6.0 * number_datasets)
    )


# ============================================================
# FIGURE 3
# ============================================================

def draw_rank_panel(
    ax,
    dataframe,
    metric,
    title,
):
    """
    Painel visualmente próximo da Figura 3 do artigo.

    O artigo apresenta:
        - métodos no eixo X;
        - average rank no eixo Y;
        - ponto central;
        - barra correspondente à critical difference;
        - grade horizontal pontilhada.
    """

    table = build_metric_table(
        dataframe,
        metric
    )

    methods = list(table.columns)

    if len(methods) < 2:
        raise RuntimeError(
            f"Métodos insuficientes para {metric}"
        )

    average_ranks = calculate_average_ranks(
        table
    )

    cd = calculate_cd(
        number_datasets=len(table),
        number_methods=len(methods),
    )

    # Friedman
    if len(methods) >= 3:
        values = [
            table[method].values
            for method in methods
        ]

        statistic, p_value = (
            friedmanchisquare(*values)
        )

        print(
            f"{title}: "
            f"Friedman={statistic:.6f}, "
            f"p={p_value:.6g}, "
            f"CD={cd:.4f}"
        )

    x = np.arange(len(methods))

    # No artigo, a barra vertical representa visualmente a
    # diferença crítica em torno do rank.
    ax.errorbar(
        x,
        average_ranks,
        yerr=cd / 2.0,
        fmt="o",
        capsize=5,
        markersize=4,
        linewidth=1.0,
        elinewidth=1.0,
    )

    ax.set_title(
        title,
        fontsize=10,
        pad=4,
    )

    ax.set_ylabel(
        "Average Ranks",
        fontsize=8,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        methods,
        fontsize=7,
    )

    ax.tick_params(
        axis="y",
        labelsize=7,
    )

    # Rank 1 = melhor.
    # Mantemos os valores menores visualmente embaixo, como
    # na Figura 3 original.
    lower = max(
        0.5,
        np.floor(
            np.min(
                average_ranks - cd / 2
            )
            * 2
        ) / 2
    )

    upper = min(
        len(methods) + 0.5,
        np.ceil(
            np.max(
                average_ranks + cd / 2
            )
            * 2
        ) / 2
    )

    if upper - lower < 1:
        upper = lower + 1

    ax.set_ylim(lower, upper)

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.7,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def generate_figure_3(dataframe):
    """
    Reproduz a estrutura da Figura 3:

        (a) False alarms
        (b) Drift detection delay
        (c) Miss-detection rates
    """

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(5.0, 7.0),
    )

    draw_rank_panel(
        axes[0],
        dataframe,
        "false_alarms",
        "Number of False Alarms",
    )

    axes[0].text(
        0.5,
        -0.30,
        "(a)",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=9,
    )

    draw_rank_panel(
        axes[1],
        dataframe,
        "detection_delay",
        "Drift Detection Delay",
    )

    axes[1].text(
        0.5,
        -0.30,
        "(b)",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
    )

    draw_rank_panel(
        axes[2],
        dataframe,
        "miss_detection_rate",
        "Miss-detection Rates",
    )

    axes[2].text(
        0.5,
        -0.30,
        "(c)",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
        hspace=0.85
    )

    path = FIGURES_DIR / "figure_3.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figura 3 salva em:")
    print(path)


# ============================================================
# FIGURE 4
# ============================================================

def get_boxplot_values(
    dataframe,
    drift_type,
    metric,
):
    values = []

    subset = dataframe[
        dataframe["drift_type"]
        == drift_type
    ]

    for method in METHOD_ORDER:

        method_values = subset[
            subset["method"] == method
        ][metric].to_numpy()

        values.append(method_values)

    return values


def style_boxplot_axis(
    ax,
    title,
    ylabel,
):
    ax.set_title(
        title,
        fontsize=9,
        pad=3,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=7,
    )

    ax.tick_params(
        axis="x",
        labelsize=6,
    )

    ax.tick_params(
        axis="y",
        labelsize=6,
    )

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.7,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def create_boxplot(
    ax,
    dataframe,
    drift_type,
    metric,
    title,
    ylabel,
):
    values = get_boxplot_values(
        dataframe,
        drift_type,
        metric,
    )

    ax.boxplot(
        values,
        tick_labels=METHOD_ORDER,
        widths=0.55,
        patch_artist=False,
        showfliers=True,

        medianprops={
            "linewidth": 1.0
        },

        boxprops={
            "linewidth": 0.8
        },

        whiskerprops={
            "linewidth": 0.8
        },

        capprops={
            "linewidth": 0.8
        },

        flierprops={
            "marker": "+",
            "markersize": 4,
        },
    )

    style_boxplot_axis(
        ax,
        title,
        ylabel,
    )


def generate_figure_4(dataframe):
    """
    Reproduz a disposição exata da Figura 4:

                          Abrupt       Gradual

        False alarms       [ ]           [ ]

        Detection delay    [ ]           [ ]

        Miss detection     [ ]           [ ]
    """

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8.0, 7.0),
    )

    # --------------------------------------------------------
    # (a) FALSE ALARMS
    # --------------------------------------------------------

    create_boxplot(
        axes[0, 0],
        dataframe,
        "abrupt",
        "false_alarms",
        "Number of False Alarms: Abrupt Drifts",
        "False alarms",
    )

    create_boxplot(
        axes[0, 1],
        dataframe,
        "gradual",
        "false_alarms",
        "Number of False Alarms: Gradual Drifts",
        "False alarms",
    )

    # --------------------------------------------------------
    # (b) DETECTION DELAY
    # --------------------------------------------------------

    create_boxplot(
        axes[1, 0],
        dataframe,
        "abrupt",
        "detection_delay",
        "Drift Detection Delay: Abrupt Drifts",
        "Delay (instances)",
    )

    create_boxplot(
        axes[1, 1],
        dataframe,
        "gradual",
        "detection_delay",
        "Drift Detection Delay: Gradual Drifts",
        "Delay (instances)",
    )

    # --------------------------------------------------------
    # (c) MISS DETECTION
    # --------------------------------------------------------

    create_boxplot(
        axes[2, 0],
        dataframe,
        "abrupt",
        "miss_detection_rate",
        "Miss-detection Rates: Abrupt Drifts",
        "Rate (%)",
    )

    create_boxplot(
        axes[2, 1],
        dataframe,
        "gradual",
        "miss_detection_rate",
        "Miss-detection Rates: Gradual Drifts",
        "Rate (%)",
    )

    # Como existem exatamente 3 drifts, os valores possíveis
    # são aproximadamente:
    # 0, 33.33, 66.67, 100.
    axes[2, 0].set_ylim(-5, 105)
    axes[2, 1].set_ylim(-5, 105)

    axes[2, 0].set_yticks(
        [0, 20, 40, 60, 80, 100]
    )

    axes[2, 1].set_yticks(
        [0, 20, 40, 60, 80, 100]
    )

    # Rótulos (a), (b), (c), posicionados entre as linhas,
    # como na publicação.
    axes[0, 0].text(
        1.05,
        -0.28,
        "(a)",
        transform=axes[0, 0].transAxes,
        ha="center",
        fontsize=9,
    )

    axes[1, 0].text(
        1.05,
        -0.28,
        "(b)",
        transform=axes[1, 0].transAxes,
        ha="center",
        fontsize=9,
    )

    axes[2, 0].text(
        1.05,
        -0.28,
        "(c)",
        transform=axes[2, 0].transAxes,
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
        hspace=0.55,
        wspace=0.28,
    )

    path = FIGURES_DIR / "figure_4.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figura 4 salva em:")
    print(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Reading experiment results...")
    print()

    dataframe = collect_metrics()

    print()
    print(
        "Total de combinações "
        "dataset/método:",
        len(dataframe),
    )

    print()

    print(
        dataframe.groupby(
            ["drift_type", "method"]
        )[
            [
                "false_alarms",
                "detection_delay",
                "miss_detection_rate",
            ]
        ].mean()
    )

    print()
    print("Generating Figure 3...")

    generate_figure_3(
        dataframe
    )

    print()
    print("Generating Figure 4...")

    generate_figure_4(
        dataframe
    )

    print()
    print("=" * 60)
    print("Finished")
    print("=" * 60)

    print(
        "\nSomente duas figuras foram geradas:"
    )

    print(
        FIGURES_DIR / "figure_3.png"
    )

    print(
        FIGURES_DIR / "figure_4.png"
    )


if __name__ == "__main__":
    main()