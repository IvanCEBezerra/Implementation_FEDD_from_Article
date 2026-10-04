import numpy as np


class ECDDDetector:

    def __init__(
        self,
        lambda_param=0.2,
        W=1.0,
        C=1.5
    ):
        self.lambda_param = lambda_param
        self.W = W
        self.C = C

        self.reset()

    def reset(self):
        """
        Reinicia o ECDD para monitorar uma nova sequência
        de distâncias.
        """

        self.distances = []

        # EWMA
        self.Z_t = None

        # Estatísticas das distâncias
        self.mu_d = None
        self.sigma_d = None

        # Desvio padrão da EWMA
        self.sigma_Z = None

        # Limites
        self.warning_threshold = None
        self.drift_threshold = None

    def update(self, distance):
        """
        Recebe uma nova distância e atualiza o ECDD.

        Retorna:
            warning_signal
            drift_signal
            below_warning
        """

        distance = float(distance)

        # 1. Armazena a nova distância
        self.distances.append(distance)

        t = len(self.distances)

        # 2. Média das distâncias
        self.mu_d = np.mean(self.distances)

        # 3. Desvio padrão das distâncias
        self.sigma_d = np.std(
            self.distances,
            ddof=0
        )

        # 4. Atualiza a EWMA
        if self.Z_t is None:
            self.Z_t = distance
        else:
            self.Z_t = (
                (1.0 - self.lambda_param) * self.Z_t
                + self.lambda_param * distance
            )

        # 5. Desvio padrão da EWMA
        factor = (
            self.lambda_param
            / (2.0 - self.lambda_param)
        )

        correction = (
            1.0
            - (1.0 - self.lambda_param) ** (2 * t)
        )

        self.sigma_Z = (
            np.sqrt(
                factor * correction
            )
            * self.sigma_d
        )

        # 6. Warning threshold
        self.warning_threshold = (
            self.mu_d
            + self.W * self.sigma_Z
        )

        # 7. Drift threshold
        self.drift_threshold = (
            self.mu_d
            + self.C * self.sigma_Z
        )

        # 8. Testes
        warning_signal = (
            self.Z_t > self.warning_threshold
        )

        drift_signal = (
            self.Z_t > self.drift_threshold
        )

        below_warning = (
            self.Z_t < self.warning_threshold
        )

        return (
            warning_signal,
            drift_signal,
            below_warning
        )