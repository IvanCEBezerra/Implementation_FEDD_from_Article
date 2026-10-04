import numpy as np

from fedd.drift_detection.ecdd import ECDDDetector
from fedd.features_extration.distances import cosine_distance
from fedd.features_extration.distances import pearson_distance
from fedd.features_extration.features import extract_features


class FEDDDetector:

    def __init__(
        self,
        m=300,
        lambda_param=0.2,
        W=1.0,
        C=1.5,
        distance="cosine"
    ):
        self.m = m
        self.lambda_param = lambda_param
        self.W = W
        self.C = C
        self.distance_name = distance

        self.samples = []

        # s = início do conceito conhecido
        self.s = 0

        # Vetor de características de referência
        self.fv0 = None

        # Índice temporal
        self.t = -1

        # Instante do primeiro warning
        self.warn = 0

        # Quantidade de vezes abaixo do warning threshold
        self.below_warn = 0

        self.ecdd = ECDDDetector(
            lambda_param=lambda_param,
            W=W,
            C=C
        )

    def update(self, value):

        self.samples.append(value)
        self.t += 1

        result = {
            "t": self.t,
            "distance": None,
            "warning": False,
            "warning_state": False,
            "drift": False,
            "Z_t": None,
            "mu_d": None,
            "warning_threshold": None,
            "drift_threshold": None
        }

        # ====================================================
        # Steps 4-5:
        # cria o vetor de características de referência
        #
        # Artigo:
        # t - s = m - 1
        # ====================================================

        if (
            self.fv0 is None
            and self.t - self.s == self.m - 1
        ):
            initial_window = np.asarray(
                self.samples[
                    self.s:self.t + 1
                ]
            )

            self.fv0 = extract_features(
                initial_window
            )

        # ====================================================
        # Steps 6-9:
        # processamento online
        #
        # Artigo:
        # t - s > m - 1
        # ====================================================

        elif (
            self.fv0 is not None
            and self.t - self.s > self.m - 1
        ):
            new_window = np.asarray(
                self.samples[
                    self.t - self.m + 1:self.t + 1
                ]
            )

            fvt = extract_features(
                new_window
            )

            # Step 8: distância entre fv0 e fvt

            if self.distance_name == "cosine":
                distance = cosine_distance(
                    self.fv0,
                    fvt
                )

            elif self.distance_name == "pearson":
                distance = pearson_distance(
                    self.fv0,
                    fvt
                )

            else:
                raise ValueError(
                    f"Unknown distance: {self.distance_name}"
                )

            # Step 9: atualiza estatísticas ECDD

            (
                warning_signal,
                drift_signal,
                below_warning
            ) = self.ecdd.update(distance)

            result["distance"] = distance
            result["warning"] = warning_signal
            result["drift"] = drift_signal
            result["Z_t"] = self.ecdd.Z_t
            result["mu_d"] = self.ecdd.mu_d
            result["warning_threshold"] = (
                self.ecdd.warning_threshold
            )
            result["drift_threshold"] = (
                self.ecdd.drift_threshold
            )

            # =================================================
            # Steps 10-13:
            # registra o primeiro warning
            # =================================================

            if (
                self.warn == 0
                and warning_signal
            ):
                self.warn = self.t

            # =================================================
            # Steps 14-17:
            # drift
            # =================================================

            if drift_signal:

                # Artigo: s = warn
                self.s = self.warn

                self.warn = 0
                self.below_warn = 0

                # Reinicializa o processo para o novo conceito
                self.fv0 = None
                self.ecdd.reset()

            # =================================================
            # Steps 18-26:
            # saída do estado de warning
            # =================================================

            elif self.warn > 0:

                if below_warning:
                    self.below_warn += 1

                if self.below_warn == 10:
                    self.warn = 0
                    self.below_warn = 0

            result["warning_state"] = (
                self.warn > 0
            )

        return result