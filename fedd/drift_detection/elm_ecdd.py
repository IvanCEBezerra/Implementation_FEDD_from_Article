import numpy as np

from fedd.drift_detection.ecdd import ECDDDetector
from fedd.drift_detection.elm import ELMRegressor


class ELM_ECDD_Detector:
    def __init__(
        self, 
        p_lags=5,           # Past observations for prediction
        h_neurons=10,       # Hidden neurons
        initial_train=1000, # Initial training size
        n_retrain=400,      # Samples before retrain
        lambda_param=0.2, 
        W=1.0, 
        C=1.5
    ):
        self.p_lags = p_lags
        self.initial_train = initial_train
        self.n_retrain = n_retrain
        
        # Init ELM predictor
        self.elm = ELMRegressor(n_hidden=h_neurons)
        
        # Init ECDD monitor
        self.ecdd = ECDDDetector(
            lambda_param=lambda_param, 
            W=W, 
            C=C, 
        )
        
        self.samples = []
        self.buffer_concept = [] # Buffer for retraining after drift
        self.t = -1
        self.is_trained = False
        self.warn = 0
        
    def _create_dataset(self, series):
        """
        Formata uma série 1D em matriz X (p_lags) e vetor y (target) para treinar o ELM.
        """
        X, y = [], []
        for i in range(len(series) - self.p_lags):
            X.append(series[i : i + self.p_lags])
            y.append(series[i + self.p_lags])
        return np.array(X), np.array(y)

    def update(self, value):
        self.samples.append(value)
        self.buffer_concept.append(value)
        self.t += 1
        
        result = {
            "t": self.t, 
            "error": None, 
            "warning": False, 
            "warning_state": False,
            "drift": False, 
            "retrained": False
        }

        # 1. Initial training phase
        if not self.is_trained:
            # Wait for enough samples
            if len(self.samples) == self.initial_train + self.p_lags:
                X_train, y_train = self._create_dataset(self.samples)
                self.elm.fit(X_train, y_train)
                self.is_trained = True
                
                # Clear buffer after learning initial concept
                self.buffer_concept = [] 
            return result
            
        # 2. Online prediction
        if len(self.samples) >= self.p_lags + 1:
            # Use past observations as input
            x_input = np.array(self.samples[-(self.p_lags + 1): -1]).reshape(1, -1)
            y_true = value
            
            # Predict next value
            y_pred = self.elm.predict(x_input)[0]
            
            # Prediction error replaces feature distance
            error = abs(y_true - y_pred)
            result["error"] = error
            
            # 3. ECDD test
            warning_signal, drift_signal, _ = self.ecdd.update(error)
            
            result["warning"] = warning_signal
            result["drift"] = drift_signal
            
            if warning_signal and self.warn == 0:
                self.warn = self.t
                
            if drift_signal:
                self.warn = 0
                self.ecdd.reset()
                
                # 4. Retrain after drift
                if len(self.buffer_concept) >= self.n_retrain + self.p_lags:
                    # Use only recent samples to avoid old concept
                    clean_data = self.buffer_concept[-self.n_retrain - self.p_lags:]
                    X_retrain, y_retrain = self._create_dataset(clean_data)
                    
                    self.elm.fit(X_retrain, y_retrain)
                    result["retrained"] = True
                
                # Clear buffer after confirmed drift
                self.buffer_concept = []

            result["warning_state"] = (self.warn != 0)

        return result