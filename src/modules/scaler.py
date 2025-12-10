# Template scaler from scratch
import numpy as np
class MyScaler: #Scaler is so confusing
    def __init__(self):
        self.mean = None
        self.std = None
    # Tính mean và std
    def fit(self, x):
        x = np.array(x, dtype=float)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return self
    
    # Standardize dữ liệu dựa theo mean và std
    def transform(self, x):
        return (x - self.mean) / self.std
    
    # Kết hợp fit và transform
    def fit_transform(self, x):
        return self.fit(x).transform(x)