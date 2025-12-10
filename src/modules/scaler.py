#Template scaler for 1 dimensional only
#Might need to improve for multi-dimensional data
import numpy as np
class MyScaler: #Scaler is so confusing
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x):
        x = np.array(x, dtype=float)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return self
    def transform(self, x):
        return (x - self.mean) / self.std
    def fit_transform(self, x):
        return self.fit(x).transform(x)