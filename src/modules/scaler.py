#Template scaler for 1 dimensional only
#Might need to improve for multi-dimensional data
import numpy as np
class MyScaler: #Scaler is so confusing
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, x): #Chuẩn hóa từ 0 đến 1
        x = list(x)
        self.mean = np.mean(x)
        self.std = np.std(x)
        return self
    def transform(self, x):
        return (x - self.mean) / self.std
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)