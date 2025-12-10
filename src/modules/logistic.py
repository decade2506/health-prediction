# Template to improve upon
# Logistic Regression from scratch
import numpy as np
class MyLogisticRegression:
    def __init__(self, lr=0.1, steps=1000):
        self.lr = lr
        self.steps = steps
        self.W = None
        self.bias = None
    def sigmoid(self,z): #Sigma boy :))
        return 1/(1 + np.exp(-z))
    
    #Like the name brooo
    def logisticRegression(self, X, Y):
        m, n = X.shape
        W = np.zeros(n)
        bias = 0
        for _ in range(self.steps):
            z = np.dot(X,W) + bias
            y_hat = self.sigmoid(z)
            descent_w = (1/m) * np.dot(X.T, (y_hat - Y))
            descent_b = (1/m) * np.sum(y_hat - Y)
            W = W - self.lr * descent_w
            bias = bias - self.lr * descent_b
        self.W = W
        self.bias = bias
    
    # Tỉ lệ dự đoán
    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.bias
        prob = self.sigmoid(z)
        return prob
    
    # Dự đoán nếu xác suất >= 0.5 thì là 1, ngược lại là 0
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)