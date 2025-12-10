#Template to improve upon
#class cuz why not, right???
import pandas as pd
class MyLogisticRegression:
    def __init__(self, lr=0.1, steps=1000):
        self.lr = lr
        self.steps = steps
        self.W = None
        self.bias = None
    def sigmoid(self,z): #Sigma boy :))
        return 1/(1 + pd.exp(-z))
    def logisticRegression(self, X, Y):
        m, n = X.shape
        W = pd.zeros(n)
        bias = 0
        for _ in range(self.steps):
            z = pd.dot(X,W) + bias
            y_hat = self.sigmoid(z)
            descent_w = (1/m) * pd.dot(X.T, (y_hat - Y))
            descent_b = (1/m) * pd.sum(y_hat - Y)
            W = W - self.lr * descent_w
            bias = bias - self.lr * descent_b
        self.W = W
        self.bias = bias
    def predict_proba(self, X):
        z = pd.dot(X, self.W) + self.bias
        prob = self.sigmoid(z)
        return prob
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)