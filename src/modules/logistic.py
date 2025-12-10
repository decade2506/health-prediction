#Template to improve upon
import pandas as pd
def sigmoid(z): #Sigma boy :))
    return 1/(1 + pd.exp(-z))
def logisticRegression(X, Y, lr = 0.1, steps = 1000):
    m, n = X.shape
    W = pd.zeros(n)
    bias = 0
    for _ in range(steps):
        z = pd.dot(X,W) + bias
        y_hat = sigmoid(z)
        descent_w = (1/m) * pd.dot(X.T, (y_hat - Y))
        descent_b = (1/m) * pd.sum(y_hat - Y)
        W = W - lr * descent_w
        bias = bias - lr * descent_b

    return W, bias
def predict(X, W, bias):
    z = pd.dot(X, W) + bias
    prob = sigmoid(z)
    return (prob >= 0.5).asType(int)