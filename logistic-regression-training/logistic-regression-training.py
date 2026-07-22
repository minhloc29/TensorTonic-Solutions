import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    z_pred = X @ w + b
    p = _sigmoid(z_pred)
    for step in range(steps):
        
        z_pred = X@w + b
        p = _sigmoid(z_pred)
        y_pred = (1 / (1 + np.exp(-z_pred)))

        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = (1 / n_samples) * np.sum(y_pred - y)

        w -= lr * dw
        b -= lr * db
    return w, b