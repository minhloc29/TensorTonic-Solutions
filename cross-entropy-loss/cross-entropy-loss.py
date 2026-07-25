import numpy as np

def cross_entropy_loss(y_true, y_pred):

    # y_pred: (N, C)
    # y_true: (N, )

    y_pred = np.asarray(y_pred, dtype = float)
    y_true = np.asarray(y_true, dtype = int)

    total = 0
    n = len(y_true)
    for i, _ in enumerate(range(n)):


        eps = 1e-15
        idx = y_true[i]
        p = np.clip(y_pred[i][idx], eps, 1 - eps)
        total += np.log(p)

    total = (-total) / n
    return total