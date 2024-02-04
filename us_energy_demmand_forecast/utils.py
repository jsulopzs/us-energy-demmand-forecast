import numpy as np

def create_sequences(data, n_windows=24):
    X, y = [], []
    for i in range(len(data) - n_windows):
        X.append(data[i:i+n_windows])
        y.append(data[i+n_windows])
    return np.array(X), np.array(y)