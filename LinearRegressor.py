import numpy as np

class LinearRegressor():
    
    def __init__(self, lr = 0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        # Initiate weights and bias
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features, 1))
        self.b = 0

        for _ in self.n_iterations:
            # Calculate y_hat
            y_hat = np.dot(self.W, X) + self.b

            # Calculate the partial derivatives w.r.t weights and bias
            dw = (2/n_samples) * np.dot(X.T, (y_hat - y))
            db = (2/n_samples) * np.sum(y_hat - y)

            # Updates weights and bias
            self.W -= self.lr * dw
            self.b -= self.lr * db


    def predict(self, X):
        return np.dot(self.W, X) + self.b
