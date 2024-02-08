import numpy as np

class LinearRegression:
    
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        # Initiate weights and bias
        n_samples, n_features = X.shape
        self.W = np.zeros((n_features,1)) # dim is (n_features, 1)
        self.b = 0

        for _ in range(self.n_iterations):
            # Calculate y_hat
            y_hat = np.dot(X, self.W) + self.b  # dim is (n_samples, 1)

            # Calculate the partial derivatives w.r.t weights and bias
            # y is dim (n_samples,) so np.newaxis used to correct dimension.
            dw = (2/n_samples) * np.dot(X.T, (y_hat - y[:, np.newaxis])) # dim is (n_features, 1)
            db = (2/n_samples) * np.sum(y_hat - y[:, np.newaxis])

            # Update weights and bias
            self.W -= self.lr * dw # dim is (n_features, 1)
            self.b -= self.lr * db


    def predict(self, X):
        return np.dot(X, self.W) + self.b