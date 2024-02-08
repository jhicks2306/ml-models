import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        # Initiate weights and bias
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features) # shape (n_features,)
        self.b = 0

        for _ in range(self.n_iters):
            # Calculate y_hat
            linear_preds = np.dot(X, self.W) + self.b # shape (n_samples,)
            y_hat = sigmoid(linear_preds) # shape (n_samples,)

            # Calculate partial differentials
            dw = (2/n_samples) * np.dot(X.T, (y_hat - y))
            db = (2/n_samples) * np.sum(y_hat - y)

            # Update weights
            self.W -= self.lr * dw
            self.b -= self.lr * db


    def predict(self, X, threshold=0.5):
        linear_preds = np.dot(X, self.W) + self.b # shape (n_samples,)
        y_probs = sigmoid(linear_preds) # shape (n_samples,)
        y_preds = (y_probs > threshold).astype(int)
        return y_preds, y_probs
