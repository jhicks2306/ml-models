import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Calculate the covariance of X
        cov = np.cov(X.T)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix.
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors by descending eigenvalue
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[:, idxs]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project data onto the n_components.
        X = X - self.mean
        return np.dot(X, self.components)