import numpy as np

class PCA:
    '''
    A Principal Component Analysis class.

    Parameters
    ----------
    n_components: int
        The number of components onto which the data will be projected 
    Methods
    -------
    fit(X)
        Fit a classifier to array of features X.
    transform(X)
        Project matrix X (n_samples, n_features) onto matrix (n_samples, n_components).

    '''    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        '''
        Fit a PCA classifier to array of features X. 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)

        Returns
        -------
        Fitted PCA classifer ready for runnning .transform(X)
         
        '''
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
        '''
        Project matrix X (n_samples, n_features) onto matrix (n_samples, n_components).

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)

        Returns
        -------
        Transformed matrix: Matrix (n_samples, n_components).

        ''' 
        # Project data onto the n_components.
        X = X - self.mean
        return np.dot(X, self.components)