import numpy as np

class LinearRegression:
    '''
    A Linear Regression classifier class.

    Parameters
    ----------
    lr: float
        Learning rate. A hyperparameter to control the size of parameter updates
    n_iterations: int
        Number of training iterations. 

    Methods
    -------
    fit(X, y)
        Fit a Linear Regression classifier to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    
    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        '''
        Fit a Linear Regression classifier to array of features X and target values y 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)
        y: 1-d np.array of target values, shape (n_samples,). Target values should be 0 or 1.

        Returns
        -------
        Fitted Linear Regression classifer ready for runnning .predict(X)
         
        '''
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
        '''
        Predict label values given array of features X.

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)

        Returns
        -------
        Predictions: 1-d np.array of predicted labels.

        ''' 
        return np.dot(X, self.W) + self.b
