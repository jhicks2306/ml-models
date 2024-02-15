import numpy as np

def step_function(x):
    return np.where(x > 0, 1, 0)


class Perceptron:
    '''
    A Perceptron classifier class.

    Parameters
    ----------
    learning_rate: float
        Hyperparameter to control the size of parameter updates
    n_iters: int
        Number of training iterations. 

    Methods
    -------
    fit(X, y)
        Fit a Perceptron classifier to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.activation = step_function

    def fit(self, X, y):
        '''
        Fit a Perceptron classifier to array of features X and target values y 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)
        y: 1-d np.array of target values, shape (n_samples,). Target values should be 0 or 1.

        Returns
        -------
        Fitted Perceptron classifer ready for runnning .predict(X)
         
        '''
        n_samples, n_features = X.shape

        # Initiate parameters
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0

        # Learn weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)

                # Update weights and bias term using perceptron update rule.
                self.weights += self.learning_rate * (y[idx] - y_pred) * x_i
                self.bias += self.learning_rate * (y[idx] - y_pred)

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
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation(linear_output)
        return y_pred
