import numpy as np

class SVM:
    '''
    An SVM classifier class.

    Methods
    -------
    fit(X, y)
        Fit an SVM classifier to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        '''
        Fit a SVM classifier to array of features X and target values y 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)
        y: 1-d np.array of target values, shape (n_samples,). Target values should be -1 or 1.

        Returns
        -------
        Fitted SVM classifer ready for runnning .predict(X)
         
        '''
        n_samples, n_features = X.shape

        # Initiate weights and bias.
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = 0

        # Make sure y labels are 1 or -1.
        y_ = np.where(y <= 0, -1, 1)

        # Apply SVM update rules to weight and bias terms.
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

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
        approx = np.dot(X, self.weights) - self.bias
        y_pred = np.sign(approx)
        return y_pred

                                               