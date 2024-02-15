import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class LogisticRegression:
    '''
    A Logistic Regression classifier class.

    Parameters
    ----------
    lr: float
        Learning rate. A hyperparameter to control the size of parameter updates
    n_iters: int
        Number of training iterations. 

    Methods
    -------
    fit(X, y)
        Fit a Logistic Regression classifier to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        '''
        Fit a Logistic Regression classifier to array of features X and target values y 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)
        y: 1-d np.array of target values, shape (n_samples,). Target values should be 0 or 1.

        Returns
        -------
        Fitted Logistic Regression classifer ready for runnning .predict(X)
         
        '''
        # Initiate weights and bias
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features) # shape (n_features,)
        self.b = 0

        for _ in range(self.n_iters):
            # Calculate y_hat
            Logistic_preds = np.dot(X, self.W) + self.b # shape (n_samples,)
            y_hat = sigmoid(Logistic_preds) # shape (n_samples,)

            # Calculate partial differentials
            dw = (2/n_samples) * np.dot(X.T, (y_hat - y))
            db = (2/n_samples) * np.sum(y_hat - y)

            # Update weights
            self.W -= self.lr * dw
            self.b -= self.lr * db


    def predict(self, X, threshold=0.5):
        '''
        Predict label values given array of features X.

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)

        Returns
        -------
        Predictions: 1-d np.array of predicted labels.

        ''' 
        Logistic_preds = np.dot(X, self.W) + self.b # shape (n_samples,)
        y_probs = sigmoid(Logistic_preds) # shape (n_samples,)
        y_preds = (y_probs > threshold).astype(int)
        return y_preds, y_probs
