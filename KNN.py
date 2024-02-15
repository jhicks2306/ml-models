import numpy as np

def euclidian_distance(x1, x2):
    return np.sum(np.sqrt((x1-x2)**2))
    
class KNN():
    '''
    A K Nearest Neighbours classifier class.

    Parameters
    ----------
    k: int
        Number of neighbouring points to consider when classifiying a sample.

    Methods
    -------
    fit(X, y)
        Fit K Nearest Neighbours class to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, Y_train):
        '''
        Fit a K Nearest Neighbours classifier to array of features X and target values y 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)
        y: 1-d np.array of target values, shape (n_samples,). Target values should be -1 or 1.

        Returns
        -------
        Classifer ready for runnning .predict(X)
         
        '''
        # Take training inputs X and labels Y.
        self.X_train = X_train
        self.Y_train = Y_train
    
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
        preds = []
        for x in X:
            # Compute distances with other points in the training set.
            distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

            # Find nearest k points and their classes.
            sorted_idxs = np.argsort(distances)
            k_nearest_idxs = sorted_idxs[:self.k]
            k_nearest_classes = self.Y_train[k_nearest_idxs]
            
            # Take majority vote:
            unique_vals, counts = np.unique(k_nearest_classes, return_counts=True)
            max_count_idx = np.argmax(counts)
            most_common_class = unique_vals[max_count_idx]

            # Add most common class to predictions.
            preds.append(most_common_class)

        return np.array(preds)