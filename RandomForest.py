import numpy as np
from DecisionTree import DecisionTree
from collections import Counter

class RandomForest:
    '''
    A Random Forest classifier class.

    Parameters
    ----------
    n_trees: int
        Number of Decision Tree classifiers to generate in the Random Forest.
    min_samples_split: int
        Minimum number of samples that must be within a leaf node, otherwise a split will not take place.
    max_depth: int
        Maximum depth a tree is allowed to reach.
    n_features: int
        Number of features used to build the Decision Tree.
    subsample: float between 0.0 and 1.0.

    Methods
    -------
    fit(X, y)
        Fit a Decision Tree to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    def __init__(self, n_trees=10, max_depth=5, min_sample_split=2, n_features=None, subsample=1.0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_sample_split
        self.n_features = n_features
        self.subsample = subsample
        self.random_generator = np.random.default_rng(seed=23)
        self.trees = []

    def fit(self, X, y):
        '''
        Fit a Random Forest class to array of features X and target values y 

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
        y: 1-d array of target values, shape (n_samples,)

        Returns
        -------
        Fitted Random Forest class ready for runnning .predict(X)
         
        '''
        self.trees = []
        for _ in range(self.n_trees):
                
            # Sample the data
            X_sample, y_sample = self._sample_data(X, y)

            # Make tree and fit to sample
            tree = DecisionTree(self.min_samples_split, self.max_depth, self.n_features)
            tree.fit(X_sample, y_sample)

            # Add to list of trees
            self.trees.append(tree)

    def predict(self, X):
        '''
        Predict label values given array of features X.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)

        Returns
        -------
        Predictions: 1-d numpy array of predicted labels.

        '''     
        # Make prediction for each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        print(predictions.shape)
        predictions = np.swapaxes(predictions, 0, 1) # Switch from lists of predictions per tree to lists of predictions per sample.
        print(predictions.shape)
        # Take majority vote of the trees
        return np.array([self._most_common_label(preds) for preds in predictions])

    def _most_common_label(self, y):
        '''
        Returns most common label in array of labels, y.
        '''    
        counter = Counter(y)
        label = counter.most_common(1)[0][0] # most_common outputs an ordered list of tuples [(1st_most_common_label, count), (2nd_most_common_labe, count)]
        return label
    
    def _sample_data(self, X, y):
        '''
        Returns a random subsample of training samples, X and target data, y.
        '''
        n_samples = X.shape[0]
        sample_idxs = self.random_generator.choice(n_samples, int(n_samples * self.subsample), replace=True)
        return X[sample_idxs, :], y[sample_idxs]
