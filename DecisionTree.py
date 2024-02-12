import numpy as np

class Node:
    def __init__(self, feature=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = None
    
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Use minimum of available features and optionally specified by user.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_treee(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if n_samples<=self.min_samples_split or depth>=self.max_depth or n_labels==1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find the best split

        # Create the child (recursively)

    def _most_common_label(self, y):
        pass 
    
    def predict(self, ):
        pass