import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.left = left
        self.right = right
        self.value = None
    
    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree:
    '''
    '''
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Use minimum of available features and optionally specified by user.
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        '''
        Checks to see if any stopping criteria are met.
        Otherwise, continues building the decision tree.
        '''
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check the stopping criteria
        if n_samples<=self.min_samples_split or depth>=self.max_depth or n_labels==1:
            leaf_value = self._most_common_label(y) # A leaf's value is the most common label.
            return Node(value=leaf_value)
        
        feat_idxs = np.random.Generator.choice(a=n_features, size=self.n_features, replace=True)

        # Find the best split
        best_feature_idx, best_split_thr = self._best_split(X, y, feat_idxs)

        # Create the child (recursively)
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_split_thr)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature_idx, best_split_thr, left, right)



    def _best_split(self, X, y, feat_idxs):
        '''
        Calculates the best split for each feature provided based on the information gain.
        Then returns the best split.
        '''
        best_gain = -1
        split_idx, split_threshold = None, None

        for idx in feat_idxs:
            X_col = X[:,idx]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                gain = self._information_gain(y, X_col, thr)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_col, thr):
        '''Calculates the information gain between the parent and child Nodes.'''

        # Parent entropy
        parent_entropy = self._entropy

        # Create children
        left_idxs, right_idxs = self._split(X_col, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            # No information gain.
            return 0

        # Calculate weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy =  (n_l/n) * e_l + (n_r/n) * e_r

        # Calculate information gain.
        information_gain = parent_entropy - child_entropy
        return information_gain
         

    def _split(self, X_col, thr):
        '''
        Returns left and right indexes having split X_col using threshold (thr).
        '''
        left_idx = np.argwhere(X_col<=thr).flatten()
        right_idx = np.argwhere(X_col>thr).flatten()
        return left_idx, right_idx


    def _entropy(self, y):
        '''
        Calculates the entropy within within an array of labels.
        ''' 
        hist = np.bincount(y) # counts occurances of int class labels
        ps = hist / len(y) # an array of probabilities for each value in y.
        return - np.sum([p * np.log(p) for p in ps if p>0]) # the entropy.


    def _most_common_label(self, y):
        '''
        Returns most common label in array of labels, y.
        '''    
        counter = Counter(y)
        label = counter.most_common(1)[0][0] # most_common outputs an ordered list of tuples [(1st_most_common_label, count), (2nd_most_common_labe, count)]
        return label


    def predict(self, X):
        '''
        Predict label values given array X.
        '''
        np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        '''
        Traverses the tree nodes until a leaf node is found. Then it returns leaf's value.
        '''
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
