import numpy as np

class NaiveBayes:
    '''
    A Naive Bayes classifier class.

    Methods
    -------
    fit(X, y)
        Fit a Decision Tree to array of features X and target values y.
    predict(X)
        Predict target values on a set of features X.

    '''
    def fit(self, X, y):
        '''
        Fit a Naive Bayes classifier to array of features X and target values y 

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)
        y: 1-d np.array of target values, shape (n_samples,)

        Returns
        -------
        Fitted Naive Bayers classifer ready for runnning .predict(X)
         
        '''
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initiate arrays to be populated
        self._means = np.zeros((n_classes, n_features), dtype=np.float64)
        self._vars = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Calculate the mean, var and prior probability (frequency) for each class.
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._means[idx, :] = X_c[idx, :].mean(axis=0)
            self._vars[idx, :] = X_c[idx, :].var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

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
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        '''
        Predict the class label for a single sample

        Parameters
        ----------
        x: np.array of shape (1, n_features)

        Returns
        -------
        Prediction: int
        '''
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            prior_class_prob = self._priors[idx]
            posterior_feature_probs = np.log(self._pdf(idx, x)) # an array of posterier feature probs
            posterior_class_prob = np.sum(posterior_feature_probs) + prior_class_prob
            posteriors.append(posterior_class_prob)
        
        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        '''
        Uses a Gaussian distribution (probability density function)
        to return posterior probabilities
        '''
        mean = self._means[class_idx]
        var = self._vars[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
