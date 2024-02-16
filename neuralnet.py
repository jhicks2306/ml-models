import numpy as np

def sigmoid(z):
    # Activation function for the neural net.
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    # Converts vector y with n_labels into matrix of shape (len(y), num_labels)
    arr = np.zeros((y.shape[0], num_labels))
    for idx, val in enumerate(y):
        arr[idx, val] = 1
    return arr

class NeuralNetMLP:

    def __init__(self, n_features, n_hidden, n_classes, random_seed=123):
        # super().__init__() <-- is this needed?
        self.n_classes = n_classes

        # Random number generator for initialization
        rng = np.random.default_rng(seed=random_seed)

       
        # Hidden layer - weights and bias.
        self.w_h = rng.normal(
            loc=0.0, scale=0.1, size=(n_hidden, n_features)
        )
        self.b_h = np.zeros(n_hidden)

        # Output layer - weights and bias.
        self.w_out = rng.normal(
            loc=0.0, scale=0.1, size=(n_classes, n_hidden)
        )
        self.b_out = np.zeros(n_classes)       

    def forward(self,x):
        # Hidden layer
        # Input dimensions
        # x     shape (n_examples, n_features)
        # w_h   shape (n_hidden, n_features)
        # b_h   shape (n_hidden,)
    
        # Output dimensions
        # z_h   shape (n_examples, n_hidden)
        # a_h   shape (n_examples, n_hidden)
        z_h = np.dot(x, self.w_h.T) + self.b_h
        a_h = sigmoid(z_h)

        # Output layer
        # Input dimensions
        # a_h   shape (n_examples, n_hidden)
        # w_out shape (n_classes, n_hidden)
        # b_out shape (n_classes,)
    
        # Output dimensions
        # z_h   shape (n_examples, n_classes)
        # a_h   shape (n_examples, n_classes)
        z_out= np.dot(a_h, self.w_out.T) + self.b_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        # One hot encoding

        # PART 1 dLoss/dOutWeights
        # Shape out (n_examples, n_classes)
        y_onehot = int_to_onehot(y, self.n_classes)

        # Shape out (n_examples, n_classes)
        dLoss_da_out = 2. * (a_out - y_onehot) / y.shape[0]

        # Shape out (n_examples, n_classes)
        da_out__dz_out = a_out * (1. - a_out)

        # Shape out (n_examples, n_classes)
        dLoss_dz_out = dLoss_da_out * da_out__dz_out

        # Shape out (n_examples, n_hidden)
        dz_out__dw_out = a_h

        # Shape in (n_examples, n_classes).T dot (n_examples, n_hidden)
        # Shape out (n_classes, n_hidden)
        dLoss_dw_out = np.dot(dLoss_dz_out.T, dz_out__dw_out)
        dLoss_db_out = np.sum(dLoss_dz_out, axis=0)

        # PART 1 dLoss/dHiddenWeights
        # Shape out (n_classes, n_hidden)
        dz_out__a_h = self.w_out

        # Shape out (n_examples, n_hidden)
        dLoss_da_h = np.dot(dLoss_dz_out, dz_out__a_h)

        # Shape out (n_examples, n_hidden)
        da_h__dz_h = a_h * (1. - a_h) # sigmoid derivative.

        # Shape out (n_examples, n_features)
        dz_h__dw_h = x

        # Shape in (n_examples, n_hidden).T dot (n_examples, n_features)
        # Shape out (n_hidden, n_features)
        dLoss_dw_h = np.dot((dLoss_da_h * da_h__dz_h).T, dz_h__dw_h )
        dLoss_db_h = np.sum((dLoss_da_h * da_h__dz_h), axis=0)

        return (dLoss_dw_out, dLoss_db_out, dLoss_dw_h, dLoss_db_h)




