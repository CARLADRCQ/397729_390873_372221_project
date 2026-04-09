import numpy as np


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.w = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        X = training_data
        y = training_labels

        # Add bias term (column of 1s)
        N = X.shape[0]
        X = np.c_[np.ones(N), X]

        # Closed-form solution
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)   # safer than inv
        Xt_y = X.T @ y

        self.w = XtX_inv @ Xt_y

        # Return predictions on training data
        pred_labels = X @ self.w
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        X = test_data

        # Add bias term
        N = X.shape[0]
        X = np.c_[np.ones(N), X]

        pred_labels = X @ self.w
        return pred_labels
