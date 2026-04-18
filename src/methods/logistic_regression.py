import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
   
        N, D = training_data.shape
        C = get_n_classes(training_labels)

        y_onehot = label_to_onehot(training_labels, C)

        self.weights = np.zeros((D, C))

        for i in range(self.max_iters):
            scores = training_data @ self.weights
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            gradient = (training_data.T @ (probs - y_onehot)) / N
            self.weights -= self.lr * gradient

        scores = training_data @ self.weights
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        pred_onehot = np.zeros_like(probs)
        pred_onehot[np.arange(N), np.argmax(probs, axis=1)] = 1
        pred_labels = onehot_to_label(pred_onehot)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        N = test_data.shape[0]
        scores = test_data @ self.weights
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        pred_onehot = np.zeros_like(probs)
        pred_onehot[np.arange(N), np.argmax(probs, axis=1)] = 1
        pred_labels = onehot_to_label(pred_onehot)

        return pred_labels
