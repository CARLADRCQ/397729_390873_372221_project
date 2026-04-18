import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        self.datatrain= training_data
        self.labelstrain= training_labels

        pred_labels= self.predict(training_data)
        
        return pred_labels

    def predict(self, test_data):
        points_nbr = test_data.shape[0]
        test_labels = np.zeros(points_nbr)

        for i in range(points_nbr):
            distances = np.linalg.norm(self.datatrain - test_data[i], axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.labelstrain[k_indices]

            
            if self.task_kind == "classification":
                counts = np.bincount(k_nearest_labels.astype(int))
                test_labels[i] = np.argmax(counts)
            
            elif self.task_kind == "regression":
                test_labels[i] = np.mean(k_nearest_labels)

        return test_labels