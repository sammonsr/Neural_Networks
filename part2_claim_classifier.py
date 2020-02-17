import numpy as np
import pickle

import torch
from torch import nn


class ClaimClassifier():

    def __init__(self, num_layers=3, neurons_per_layer=3):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        pass

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE

        return  # YOUR CLEAN DATA AS A NUMPY ARRAY

    def _xavier_init(self, size, gain=1.0):
        """
        Xavier initialization of network weights.
        """
        low = -gain * np.sqrt(6.0 / np.sum(size))
        high = gain * np.sqrt(6.0 / np.sum(size))
        return np.random.uniform(low=low, high=high, size=size)

    def _init_weights(self, layer):
        rows, cols = layer.weight.data.shape
        random_weight = self._xavier_init((rows, cols))
        layer.weight.data = torch.as_tensor(random_weight)

    def load_data(self, filename, has_header=True):
        skip_rows = 1 if has_header else 0
        data = np.loadtxt(filename, delimiter=',', skiprows=skip_rows)
        # Split into x and y
        X, y = np.split(data, [-1], axis=1)
        return X, y

    def remove_colmn(self, X, index):
        return np.delete(X, [index], axis=1)

    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # Setup preprocessing
        self.col_mins = np.amin(X_raw, axis=0)
        self.col_maxs = np.amax(X_raw, axis=0)

        num_inputs = len(X_raw[0])
        num_outputs = y_raw.shape[1]

        assert num_inputs == 9
        assert num_outputs == 1
        layers = self.create_net_layers(num_inputs, num_outputs)

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE
        pass

    def create_net_layers(self, num_inputs, num_outputs):
        layers = []

        # Input layer
        layer = nn.Linear(num_inputs, self.neurons_per_layer)
        self._init_weights(layer)
        layers.append(("lin1", layer))

        # Hidden layers
        for i in range(2, self.num_layers):
            layer = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
            self._init_weights(layer)
            layer_name = "lin{}".format(i)
            layers.append((layer_name, layer))

        # Output layer
        layer = nn.Linear(self.neurons_per_layer, num_outputs)
        self._init_weights(layer)
        layer_name = "lin{}".format(self.num_layers)
        layers.append((layer_name, layer))

        print(layers)

        return layers

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE

        return  # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model


# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters


if __name__ == "__main__":
    classifier = ClaimClassifier()

    X_raw, y_raw = classifier.load_data('part2_training_data.csv')
    # Remove 'claim_amount' column
    claim_amount_col_index = 9
    X_raw = classifier.remove_colmn(X_raw, claim_amount_col_index)

    # Train network
    model = classifier.fit(X_raw, y_raw)
