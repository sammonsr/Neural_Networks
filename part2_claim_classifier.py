from collections import OrderedDict

import numpy as np
import pickle

import torch
from torch import nn
from sklearn import metrics


class ClaimClassifier:
    # Hyperparameters
    EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8

    def __init__(self, num_layers=3, neurons_per_layer=3):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.col_mins = []
        self.col_maxs = []
        self.network = None

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

        return (X_raw - self.col_mins) / (self.col_maxs - self.col_mins)

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
        layer.weight.data = torch.as_tensor(random_weight).float()

    def load_data(self, filename, has_header=True, shuffle=False):
        skip_rows = 1 if has_header else 0
        data = np.loadtxt(filename, delimiter=',', skiprows=skip_rows)
        if shuffle:
            np.random.shuffle(data)
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
        self.network = self.create_network(num_inputs, num_outputs)

        # Train neural network

        # binary cross entropy loss
        loss_fun = torch.nn.BCELoss()

        # Adam optimizer
        opt = torch.optim.Adam(self.network.parameters(), lr=self.LEARNING_RATE)

        # Preprocess X
        X_clean = self._preprocessor(X_raw)

        # Convert X and Y to tensor floats
        X_clean = torch.as_tensor(X_clean).float()
        y_raw = torch.as_tensor(y_raw).float()

        # Shuffle data for batching
        permutation = torch.randperm(X_clean.size()[0])

        # training
        for epoch in range(self.EPOCHS):
            print('at epoch ', epoch)
            for i in range(0, X_clean.size()[0], self.BATCH_SIZE):
                indices = permutation[i:i + self.BATCH_SIZE]
                batch_x, batch_y = X_clean[indices], y_raw[indices].view(self.BATCH_SIZE)

                opt.zero_grad()
                y_pred_val = self.network(batch_x.float()).view(self.BATCH_SIZE)
                loss = loss_fun(y_pred_val, batch_y)
                loss.backward()
                opt.step()

    def create_network(self, num_inputs, num_outputs):
        layers = []

        # Input layer
        layer = nn.Linear(num_inputs, self.neurons_per_layer)
        self._init_weights(layer)
        layers.append(("lin1", layer))
        layers.append(("relu1", nn.ReLU()))

        # Hidden layers
        for i in range(2, self.num_layers):
            layer = nn.Linear(self.neurons_per_layer, self.neurons_per_layer)
            self._init_weights(layer)
            layer_name = "lin{}".format(i)
            layers.append((layer_name, layer))
            layers.append(("relu{}".format(i), nn.ReLU()))

        # Output layer
        layer = nn.Linear(self.neurons_per_layer, num_outputs)
        self._init_weights(layer)
        layer_name = "lin{}".format(self.num_layers)
        layers.append((layer_name, layer))
        layers.append(("sig{}".format(self.num_layers), nn.Sigmoid()))

        print(layers)

        return nn.Sequential(OrderedDict(layers))

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

        X_clean = torch.as_tensor(self._preprocessor(X_raw)).float()

        self.network.eval()

        return self.network(X_clean).detach().numpy().astype('int')


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


def get_train_test_split(X_raw, y_raw):
    split_point = int(0.8 * total_rows)
    train_X_raw = np.array(X_raw[: split_point])
    train_y_raw = np.array(y_raw[: split_point])
    test_X_raw = np.array(X_raw[split_point:])
    test_y_raw = np.array(y_raw[split_point:])

    return train_X_raw, train_y_raw, test_X_raw, test_y_raw


if __name__ == "__main__":
    classifier = ClaimClassifier()

    X_raw, y_raw = classifier.load_data('part2_training_data.csv', shuffle=True)
    # Remove 'claim_amount' column
    claim_amount_col_index = 9
    X_raw = classifier.remove_colmn(X_raw, claim_amount_col_index)

    total_rows = len(X_raw)

    train_X_raw, train_y_raw, test_X_raw, test_y_raw = get_train_test_split(X_raw, y_raw)

    # Train network
    classifier.fit(X_raw, y_raw)

    # Evaluate
    predictions = classifier.predict(test_X_raw)
    confusion_matrix = metrics.confusion_matrix(test_y_raw, predictions)
    report = metrics.classification_report(test_y_raw, predictions)
    print(confusion_matrix)
    print(report)
