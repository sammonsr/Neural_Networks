from collections import OrderedDict

import numpy as np
import pickle

import torch
from torch import nn
from sklearn import metrics


class ClaimClassifier:

    def __init__(self, num_layers, neurons_per_layer, num_epochs, learning_rate, batch_size):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Force use of cpu for now
        self.device = "cpu"

        self.network = None

        # Hyperparameters
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neurons_per_layer = neurons_per_layer

        # Preprocessing
        self.col_mins = []
        self.col_maxs = []

    def __str__(self):
        return "ClaimClassifier(num_layers = {}, num_epochs={}, lr={}, batch_size={}, neurons_per_layer={})".format(
            self.num_layers, self.num_epochs, self.learning_rate, self.batch_size, self.neurons_per_layer)

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

        # binary cross entropy loss
        loss_fun = torch.nn.BCELoss().to(self.device)

        # Adam optimizer
        opt = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Preprocess X
        X_clean = self._preprocessor(X_raw)

        # Convert X and Y to tensor floats
        X_clean = torch.as_tensor(X_clean).float().to(self.device)
        y_raw = torch.as_tensor(y_raw).float().to(self.device)

        # Shuffle data for batching
        permutation = torch.randperm(X_clean.size()[0])

        # keeping the network in training mode
        self.network.train()

        # training
        for epoch in range(self.num_epochs):
            print('at epoch ', epoch)
            for i in range(0, X_clean.size()[0], self.batch_size):
                # TODO: make sure not missing anything
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = X_clean[indices], y_raw[indices].view(self.batch_size)

                # Setup batches to use GPU if available
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                opt.zero_grad()
                y_pred_val = self.network(batch_x.float()).view(self.batch_size)
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

        return nn.Sequential(OrderedDict(layers)).to(self.device)

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

        # Setup data to use GPU if available
        X_clean = X_clean.to(self.device)

        self.network.eval()

        return self.network(X_clean).cpu().detach().numpy().astype('int')

    def evaluate_architecture(self, X_test, y_test, verbose=True):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        predictions = self.predict(X_test)

        if verbose:
            confusion_matrix = metrics.confusion_matrix(y_test, predictions)
            norm_confusion_matrix = metrics.confusion_matrix(y_test, predictions, normalize='true')
            report = metrics.classification_report(y_test, predictions)

            print("Confusion matrix:")
            print(confusion_matrix)
            print("\nNormalized Confusion matrix:")
            print(norm_confusion_matrix)
            print(report)

        return metrics.f1_score(y_test, predictions, average='macro')

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
def ClaimClassifierHyperParameterSearch(X_train, y_train, X_test, y_test):
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class.

    The function should return your optimised hyper-parameters.
    """

    num_layer_space = list(range(2, 6))
    neurons_per_layer_space = list(range(1, 6))
    num_epochs_space = list(range(1, 6))
    lr_space = [10 ** - i for i in range(1, 5)]
    batch_size_space = [2 ** i for i in range(3, 8)]

    best_model = None
    best_score = -1

    for num_layers in num_layer_space:
        for neurons_per_layer in neurons_per_layer_space:
            for num_epochs in num_epochs_space:
                for lr in lr_space:
                    for batch_size in batch_size_space:
                        print("\n\n\n==========================================================")

                        model = ClaimClassifier(num_layers, neurons_per_layer, num_epochs, lr, batch_size)

                        print("Evaluating", model)

                        model.fit(X_train, y_train)

                        score = model.evaluate_architecture(X_test, y_test, verbose=False)

                        print(model, "score =", score)
                        print("==========================================================")

                        if score > best_score:
                            best_model = model
                            best_score = score

    return str(best_model)


def get_train_test_split(X_raw, y_raw):
    total_rows = len(X_raw)
    split_point = int(0.8 * total_rows)
    train_X_raw = np.array(X_raw[: split_point])
    train_y_raw = np.array(y_raw[: split_point])
    test_X_raw = np.array(X_raw[split_point:])
    test_y_raw = np.array(y_raw[split_point:])

    return train_X_raw, train_y_raw, test_X_raw, test_y_raw


def remove_column(X, index):
    return np.delete(X, [index], axis=1)


def load_data(has_header=True, shuffle=False):
    filename = 'part2_training_data.csv'
    skip_rows = 1 if has_header else 0
    data = np.loadtxt(filename, delimiter=',', skiprows=skip_rows)
    if shuffle:
        np.random.shuffle(data)
    # Split into x and y
    X, y = np.split(data, [-1], axis=1)

    # Remove 'claim_amount' column
    claim_amount_col_index = 9
    X = remove_column(X, claim_amount_col_index)
    return X, y


if __name__ == "__main__":
    # Load data
    X_raw, y_raw = load_data(shuffle=True)
    train_X_raw, train_y_raw, test_X_raw, test_y_raw = get_train_test_split(X_raw, y_raw)

    # best_hyper_params = ClaimClassifierHyperParameterSearch(train_X_raw, train_y_raw, test_X_raw, test_y_raw)
    # print("Best params: \n", best_hyper_params)

    classifier = ClaimClassifier(num_layers=3, neurons_per_layer=3, num_epochs=5, learning_rate=0.1, batch_size=8)

    # Train network
    classifier.fit(train_X_raw, train_y_raw)

    # Evaluate
    classifier.evaluate_architecture(test_X_raw, test_y_raw)
