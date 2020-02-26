from random import random
from random import shuffle as randshuffle
import math
import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = {}

    def __str__(self):
        return "SigmoidLayer"

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, x):
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current['x'] = x.copy()

        return self.sigmoid(x)
        #######################################################################
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        dl_da = grad_z
        g_prime_z = self.sigmoid_prime(self._cache_current['x'])
        # Compute Hadamard product (element-wise)
        dl_dz = np.multiply(dl_da, g_prime_z)

        return dl_dz

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = {}

    def __str__(self):
        return "ReluLayer"

    def relu_prime(self, x):
        result = x.copy()

        relu_prime_func = lambda v: 1 if v > 0 else 0
        relu_prime_func = np.vectorize(relu_prime_func)

        return relu_prime_func(result)

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        result = np.maximum(x, 0)
        self._cache_current['x'] = x
        return result

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        dl_da = grad_z
        g_prime_z = self.relu_prime(self._cache_current['x'])
        # Compute Hadamard product (element-wise)
        dl_dz = np.multiply(dl_da, g_prime_z)

        return dl_dz

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out, layer_num=-1):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._W = xavier_init((n_in, n_out))
        self._b = xavier_init(n_out)
        self.layer_num = layer_num

        # Dictionary of cached values
        self._cache_current = {'x': None}
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __str__(self):
        return "LinearLayer #{} (In:{}, Out:{})".format(self.layer_num, self.n_in, self.n_out)

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self._cache_current['x'] = x

        # Tile bias to the batch size
        B = np.tile(self._b, (x.shape[0], 1))

        XW_b = np.matmul(x, self._W) + B

        return XW_b

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        dl_dz = grad_z
        ones_row_vector = np.ones(dl_dz.shape[0])

        dl_dw = np.matmul(self._cache_current['x'].T, dl_dz)
        dl_db = np.matmul(ones_row_vector, dl_dz)

        # Set gradients for update_params usage
        self._grad_W_current = dl_dw
        self._grad_b_current = dl_db

        dl_dx = np.matmul(dl_dz, self._W.T)

        assert dl_dx.shape == self._cache_current['x'].shape
        assert dl_dw.shape == self._W.shape
        assert dl_db.shape == self._b.shape

        return dl_dx
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Save old values of b and w for assertion checks
        old_w_shape = self._W.shape
        old_b_shape = self._b.shape

        # Perform gradient descent
        self._W = self._W - (learning_rate * self._grad_W_current)
        self._b = self._b - (learning_rate * self._grad_b_current)

        # Check that W and b haven't changed their shape
        assert self._W.shape == old_w_shape
        assert self._b.shape == old_b_shape
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._layers = []

        num_layers = len(self.neurons)

        last_layer_num_out = input_dim
        for i in range(num_layers):
            num_in = last_layer_num_out
            num_out = self.neurons[i]

            last_layer_num_out = num_out

            self._layers.append(LinearLayer(num_in, num_out, layer_num=i))
            if activations[i] != "identity":
                self._layers.append(self.str_to_activation_layer(activations[i]))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __str__(self):
        ret = "\nMultiLayerNetwork \n"
        for layer in self._layers:
            ret += "    ↓\n " + str(layer) + "\n"

        return ret

    def str_to_activation_layer(self, layer_name):
        if layer_name == "relu":
            return ReluLayer()
        if layer_name == "sigmoid":
            return SigmoidLayer()
        raise Exception('Unknown layer: ' + layer_name)

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        prev_out = x
        for layer in self._layers:
            prev_out = layer.forward(prev_out)

        return prev_out
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        prev_grad_z = grad_z
        for layer in reversed(self._layers):
            prev_grad_z = layer.backward(prev_grad_z)

        return prev_grad_z

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for layer in self._layers:
            layer.update_params(learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
            self,
            network,
            batch_size,
            nb_epoch,
            learning_rate,
            loss_fun,
            shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = self.name_to_loss_layer(loss_fun)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def name_to_loss_layer(self, name):
        if name == "mse":
            return MSELossLayer()
        if name == "cross_entropy":
            return CrossEntropyLossLayer()
        raise Exception("Unknown loss layer ", name)

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        shuffled_indicies = list(range(len(input_dataset)))
        randshuffle(shuffled_indicies)

        input_shuffled = []
        target_shuffled = []

        for i in shuffled_indicies:
            input_shuffled.append(input_dataset[i])
            target_shuffled.append(target_dataset[i])

        return np.array(input_shuffled), np.array(target_shuffled)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.
        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        assert len(input_dataset) == len(target_dataset)

        for i in range(self.nb_epoch):
            # Shuffle input data
            if self.shuffle_flag:
                input_dataset, target_dataset = self.shuffle(input_dataset, target_dataset)

            # Split into batches
            batches = []
            for j in range(0, len(input_dataset), self.batch_size):
                input_batch = input_dataset[j: min(len(input_dataset) - 1, j + self.batch_size)]
                target_batch = target_dataset[j: min(len(target_dataset) - 1, j + self.batch_size)]
                batches.append((input_batch, target_batch))

            for (input, target) in batches:
                # Forward pass for batch
                predictions = self.network.forward(input)
                # Compute loss
                self.eval_loss(input_dataset, target_dataset)
                # Perform backward pass
                self.network.backward(self._loss_layer.backward())
                # Perform one step of gradient descent
                self.network.update_params(self.learning_rate)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        predictions = self.network.forward(input_dataset)
        loss = self._loss_layer.forward(predictions, target_dataset)
        return loss

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        self.data = data

        self.col_mins = np.amin(data, axis=0)
        self.col_maxs = np.amax(data, axis=0)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Scale between 0 and 1
        return (data - self.col_mins) / (self.col_maxs - self.col_mins)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        return data * (self.col_maxs - self.col_mins) + self.col_mins

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "sigmoid"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=128,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print(preds)
    print(targets)

    print(net)
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
