from abc import abstractmethod
import numpy as np

class InputLayer:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = inputs


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.inputs = None
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        self.dinputs = None
        self.dbiases = None
        self.dweights = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def __init__(self):
        self.inputs = None
        self.output = None

        self.dinputs = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    @staticmethod
    def predictions(outputs):
        return outputs


# Will Fill In Later
class ActivationSigmoid:
    @staticmethod
    def predictions(outputs):
        return (outputs > 0.5) * 1


class ActivationSoftmax:
    def __init__(self):
        self.dinputs = None
        self.output = None

    def forward(self, inputs):
        # un-normalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array. Note dvalues is 2D
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Softmax jacobian array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient and add it to the array of sample gradients.
            # single_dvalues is a vector
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    @staticmethod
    def predictions(outputs):
        return np.argmax(outputs, axis=1)


class Loss:
    def calculate(self, output, expectedOutput):
        sample_losses = self.forward(output, expectedOutput)
        data_loss = np.mean(sample_losses)
        return data_loss

    @abstractmethod
    def forward(self, output, expectedOutput):
        pass


class LossCategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.dinputs = None

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        # Clip data to prevent division by 0 and
        # dragging the mean towards a certain value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Sparse (Categorical)
        if len(y_true.shape) == 1:
            # [[0,1,2], [y_true]]
            correct_conf = y_pred_clipped[range(samples), y_true]
        # One-hot
        elif len(y_true.shape) == 2:
            correct_conf = np.sum(y_pred_clipped * y_true, axis=1)
            print(f'conf: {correct_conf}')
            print(f'true: {y_true}')

        else:
            raise ValueError("Wrong dimensions for y_true")

        neg_log_likelihood = -np.log(correct_conf)
        return neg_log_likelihood

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # Number of labels in every sample
        labels = len(dvalues[0])

        # if labels are spare, turn them into a one hot vector
        if len(y_true.shape) == 1:
            # eye creates identity matrix
            y_true = np.eye(labels)[y_true]

        # Calculate Gradient
        self.dinputs = -y_true / dvalues

        # Normalize Gradient
        self.dinputs = self.dinputs / samples


# Combined softmax and cross entropy for faster backward step
class ActivationSoftmax_Loss_CategoricalCrossEntropy:
    def __init__(self):
        self.dinputs = None

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # Calculate Gradients (Taking advantage of one-hot encoded y-true)
        self.dinputs[range(samples), y_true] -= 1

        # Normalize
        self.dinputs /= samples


class OptimizerSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    # Given layer of an object, this will adjust the weights and biases
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


class Accuracy:

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        return accuracy

    @abstractmethod
    def compare(self, predictions, y):
        pass


class AccuracyCategorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    # Does not do anything
    def init(self, y):
        pass

    # Return a list of true and false values (1s and 0s)
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y