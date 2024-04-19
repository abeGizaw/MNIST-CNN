import numpy as np
from scipy import signal
class ConvolutionLayer:
    # Note this Convolution layer does not account for varying strides
    def __init__(self, *, depth, kernel_size, padding='valid', input_size):
        """
        :param depth: amount of filters
        :param kernel_size: filter size
        :param padding: padding mode
        :param input_size: (height, width, depth)
        """
        self.input_height, self.input_width, self.input_depth = input_size
        self.depth = depth

        self.kernel_height, self.kernel_width = kernel_size
        self.kernel_shape = (self.kernel_height, self.kernel_width, self.input_depth, self.depth)

        self.padding = padding
        if self.padding == 'same':
            # For 'same' padding, output dimensions should be the same as input dimensions
            self.output_height = self.input_height
            self.output_width = self.input_width
            self.dpadding = 'same'

        elif self.padding == 'full':
            # For 'full' padding (output is the largest possible size)
            self.output_height = self.input_height + self.kernel_height - 1
            self.output_width = self.input_width + self.kernel_width - 1
            self.dpadding = "valid"

        elif self.padding == 'valid':
            # For 'valid' padding (no padding is applied)
            self.output_height = self.input_height - self.kernel_height + 1
            self.output_width = self.input_width - self.kernel_width + 1
            self.dpadding = "full"
        else:
            raise ValueError("Padding mode must be 'same', 'full', or 'valid'.")

        # Weights and biases (filters and biases). Using tied bias
        self.weights = self.glorot_init(self.kernel_shape)
        self.biases = np.zeros(self.depth)
        self.dbiases = np.zeros(self.depth)
        self.dweights = np.zeros(self.kernel_shape)

        self.inputs = None
        self.output = None
        self.dinputs = None


    # Xavier/Glorot initialization for layers using sigmoid or tanh
    @staticmethod
    def glorot_init(shape):
        """
        :param shape: kernel shape -> (kernel_height, kernel_width, input_depth, number_of_filters)
        :return: fill the matrix using glorot
        """

        fan_in = shape[0] * shape[1] * shape[2]  # product of kernel dimensions and input depth
        fan_out = shape[0] * shape[1] * shape[3]  # product of kernel dimensions and number of filters

        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

    def forward(self, inputs):
        """
        :param inputs: image input. Dimensions -> {batch_size, image height, image width, image depth}
        :return: set the output
        """
        self.inputs = inputs
        batch_size = inputs.shape[0]
        output_shape = (batch_size, self.output_height, self.output_width, self.depth)
        self.output = np.zeros(output_shape)

        depth_size = self.kernel_shape[-1]
        # For each image in the batch
        for image in range(batch_size):
            # For each filter
            for filt in range(depth_size):
                # For each sub image in out image/filter
                for dep in range(self.input_depth):
                    # Cross-correlation
                    self.output[image, :, :, filt] = signal.correlate2d(
                        inputs[image, :, :, dep],
                        self.weights[:, :, dep, filt],
                        mode=self.padding
                    )
                # Add the bias for each filter
                self.output[image, :, :, filt] += self.biases[filt]



    def backward(self, dvalues):
        """
        :param dvalues: image derivative input. Dimensions -> {batch_size, image height, image width, image depth}
        :return: set the dweights, dinputs, and dbiases
        """
        self.dinputs = np.zeros(self.inputs.shape)

        batch_size = dvalues.shape[0]
        depth_size = self.kernel_shape[-1]
        # For each image in the batch
        for image in range(batch_size):
            # For each filter
            for filt in range(depth_size):
                # For each sub image in out image/filter
                for dep in range(self.input_depth):
                    # Cross-correlation

                    # Change of weights is a cross correlation of input image and corresponding dval
                    self.dweights[:, :, dep, filt] += signal.correlate2d(
                        self.inputs[image, :, :, dep],
                        dvalues[image, :, :, dep],
                        mode="valid"
                    )

                    rotated_weights = np.rot90(self.weights[:, :, dep, filt], 2)

                    # Change of inputs is a convolution of dval and the kernel (which has been rotated 180 degrees)
                    self.dinputs[image, :, :, dep] += signal.convolve2d(
                        dvalues[image, :, :, dep],
                        rotated_weights,
                        mode = self.dpadding
                    )

                self.dbiases[filt] += np.sum(dvalues[image, :, :, filt])

        # Batch Normalization
        self.dweights /= batch_size
        self.dbiases /= batch_size



class AveragePooling:
    def __init__(self, *, window_size=2, stride = 2):
        self.input = None
        self.output = None
        self.dinputs = None
        self.window_size = window_size
        self.stride = stride

    def forward(self, inputs):
        """
        :param inputs: image input. Dimensions -> {batch_size, image height, image width, image depth}
        :return: reduce the size with average pooling
        """
        self.input = inputs

        batch_size, height, width, channels = inputs.shape

        # Calculate the dimensions of the output array
        output_height = (height - self.window_size) // self.stride + 1
        output_width = (width - self.window_size) // self.stride + 1

        # Initialize the output array with zeros
        self.output = np.zeros((batch_size, output_height, output_width, channels))

        # Apply the pooling operation
        for k in range(batch_size):  # Loop over each item in the batch
            for i in range(0, height - self.window_size + 1, self.stride):
                for j in range(0, width - self.window_size + 1, self.stride):
                    # Pulls a window out of our input
                    window = inputs[:, i:i + self.window_size, j:j + self.window_size, :]

                    # Average over the height and width of the window and place in output
                    self.output[:, i // self.stride, j // self.stride, :] = np.mean(window, axis=(1, 2))


    def backward(self, dvalues):
        self.dinputs = np.zeros(self.input.shape)
        batch_size, input_height, input_width, input_depth = self.input.shape

        # Loop over the batch size and depth dimensions
        for i in range(batch_size):
            for d in range(input_depth):
                # Loop over the spatial dimensions of the dvalues
                for j in range(dvalues.shape[1]):  # Assuming dvalues.shape[1] == dvalues.shape[2]
                    for k in range(dvalues.shape[2]):

                        start_j = j * self.stride
                        end_j = start_j + self.window_size
                        start_k = k * self.stride
                        end_k = start_k + self.window_size

                        # Spread the gradient from dvalues to the dinputs
                        gradient = dvalues[i, j, k, d] / (self.window_size ** 2)
                        self.dinputs[i, start_j:end_j, start_k:end_k, d] += gradient


class Flatten:
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.input_shape = None

    def forward(self, inputs):
        """
        :param inputs: image with size -> (batchSize, height, width, depth)
        :return: a flattened output, keeping the batch-size untouched
        """
        self.input_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0], -1))

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)
