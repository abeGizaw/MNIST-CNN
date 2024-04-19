import numpy as np
from scipy import signal
class ConvolutionLayer:
    # Note this Convolution layer does not account for varying strides
    def __init__(self, *, depth, kernel_size, padding='valid', input_size):
        # input size -> [height, width, depth]
        self.firstTime = True
        self.moutput = None
        self.input_height, self.input_width, self.input_depth = input_size
        self.depth = depth # Number of filters

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

        print(f'kernel shape: {self.kernel_shape}')
        print(f'bias shape: {self.biases.shape, self.biases}')
        print(f'weights shape: {self.weights.shape}')

        self.inputs = None
        self.output = None
        self.dinputs = None
        self.dbiases = None
        self.dweights = None


    # Xavier/Glorot initialization for layers using sigmoid or tanh
    @staticmethod
    def glorot_init(shape):
        # shape: (kernel_height, kernel_width, input_depth, number_of_filters)
        fan_in = shape[0] * shape[1] * shape[2]  # product of kernel dimensions and input depth
        fan_out = shape[0] * shape[1] * shape[3]  # product of kernel dimensions and number of filters

        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

    def forward(self, inputs):
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

        # if self.firstTime:
        #     print()
        #     print(f'indexes for image: {image, filt}')
        #     print(f'indexes for filter: {dep, filt}')
        #     print(f'input shape: {inputs.shape} -> {inputs[image, :, :, dep].shape}')
        #     print(f'choosing filter: {self.w[:, :, dep, filt].shape}')
        #     print(f'output shape: {output_shape}')
        #     self.firstTime = False


    def backward(self, dvalues):
        self.dweights = np.zeros(self.kernel_shape)
        self.dbiases = np.zeros(self.depth)
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

                    self.dweights[:, :, dep, filt] += signal.correlate2d(
                        self.inputs[image, :, :, dep],
                        dvalues[image, :, :, dep],
                        mode="valid"
                    )

                    rotated_weights = np.rot90(self.weights[:, :, dep, filt], 2)

                    self.dinputs[image, :, :, dep] += signal.convolve2d(
                        dvalues[image, :, :, dep],
                        rotated_weights,
                        mode = self.dpadding
                    )

                self.dbiases[filt] += np.sum(dvalues[image, :, :, filt])


        self.dweights /= batch_size
        self.dbiases /= batch_size



class AveragePooling:
    def __init__(self, *, window_size=2, stride = 2):
        self.moutput = None
        self.output = None
        self.dinputs = None
        self.window_size = window_size
        self.stride = stride

    def forward(self, inputs):
        print(f"PooL Input size: {inputs.shape}")

        batch_size, height, width, channels = inputs.shape

        # Calculate the dimensions of the output array
        output_height = (height - self.window_size) // self.stride + 1
        output_width = (width - self.window_size) // self.stride + 1

        # Initialize the output array with zeros
        self.output = np.zeros((batch_size, output_height, output_width, channels))

        # Apply the pooling operation
        for i in range(0, height - self.window_size + 1, self.stride):
            for j in range(0, width - self.window_size + 1, self.stride):
                # Pulls a window out of our input
                window = inputs[:, i:i + self.window_size, j:j + self.window_size, :]

                # Average over the height and width of the window and place in output
                self.output[:, i // self.stride, j // self.stride, :] = np.mean(window, axis=(1, 2))

        print(f'pool output size: {self.output.shape}')
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues


class Flatten:
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.input_shape = None

    def forward(self, inputs):
        # print(f'flatten in: {inputs.shape}')
        self.input_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0], -1))
        # print(f'flatten out: {self.output.shape}')


    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)
