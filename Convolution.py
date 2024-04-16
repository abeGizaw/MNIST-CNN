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
        elif self.padding == 'full':
            # For 'full' padding (output is the largest possible size)
            self.output_height = self.input_height + self.kernel_height - 1
            self.output_width = self.input_width + self.kernel_width - 1
        elif self.padding == 'valid':
            # For 'valid' padding (no padding is applied)
            self.output_height = self.input_height - self.kernel_height + 1
            self.output_width = self.input_width - self.kernel_width + 1
        else:
            raise ValueError("Padding mode must be 'same', 'full', or 'valid'.")

        # Weights and biases (filters and biases). Using tied bias
        self.w = self.glorot_init(self.kernel_shape)
        self.biases = np.zeros(depth)

        print(f'kernel shape: {self.kernel_shape}')
        print(f'bias shape: {self.biases.shape, self.biases}')
        print(f'weights shape: {self.w.shape}')

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
                        self.w[:, :, dep, filt],
                        mode=self.padding
                    )
                # Add the bias for each filter
                self.output[image, :, :, filt] += self.biases[filt]

        self.moutput = self.output
        self.output = inputs

        # if self.firstTime:
        #     print()
        #     print(f'indexes for image: {image, filt}')
        #     print(f'indexes for filter: {dep, filt}')
        #     print(f'input shape: {inputs.shape} -> {inputs[image, :, :, dep].shape}')
        #     print(f'choosing filter: {self.w[:, :, dep, filt].shape}')
        #     print(f'output shape: {output_shape}')
        #     self.firstTime = False


    def backward(self, dvalues):
        self.dinputs = dvalues

class AveragePooling:
    def __init__(self, *, window_size=2, stride = 1):
        self.moutput = None
        self.output = None
        self.dinputs = None
        self.window_size = window_size
        self.stride = stride

    def forward(self, inputs):
        output_height = (inputs.shape[1] - self.window_size) // self.stride + 1
        output_width = (inputs.shape[2] - self.window_size) // self.stride + 1
        self.output = np.zeros((inputs.shape[0], output_height, output_width, inputs.shape[3]))

        for i in range(output_height):
            for j in range(output_width):
                for k in range(inputs.shape[3]):
                    self.output[i, j, k] = np.mean(inputs[i * self.stride:i * self.stride + self.window_size,
                                              j * self.stride:j * self.stride + self.window_size,
                                              k])


        self.moutput = self.output
        self.output = inputs



    def backward(self, dvalues):
        self.dinputs = dvalues


class Flatten:
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.input_shape = None

    def forward(self, inputs):
        #print(f'flatten in: {inputs.shape}')
        self.input_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0], -1))
        #print(f'flatten out: {self.output.shape}')


    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)
