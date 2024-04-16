import numpy as np
from scipy import signal
class ConvolutionLayer:
    def __init__(self, *, depth, kernel_size, stride = 1, padding=0, input_size):
        # input size -> [height, width, depth]
        self.input_height, self.input_width, self.input_depth = input_size
        self.depth = depth # Number of filters

        self.kernel_height, self.kernel_width = kernel_size
        self.kernel_shape = (self.kernel_height, self.kernel_width, self.input_depth, self.depth)

        self.output_height = self.input_height - self.kernel_height + 1
        self.output_width = self.input_width - self.kernel_width + 1

        # Weights and biases (filters and biases)
        self.w = np.random.randn(*self.kernel_shape)
        self.biases = np.zeros(depth)

        print(f'kernel shape: {self.kernel_shape}')
        print(f'bias shape: {self.biases.shape, self.biases}')
        print(f'weights shape: {self.w.shape}')


        self.output = None
        self.dinputs = None
        self.dbiases = None
        self.dweights = None
        self.count = 0

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
                        mode='valid'
                    )


        self.output = inputs

        # if self.firstTime:
        #     # print()
        #     # print(f'input shape: {inputs.shape} -> {inputs[image, :, :, filt].shape}')
        #     # print(f'choosing filter: {self.w[:, :, 0, filt].shape}')
        #     # print(f'output shape: {self.output_shape}')
        #     # print(f'indexes for image: {image, filt}')
        #     # print(f'indexes for filter: {dep, filt}')
        #     mock = signal.correlate2d(
        #         inputs[image, :, :, dep],
        #         self.w[:, :, dep, filt],
        #         mode='valid'
        #     )
        #     # print(f'output example shape: {mock.shape}')
        #
        #     self.count += 1


    def backward(self, dvalues):
        self.dinputs = dvalues

class AveragePooling:
    def __init__(self, *, filters=None, kernel_size=None, stride = 1):
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.output = inputs


    def backward(self, dvalues):
        self.dinputs = dvalues


class Flatten:
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0], -1))

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.input_shape)
