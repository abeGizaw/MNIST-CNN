import numpy as np
from FeedForward import ActivationReLU
class ConvolutionLayer:
    def __init__(self):
        self.output = None

        self.dinputs = None
        self.dbiases = None
        self.dweights = None
    def forward(self, inputs):
        self.output = inputs

    def backward(self, dvalues):
        pass

class AveragePooling:
    def __init__(self):
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
