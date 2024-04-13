import numpy as np
from LoadData import load_mnist_images, load_mnist_labels
from nnfs.datasets import spiral_data
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical)
from Model import Model


# Load the data
train_images = load_mnist_images('MNISTdata\\train-images-idx3-ubyte')
train_labels = load_mnist_labels('MNISTdata\\train-labels-idx1-ubyte')
test_images = load_mnist_images('MNISTdata\\t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('MNISTdata\\t10k-labels-idx1-ubyte')


print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# Dataset
X , y = spiral_data(samples = 100, classes = 2)
X_test , y_test = spiral_data(samples = 100, classes = 2)

# Reshape to be a list of lists. Inner lists contain an output
# either 1 or 0 per each output neuron =
y.reshape(-1, 1)
y_test.reshape(-1, 1)

# Creating layers and their activation functions
model = Model()
model.add(DenseLayer(2, 64))
model.add(ActivationReLU())
model.add(DenseLayer(64, 3))
model.add(ActivationSoftmax())
model.set(loss = LossCategoricalCrossEntropy(),
          optimizer = OptimizerSGD(learning_rate = 0.45),
          accuracy = AccuracyCategorical())
model.finalize()
model.train(X, y, epochs=7000,
                  print_every=1000,
                  validation_data = (X_test, y_test)
            )


