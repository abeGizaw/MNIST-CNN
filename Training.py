import numpy as np
from LoadData import load_mnist_images, load_mnist_labels, formatData
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical)
from Model import Model


# Load and format the data
train_images = load_mnist_images('MNISTdata\\train-images-idx3-ubyte')
train_labels = load_mnist_labels('MNISTdata\\train-labels-idx1-ubyte')
test_images = load_mnist_images('MNISTdata\\t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('MNISTdata\\t10k-labels-idx1-ubyte')
print(f'X_train shape: {train_images.shape}')
print(f'y_train shape: {train_labels.shape}')
print(f'X_test shape: {test_images.shape}')
print(f'y_test shape: {test_labels.shape}')
X_train, X_test, train_labels, test_labels = formatData(train_images, test_images, train_labels, test_labels)


# Creating layers and their activation functions
model = Model()
model.add(DenseLayer(784, 64))
model.add(ActivationReLU())
model.add(DenseLayer(64, 10))
model.add(ActivationSoftmax())
model.set(loss = LossCategoricalCrossEntropy(),
          optimizer = OptimizerSGD(learning_rate = 0.45),
          accuracy = AccuracyCategorical())
model.finalize()
model.train(X_train, train_labels, epochs=100,
                  print_every=100,
                  validation_data = (X_test, test_labels)
            )
