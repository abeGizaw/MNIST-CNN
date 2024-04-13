import numpy as np
from nnfs.datasets import spiral_data
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical)
from Model import Model


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
model.train(X, y, epochs=37000,
                  print_every=1000,
                  validation_data = (X_test, y_test)
            )


