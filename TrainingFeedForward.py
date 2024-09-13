import time
import matplotlib.pyplot as plt
from LoadData import load_mnist_images, load_mnist_labels, formatData, plot_mnist_image
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical, ActivationTanh)
from Model import Model
from StatisticsTracker import StatisticsTracker

# Load and format the data
train_images = load_mnist_images('MNISTdata\\train-images-idx3-ubyte')
train_labels = load_mnist_labels('MNISTdata\\train-labels-idx1-ubyte')
test_images = load_mnist_images('MNISTdata\\t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('MNISTdata\\t10k-labels-idx1-ubyte')
print(f'X_train shape: {train_images.shape}')
print(f'y_train shape: {train_labels.shape}')
print(f'X_test shape: {test_images.shape}')
print(f'y_test shape: {test_labels.shape}\n')
X_train, X_test, train_labels, test_labels = formatData(train_images, test_images, train_labels, test_labels, flatten=True)

# Example of plotting images
plot_mnist_image(X_train[0].reshape(28, 28), train_labels[0])  # Reshape is necessary if the image was flattened

# Creating LetNet Model
stats = StatisticsTracker("FeedForward", "feedforward_stats.json")
model = Model(stats)

# Feedforward (Classification)
model.add(DenseLayer(784, 120))
model.add(ActivationTanh())
model.add(DenseLayer(120, 84))
model.add(ActivationTanh())
model.add(DenseLayer(84, 10))
model.add(ActivationSoftmax())


model.set(loss = LossCategoricalCrossEntropy(),
          optimizer = OptimizerSGD(learning_rate = 0.5, decay=0.0003),
          accuracy = AccuracyCategorical())

# Finishing touches
model.finalize()

# Time model.train()
start_train = time.time()
model.train(X_train, train_labels, epochs=5, batch_size=64)
end_train = time.time()

start_validate = time.time()
model.validate(validation_data = (X_test, test_labels), batch_size=64)
end_validate = time.time()

train_time = end_train - start_train
validate_time = end_validate - start_validate
print(f'train time: {train_time:.2f} seconds')
print(f'finalize time: {validate_time:.2f} seconds')
print(f'combined time: {train_time + validate_time:.2f} seconds')

stats.add_time('validation', validate_time)
stats.add_time('training', train_time)
stats.add_time('combined', train_time + validate_time)
stats.save_statistics()