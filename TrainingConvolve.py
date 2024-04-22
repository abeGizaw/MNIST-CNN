import time
from LoadData import load_mnist_images, load_mnist_labels, formatData
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical)
from Model import Model
from Convolution import ConvolutionLayer, Flatten
from StatisticsTracker import StatisticsTracker

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


# Creating LetNet Model
stats = StatisticsTracker("Convolve","convolve_stats.json")
model = Model(stats)

# Convolution step (Feature Extraction)
model.add(ConvolutionLayer(depth=6, kernel_size=(5, 5), padding='same', input_size = (28, 28, 1)))
model.add(ActivationReLU())

# Optional second convolution layer (input of first dense layer goes from 4704 -> 9216)
# model.add(ConvolutionLayer(depth=16, kernel_size=(5, 5), input_size = (28, 28, 6)))
# model.add(ActivationReLU())

# Prepare to pass into feedforward
model.add(Flatten())

# Feedforward (Classification)
model.add(DenseLayer(4704, 120))
model.add(ActivationReLU())
model.add(DenseLayer(120, 84))
model.add(ActivationReLU())
model.add(DenseLayer(84, 10))
model.add(ActivationSoftmax())


model.set(loss = LossCategoricalCrossEntropy(),
          optimizer = OptimizerSGD(learning_rate = 0.5),
          accuracy = AccuracyCategorical())

# Finishing touches
model.finalize()

# Time model.train()
start_train = time.time()
model.train(X_train, train_labels, epochs=3, print_every=64, batch_size=128)
end_train = time.time()

start_validate = time.time()
model.validate(validation_data = (X_test, test_labels), batch_size=128, print_every=64)
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
