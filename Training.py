from LoadData import load_mnist_images, load_mnist_labels, formatData
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax, OptimizerSGD,AccuracyCategorical)
from Model import Model
from Convolution import ConvolutionLayer, AveragePooling, Flatten

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
model = Model()

# Convolution step (Feature Extraction)
model.add(ConvolutionLayer(depth=6, kernel_size=(5, 5), padding='same', input_size = (28, 28, 1)))
model.add(ActivationReLU())
#model.add(AveragePooling())
#model.add(ConvolutionLayer(depth=16, kernel_size=(5, 5), input_size = (14, 14, 6)))
#model.add(ActivationReLU())
#model.add(AveragePooling())

# Prepare to pass into feedforward
model.add(Flatten())

# Feedforward (Classification)
#model.add(DenseLayer(400, 120))
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
model.train(X_train, train_labels, epochs=1,print_every=1000, batch_size =128)
model.validate(validation_data = (X_test, test_labels))
