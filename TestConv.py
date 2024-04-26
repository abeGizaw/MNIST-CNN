from LoadData import load_mnist_images, load_mnist_labels, formatData
from FeedForward import (DenseLayer, ActivationReLU, LossCategoricalCrossEntropy,
                            ActivationSoftmax,ActivationSoftmax_Loss_CategoricalCrossEntropy)
from Convolution import ConvolutionLayer, Flatten, AveragePooling
from scipy import signal
import numpy as np
"""
This file was simply for debugging purposes. With more layers affecting the accuracy, I decided to see what my values were passing layer to layer
From what I can tell, it seems all operations are done correctly.
Leaving this here if decide to come back and mess with things
Make sure to uncomment the import in convolution to be able to set the random seed 
"""
# Load and format the data
train_images = load_mnist_images('MNISTdata\\train-images-idx3-ubyte')
train_labels = load_mnist_labels('MNISTdata\\train-labels-idx1-ubyte')
test_images = load_mnist_images('MNISTdata\\t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('MNISTdata\\t10k-labels-idx1-ubyte')
print(f'X_train shape: {train_images.shape}')
print(f'y_train shape: {train_labels.shape}')
print(f'X_test shape: {test_images.shape}')
print(f'y_test shape: {test_labels.shape}\n')
X_train, X_test, train_labels, test_labels = formatData(train_images, test_images, train_labels, test_labels)

print(f'X_train shape: {X_train[0].shape}')
print(f'y_train expected: {train_labels[0]}')
print(X_train[0:2, 5:12, 6:13 ,0])
print(f'I will test with shape: {X_train[0:2, 5:12, 6:13 ,:].shape}')

c1 = ConvolutionLayer(depth=2, kernel_size=(6, 6), padding='same', input_size = (7, 7, 1))
c2 = ActivationReLU()
c3 = AveragePooling()
c4 = Flatten()
c5 = DenseLayer(18, 10)
c6 = ActivationSoftmax()

c7 = LossCategoricalCrossEntropy()
c8 = ActivationSoftmax_Loss_CategoricalCrossEntropy()
#c8 =  OptimizerSGD(learning_rate = 0.5)


c1.forward(X_train[0:2, 5:12, 6:13 ,:]) # Conv
c2.forward(c1.output) # Act
c3.forward(c2.output) # Pool
c4.forward(c3.output) # Flatten
c5.forward(c4.output) # Dense
c6.forward(c5.output)
loss = c7.calculate(c6.output, train_labels[0:2])

c8.backward(c6.output, train_labels[0:2])
c6.dinputs = c8.dinputs
c5.backward(c6.dinputs) # Dense
c4.backward(c5.dinputs) # Flatten
c3.backward(c4.dinputs) # Pooling
c2.backward(c3.dinputs) # Pooling
c1.backward(c2.dinputs) # Pooling

print(f'feeding flatten layer output shape of: {c3.output.shape}')

# for k in range(c3.output.shape[3]):
#     print(f'channel {k+1}')
#     print(c3.output[:,:,:,k])

print()
# print(f'out of dense is {c5.output}\n')
# print(f'out of softmax is {c6.output}\n')
print(loss)
# print(f'flatten had output :\n {c4.output} \n'
#       f'\n and gets dvalues :\n {c5.dinputs}\n'
#       f'\n and converts to: \n {c4.dinputs}')
# print(f'pooling gets dvalues :\n {c4.dinputs}\n'
#       f'\n and converts to: \n {c3.dinputs}')
# print(f'conv had output :\n {c1.output} \n'
#       f'\n and gets dvalues :\n {c2.dinputs}\n'
#       f'\n and converts to: \n {c1.dinputs}')
print(c1.inputs.shape)
print(c1.output.shape)
print(c2.dinputs.shape)
print(c1.dinputs.shape)
print(c1.weights.shape)


t1 = np.array([[ 0.02761309,  0.05813174, -0.04318737, -0.035306,0.26228727],
 [0.1650246, 0.03849182, -0.24265865, -0.27140546,  0.15734922],
 [ 0.27074742, -0.02179056, -0.21593659, -0.20174985,  0.01235928],
 [-0.13318746, -0.02480512, -0.2722136,   0.06341092,  0.25102182],
 [-0.07947433,  0.11179709,  0.0943375,  -0.16383236, -0.10440949]])

t2 = np.array([
[ 0.12172949,  0.02538976,  0.08253017,  0.22162028, -0.06593543],
 [ 0.01634543,  0.24075382, -0.23355494,  0.1881582,   0.20931048],
 [ 0.16922964, 0.15869127, 0.07915128,  0.25154273, -0.0482745 ],
 [ 0.15513,     0.03871209,  0.06654469,  0.06614786,  0.10285309],
 [-0.03562011, -0.24877404,  0.09652736, -0.20991098, -0.07709683]
])
print(X_train[0, 5:12, 6:13 ,0].shape)

#print(np.maximum(0, signal.correlate2d(X_train[1, 5:12, 6:13 ,0],t1, mode='same')))
#print(np.maximum(0, signal.correlate2d(X_train[0, 5:12, 6:13 ,0],t1, mode='same')))
#print()
#print(np.maximum(0, signal.correlate2d(X_train[0, 5:12, 6:13 ,0],t2, mode='same')))
#print(np.maximum(0, signal.correlate2d(X_train[1, 5:12, 6:13 ,0],t2, mode='same')))



#model.set(loss = LossCategoricalCrossEntropy(),
#           optimizer = OptimizerSGD(learning_rate = 0.5),
#           accuracy = AccuracyCategorical())
#
# model.finalize()
# model.train(X_train, train_labels, epochs=1, print_every=10000, batch_size=None)
