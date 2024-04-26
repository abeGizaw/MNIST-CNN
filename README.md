# MNIST-CNN
This repo features my implementation of a LeNet Convolutional Neural Network (CNN) model built from scratch in Python. It recreates the architecture of LeNet, specifically designed for the classification of handwritten digits. The focus lies on replicating the original structure and parameters of LeNet, offering a detailed walkthrough of the results of different architectures

This is an extensions of my other project: [Neural Network From Scratch](https://github.com/abeGizaw/NeuralNetworkStarter)
Report can be found ![here](/stuffForReadMe/CNNReport.pdf)

Goal:  
![Your result image placeholder](/stuffForReadMe/LeNetArchitecture.png)

LeNet
- Image: 28 (height) x 28 (width) x 1 (channel)
- Convolution with 5x5 kernel + 2 padding: 28x28x6
- Pool with 2x2 average kernel + 2 stride: 14x14x6
- Convolution with 5x5 kernel (no pad): 10x10x16
- Pool with 2x2 average kernel + 2 stride: 5x5x16
- Dense: 120 fully connected neurons
- Dense: 84 fully connected neurons
- Dense: 10 fully connected neurons
- Output: 1 of 10 classes

LeNet uses 2 iterations of convolution, with the sigmoid activation function between each layer, and average pooling.

The Feedforward consists of 3 dense layers also using the sigmoid activation function.

Result:

![Your result image placeholder](/stuffForReadMe/myArchitecture.png)

## Training Files

1. TrainingFeedForward.py
   - Using the model without any convolution. Just using a basic FeedForward network.

2. TrainingConvolv.py
   - Using the model without pooling. The file has options to use 1 or 2 convolution layers.

3. TrainingFull.py
   - The full LeNet Architecture (As close as I could get 😅). Includes Average Pooling and the Convolution process. The file has options to use 1 or 2 convolution layers.

## Computation Files
1. LoadData.py
   - Loads the MNIST data and formats it for the model.

2. Model.py
   - Model Code. Allows the user to build their Neural Network using a very similar approach to Keras.

3. FeedForward.py
   - Code for the FeedForward Network. Lots of inspiration from author/Youtuber Harrison Linsley (Sentdex) and his book [Neural Networks from Scratch in Python](https://nnfs.io/).
  
4. Convolution.py
   - Code for the Convolution Process. Contains the pooling and flattening layers.

5. StatisticsTracker.py
   - This file keeps track of the average run time of the model’s steps, epochs, training, validation, and total runtime
   - It also keeps track of the average accuracy of each model
   - Saves everything to a JSON file, and writes averages to a Txt file
     
## Other Files
   1. RunNetworks.py
      - Runs a network x times (for training purposes)
      - I don’t recommend using this file when individually training the more complex models since we can run into defect iterations that we won’t be able to back out of.

   2. TestConv.py
      - My brute force attempt of debugging the conv and pooling layers, and making sure they are computing on the right values  


## Baisc Code Setup  
```  
train_images = load_mnist_images('MNISTdata\\train-images-idx3-ubyte')
train_labels = load_mnist_labels('MNISTdata\\train-labels-idx1-ubyte')
test_images = load_mnist_images('MNISTdata\\t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('MNISTdata\\t10k-labels-idx1-ubyte')
X_train, X_test, train_labels, test_labels = formatData(train_images, test_images, train_labels, test_labels)

stats = StatisticsTracker("FullNetwork", 'full_network_stats.json')
model = Model(stats)

model.add(ConvolutionLayer(depth=6, kernel_size=(5, 5), padding='same', input_size = (28, 28, 1)))
model.add(ActivationReLU())
model.add(AveragePooling())
model.add(ConvolutionLayer(depth=16, kernel_size=(5, 5), input_size = (14, 14, 6)))
model.add(ActivationReLU())
model.add(AveragePooling())

model.add(Flatten())

model.add(DenseLayer(400 , 120))
model.add(ActivationReLU())
model.add(DenseLayer(120, 84))
model.add(ActivationReLU())
model.add(DenseLayer(84, 10))
model.add(ActivationSoftmax())

model.set(loss = LossCategoricalCrossEntropy(),
          optimizer = OptimizerSGD(learning_rate = 0.5),
          accuracy = AccuracyCategorical())
model.finalize()
model.train(X_train, train_labels, epochs=3, print_every=64, batch_size=128)
model.validate(validation_data = (X_test, test_labels), batch_size=128, print_every=64)  
```  
## Data and Results

This table documents how fast the each model ran and how accurate it was when finished. Note that the model had a lot of run-time variation per step and epoch. Validation is much quicker than training because you only do 1 iteration with no backpropagation. Run time will vary by computer (obv.).

| Networks                | Params                            | Run time                                                        | Accuracy                                                             |
|-------------------------|-----------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------|
| **FeedForward**         | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 0.00 s <br> Training: 0.6 - 0.8 s <br> Validation: 0.03 s <br> Total: 2.13 s | Times Ran: 48 <br> Epoch: 88% -> 95% -> 97% <br> Total: 95%          |
| **Convolution (Single Conv)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 0.08 s <br> Training: 1.9 min <br> Validation: 2.45 s <br> Total: 1.9 min | Times Ran: 22 <br> Epoch: 86% -> 95% -> 97% <br> Total: 96%          |
| **Convolution (Double Conv)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 1.2 - 2 s <br> Training: 35 min <br> Validation: 28 s <br> Total: 35.4 min | Times Ran: 5 <br> Epoch: 70% -> 85% -> 94% <br> Total: 91%          |
| **Convolution + Avg. Pooling (Single)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 1 - 3.5 s <br> Training: 30 min <br> Validation: 42.3 s <br> Total: 31 min | Times Ran: 3 <br> Epoch: 85% -> 94% -> 96% <br> Total: 92%          |
| **Convolution + Avg. Pooling (Double)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 1 - 15 min <br> Training: 22 min <br> Validation: 1.1 h <br> Total: 2.1 h | Times Ran: 9 <br> Epoch: 49% -> 72% -> 84% <br> Total: 81%          |      
  
If we were to do more than 3 iterations for the more complex networks, I am very confident that they would become just as accurate as the feedforward network or the one convolution network. But I wanted to show that the increase of parameters leads to slower learning without more up-to-date optimization techniques (also because I am impatient)

*Note that if running one of the more complex models, you notice that its not learning within the first epoch (using print_every), then restart the model! It will most likely not fix itself within the 3 iterations. Some reasons for it breaking are below*  

## Further Work  
I plan on doing some more research on why the model does end up breaking. This will just be an in depth look at what’s happening with the weights and biases when my accuracies aren’t increasing, or suddenly decrease a lot. With this information, I will probably add some extra features such as reducing the learning rate as the training progresses, regularization, etc.  

I also want my stats tracker to flag these defect iterations, and not include them (at user’s discretion) in the model’s statistics. I could use this to track the weights and biases and track them seperatly from the good runs. 
# References

1. [LeCun's article on using CNNs for MNIST recognition](https://www.rose-hulman.edu/class/cs/csse413/assignments/cnn/Lecun98.pdf)
2. [Textbook on Neural Networks from Scratch in Python](https://nnfs.io/)
3. [Some Youtube on above book](https://www.youtube.com/watch?v=Wo5dMEP_BbI)
4. [LeNet’s architecture](https://en.wikipedia.org/wiki/File:Comparison_image_neural_networks.svg)
5. [Visual on how CNNs learn](https://www.youtube.com/watch?v=xvFZjo5PgG0)
6. [Math behind CNN backpropagation](https://www.youtube.com/watch?v=z9hJzduHToc)
7. [Article about pooling](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/)
8. [Article about calculating gradients for pooling](https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/)
9. [Another video about convolution](https://www.youtube.com/watch?v=Lakz2MoHy6o)
10. [The most helpful link. Very in depth exclamation about CNNs and how to use numpy to make everything efficient and easy](https://social.mtdv.me/watch?v=-0X4TxgEYh)

