# MNIST-CNN
This repo features my implementation of a LeNet Convolutional Neural Network (CNN) model built from scratch in Python. It recreates the architecture of LeNet, specifically designed for the classification of handwritten digits. The focus lies on replicating the original structure and parameters of LeNet, offering a detailed walkthrough of the results of different architectures

This is an extensions of my other project: [Neural Network From Scratch](https://github.com/abeGizaw/NeuralNetworkStarter)

# Architecture

Goal:  
![Your image placeholder](image-link.jpg)

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

![Your result image placeholder](result-image-link.jpg)

# Training Files

1. TrainingFeedForward.py
   - Using the model without any convolution. Just using a basic FeedForward network.

2. TrainingConvolv.py
   - Using the model without pooling. The file has options to use 1 or 2 convolution layers.

3. TrainingFull.py
   - The full LeNet Architecture (As close as I could get 😅). Includes Average Pooling and the Convolution process. The file has options to use 1 or 2 convolution layers.

# Computation Files
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
  

## Data and Results

This table documents how fast the each model ran and how accurate it was when finished. Note that the model had a lot of run-time variation per step and epoch. Validation is much quicker than training because you only do 1 iteration with no backpropagation. Run time will vary by computer (obv.).

| Networks                | Params                            | Run time                                                        | Accuracy                                                             |
|-------------------------|-----------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------------|
| **FeedForward**         | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 0.00 s <br> Training: 0.6 - 0.8 s <br> Validation: 0.03 s <br> Total: 2.13 s | Times Ran: 6 <br> Epoch: 87% -> 96% -> 97% <br> Total: 95%          |
| **Convolution (Single Conv)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 0.08 s <br> Training: 1.9 min <br> Validation: 2.45 s <br> Total: 1.9 min | Times Ran: 5 <br> Epoch: 88% -> 96% -> 97% <br> Total: 96%          |
| **Convolution (Double Conv)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 1.2 - 2 s <br> Training: 35 min <br> Validation: 28 s <br> Total: 35.4 min | Times Ran: 1 <br> Epoch: 55% -> 85% -> 90% <br> Total: 84%          |
| **Convolution + Avg. Pooling (Single)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 1 - 3.5 s <br> Training: 30 min <br> Validation: 42.3 s <br> Total: 31 min | Times Ran: 1 <br> Epoch: 87% -> 95% -> 97% <br> Total: 97%          |
| **Convolution + Avg. Pooling (Double)** | Lr = 0.5 <br> Epochs = 3 <br> Batch size = 128 | Step: 1 - 15 min <br> Training: 22 min <br> Validation: 1.1 h <br> Total: 2.1 h | Times Ran: 1 <br> Epoch: 18% -> 77% -> 85% <br> Total: 76%          |
