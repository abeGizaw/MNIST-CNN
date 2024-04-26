import time
from FeedForward import (LossCategoricalCrossEntropy, ActivationSoftmax,
                         ActivationSoftmax_Loss_CategoricalCrossEntropy, InputLayer)
from colorama import Fore, Style
class Model:
    def __init__(self, statTracker):
        self.layers = []
        self.input_layer = None
        self.trainable_layers = []
        self.statTracker = statTracker

        # Layers we want to save off
        self.accuracy = None
        self.optimizer = None
        self.loss = None
        self.softmax_classifier_output = None
        self.output_layer_activation = None

    def add(self, layer):
        self.layers.append(layer)

    # Asterisk requires loss and optimizer to be specified when set is called
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy



    def finalize(self):
        """
        Connects all the layers, setting their next and previous
        Keeps track of all trainable layers
        :return:
        """
        self.input_layer = InputLayer()
        layer_count = len(self.layers)

        for i in range(layer_count):
            # If it is the 1st layer then
            # the prev layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # Hidden layer to hidden layer
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # If we are at the last layer, the next object is the loss func
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If the layer has weights, then it is trainable
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])


        # If the model uses softmax and categorical cross entropy
        # Save off the combined model for faster backward pass
        if isinstance(self.layers[-1], ActivationSoftmax) and \
                isinstance(self.loss, LossCategoricalCrossEntropy):
            self.softmax_classifier_output = \
                ActivationSoftmax_Loss_CategoricalCrossEntropy()

    # Size of x is # inputs x flattened input
    def train(self, X, y, *, epochs=1, batch_size = None, print_every=100):
        """

        :param X: train data inputs (# of inputs x flattened input)
        :param y: the trained data expected
        :param epochs: iterations
        :param batch_size
        :param print_every
        :return:
        """
        print(f'starting to train\n')
        # Calculate number of steps
        train_steps = 1
        if batch_size is not None:
            # Number of batches being trained
            train_steps = len(X) // batch_size

            # Add a step for leftover data
            if train_steps * batch_size < len(X):
                train_steps += 1
            print(f'running {len(X)} pieces of data in {train_steps} steps\n')

        for epoch in range(1, epochs + 1):
            epoch_time_s = time.time()
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                # Figure out data to train on
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                train_time_s = time.time()
                # Perform the forward pass
                output = self.forward(batch_X)

                # Calculate loss
                loss = self.loss.calculate(output, batch_y)

                # Get predictions and calculate accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Backward pass
                self.backward(output, batch_y)

                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)

                total_train_time = time.time() - train_time_s
                if batch_size is not None and step % print_every == 0 and step != 0:
                    self.statTracker.add_time("step", total_train_time)
                    print(f'step:{step}, acc:{accuracy:.3f}, loss:{loss:.3f}')
                    print(f'step took {total_train_time:.2f} seconds')
                    if accuracy <= .1 and step >= 64:
                        print(f'{Fore.RED} WARNING: Most Likely Need to Restart This run :( {Style.RESET_ALL}')


            epoch_data_loss = self.loss.calculate_accumulated()
            epoch_accuracy = self.accuracy.calculate_accumulated()

            total_epoch_time = time.time()- epoch_time_s
            self.statTracker.add_time("epoch", total_epoch_time)
            self.statTracker.record_epoch_accuracy(epoch, epoch_accuracy)
            print(f'epoch:{epoch}, acc:{epoch_accuracy:.3f}, loss:{epoch_data_loss:.3f}')
            print(f'epoch took {total_epoch_time:.2f} seconds\n')
            if epoch_accuracy < .2:
                print(f'{Fore.RED} WARNING: Most Likely Need to Restart This run :( {Style.RESET_ALL}')


    def validate(self, *, validation_data, batch_size = None, print_every = 100):
        """
        :param print_every:
        :param validation_data: inputs and expected out values
        :param batch_size
        :return:
        """
        print(f'starting to validate\n')

        self.loss.new_pass()
        self.accuracy.new_pass()
        X_val, y_val = validation_data

        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        for step in range(validation_steps):
            # Figure out data to train on
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            valid_time_s = time.time()
            output = self.forward(batch_X)
            loss = self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            valid_time_e = time.time()

            if batch_size is not None and step % print_every == 0 and step != 0:
                print(
                    f'validation, step:{step}, acc:{accuracy:.3f}, loss:{loss:.3f}'
                )
                print(f'step took {valid_time_e - valid_time_s:.2f} seconds\n')

        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        self.statTracker.record_validation_accuracy(validation_accuracy)
        print(f'final validation - acc:{validation_accuracy}, loss:{validation_loss}')


    def forward(self, X):
        """
        :param X: input to pass forward
        Call forward on every hidden layer
        Output of prev layer is input of the next
        :return:
        """
        # Pass in the input to the input layer
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        # The final layers output
        return self.layers[-1].output


    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # Sets the first dinputs
            self.softmax_classifier_output.backward(output, y)

            # Since we combined softmax and CrossEntropy, they will share dinputs
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward on every layer but the last in reversed order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

        # Is not a softmax classifier
        else:
            # This will set our initial dinputs
            self.loss.backward(output, y)

            # Go through layers in reverse order (backpropagation)
            # dinputs is now the parameter
            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)

 