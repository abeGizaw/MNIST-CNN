from FeedForward import (LossCategoricalCrossEntropy, ActivationSoftmax,
                         ActivationSoftmax_Loss_CategoricalCrossEntropy, InputLayer)
class Model:
    def __init__(self):
        self.layers = []
        self.input_layer = None

        self.trainable_layers = []
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

    # Finalizes the model, setting the next and previous layers
    def finalize(self):
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

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):
            # Perform the forward pass
            output = self.forward(X)

            # Calculate loss
            loss = self.loss.calculate(output, y)

            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            if epoch % print_every == 0:
                print(f'epoch:{epoch}, acc:{accuracy:.3f}, loss:{loss:.3f}')

        if validation_data is not None:
            # Sample, target
            X_val, y_val = validation_data

            output = self.forward(X_val)
            loss = self.loss.calculate(output, y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            print(
                f'validation, acc:{accuracy:.3f}, loss:{loss:.3f}'
            )

    def forward(self, X):
        # Pass in the input to the input layer
        self.input_layer.forward(X)

        # Call forward on every hidden layer
        # Output of prev layer is input of the next
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
