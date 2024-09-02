import numpy as np
import matplotlib.pyplot as plt

def plot_mnist_image(image, label):
    plt.imshow(image, cmap='gray')  # Plot the image - note that the data must be in 2D
    plt.title(f'Label: {label}')
    plt.colorbar()  # Optional, it shows the color bar
    plt.show()
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # magic number. We don't need to keep track of them
        int.from_bytes(f.read(4), 'big')

        number_of_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        columns = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape((number_of_images, rows, columns))
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Magic number and number of items. We don't need to keep track of them
        int.from_bytes(f.read(4), 'big')
        int.from_bytes(f.read(4), 'big')

        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def formatData(train_images, test_images, train_labels, test_labels, *, flatten = False):
    # Normalize data
    X_train = (np.array(train_images) > 64).astype(np.float32)
    X_test = (np.array(test_images) > 64).astype(np.float32)

    # Reshape to be a list of lists. Inner lists contain an output
    # from 0-9 per each output neuron
    train_labels.reshape(-1, 1)
    test_labels.reshape(-1, 1)

    # Adds a channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))


    return X_train, X_test, train_labels, test_labels