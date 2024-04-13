import numpy as np
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

def formatData(train_images, test_images, train_labels, test_labels):
    # Normalize data
    X_train = (np.array(train_images) > 0).astype(np.float32)
    X_test = (np.array(test_images) > 0).astype(np.float32)

    # Reshape to be a list of lists. Inner lists contain an output
    # from 0-9 per each output neuron
    train_labels.reshape(-1, 1)
    test_labels.reshape(-1, 1)

    # Flatten data before passing into network
    X_train = X_train.reshape((-1, 784))  # Flatten the images to 1D array of 784 features
    X_test = X_test.reshape((-1, 784))  # Image is 28X28
    return X_train, X_test, train_labels, test_labels