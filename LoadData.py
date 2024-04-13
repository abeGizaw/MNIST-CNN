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