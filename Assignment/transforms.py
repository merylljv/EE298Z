#!/usr/bin/env python3
import numpy as np


def corrupt_mnist_img(rng, img, value):
    """Corrupt a single MNIST image.
    Note that the image itself is MODIFIED.

    :param rng: instance of numpy.random.RandomState
    :param img: image to modify. ndarray or compatible
    :param value: pixel value to use for corrupting the image
    :return: modified image
    """
    # Choose square size
    s = rng.randint(7, 15)
    # Choose top-left corner position
    x = rng.randint(0, 29 - s)
    y = rng.randint(0, 29 - s)
    # Draw square
    img[..., y:y + s, x:x + s] = value
    # Return object for convenience
    return img


def corrupt_mnist_copy(x_train, value=255, seed=0):
    """Create a corrupted copy of the MNIST dataset

    :param x_train: ndarray of images. Shape: (N, ..., H, W)
    :param value: pixel value to use for corrupting the image
    :param seed: seed to use for the random number generator
    :return: ndarray of corrupted images
    :return: ndarray of corrupted images
    """
    rng = np.random.RandomState(seed)
    corrupted_x_train = x_train.copy()
    for img in corrupted_x_train:
        corrupt_mnist_img(rng, img, value)
    return corrupted_x_train


def corrupt_mnist_generator(x_train, value=255, seed=0):
    """Generator version of `corrupt_mnist_copy()`

    :param x_train:
    :param value:
    :param seed:
    :return:
    """
    rng = np.random.RandomState(seed)
    while True:
        for img in x_train:
            yield corrupt_mnist_img(rng, img.copy(), value)


class CorruptMNIST(object):
    """PyTorch transform for corrupting MNIST images

    Use after ToTensor and before Normalize.
    """

    def __init__(self, seed=0):
        self._rng = np.random.RandomState(seed)

    def __call__(self, img):
        return corrupt_mnist_img(self._rng, img, 1.)

    def __repr__(self):
        return self.__class__.__name__ + '()'
