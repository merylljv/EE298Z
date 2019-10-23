#!/usr/bin/env python3
import os.path
import numpy as np
try:
    import keras
    from keras import backend as K
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import backend as K

from .classifier_keras import model as classifier
from .structural_similarity import structural_similarity as ssim


class IdentityModel(keras.Model):
    """Model which simply returns the input"""

    def __init__(self):
        super().__init__()
        self.identity = keras.layers.Lambda(lambda x: x)

    def call(self, x):
        return self.identity(x)


def _preprocess_for_classifier(x):
    return (x - 0.1307) / 0.3081


def test_model(model, x_test, y_test, batch_size=100):
    """Run the benchmarks for the given model

    :param model:
    :param x_test: MNIST images scaled to [0, 1]
    :param y_test: MNIST labels, raw values, not one-hot vectors
    :param batch_size: batch size to use for evaluation
    :return: None
    """
    rng = np.random.RandomState(1)

    basedir = os.path.dirname(os.path.abspath(__file__))
    classifier.load_weights(os.path.join(basedir, 'mnist_cnn.h5'))

    baseline_score = 0
    correct_score = 0
    ssim_score = 0

    N = len(x_test)
    assert N % batch_size == 0, 'N should be divisible by batch_size'
    num_batches = N // batch_size

    for i in range(num_batches):
        imgs_orig = x_test[batch_size * i:batch_size * (i + 1)].astype(K.floatx())
        labels = y_test[batch_size * i:batch_size * (i + 1)]
        # Create corruption masks
        masks = []
        for _ in range(batch_size):
            # Choose square size
            s = rng.randint(7, 15)
            # Choose top-left corner position
            x = rng.randint(0, 29 - s)
            y = rng.randint(0, 29 - s)
            mask = np.zeros(imgs_orig.shape[1:], dtype=np.bool)
            # Set mask area
            mask[y:y + s, x:x + s] = True
            masks.append(mask)
        masks = np.stack(masks)

        # Add channel dimension
        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        imgs_orig = np.expand_dims(imgs_orig, channel_dim)
        masks = np.expand_dims(masks, channel_dim)

        # Generate corrupted versions
        imgs_corrupted = imgs_orig.copy()
        imgs_corrupted[masks] = 1.

        # Generate restored images
        imgs_restored = model.predict_on_batch(imgs_corrupted)

        predicted_labels_orig = classifier.predict_on_batch(_preprocess_for_classifier(imgs_orig)).argmax(axis=-1).astype(labels.dtype)
        predicted_labels_restored = classifier.predict_on_batch(_preprocess_for_classifier(imgs_restored)).argmax(axis=-1).astype(labels.dtype)
        # Calculate classifier score:
        # baseline corresponds to the original samples which the classifier is able to correctly predict
        baseline = labels == predicted_labels_orig
        # Since the classifier is NOT 100% accurate, we ignore the prediction results
        # from the original samples which were misclassified by masking it using the baseline.
        correct = (labels == predicted_labels_restored) & baseline
        baseline_score += int(baseline.sum())
        correct_score += int(correct.sum())

        # Compute SSIM over the uncorrupted pixels
        imgs_orig[masks] = 0.
        imgs_restored[masks] = 0.
        imgs_orig = imgs_orig.squeeze()
        imgs_restored = imgs_restored.squeeze()
        for j in range(batch_size):
            ssim_score += ssim(imgs_orig[j], imgs_restored[j])

    classifier_score = correct_score / baseline_score
    ssim_score /= N

    print('Classifier score: {:.2f}\nSSIM score: {:.2f}'.format(100 * classifier_score, 100 * ssim_score))


if __name__ == '__main__':
    try:
        from keras.datasets import mnist
    except ImportError:
        from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model = IdentityModel()
    x_test = x_test / 255
    test_model(model, x_test, y_test)
