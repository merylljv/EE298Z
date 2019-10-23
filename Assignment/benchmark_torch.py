#!/usr/bin/env python3
import os.path
import numpy as np
import torch
from torch import nn

from .classifier_torch import Net
from .structural_similarity import structural_similarity as ssim


class IdentityModel(nn.Module):
    """Model which simply returns the input"""

    def __init__(self):
        super().__init__()
        self.unused = nn.Linear(1, 1)

    def forward(self, x):
        return x


def _preprocess_for_classifier(x):
    return (x - 0.1307) / 0.3081


def test_model(model, dataset, batch_size=100):
    """Run the benchmarks for the given model

    :param model:
    :param dataset: MNIST test dataset
    :param batch_size: batch size to use for evaluation
    :return: None
    """
    rng = np.random.RandomState(1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    basedir = os.path.dirname(os.path.abspath(__file__))
    classifier = Net()
    classifier.load_state_dict(torch.load(os.path.join(basedir, 'mnist_cnn.pt'), map_location=device))
    classifier.to(device)
    classifier.eval()

    baseline_score = 0
    correct_score = 0
    ssim_score = 0

    N = len(dataset)
    assert N % batch_size == 0, 'N should be divisible by batch_size'
    num_batches = N // batch_size

    for i in range(num_batches):
        imgs_orig = dataset.data[batch_size * i:batch_size * (i + 1)].to(device).type(torch.float) / 255.
        labels = dataset.targets[batch_size * i:batch_size * (i + 1)].to(device)
        # Create corruption masks
        masks = []
        for _ in range(batch_size):
            # Choose square size
            s = rng.randint(7, 15)
            # Choose top-left corner position
            x = rng.randint(0, 29 - s)
            y = rng.randint(0, 29 - s)
            mask = torch.zeros(imgs_orig.shape[1:], dtype=torch.bool)
            # Set mask area
            mask[y:y + s, x:x + s] = True
            masks.append(mask)
        masks = torch.stack(masks).to(device)

        # Add channel dimension
        imgs_orig.unsqueeze_(1)
        masks.unsqueeze_(1)

        # Generate corrupted versions
        imgs_corrupted = imgs_orig.clone()
        imgs_corrupted[masks] = 1.

        # Generate restored images
        model_device = next(model.parameters()).device
        imgs_restored = model(imgs_corrupted.to(model_device)).to(device)

        predicted_labels_orig = classifier(_preprocess_for_classifier(imgs_orig)).argmax(dim=-1).type_as(labels)
        predicted_labels_restored = classifier(_preprocess_for_classifier(imgs_restored)).argmax(dim=-1).type_as(labels)
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
        imgs_orig = imgs_orig.squeeze().cpu().numpy()
        imgs_restored = imgs_restored.squeeze().cpu().numpy()
        for j in range(batch_size):
            ssim_score += ssim(imgs_orig[j], imgs_restored[j])

    classifier_score = correct_score / baseline_score
    ssim_score /= N

    print('Classifier score: {:.2f}\nSSIM score: {:.2f}'.format(100 * classifier_score, 100 * ssim_score))


if __name__ == '__main__':
    from torchvision import datasets
    data = datasets.MNIST('../data', train=False, download=True)
    model = IdentityModel()
    test_model(model, data, batch_size=100)
