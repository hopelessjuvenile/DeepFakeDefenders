import torch
import numpy as np

class PerturbCovarianceTransform:
    def __init__(self, perturbation_level=0.01, prob=0.5):
        self.perturbation_level = perturbation_level
        self.prob = prob

    def __call__(self, cov_matrix):
        if np.random.rand() < self.prob:
            perturbation = torch.randn_like(cov_matrix) * self.perturbation_level
            perturbation_matrix = torch.mm(perturbation, perturbation.T)
            cov_matrix += perturbation_matrix
            cov_matrix = (cov_matrix + cov_matrix.T) / 2
        return cov_matrix