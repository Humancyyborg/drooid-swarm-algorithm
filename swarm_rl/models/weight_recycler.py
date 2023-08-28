import torch
import numpy as np


def estimate_neuron_score(activation):
    """
    Calculates neuron score based on absolute value of activation.
    """
    reduce_axes = list(range(activation.dim() - 1))
    score = torch.mean(torch.abs(activation), dim=reduce_axes)
    # score /= torch.mean(score) + 1e-9

    return score


