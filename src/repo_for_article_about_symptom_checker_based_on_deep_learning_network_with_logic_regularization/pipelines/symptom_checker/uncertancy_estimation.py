import numpy as np
import torch
from tqdm import tqdm



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = torch.softmax(logits, -1) + 1e-8
    entropy = - probs * torch.log2(probs)
    entropy = torch.sum(entropy, -1)
    return entropy


def compute_entropy_with_norm(logits: torch.Tensor) -> torch.Tensor:
    """Compute normalized Shennon entropy according to the definition.

    Args:
        logits (torch.Tensor): Unscaled log probabilities.

    Returns:
        torch.Tensor: A tensor containing normilized the Shannon entropy in the last dimension.
    """

    entropy = compute_entropy(logits)

    even_distribution_logits = torch.ones(logits.shape[-1], device=logits.device)
    max_entropy = compute_entropy(even_distribution_logits)

    norm_entropy = entropy / max_entropy
    
    return norm_entropy
