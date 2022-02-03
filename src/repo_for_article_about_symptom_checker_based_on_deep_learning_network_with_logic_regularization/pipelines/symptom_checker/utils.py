import torch
import torch.nn as nn
import torch.nn.functional as F



def one_hot_diff(logits: torch.FloatTensor, mode: str, threshold: float = 0.5) -> torch.FloatTensor:

    assert mode in ['sigmoid', 'softmax']
    assert logits.ndim == 2

    if mode == 'sigmoid':
        probs = torch.sigmoid(logits)
        oh_mask = torch.where(probs >= threshold, 1, 0).to(probs.device)
    elif mode == 'softmax':
        probs = torch.softmax(logits, dim=1)
        oh_mask = F.one_hot(probs.argmax(dim=1), num_classes=probs.shape[1])

    # binnarization trick (like in Gumbel-Softmax)
    oh_diff = oh_mask - probs.detach() + probs

    return oh_diff



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
