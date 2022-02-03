import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from https://github.com/yandexdataschool/gumbel_lstm/blob/master/gumbel_sigmoid.py
def gumbel_sigmoid(
        logits,
        temperature=1.0,
        hard=False,
        eps=1e-20
    ):
    """
    A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
    In short, it's a function that mimics #[a>0] indicator where a is the logit
    Explaination and motivation: https://arxiv.org/abs/1611.01144
    Math:
    Sigmoid is a softmax of two logits: a and 0
    e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
    Gumbel-sigmoid is a gumbel-softmax for same logits:
    gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
    where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
    gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
    gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
    
    For computation reasons:
    gumbel1-gumbel2 = -log(-log(uniform1(0,1)) + log(-log(uniform2(0,1)) = log( log(uniform2(0,1)) / log(uniform1(0,1)) )
    gumbel_sigm(a) = sigm([ a - log(log(uniform2(0,1)) / log(uniform1(0,1))) ] / t)
    Args:
        logits: [batch_size, ] unnormalized log-probs
        temperature: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        eps: a small number used for numerical stability
    Returns:
        a callable that can (and should) be used as a nonlinearity
    
    """

    assert temperature != 0

    # computes a gumbel softmax sample

    #sample from Gumbel(0, 1)
    uniform1 = torch.rand(logits.shape, device=logits.device)
    uniform2 = torch.rand(logits.shape, device=logits.device)

    noise = torch.log(
        torch.log(uniform2 + eps) / torch.log(uniform1 + eps) + eps
    )

    #draw a sample from the Gumbel-Sigmoid distribution
    y = torch.sigmoid((logits + noise) / temperature)

    if not hard:
        return y

    y_hard = torch.zeros_like(y)  # [Batchsize,]
    y_hard[y>0.5] = 1
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))