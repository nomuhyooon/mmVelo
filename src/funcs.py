import math
import torch
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
import numpy as np
from torch.distributions import Distribution
from scipy.special import iv

def kl_divergence(d1, d2, K=100):
    """Computes closed-form KL if available, else computes a MC estimate."""
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)

def convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    """Converts negative binomial mean and dispersion to count and logit parametrization."""
    # theta is inverse dispersion parameter.
    # because the var can be foramulated as var[x] = E[x] - E[x]^2/r
    # where 1/r is the dispersion parameter.
    logits = torch.log(mu+eps) - torch.log(theta)
    counts = theta
    return counts, logits

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))

class ZINB_logits(ZeroInflatedNegativeBinomial):
    def __init__(self, total_count, logits, gate_logits):
        super(ZINB_logits, self).__init__(total_count=total_count, logits=logits, gate_logits=gate_logits, validate_args=False)


"""
def iv(v, z):
    # Compute the modified Bessel function of the first kind using the asymptotic expansion
    # The asymptotic expansion is accurate for large v and large z
    # We need to handle the case when z is less than or equal to zero separately, since the
    # asymptotic expansion is not valid for z <= 0
    print(v)
    if z <= 0:
        return torch.tensor(float('inf'))
    else:
        # Compute the first term in the series expansion
        term = torch.exp(z) / torch.sqrt(2 * np.pi * z)

        # Compute the remaining terms in the series expansion using a recursive formula
        total = term
        for k in range(1, 100):
            term *= ((4 * v ** 2 - 1) / (8 * z ** 2 * k) - 1 / (2 * z)) * term
            total += term
            if torch.abs(term) < torch.finfo(torch.float32).eps:
                break
        return total

def iv(v, z):
    # Compute the modified Bessel function of the first kind using the power series expansion
    # This implementation is accurate for small and large values of v and z
    term = torch.tensor(1.)
    total = term
    for k in range(1, 1000):
        term *= z / (2 * (v + k))
        total += term ** 2
        if torch.abs(term) < torch.finfo(torch.float32).eps:
            break
    return total
"""

def iv_approx(v, z):
    # Compute the leading term and the sum of the remaining terms in the square brackets
    leading = torch.exp(z) / torch.sqrt(2 * np.pi * v)
    sum_remaining = torch.tensor(0.0, dtype=z.dtype, device=z.device)
    term = 1 / (8 * v)
    for i in range(1, 5):
        sum_remaining += term
        term *= -(2*i-1) / (16 * v)

    # Combine the terms and return the result
    return leading * (1 - sum_remaining)





class VonMisesFisher(Distribution):
    def __init__(self, mu, kappa):
        super().__init__()
        self.mu = mu
        self.kappa = kappa.view(-1)
        self.dim = mu.shape[-1]

    def log_prob(self, x, eps=1e-8):
        v = torch.tensor(self.dim / 2 - 1)
        l2_norm = torch.norm(x, dim=1)

        x = x / l2_norm.view(-1,1)        
        z = self.kappa * torch.sum(self.mu * x, dim=-1)

        norm_const = iv_approx(v, z)
        log_prob = torch.log(norm_const+eps) + z

        # to do
        log_prob = torch.where(torch.isinf(log_prob), 0., log_prob)
        return log_prob


    def entropy(self):
        raise NotImplementedError()

    def rsample(self, sample_shape=torch.Size()):
        # Not implemented as the von Mises-Fisher distribution does not have a straightforward way to generate samples
        raise NotImplementedError()

    def sample(self, sample_shape=torch.Size()):
        # Not implemented as the von Mises-Fisher distribution does not have a straightforward way to generate samples
        raise NotImplementedError()
