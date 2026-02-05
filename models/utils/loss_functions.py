"""
This file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/tree/master/models/utils
"""

import numpy as np
import torch
from torch.nn import functional as F


##-------------------------------------------------------------------------------------------------------------------##

####################
## Loss functions ##
####################

def loss_fn_kd(scores, target_scores, T=2., weights=None):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be <2D-tensors>, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # If [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    n = scores.size(1)
    if n > target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -targets_norm * log_scores_norm

    # Sum over the prob-scores of all classes (1) and then average over all elements in the batch (2)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # -> sum over classes
    KD_loss_unnorm = weighted_average(KD_loss_unnorm, weights=weights, dim=0)  # -> average over batch

    # Normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


##-------------------------------------------------------------------------------------------------------------------##

######################
## Helper functions ##
######################

def weighted_average(tensor, weights=None, dim=0):
    '''Computes weighted average of [tensor] over dimension [dim].'''
    if weights is None:
        mean = torch.mean(tensor, dim=dim)
    else:
        batch_size = tensor.size(dim) if len(tensor.size()) > 0 else 1
        assert len(weights) == batch_size
        # sum_weights = sum(weights)
        # norm_weights = torch.Tensor([weight/sum_weights for weight in weights]).to(tensor.device)
        norm_weights = torch.tensor([weight for weight in weights]).to(tensor.device)
        mean = torch.mean(norm_weights * tensor, dim=dim)
    return mean


def to_one_hot(y, classes, device=None):
    '''Convert <nd-array> or <tensor> with integers [y] to a 2D "one-hot" <tensor>.'''
    if type(y) == torch.Tensor:
        device = y.device
        y = y.cpu()
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c if device is None else c.to(device)


##-------------------------------------------------------------------------------------------------------------------##

########################################################
## Calculate log-likelihood for various distributions ##
########################################################

def log_Normal_standard(x, mean=0, average=False, dim=None):
    '''Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=I.
    NOTES: [dim]=-1    summing / averaging over all but the first dimension
           [dim]=None  summing / averaging is done over all dimensions'''
    log_normal = -0.5 * torch.pow(x - mean, 2)
    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1
    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)


# def log_Normal_diag(x, mean, log_var, average=False, dim=None):
#     '''Calculate log-likelihood of sample [x] under Gaussian distribution(s) with mu=[mean], diag_var=exp[log_var].
#     NOTES: [dim]=-1    summing / averaging over all but the first dimension
#            [dim]=None  summing / averaging is done over all dimensions'''

#     log_var[torch.isnan(log_var)] = torch.mean(log_var[~torch.isnan(log_var)])  # TODO: NaN handling
#     log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
#     log_normal[torch.isnan(log_normal)] = torch.mean(log_normal[~torch.isnan(log_normal)])  # TODO: NaN handling
#     if torch.isnan(log_normal).any().item():
#         raise ValueError("NaN is encountered")
#     if dim is not None and dim == -1:
#         log_normal = log_normal.view(log_normal.size(0), -1)
#         dim = 1
#     if average:
#         return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
#     else:
#         return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)
def safe_sanitize(x, nan=0.0, clamp_min=-1e3, clamp_max=1e3):
    # Replace NaNs (autograd-safe)
    x = torch.where(torch.isnan(x), torch.full_like(x, nan), x)
    # Clamp handles +/-inf and huge magnitudes (autograd-safe)
    x = torch.clamp(x, min=clamp_min, max=clamp_max)
    return x


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    """
    Stable log N(x | mean, diag(exp(log_var))) compatible with older PyTorch autograd.
    """

    # sanitize inputs (avoid NaNs and extreme values)
    x = safe_sanitize(x, nan=0.0, clamp_min=-1e3, clamp_max=1e3)
    mean = safe_sanitize(mean, nan=0.0, clamp_min=-1e3, clamp_max=1e3)
    log_var = safe_sanitize(log_var, nan=0.0, clamp_min=-20.0, clamp_max=20.0)

    # clamp log-variance to prevent exp overflow / underflow
    log_var = torch.clamp(log_var, min=-8.0, max=8.0)

    # stable form: use exp(-log_var)
    inv_var = torch.exp(-log_var)
    diff2 = (x - mean).pow(2)

    log_normal = -0.5 * (log_var + diff2 * inv_var)

    # final sanitize
    log_normal = safe_sanitize(log_normal, nan=0.0, clamp_min=-1e3, clamp_max=1e3)

    if dim is not None and dim == -1:
        log_normal = log_normal.view(log_normal.size(0), -1)
        dim = 1

    if average:
        return torch.mean(log_normal, dim) if dim is not None else torch.mean(log_normal)
    else:
        return torch.sum(log_normal, dim) if dim is not None else torch.sum(log_normal)
