import numpy as np
import torch


class PermManager:

    # helper class for batching
    def __init__(self, n, batch_size, rng: np.random.RandomState,
                 perm=None, perm_index=0, epoch=0):
        self.n = n
        self.batch_size = batch_size
        self._rng = rng
        if perm is None:
            self.perm = self._rng.permutation(self.n)
        else:
            self.perm = perm
        self.perm_index = perm_index
        self.epoch = epoch

    def get_indices(self):
        indices = np.zeros(shape=(self.batch_size,), dtype=np.int32)
        n_stored = 0
        while n_stored < self.batch_size:
            # Take either to end of batch or end of epoch, whichever comes first
            n_to_take = min(self.batch_size - n_stored, self.n - self.perm_index)
            indices[n_stored: n_stored + n_to_take] = \
                self.perm[self.perm_index: self.perm_index + n_to_take]
            # Update these to reflect elements taken from this epoch
            self.perm_index += n_to_take
            n_stored += n_to_take
            # If this assertion is violated this method must be incorrect
            assert self.perm_index <= self.n
            # Reshuffle indices and reset perm_index if we have reached the end of an epoch
            if self.perm_index == self.n:
                self.perm_index = 0
                self.epoch += 1
                self.perm = self._rng.permutation(self.n)

        # If this is violated then this method is incorrect
        assert n_stored == self.batch_size

        return indices

def stable_log(x, eps=10e-6):
    # logs are slightly modified for numerical stability
    # return torch.log(torch.clamp(x, min=eps))
    return torch.log(torch.add(x, eps))


def sb(x, device):
    """
    Applies the 'stick-breaking' procedure to a vector, x, of reals in (0, 1).
    Produces a vector that sums to < 1: (x[0], (1-x[0])*x[1], (1-x[0])*(1-x[1])*x[2], ... )
    """
    cum_prod = torch.cumprod(1. - x[:, :-1], dim=-1)
    ones_shape = list(x.shape)
    ones_shape[-1] = 1
    cum_prod = torch.cat([torch.ones(ones_shape).to(device), cum_prod], axis=-1)
    return x * cum_prod


def check_new_option(x, tol=0.5):
    """
    Check if the cumulative sum of probas in x reaches tol.
    """
    # with torch.no_grad():
    #     cum_sum = torch.cumsum(x, dim=0)
    #     ind_vector = cum_sum < tol
    #     return ind_vector[-1]
    if min(x) < tol/len(x):
        return False
    return True


def relaxed_delta_binary(b):
    """relaxes delta(b=1) from binary b to b in (0, 1) as delta(b=1) ~ b"""
    return b


def relaxed_policy_eval(eta, h):
    """
    relaxes evaluating a discrete distribution eta at an index h (thought of as a one-hot) to when h lives in the
    simplex along the last dimension as eta(index(h)) ~ sum_i eta_i * h_i
    """
    return torch.sum(eta * h, axis=-1, keepdim=True)


def relaxed_delta_one_hot(h1, h2):
    """
    relaxes delta(h1=h2) from one-hot h1 and h2 to simplex-valued h1 and h2 along the last dimension as
    delta(h1=h2) ~ 1 - ||h1-h2||_1 / 2
    """
    return 1. - torch.sum(torch.abs(h1 - h2), axis=-1, keepdim=True) / 2.


def perfect_encoder(batch, K, device):
    """
    hand-crafted encoder for line env that outputs perfect zeta.
    """
    enc = torch.zeros((batch.shape[0],batch.shape[1], K+1))
    enc += -10
    for i, traj in enumerate(batch):
        for j, sa in enumerate(traj):
            if sa[-1] == 1:
                enc[i, j, 1] = 10
            else:
                enc[i, j, 0] = 10
            if sa[0] == 0 or sa[0] == 9:
                enc[i, j, -1] = 10
    return enc.to(device)
