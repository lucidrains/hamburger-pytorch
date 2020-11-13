import torch
from torch import nn, einsum
import torch.nn.functional as F
from contextlib import contextmanager
from einops import repeat, rearrange

# helper fn

@contextmanager
def null_context():
    yield

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class NMF(nn.Module):
    def __init__(
        self,
        dim,
        n,
        ratio = 8,
        K = 6,
        eps = 2e-8
    ):
        super().__init__()
        r = dim // ratio

        D = torch.zeros(dim, r).uniform_(0, 1)
        C = torch.zeros(r, n).uniform_(0, 1)

        self.K = K
        self.D = nn.Parameter(D)
        self.C = nn.Parameter(C)

        self.eps = eps

    def forward(self, x):
        b, D, C, eps = x.shape[0], self.D, self.C, self.eps

        # x is made non-negative with relu as proposed in paper
        x = F.relu(x)

        D = repeat(D, 'd r -> b d r', b = b)
        C = repeat(C, 'r n -> b r n', b = b)

        # transpose
        t = lambda tensor: rearrange(tensor, 'b i j -> b j i')

        for k in reversed(range(self.K)):
            # only calculate gradients on the last step, per propose 'One-step Gradient'
            context = null_context if k == 0 else torch.no_grad
            with context():
                C_new = C * ((t(D) @ x) / ((t(D) @ D @ C) + eps))
                D_new = D * ((x @ t(C)) / ((D @ C @ t(C)) + eps))
                C, D = C_new, D_new

        return D @ C

class Hamburger(nn.Module):
    def __init__(
        self,
        *,
        dim,
        n,
        inner_dim = None,
        ratio = 8,
        K = 6
    ):
        super().__init__()
        inner_dim = default(inner_dim, dim)

        self.lower_bread = nn.Conv1d(dim, inner_dim, 1, bias = False)
        self.ham = NMF(inner_dim, n, ratio = ratio, K = K)
        self.upper_bread = nn.Conv1d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        shape = x.shape
        x = x.flatten(2)

        x = self.lower_bread(x)
        x = self.ham(x)
        x = self.upper_bread(x)
        return x.reshape(shape)
