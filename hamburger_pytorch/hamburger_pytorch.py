import torch
from torch import nn, einsum
import torch.nn.functional as F
from contextlib import contextmanager
from einops import rearrange

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
        K = 6
    ):
        super().__init__()
        r = dim // ratio

        D = torch.zeros(dim, r).uniform_(0, 1)
        C = torch.zeros(r, n).uniform_(0, 1)

        self.K = K
        self.D = nn.Parameter(D)
        self.C = nn.Parameter(C)

    def forward(self, x):
        D, C = self.D, self.C

        # x is made non-negative with relu as proposed in paper
        x = F.relu(x)

        C_out = C
        D_out = D

        for k in reversed(range(self.K)):
            # only calculate gradients on the last step, per propose 'One-step Gradient'
            context = null_context if k == 0 else torch.no_grad
            with context():
                C_out = C_out * ((D.t() @ x) / (D.t() @ D @ C))
                D_out = D_out * ((x @ C.t()) / (D @ C @ C.t()))

        return D_out @ C_out

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
