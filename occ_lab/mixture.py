"""A joint head for Task 2: a mixture over queue configurations, and a decision that never assumes independence.

Every Task 2 member we have emits per-cell marginals and the decision takes a top-k set. `occ_lab/
selector_posterior.py` priced what that costs: keeping the posterior's per-cell order and only choosing the right
k per window is worth **+0.0501 (true |Y|) to +0.0762 (best prefix)** on the ongoing family, and `occ_lab/
budget_regressor.py` then showed the right k cannot be predicted from the marginals (CV R2 0.018-0.028, and it
loses at every strength on the scored months). That is what a marginal model must look like: how many links break
together is a property of the JOINT, and marginals do not carry it.

Note 31 reached the same wall analytically - the expected-IoU ratio rule gains +0.05..+0.09 on onset and LOSES on
ongoing, because ongoing cells are strongly correlated and the rule's independent-Bernoulli calculus is wrong.
The prefix structure was never the problem; deriving k from independence was.

So: keep the backbone, replace the head and the decision.

    p(Y | X) = sum_k pi_k(X) prod_cells Bernoulli(y | p_k(X))

K components, each internally independent, the mixture correlated. Sampling a component and then its cells gives
configurations whose |Y| varies the way the truth's does, and whose modes can disagree about WHERE the queue is -
which is the shape note 51 found on the onset misses, where the model is confidently wrong and the true
bottleneck sits a median of 26 links from the chosen one.

The decision is then a Monte Carlo argmax, with no closed form and no independence:

    k* = argmax_k  (1/M) sum_m IoU(top-k by marginal, Y^m)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MixtureHead(nn.Module):
    """K sets of cell logits plus the mixture weights, from a (B, C, S) feature map."""

    def __init__(self, c: int, out: int, k: int = 8):
        super().__init__()
        self.k, self.out = k, out
        self.cell = nn.Linear(c, k * out)
        self.pi = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, k))

    def forward(self, h):                                     # h: (B, S, C)
        B, S, _ = h.shape
        lg = self.cell(h).view(B, S, self.k, self.out).permute(0, 2, 3, 1)   # (B, K, out, S)
        pi = torch.log_softmax(self.pi(h.mean(1)), -1)                        # (B, K)
        return lg, pi


def mixture_nll(lg, pi, y, m):
    """-log sum_k pi_k prod_cells p_k, with the cells masked. lg (B,K,out,S), y/m (B,out,S)."""
    y_ = y.unsqueeze(1)
    m_ = m.unsqueeze(1)
    ll = -(nn.functional.binary_cross_entropy_with_logits(lg, y_.expand_as(lg), reduction="none")
           * m_).sum((2, 3))                                   # (B, K) log-likelihood per component
    return -(torch.logsumexp(pi + ll, -1)).mean()


def mcl_loss(lg, pi, y, m, eps=0.05):
    """Multiple-choice learning. The mixture likelihood degenerates here: a component's log-likelihood is a sum
    over hundreds of cells, so the responsibility softmax saturates, one component takes every window, and the
    head collapses to the marginal it replaced (measured: identical decisions to four decimals). Winner-take-all
    instead updates the component that already fits a window best and leaves the rest a floor weight `eps`, which
    forces them to divide the space. pi is trained to match which component wins."""
    y_, m_ = y.unsqueeze(1), m.unsqueeze(1)
    ce = (nn.functional.binary_cross_entropy_with_logits(lg, y_.expand_as(lg), reduction="none")
          * m_).sum((2, 3)) / m_.sum((2, 3)).clamp(min=1)     # (B, K) mean CE per component
    win = ce.argmin(1)
    w = torch.full_like(ce, eps / max(ce.shape[1] - 1, 1))
    w.scatter_(1, win[:, None], 1.0 - eps)
    return (w.detach() * ce).sum(1).mean() - (pi.gather(1, win[:, None]).squeeze(1)).mean()


def spread(p_k, pi, n, rng):
    """sd of |Y| under the mixture against the same marginals treated as independent - the quantity a joint head
    has to move, since `occ_lab/selector_posterior.py` priced knowing |Y| at +0.0501."""
    joint = sample_configs(p_k, pi, n, rng).reshape(n, -1).sum(1)
    marg = (pi[:, None, None] * p_k).sum(0).ravel()
    ind = (rng.random((n, len(marg))) < marg[None]).sum(1)
    return float(joint.std()), float(ind.std())


def responsibilities(lg, pi, y, m):
    y_, m_ = y.unsqueeze(1), m.unsqueeze(1)
    ll = -(nn.functional.binary_cross_entropy_with_logits(lg, y_.expand_as(lg), reduction="none")
           * m_).sum((2, 3))
    return torch.softmax(pi + ll, -1)


def sample_configs(p_k, pi, n, rng):
    """p_k (K, out, S) probabilities, pi (K,) weights -> (n, out, S) boolean configurations."""
    comp = rng.choice(len(pi), size=n, p=pi / pi.sum())
    u = rng.random((n,) + p_k.shape[1:])
    return u < p_k[comp]


def decide(marg, samples, e, kmax=None):
    """The Monte Carlo prefix rule: take the top k cells by the marginal, with k chosen to maximise the
    expected IoU under the sampled configurations rather than under an independence formula."""
    idx = np.flatnonzero(e.ravel())
    if not len(idx):
        return np.zeros_like(e)
    order = idx[np.argsort(-marg.ravel()[idx])]
    S = samples.reshape(len(samples), -1)[:, order]            # (M, n_cells) in prefix order
    hit = np.cumsum(S, 1).astype(np.float64)                   # |A_k and Y|
    tot = S.sum(1, keepdims=True).astype(np.float64)           # |Y|
    kk = np.arange(1, len(order) + 1)[None, :]
    iou = hit / np.maximum(kk + tot - hit, 1.0)
    mean = iou.mean(0)
    kb = int(np.argmax(mean[:kmax] if kmax else mean)) + 1
    out = np.zeros(e.size, bool)
    out[order[:kb]] = True
    return out.reshape(e.shape)
