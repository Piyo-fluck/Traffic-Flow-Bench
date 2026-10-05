"""Structured link identity for the onset STID (ST-MLP / STEMLP, adapted to our corridors).

Our Task 2 panels are mainline corridors: the links form a chain ordered by milepost (ramps are separate data and
already enter as static input channels). So the "graph" here is a path, and its normalized-Laplacian eigenvectors
reduce to a cosine basis along the corridor - a positional encoding rather than network structure. That is still
something the current onset STID does not have: its identity is a free per-link embedding with no notion of
where a link sits or which links are next to it.

    adj(ctx)    the row-normalised adjacency of the chain, used for ST-MLP's A B (neighbourhood identity)
    lappe(ctx)  the k lowest non-trivial eigenvectors of L = I - D^-1/2 A D^-1/2 (STEMLP's spectral identity)
"""
from __future__ import annotations

import numpy as np


def order(ctx):
    """Link order along the corridor.

    Mileposts give the physical order, but `common.windows` strips them from ctx.z to save memory. Checked on
    every panel: the released link order already equals the milepost order, so the file order is the fallback and
    the two paths agree.
    """
    mp = ctx.z.get("milepost") if isinstance(ctx.z, dict) else None
    if mp is None:
        return np.arange(ctx.S)
    return np.argsort(np.nan_to_num(np.asarray(mp, float), nan=np.inf), kind="stable")


def adj(ctx) -> np.ndarray:
    """(S, S) symmetric chain adjacency with self-loops, row-normalised."""
    S = ctx.S
    o = order(ctx)
    A = np.zeros((S, S), np.float32)
    A[o[:-1], o[1:]] = 1.0
    A[o[1:], o[:-1]] = 1.0
    A[np.arange(S), np.arange(S)] = 1.0
    return A / A.sum(1, keepdims=True)


def lappe(ctx, k: int = 12) -> np.ndarray:
    """(S, k) Laplacian positional encoding, the k lowest non-trivial eigenvectors, unit-scaled."""
    S = ctx.S
    o = order(ctx)
    A = np.zeros((S, S), np.float64)
    A[o[:-1], o[1:]] = 1.0
    A[o[1:], o[:-1]] = 1.0
    d = np.maximum(A.sum(1), 1e-6)
    L = np.eye(S) - A / np.sqrt(np.outer(d, d))
    w, v = np.linalg.eigh(L)
    v = v[:, np.argsort(w)][:, 1:k + 1]                      # drop the constant eigenvector
    if v.shape[1] < k:                                       # a very short corridor
        v = np.pad(v, ((0, 0), (0, k - v.shape[1])))
    v = v / np.maximum(np.abs(v).max(0, keepdims=True), 1e-6)
    return v.astype(np.float32)
