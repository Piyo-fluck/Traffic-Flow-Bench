# 113. The degree of freedom S_state cannot see: the correlation of the speed and flow errors

Date 2026-09-24. Code `analysis/t3_qv_error_correlation.py`.

## The argument

`S_state` is two separate RMSEs, one on v and one on q, so it depends only on the MARGINAL accuracy of the two
channels. `S_LWR` is scored on k = q/v, and

    dk/k ~= dq/q - dv/v,    Var(dk/k) = s_q^2 + s_v^2 - 2 rho s_q s_v

so it also depends on **rho, the correlation between the two errors - a quantity S_state is completely blind
to.** Note 112's length weighting failed because it asked the network to reallocate accuracy it does not have.
This asks for no accuracy at all, only that the two errors it is already making point the same way.

At rho = 0 the derivative of Var(dk/k) with respect to rho is -2 s_q s_v, which is not zero, while the marginal
cost of the perturbation is second order. A first-order gain at zero first-order cost.

## Measured: rho is zero

Production predictions, 10 panels, local test block, relative errors at target cells:

| | rho(dv, dq) | sd dv/v | sd dq/q | sd dk/k |
|---|---|---|---|---|
| mean over panels | **+0.003** | 0.0153 | 0.0278 | 0.0310 |

Per panel it runs from -0.012 to +0.013. **The two heads make completely uncorrelated errors**, and the single
member and the 3-seed ensemble are identical on this.

What the density error would be at a higher correlation, marginals held fixed:

| rho | sd(dk/k) | change |
|---|---|---|
| +0.00 | 0.0316 | - |
| +0.20 | 0.0289 | -8.7% |
| +0.40 | 0.0258 | **-18.5%** |
| +0.60 | 0.0222 | **-29.7%** |

E_LWR is proportional to \|e\|, so at rho 0.40 it would fall 0.351 -> 0.286, S_LWR 0.6486 -> 0.714, about
**+0.0065 on the total**; at rho 0.60, about **+0.0104**. Larger than anything else currently on the board.

## Why this is not one of the three losses already tried

| flag | what it does | result |
|---|---|---|
| `M3_KHEAD` | density as a THIRD output head | separate head, does not couple v and q |
| `M3_DNLOSS` | conservation proxy on the density INCREMENT | note 05: lambda 0.05 net 0 |
| `M3_LWR` | the metric's own numerator, also an increment, L1 | note 105: lambda 0.1 lowered S_LWR to 0.640 |

All three go through the increment, which is non-smooth and was unstable. **The density LEVEL loss on k derived
from the two main heads has never been run.** `M3_KRHO=lambda` adds
`lambda * mean((k_hat/k_true - 1)^2)` at target cells, no new inputs or outputs, production behaviour at
lambda 0.

Sweep running at lambda 3 / 10 / 30 against the lambda-0 control (single seed, 25 epochs, same settings):
S_state 0.95091, S_FD 0.98749, S_LWR 0.63641, 0.35s+0.05f+0.10l = 0.44583.

## Result: the correlation will not move (2026-09-24)

Single runs, 25 epochs, seed 0, 10 panels, local test block:

| run | S_state | S_FD | S_LWR | 0.35 s + 0.05 f + 0.10 l | vs control | krho at epoch 25 |
|---|---|---|---|---|---|---|
| lambda 0 (control) | 0.95091 | 0.98749 | 0.63641 | **0.44583** | - | ~0.00096 |
| lambda 3 | 0.94990 | 0.98747 | 0.63041 | 0.44488 | -0.00095 | 0.00095 |
| lambda 10 | 0.94960 | 0.98750 | 0.63490 | 0.44522 | -0.00061 | 0.00093 |
| lambda 30 | 0.94941 | 0.98752 | **0.63777** | 0.44545 | -0.00039 | 0.00092 |

**The decisive number is the last column, not the score.** `krho` is `mean((k_hat/k_true - 1)^2)`, so it is the
quantity the argument is about. At lambda 30 it makes up 46% of the whole training loss, and it falls from about
0.00096 to 0.00092 - **sd(dk/k) 0.0310 -> 0.0303, a 1% reduction where 18% was needed.** The network cannot align
the two errors even when told to care about almost nothing else.

S_state falls monotonically (-0.0015 by lambda 30) and the total is negative at every lambda, improving with
lambda but extrapolating to roughly the control rather than past it. S_LWR is non-monotone (0.6304 at lambda 3,
below the control) and the spread across the three runs is 0.0074, which is the size of plausible seed noise on a
single training run - so the lambda 30 S_LWR gain of +0.0014 should not be read as real.

## Reading

The argument was sound and the measurement behind it stands: rho really is +0.003, S_state really is blind to it,
and the first-order gain really would be worth +0.0065 to +0.0104. What is false is the assumption that rho is
**reachable**. The speed error and the flow error are not two views of one mistake the network could align; they
are separate irreducible components, and a loss pointed straight at their product compresses it by 1%.

This is the ninth independent direction to land in the same place (notes 96-107, 111, 112). The measured
irreducibility of the Task 1 residual now extends to the derived density: sd(dk/k) ~ 0.030 is a floor under
direct optimisation.
