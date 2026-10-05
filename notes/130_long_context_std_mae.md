# 130 - Long context for ongoing (STD-MAE-style branch on the SSL w192 CNN): negative

Hypothesis: every Task 2 member reads 60 minutes, so for ongoing windows it cannot see how long the queue has
lasted or how this run behaved at this place on earlier days; the rules (2026-09-20) allow both.

`occ_lab/longctx.py`: the SSL w192 T2 CNN (p_PS backbone, note 123) plus an STD-MAE (IJCAI 2024) branch -
the same day's 5 h before the short history, the 3 earlier days of the same split at T-180 .. T+115, and the
queue's age at T-5 (+-2 links max); two dilated-CNN encoders pretrained by masked reconstruction with links hidden
(S-mask) and 12-slot patches hidden (T-mask), ratio 0.25, MAE, train truth, 20k steps; fused as H_short + MLP(...)
with a zero-initialised last layer; 8k fine-tuning steps; blanks of the scored months simulated. Control: the same
fine-tuning with the branch off. Lab: note 106's causal sup+rec adaptation.

| ongoing family IoU, sup+rec (val / priv / mean) | s20 | s21 | s22 |
|---|---|---|---|
| p_PS as shipped | 0.8364 | 0.8422 | 0.8400 |
| control (no branch, same fine-tuning) | 0.8380 | 0.8427 | |
| **long context** | 0.8407 | 0.8433 | 0.8411 |
| long context without the pretraining | 0.8385 | | |

Frozen (no adaptation), long vs control: +0.0011 / +0.0004 (s20 val / priv). ES BCE on the train tail equal
(0.02001 vs 0.02007, 0.01948 vs 0.01944). **The long context carries almost nothing the 60 minutes do not.**

In AU's stack (`occ_lab/lc_stack_eval.py`): long seeds replacing p_PS seeds -0.0002 (early days -0.0011); long at
0.10 / 0.20 of p0 +0.0001 / +0.0002; the control replacing seeds +0.0005. Negative. With note 122 (onset, train run)
this closes the context-length axis for both halves of Task 2.
