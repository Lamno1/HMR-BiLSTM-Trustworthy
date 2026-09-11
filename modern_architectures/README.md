# Modern Architecture Benchmark (isolated experiment)

**Scope:** compares 4 backbones under the *identical* training protocol (focal loss,
class weights, cosine LR, FGSM adversarial training, grad clip) on the *same*
`data/processed/{train,val,test}.npz` splits used by the paper's official baseline.
Only the sequence-modeling backbone changes between runs — everything else is held
fixed so any F1 difference is attributable to the backbone, not to protocol drift.

This folder does **not** read or write anything outside itself except read-only
access to `../data/processed/*.npz` and `../data/processed/class_weights.npy`.
It does not touch `../results/`, `../hmr_bilstm.py`, or any paper-official output.

## Why this exists

Literature search (2023-2026, inter-patient MIT-BIH beat classification) found:
- No paper demonstrates Transformer/Mamba beating BiLSTM at this sequence length
  (~187 samples/beat) — SSM/Transformer wins in ECG literature are all at
  10s/12-lead rhythm scale, a different regime.
- Field-wide, F-class (fusion beat) F1 remains largely unsolved regardless of
  backbone (a 2026 paper purpose-built for it scored F1(F)=0.066, worse than
  this project's CNN+BiLSTM baseline of 0.296).

So this benchmark exists to test that empirically rather than assume it — see
results below once run.

## Backbones

| Model             | File          | Description                                                   |
|-------------------|---------------|-----------------------------------------------------------------|
| `bilstm`           | `models.py`   | Same CNN + ResMemECG as the paper baseline, re-run in this harness for an apples-to-apples control |
| `cnn_transformer`  | `models.py`   | Same CNN front-end, BiLSTM replaced by a TransformerEncoder      |
| `cnn_xlstm`        | `models.py`   | Same CNN front-end, BiLSTM replaced by bidirectional sLSTM (Beck et al. 2024, exponential gating + log-space stabilizer) |
| `pure_transformer` | `models.py`   | No CNN — raw waveform patch-embedded directly into a TransformerEncoder |
| ~~`cnn_mamba`~~    | `models.py`   | **Excluded from the main benchmark** — see below                |

`cnn_xlstm` implements only the single-head scalar-memory **sLSTM** variant from
the xLSTM paper (exponential input gate, sigmoid forget gate, `m_t`/`n_t`
stabilizer+normalizer state) — not the full paper's multi-head sLSTM + matrix-
memory mLSTM block stack. Chosen as the variant structurally closest to a
direct "modernized LSTM gating" comparison against ResMemECG's own custom
gating. Smoke-tested (3 epochs, real inter-patient data): F1-macro=0.481,
~300-355s/epoch (~2x bilstm's per-epoch cost, but full-run feasible, unlike
cnn_mamba).

### Why `cnn_mamba` is excluded

Code is implemented and passes smoke tests (forward/backward + 3-epoch training,
no crashes), but excluded from the multi-seed comparison:

- The pure-PyTorch loop-based S6 scan (no `mamba-ssm` CUDA kernel — that package
  doesn't install cleanly on Windows) is ~80x slower per epoch than
  `cnn_transformer` (≈1180s vs ≈15s/epoch on this GPU). A full 45-epoch run
  would take ≈15h *per seed*; multi-seed validation would take days.
- At matched epoch count (3), it scored lowest of all four backbones on val
  F1-macro (0.295 vs 0.42-0.52), with no early sign of a "slow starter" pattern
  that would justify the extra compute.
- No literature evidence was found (see literature-search notes in the parent
  conversation) that Mamba/SSM backbones beat BiLSTM at this sequence length
  (~187 samples/beat) — the theoretical advantage of SSMs (linear-time long-range
  modeling) is aimed at sequences orders of magnitude longer than this.

Net assessment: low expected payoff (~15-25%, qualitative) for the engineering
cost of a proper chunked/parallel-scan rewrite. Decision: drop from the
headline comparison; code stays in `models.py` as a tried-and-documented
negative result, not deleted.

## Usage

```
python train_modern.py --model bilstm --seed 42 --out-dir results/bilstm_s42
python train_modern.py --model cnn_transformer --seed 42 --out-dir results/cnn_transformer_s42
python train_modern.py --model pure_transformer --seed 42 --out-dir results/pure_transformer_s42

# or run the full multi-seed benchmark (bilstm, cnn_transformer, pure_transformer only):
python run_benchmark.py --seeds 42 43 44 --epochs 45
```

All commands default to `../data/processed/splits` with the `inter_` prefix
(patient-disjoint protocol — do not switch to `intra_`, it is leakage-inflated).

## Status

Prototype stage — not multi-seed validated yet. Do not treat any single-seed
result here as conclusive (see `[[feedback_research_rigor]]`: this project's
minority-class F1 is highly seed-volatile).
