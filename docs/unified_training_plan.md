# Unified HRM–TensorLNN–RWKV Training Notes

## Tensor Shapes

- Hidden state `h_t`: `(batch, d_h)` where `d_h = hidden_size`.
- Temporal cache `time_state_t`: `(batch, d_h)` mirrors `h_t` for RWKV decay.
- Relational tensor `R_t`: `(batch, S, K, d_r)` for `S` predicate slots, `K` roles, and role width `d_r`.
- Causal mask `C_t`: `(batch, S, S)` normalized row-wise to gate predicate influence.
- Abstraction stack `A_t = {a_t^(ℓ)}`: each `a_t^(ℓ)` is `(batch, d_ℓ)`; first layer satisfies `d_0 = d_h`.

## Forward Pass Summary

1. **RWKV update** – mixes input `x_t` and prior hidden `h_t` with learned decay and mix gates to produce `h̃_t` and `time_state_{t+1}`.
2. **Tensor binding** – projects `h̃_t` into new predicate-role tensors and causal masks, then blends them with previous `R_t` and `C_t` via learnable mixing ratios.
3. **HRM reduction** – aggregates relational context using `C_t`, compresses it with `h̃_t`, and pushes the result through stacked abstraction layers to emit `h_{t+1}` and updated abstractions.

The fused kernel `UnifiedCell` executes these steps sequentially without cross-module synchronization. Setting `debug_mode=True` exposes `RWKVBlock`, `TensorBindBlock`, and `HRMReducer` for isolated unit tests.

## Training Objectives

- **Primary sequence loss**: Task-specific criterion (e.g., next-token cross-entropy, regression) computed on predictions derived from `h_t`.
- **Symbolic consistency**: Encourage accurate tensor bindings via cross-entropy or contrastive loss on relational labels where available.
- **Hierarchical sparsity**: Apply L1 or entropy penalties on abstraction activations to prevent redundancy.
- **Optional abstraction heads**: Attach lightweight predictors to selected `a_t^(ℓ)` layers and supervise intermediate reasoning targets.
- **Relational fidelity metric**: Measure reconstruction accuracy of predicates decoded from `R_t`; incorporate as auxiliary loss if ground-truth relations exist.

## Dataset Usage

- `Data/GSM8K` – arithmetic and reasoning sequences for supervising the primary loss.
- `Data/CoT` – chain-of-thought exemplars suited for abstraction-head supervision.
- `Data/HRDoc` – long-form documents for stress-testing hierarchical compression; sample spans for contextual pretraining.
- `Data/wikipedia` – broad knowledge source for representative language patterns.
- `Data/LL` and `Data/complete_datasets` – leverage structured or lexical resources to derive relational labels for symbolic consistency losses.

## Training Loop Tips

- Initialize states via `UnifiedCell.initial_state(batch, device)`; masks start as identity matrices.
- Use automatic mixed precision to reduce memory pressure; checkpoint between blocks when sequences are long.
- Start with `debug_mode=True` to validate each block individually, then switch off for production runs.
- Schedule higher weighting on symbolic and sparsity losses late in training to avoid destabilizing early convergence.
- Track abstraction-layer metrics (activation norms, prediction accuracy) to verify hierarchical behavior.

## Reference Script

- `src/prernk/training.py` exposes `train_text_corpus` and a CLI (`python -m prernk.training`) that streams the unified cell through a simple text dataset.
- Default corpus leverages `Data/LL/Random English Sentences.txt`; override with `--data path1 path2` to point at other resources (e.g., curated subsets of `GSM8K` or `HRDoc`).
- Adjust model hyperparameters by editing `TrainingConfig` or extending the CLI; the wrapper automatically builds `UnifiedConfig` for the cell.

