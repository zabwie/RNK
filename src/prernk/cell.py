"""Unified differentiable cell combining HRM, TensorLNN, and RWKV principles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass
class UnifiedConfig:
    """Configuration for the unified recurrent cell.

    Args:
        input_size: Dimensionality of the external input `x_t`.
        hidden_size: Dense hidden dimensionality `d_h` propagated between steps.
        symbol_slots: Number of symbolic predicates tracked in the relational tensor `S`.
        role_slots: Number of roles per predicate (`K`).
        relation_size: Dimensionality of each role embedding (`d_r`).
        abstraction_sizes: Ordered list of abstraction layer sizes. The first entry MUST
            equal ``hidden_size`` so that the lowest abstraction rewrites ``h_t``.
        activation: Name of the elementwise activation used in the RWKV candidate path.
        debug_mode: When ``True`` exposes submodules separately to aid unit testing.
    """

    input_size: int
    hidden_size: int
    symbol_slots: int
    role_slots: int
    relation_size: int
    abstraction_sizes: Sequence[int]
    activation: str = "gelu"
    debug_mode: bool = False

    def __post_init__(self) -> None:
        if not self.abstraction_sizes:
            raise ValueError("abstraction_sizes must contain at least one entry")
        if self.abstraction_sizes[0] != self.hidden_size:
            raise ValueError(
                "The first abstraction size must match hidden_size to rewrite h_t"
            )


@dataclass
class UnifiedState:
    """Container for recurrent state.

    Attributes:
        h: Dense hidden tensor with shape ``(B, d_h)``.
        time_state: Cached temporal accumulator with shape ``(B, d_h)`` used by the
            RWKV-inspired time mixing mechanism.
        relations: Tensor capturing bound predicates with shape
            ``(B, S, K, d_r)``.
        causal_mask: Dense causal mask with shape ``(B, S, S)`` constraining which
            predicates influence each other.
        abstractions: Tuple of abstraction tensors ``a_t^{(ℓ)}`` where each entry has
            shape ``(B, d_ℓ)`` matching ``abstraction_sizes``.
    """

    h: Tensor
    time_state: Tensor
    relations: Tensor
    causal_mask: Tensor
    abstractions: Tuple[Tensor, ...]


class RWKVBlock(nn.Module):
    """RWKV-inspired update mixing current input with the hidden trace."""

    def __init__(self, config: UnifiedConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        merged_dim = config.input_size + config.hidden_size
        self.norm = nn.LayerNorm(merged_dim)
        self.proj = nn.Linear(merged_dim, 3 * config.hidden_size)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        if config.activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif config.activation.lower() == "silu":
            self.activation = nn.SiLU()
        elif config.activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")

        # Learnable time decay and mixing logits per hidden channel.
        self.time_decay_logits = nn.Parameter(torch.full((config.hidden_size,), -1.0))
        self.time_mix_logits = nn.Parameter(torch.zeros(config.hidden_size))
        with torch.no_grad():
            self.time_decay_logits.add_(0.01 * torch.randn_like(self.time_decay_logits))
            self.time_mix_logits.add_(0.01 * torch.randn_like(self.time_mix_logits))

    def forward(self, state: UnifiedState, x_t: Tensor) -> Tuple[Tensor, Tensor]:
        h_prev = state.h
        assert h_prev.shape == state.time_state.shape, "hidden and time_state shapes mismatch"
        assert x_t.shape[0] == h_prev.shape[0], "batch mismatch between input and state"

        h_tilde, time_state_next, _ = self.forward_debug(state, x_t)
        return h_tilde, time_state_next

    def forward_debug(
        self, state: UnifiedState, x_t: Tensor
    ) -> Tuple[Tensor, Tensor, dict[str, Tensor]]:
        h_prev = state.h
        concat = torch.cat([x_t, h_prev], dim=-1)
        concat = self.norm(concat)
        k_raw, v_raw, candidate_raw = self.proj(concat).chunk(3, dim=-1)

        k = torch.sigmoid(k_raw)
        v = torch.sigmoid(v_raw)
        candidate = self.activation(candidate_raw)

        h_tilde = k * h_prev + v * candidate

        time_decay = torch.sigmoid(self.time_decay_logits).unsqueeze(0)
        time_mix = torch.sigmoid(self.time_mix_logits).unsqueeze(0)

        time_state_next = time_decay * state.time_state + (1.0 - time_decay) * h_prev
        h_mixed = time_mix * h_tilde + (1.0 - time_mix) * time_state_next

        debug = {
            "k": k,
            "v": v,
            "candidate": candidate,
            "h_tilde": h_tilde,
            "time_decay": time_decay,
            "time_mix": time_mix,
        }

        return h_mixed, time_state_next, debug


class TensorBindBlock(nn.Module):
    """Tensorized symbol-to-relation binding with causal mask generation."""

    def __init__(self, config: UnifiedConfig) -> None:
        super().__init__()
        self.symbol_slots = config.symbol_slots
        self.role_slots = config.role_slots
        self.relation_size = config.relation_size
        hidden = config.hidden_size

        self.binding_proj = nn.Linear(hidden, self.symbol_slots * self.role_slots * self.relation_size)
        self.mask_proj = nn.Linear(hidden, self.symbol_slots * self.symbol_slots)
        self.rel_mix_logits = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.mask_mix_logits = nn.Parameter(torch.zeros(1, 1, 1))

        nn.init.xavier_uniform_(self.binding_proj.weight)
        nn.init.zeros_(self.binding_proj.bias)
        nn.init.xavier_uniform_(self.mask_proj.weight)
        nn.init.zeros_(self.mask_proj.bias)
        with torch.no_grad():
            self.rel_mix_logits.add_(0.01 * torch.randn_like(self.rel_mix_logits))
            self.mask_mix_logits.add_(0.01 * torch.randn_like(self.mask_mix_logits))

        self.rel_norm = nn.LayerNorm(self.relation_size)
        self.mask_norm = nn.LayerNorm(self.symbol_slots)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, h_tilde: Tensor, state: UnifiedState) -> Tuple[Tensor, Tensor]:
        relations, causal_mask, _ = self.forward_debug(h_tilde, state)
        return relations, causal_mask

    def forward_debug(
        self, h_tilde: Tensor, state: UnifiedState
    ) -> Tuple[Tensor, Tensor, dict[str, Tensor]]:
        batch = h_tilde.shape[0]

        rel_raw = self.binding_proj(h_tilde)
        relations_new = rel_raw.view(
            batch,
            self.symbol_slots,
            self.role_slots,
            self.relation_size,
        )
        relations_new = self.rel_norm(relations_new)
        relations_new = self.dropout(relations_new)

        mask_logits = self.mask_proj(h_tilde)
        causal_mask_new = mask_logits.view(batch, self.symbol_slots, self.symbol_slots)
        causal_mask_new = self.mask_norm(causal_mask_new)
        causal_max = causal_mask_new.max(dim=-1, keepdim=True).values
        causal_mask_new = torch.softmax(causal_mask_new - causal_max, dim=-1)

        rel_mix = torch.sigmoid(self.rel_mix_logits)
        mask_mix = torch.sigmoid(self.mask_mix_logits)

        relations = rel_mix * relations_new + (1.0 - rel_mix) * state.relations
        causal_mask = mask_mix * causal_mask_new + (1.0 - mask_mix) * state.causal_mask

        debug = {
            "relations_new": relations_new,
            "causal_mask_new": causal_mask_new,
            "rel_mix": rel_mix,
            "mask_mix": mask_mix,
        }

        return relations, causal_mask, debug


class HRMLayer(nn.Module):
    """Single abstraction layer used within the HRM reducer."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        merged_dim = input_size + hidden_size
        self.norm = nn.LayerNorm(merged_dim)
        self.update = nn.Linear(merged_dim, hidden_size * 3)
        self.upper_proj = nn.Linear(merged_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

        nn.init.xavier_uniform_(self.update.weight)
        nn.init.zeros_(self.update.bias)
        nn.init.xavier_uniform_(self.upper_proj.weight)
        nn.init.zeros_(self.upper_proj.bias)

    def forward(self, lower: Tensor, previous: Tensor) -> Tuple[Tensor, Tensor]:
        merged = torch.cat([lower, previous], dim=-1)
        merged = self.norm(merged)
        gate_raw, reset_raw, candidate_raw = self.update(merged).chunk(3, dim=-1)

        gate = torch.sigmoid(gate_raw)
        reset = torch.sigmoid(reset_raw)
        candidate = self.activation(candidate_raw + reset * previous)
        candidate = self.dropout(candidate)

        updated = gate * previous + (1.0 - gate) * candidate
        updated = self.dropout(updated)
        upper = self.upper_proj(torch.cat([lower, updated], dim=-1))
        upper = self.dropout(upper)

        return updated, upper


class HRMReducer(nn.Module):
    """Recursive context compressor operating on relational and dense state."""

    def __init__(self, config: UnifiedConfig) -> None:
        super().__init__()
        self.abstraction_sizes = list(config.abstraction_sizes)

        first_input = config.hidden_size + config.relation_size
        layers: List[HRMLayer] = []
        layer_norms: List[nn.LayerNorm] = []
        in_size = first_input
        for size in self.abstraction_sizes:
            layers.append(HRMLayer(in_size, size))
            layer_norms.append(nn.LayerNorm(size))
            in_size = size
        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(layer_norms)

    def forward(
        self,
        h_tilde: Tensor,
        relations: Tensor,
        causal_mask: Tensor,
        abstractions: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        h_next, abstractions_out, _ = self.forward_debug(
            h_tilde, relations, causal_mask, abstractions
        )
        return h_next, abstractions_out

    def forward_debug(
        self,
        h_tilde: Tensor,
        relations: Tensor,
        causal_mask: Tensor,
        abstractions: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tuple[Tensor, ...], dict[str, Tensor]]:
        rel_basis = relations.mean(dim=2)
        mask_weights = torch.softmax(causal_mask, dim=-1)
        rel_context = torch.einsum("bss,bsd->bd", mask_weights, rel_basis)

        lower = torch.cat([h_tilde, rel_context], dim=-1)

        new_abstractions: List[Tensor] = []
        upper = lower
        for idx, (layer, previous) in enumerate(zip(self.layers, abstractions)):
            updated, upper = layer(upper, previous)
            updated = self.layer_norms[idx](updated)
            new_abstractions.append(updated)

        h_next = new_abstractions[0]
        new_abstractions_tuple: Tuple[Tensor, ...] = tuple(new_abstractions)

        debug = {
            "rel_basis": rel_basis,
            "mask_weights": mask_weights,
            "rel_context": rel_context,
        }

        return h_next, new_abstractions_tuple, debug


class UnifiedCell(nn.Module):
    """Unified differentiable cell implementing the combined state machine."""

    def __init__(self, config: UnifiedConfig) -> None:
        super().__init__()
        self.config = config
        self.rwkv = RWKVBlock(config)
        self.binder = TensorBindBlock(config)
        self.hrm = HRMReducer(config)

        self.debug_mode = config.debug_mode

    def forward(
        self,
        state: UnifiedState,
        x_t: Tensor,
        *,
        return_debug: bool = False,
    ) -> UnifiedState | Tuple[UnifiedState, dict[str, Tensor]]:
        assert len(state.abstractions) == len(self.hrm.layers), "abstraction depth mismatch"

        h_tilde, time_state_next, rwkv_debug = self.rwkv.forward_debug(state, x_t)
        relations, causal_mask, bind_debug = self.binder.forward_debug(h_tilde, state)
        h_next, abstractions, hrm_debug = self.hrm.forward_debug(
            h_tilde, relations, causal_mask, state.abstractions
        )

        new_state = UnifiedState(
            h=h_next,
            time_state=time_state_next,
            relations=relations,
            causal_mask=causal_mask,
            abstractions=abstractions,
        )

        debug_data = {
            "rwkv": rwkv_debug,
            "binder": bind_debug,
            "hrm": hrm_debug,
        }

        if return_debug or self.debug_mode:
            return new_state, debug_data
        return new_state

    def forward_sequence(
        self,
        inputs: Tensor,
        state: UnifiedState | None = None,
        *,
        return_debug: bool = False,
    ) -> Tuple[Tensor, UnifiedState] | Tuple[Tensor, UnifiedState, list[dict[str, Tensor]]]:
        """Iterate the cell across a sequence of inputs.

        Args:
            inputs: Tensor shaped ``(B, T, input_size)``.
            state: Optional initial `UnifiedState`. If omitted, a zero state is created.
            return_debug: When True, collects per-timestep debug payloads.

        Returns:
            Hidden states for each timestep, final state, and optionally debug info.
        """

        batch, seq_len, feature_dim = inputs.shape
        if state is None:
            state = self.initial_state(batch, inputs.device)

        outputs: List[Tensor] = []
        debug_payload: List[dict[str, Tensor]] = []
        for t in range(seq_len):
            step_input = inputs[:, t]
            if return_debug or self.debug_mode:
                state, debug = self.forward(state, step_input, return_debug=True)
                debug_payload.append(debug)
            else:
                state = self.forward(state, step_input)
            outputs.append(state.h.unsqueeze(1))

        hidden_seq = torch.cat(outputs, dim=1)

        if return_debug or self.debug_mode:
            return hidden_seq, state, debug_payload
        return hidden_seq, state

    @torch.no_grad()
    def initial_state(self, batch_size: int, device: torch.device) -> UnifiedState:
        cfg = self.config
        h = torch.zeros(batch_size, cfg.hidden_size, device=device)
        time_state = torch.zeros_like(h)
        relations = torch.zeros(
            batch_size,
            cfg.symbol_slots,
            cfg.role_slots,
            cfg.relation_size,
            device=device,
        )
        causal_mask = torch.eye(cfg.symbol_slots, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        abstractions = tuple(
            torch.zeros(batch_size, size, device=device)
            for size in cfg.abstraction_sizes
        )
        return UnifiedState(
            h=h,
            time_state=time_state,
            relations=relations,
            causal_mask=causal_mask,
            abstractions=abstractions,
        )

    def submodules(self) -> Iterable[nn.Module]:
        """Expose component blocks for debug or unit testing."""

        if self.debug_mode:
            return (self.rwkv, self.binder, self.hrm)
        return (self,)

