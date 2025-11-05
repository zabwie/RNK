import torch

from prernk.cell import UnifiedCell, UnifiedConfig, UnifiedState


def _build_config(debug: bool = False) -> UnifiedConfig:
    return UnifiedConfig(
        input_size=32,
        hidden_size=64,
        symbol_slots=8,
        role_slots=4,
        relation_size=16,
        abstraction_sizes=[64, 32, 16],
        activation="gelu",
        debug_mode=debug,
    )


def test_forward_shapes():
    cfg = _build_config()
    cell = UnifiedCell(cfg)
    device = torch.device("cpu")
    batch, steps = 3, 2
    state = cell.initial_state(batch, device)

    for _ in range(steps):
        x_t = torch.randn(batch, cfg.input_size, device=device)
        state = cell(state, x_t)

    assert state.h.shape == (batch, cfg.hidden_size)
    assert state.time_state.shape == (batch, cfg.hidden_size)
    assert state.relations.shape == (
        batch,
        cfg.symbol_slots,
        cfg.role_slots,
        cfg.relation_size,
    )
    assert state.causal_mask.shape == (batch, cfg.symbol_slots, cfg.symbol_slots)
    assert len(state.abstractions) == len(cfg.abstraction_sizes)


def test_debug_mode_exposes_submodules():
    cfg = _build_config(debug=True)
    cell = UnifiedCell(cfg)
    submodules = tuple(cell.submodules())
    assert len(submodules) == 3


def test_forward_sequence_outputs():
    cfg = _build_config()
    cell = UnifiedCell(cfg)
    batch, seq = 2, 4
    inputs = torch.randn(batch, seq, cfg.input_size)

    hidden_seq, final_state = cell.forward_sequence(inputs)

    assert hidden_seq.shape == (batch, seq, cfg.hidden_size)
    assert isinstance(final_state, UnifiedState)

