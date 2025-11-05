import torch

from prernk.cell import UnifiedConfig
from prernk.training import UnifiedSequenceModel


def test_unified_sequence_model_forward():
    vocab_size = 20
    seq_len = 5
    batch = 3
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    config = UnifiedConfig(
        input_size=16,
        hidden_size=32,
        symbol_slots=4,
        role_slots=2,
        relation_size=8,
        abstraction_sizes=[32, 16],
    )

    model = UnifiedSequenceModel(vocab_size, config)
    logits, state = model(tokens, return_state=True)

    assert logits.shape == (batch, seq_len, vocab_size)
    assert state.h.shape == (batch, config.hidden_size)


def test_sequence_model_debug_payload():
    vocab_size = 15
    seq_len = 3
    batch = 1
    tokens = torch.randint(0, vocab_size, (batch, seq_len))

    config = UnifiedConfig(
        input_size=8,
        hidden_size=16,
        symbol_slots=4,
        role_slots=2,
        relation_size=8,
        abstraction_sizes=[16, 8],
        debug_mode=True,
    )

    model = UnifiedSequenceModel(vocab_size, config)
    logits, state, debug = model(tokens, return_state=True, return_debug=True)

    assert logits.shape == (batch, seq_len, vocab_size)
    assert len(debug) == seq_len

