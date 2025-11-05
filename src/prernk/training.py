"""Training utilities for the unified HRM–TensorLNN–RWKV model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .cell import UnifiedCell, UnifiedConfig, UnifiedState
from .data import PAD_TOKEN, Vocabulary, create_text_dataloader


class UnifiedSequenceModel(nn.Module):
    """Wrapper combining embedding, unified cell, and output projection."""

    def __init__(self, vocab_size: int, cell_config: UnifiedConfig) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.cell = UnifiedCell(cell_config)
        self.embedding = nn.Embedding(vocab_size, cell_config.input_size)
        self.output_head = nn.Linear(cell_config.hidden_size, vocab_size)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        tokens: Tensor,
        state: UnifiedState | None = None,
        *,
        return_state: bool = False,
        return_debug: bool = False,
    ) -> Tensor | Tuple[Tensor, UnifiedState] | Tuple[Tensor, UnifiedState, list[dict[str, Tensor]]]:
        # tokens: (batch, seq)
        embeddings = self.embedding(tokens)
        if return_debug:
            hidden_seq, state, debug = self.cell.forward_sequence(
                embeddings, state, return_debug=True
            )
        else:
            hidden_seq, state = self.cell.forward_sequence(embeddings, state)

        logits = self.output_head(hidden_seq)

        if return_debug:
            return logits, state, debug
        if return_state:
            return logits, state
        return logits


@dataclass
class TrainingConfig:
    data_paths: Sequence[Path]
    batch_size: int = 8
    max_seq_len: int = 64
    epochs: int = 1
    lr: float = 1e-3
    min_freq: int = 1
    max_tokens: int | None = 4096
    device: str = "cpu"
    report_every: int = 25
    limit_samples: int | None = 2048

    # Model hyperparameters
    input_size: int = 128
    hidden_size: int = 256
    symbol_slots: int = 16
    role_slots: int = 4
    relation_size: int = 64
    abstraction_sizes: Sequence[int] = (256, 128, 64)
    activation: str = "gelu"
    debug_mode: bool = False


def build_model(cfg: TrainingConfig, vocab: Vocabulary) -> UnifiedSequenceModel:
    cell_cfg = UnifiedConfig(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        symbol_slots=cfg.symbol_slots,
        role_slots=cfg.role_slots,
        relation_size=cfg.relation_size,
        abstraction_sizes=cfg.abstraction_sizes,
        activation=cfg.activation,
        debug_mode=cfg.debug_mode,
    )
    return UnifiedSequenceModel(vocab_size=vocab.size, cell_config=cell_cfg)


def _shift_sequence(batch: Tensor, pad_id: int) -> tuple[Tensor, Tensor]:
    # Inputs exclude final token; targets exclude initial token.
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    # Ensure last column of inputs is pad for teacher forcing.
    inputs = torch.where(inputs == pad_id, pad_id, inputs)
    return inputs, targets


def train_epoch(
    model: UnifiedSequenceModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    pad_id: int,
    device: torch.device,
    report_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step = 0

    for batch in dataloader:
        batch = batch.to(device)
        inputs, targets = _shift_sequence(batch, pad_id=pad_id)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            targets.reshape(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        token_count = (targets != pad_id).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

        step += 1
        if report_every and step % report_every == 0:
            ppl = math.exp(total_loss / max(total_tokens, 1))
            print(f"step={step} loss={loss.item():.4f} ppl~{ppl:.2f}")

    return total_loss / max(total_tokens, 1)


def train_text_corpus(cfg: TrainingConfig) -> None:
    dataloader, vocab = create_text_dataloader(
        cfg.data_paths,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        min_freq=cfg.min_freq,
        max_tokens=cfg.max_tokens,
        limit=cfg.limit_samples,
    )

    device = torch.device(cfg.device)
    model = build_model(cfg, vocab).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_epoch(
            model,
            dataloader,
            optimizer,
            pad_id=vocab.token_to_id[PAD_TOKEN],
            device=device,
            report_every=cfg.report_every,
        )
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f} ppl~{ppl:.2f}")


def default_data_paths() -> list[Path]:
    root = Path("Data/LL")
    return [root / "Random English Sentences.txt"]


def run_cli(args: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train unified HRM–TensorLNN–RWKV model on text data")
    parser.add_argument(
        "--data",
        nargs="*",
        default=None,
        help="Paths to text files. Defaults to Data/LL/Random English Sentences.txt",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--debug-mode", action="store_true")

    parsed = parser.parse_args(args=args)
    data_paths = (
        [Path(p) for p in parsed.data]
        if parsed.data
        else default_data_paths()
    )

    cfg = TrainingConfig(
        data_paths=data_paths,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        max_seq_len=parsed.max_seq_len,
        device=parsed.device,
        lr=parsed.lr,
        debug_mode=parsed.debug_mode,
    )

    train_text_corpus(cfg)


if __name__ == "__main__":
    run_cli()

