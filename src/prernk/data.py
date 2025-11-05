"""Data utilities for training the unified HRM–TensorLNN–RWKV model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class Vocabulary:
    token_to_id: dict
    id_to_token: List[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    def decode(self, idx: int) -> str:
        return self.id_to_token[idx]


def build_vocabulary(
    corpus: Iterable[Sequence[str]],
    min_freq: int = 1,
    max_tokens: Optional[int] = None,
) -> Vocabulary:
    counter: Counter[str] = Counter()
    for tokens in corpus:
        counter.update(tokens)

    most_common = counter.most_common()
    if max_tokens is not None:
        most_common = most_common[: max_tokens - 2]  # reserve pad + unk

    id_to_token = [PAD_TOKEN, UNK_TOKEN]
    id_to_token.extend(token for token, freq in most_common if freq >= min_freq)

    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return Vocabulary(token_to_id=token_to_id, id_to_token=id_to_token)


def _read_lines(paths: Sequence[Path], limit: Optional[int]) -> List[str]:
    lines: List[str] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
                    if limit is not None and len(lines) >= limit:
                        return lines
    return lines


def tokenize_line(line: str) -> List[str]:
    return line.split()


class TextDataset(Dataset[Tensor]):
    """Simple whitespace-tokenized dataset backed by in-memory tensors."""

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        max_seq_len: int,
        pad_id: int,
    ) -> None:
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        padded = []
        for seq in sequences:
            trimmed = list(seq[: max_seq_len - 1])  # leave room for EOS/target shift
            trimmed.append(pad_id)
            length = len(trimmed)
            if length < max_seq_len:
                trimmed.extend([pad_id] * (max_seq_len - length))
            else:
                trimmed = trimmed[:max_seq_len]
            padded.append(torch.tensor(trimmed, dtype=torch.long))
        self.samples = padded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.samples[idx]


def create_text_dataloader(
    paths: Sequence[Path],
    batch_size: int,
    max_seq_len: int,
    *,
    min_freq: int = 1,
    max_tokens: Optional[int] = None,
    limit: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[DataLoader, Vocabulary]:
    lines = _read_lines(paths, limit)
    tokenized = [tokenize_line(line) for line in lines]
    vocab = build_vocabulary(tokenized, min_freq=min_freq, max_tokens=max_tokens)

    sequences = [[vocab.encode(tok) for tok in tokens] for tokens in tokenized]
    dataset = TextDataset(sequences, max_seq_len=max_seq_len, pad_id=vocab.token_to_id[PAD_TOKEN])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, vocab

