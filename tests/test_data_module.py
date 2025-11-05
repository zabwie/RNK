from pathlib import Path

import torch

from prernk.data import PAD_TOKEN, create_text_dataloader


def test_create_text_dataloader(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world\nhello unified model\n", encoding="utf-8")

    dataloader, vocab = create_text_dataloader([sample], batch_size=2, max_seq_len=6)

    batch = next(iter(dataloader))
    assert batch.shape == (2, 6)
    assert vocab.size >= 4
    pad_id = vocab.token_to_id[PAD_TOKEN]
    assert torch.any(batch == pad_id)

