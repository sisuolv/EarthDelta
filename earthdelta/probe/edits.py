"""Direct full-matrix weight edits for the approved probe."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Iterable, Mapping

import torch
from torch import Tensor


def pristine_state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class DirectWeightEditor:
    """Temporarily inject ``x @ delta.T`` at selected ``attn.proj`` modules."""

    def __init__(self, model: torch.nn.Module, deltas: Mapping[int, Tensor], *, blocks: Iterable[int]):
        self.model = model
        self.deltas = dict(deltas)
        self.blocks = tuple(int(x) for x in blocks)
        if set(self.deltas) - set(self.blocks):
            raise ValueError("delta supplied for a non-target block")
        self.handles = []

    def __enter__(self):
        for index in self.blocks:
            if index not in self.deltas:
                continue
            module = self.model.blocks[index].attn.proj
            delta = self.deltas[index]
            if delta.ndim != 2 or delta.shape != module.weight.shape:
                raise ValueError(f"delta shape mismatch at block {index}")

            def hook(_module, inputs, output, delta=delta):
                return output + torch.nn.functional.linear(inputs[0], delta)

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


@contextmanager
def build_delta_modules(model: torch.nn.Module, deltas: Mapping[int, Tensor], *, blocks: Iterable[int]):
    editor = DirectWeightEditor(model, deltas, blocks=blocks)
    with editor:
        yield editor
