import torch

from earthdelta.probe import DirectWeightEditor, pristine_state_digest


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 3, bias=False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])

    def forward(self, x):
        return self.blocks[0].attn.proj(x)


def test_zero_edit_and_restore_pristine_weights():
    torch.manual_seed(2)
    model = _Model().eval()
    x = torch.randn(2, 4, 3)
    before = pristine_state_digest(model)
    reference = model(x).detach().clone()
    with DirectWeightEditor(model, {0: torch.zeros_like(model.blocks[0].attn.proj.weight)}, blocks=(0,)):
        edited = model(x)
    assert torch.equal(reference, edited)
    assert pristine_state_digest(model) == before


def test_edit_changes_output_and_hook_is_removed():
    model = _Model().eval()
    x = torch.ones(1, 1, 3)
    delta = torch.eye(3)
    reference = model(x).detach().clone()
    with DirectWeightEditor(model, {0: delta}, blocks=(0,)):
        edited = model(x)
        assert not torch.equal(reference, edited)
    assert torch.equal(reference, model(x))
