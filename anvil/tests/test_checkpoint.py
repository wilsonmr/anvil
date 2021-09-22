import warnings

import pytest
import torch

from anvil.checkpoint import loaded_model, loaded_optimizer, loaded_scheduler


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.rand(1))


def test_model_loads():
    """Check that model state dict is loaded correctly."""
    model = SimpleModule()
    new_model = loaded_model({"model_state_dict": model.state_dict()}, SimpleModule())
    assert new_model.p == model.p


@pytest.mark.xfail(reason="See https://github.com/pytorch/pytorch/issues/65342")
@pytest.mark.parametrize(
    "optimizer,optimizer_params",
    [("Adam", {}), ("Adadelta", {}), ("Adagrad", {}), ("SGD", {"lr": 0.001})],
)
@pytest.mark.parametrize(
    "scheduler,scheduler_params",
    [
        ("CosineAnnealingLR", {"T_max": 10}),
        ("CosineAnnealingWarmRestarts", {"T_0": 10}),
    ],
)
def test_optimizer_loads(
    optimizer: str, optimizer_params: dict, scheduler: str, scheduler_params: dict
):
    """Check that optimizer and scheduler are loaded correctly."""
    model = SimpleModule()
    optim = getattr(torch.optim, optimizer)(model.parameters(), **optimizer_params)
    sched = getattr(torch.optim.lr_scheduler, scheduler)(optim, **scheduler_params)

    for _ in range(5):
        optim.step()
        sched.step()

    checkpoint = {
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
    }

    new_optim = loaded_optimizer(model, checkpoint, optimizer, optimizer_params)
    new_sched = loaded_scheduler(new_optim, checkpoint, scheduler, scheduler_params)
    # First, check that scheduler wraps optimizer
    assert new_sched.optimizer is new_optim

    # Check that state dicts were correctly loaded
    assert all(
        [
            new_optim.state_dict()[key] == optim.state_dict()[key]
            for key in optim.state_dict().keys()
        ]
    )
    assert all(
        [
            new_sched.state_dict()[key] == sched.state_dict()[key]
            for key in sched.state_dict().keys()
        ]
    )
    # Check learning rates match. See https://github.com/pytorch/pytorch/issues/65342
    assert new_optim.param_groups[0]["lr"] == optim.param_groups[0]["lr"]
    assert new_sched.get_last_lr() == sched.get_last_lr()
    with warnings.catch_warnings():  # ignore warning that we should use get_last_lr
        warnings.filterwarnings("ignore", message="To get the last learning rate")
        assert new_sched.get_lr() == sched.get_lr()
