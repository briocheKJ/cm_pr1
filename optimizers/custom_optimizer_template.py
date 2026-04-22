from __future__ import annotations


class CustomOptimizerBase:
    """
    Minimal interface for future student optimizers.

    A custom optimizer used in this project should expose:
    - zero_grad()
    - step()
    """

    def zero_grad(self) -> None:
        raise NotImplementedError("TODO: implement zero_grad() in your optimizer.")

    def step(self) -> None:
        raise NotImplementedError("TODO: implement step() in your optimizer.")


class CustomOptimizerTemplate(CustomOptimizerBase):
    """
    TODO: implement your own optimizer here.

    Suggested steps:
    1. Store the parameters you want to update.
    2. Store optimizer hyperparameters such as learning rate.
    3. In zero_grad(), clear existing gradients.
    4. In step(), update each parameter using its gradient.
    """

    def __init__(self, params, lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        # TODO: implement your own gradient reset logic here.
        raise NotImplementedError("TODO: implement your own optimizer here.")

    def step(self) -> None:
        # TODO: implement your own parameter update logic here.
        raise NotImplementedError("TODO: implement your own optimizer here.")
