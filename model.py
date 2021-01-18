from functools import partial
from typing import Callable

import torch

class Model:
    def __init__(self,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.params = torch.tensor([1.0, 1.0], requires_grad = True) # w, b
        self.loss_fn = loss_fn

    def forward(self, input):
        w, b = self.params
        return (w * input) + b

    def training_loop(self,
                      n_epochs:int,
                      learning_rate: float,
                      input_tensor: torch.Tensor,
                      label_tensor: torch.Tensor,
                      optimizer: torch.optim.Optimizer.__class__, #not sure if this is the best approach
                      interval_to_print: int = 10):
        optimizer = optimizer([self.params], lr=learning_rate)
        for epoch in range(n_epochs):
            t_p = self.forward(input_tensor)
            loss = self.loss_fn(t_p, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % interval_to_print == 0 or epoch == n_epochs - 1:
                print(f"Params: {self.params}")
                print(f"Grad: {self.params.grad[0], self.params.grad[1]}")
                print(f'Epoch {epoch}, loss {loss}')


