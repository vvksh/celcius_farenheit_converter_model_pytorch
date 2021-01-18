from functools import partial

import torch

class Model:
    def __init__(self, loss_fn, grad_fn):
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
                      interval_to_print: int = 10) -> (float, float):
        for epoch in range(n_epochs):
            if self.params.grad is not None:
                self.params.grad.zero_()

            t_p = self.forward(input_tensor)
            loss = self.loss_fn(t_p, label_tensor)
            loss.backward()
            w_grad, b_grad = self.grad_fn(input_tensor, label_tensor, t_p)
            self.w = self.w - learning_rate * w_grad
            self.b = self.b - learning_rate * b_grad
            if epoch % interval_to_print == 0 or epoch == n_epochs - 1:
                print(f"Params: {self.w, self.b}")
                print(f"Grad: {w_grad, b_grad}")
                print(f'Epoch {epoch}, loss {loss} ')
        return self.w, self.b

