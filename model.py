from functools import partial

import torch

class Model:
    def __init__(self, loss_fn, grad_fn):
        self.w = torch.ones(())
        self.b = torch.zeros(())
        self.loss_fn = loss_fn
        self.grad_fn = partial(grad_fn, self.w, self.b)

    def forward(self, input):
        return (self.w * input) + self.b

    def training_loop(self, n_epochs, learning_rate, input_tensor, label_tensor, interval_to_print=10):
        for epoch in range(n_epochs):
            t_p = self.forward(input_tensor)
            loss = self.loss_fn(t_p, label_tensor)
            w_grad, b_grad = self.grad_fn(input_tensor, label_tensor, t_p)
            self.w = self.w - learning_rate * w_grad
            self.b = self.b - learning_rate * b_grad
            if epoch % interval_to_print == 0 or epoch == n_epochs - 1:
                print(f"Params: {self.w, self.b}")
                print(f"Grad: {w_grad, b_grad}")
                print(f'Epoch {epoch}, loss {loss} ')
        return self.w, self.b

