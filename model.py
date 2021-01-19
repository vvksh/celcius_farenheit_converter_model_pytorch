from collections import OrderedDict

import torch
from torch import nn

class Model:
    def __init__(self):
        self.model = nn.Sequential(OrderedDict([
            ('hidden_linear', nn.Linear(1, 8)),
            ('hidden_activation', nn.Tanh()),
            ('output_linear', nn.Linear(8, 1))
        ]))
        self.loss_fn = nn.MSELoss()

    def forward(self, input):
        return self.model(input)

    def training_loop(self,
                      n_epochs:int,
                      learning_rate: float,
                      train_input: torch.Tensor,
                      val_input: torch.Tensor,
                      train_labels: torch.Tensor,
                      val_labels: torch.Tensor,
                      optimizer: torch.optim.Optimizer.__class__, #not sure if this is the best approach
                      interval_to_print: int = 10):
        optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        for epoch in range(n_epochs):
            train_pred = self.forward(train_input)
            train_loss = self.loss_fn(train_pred, train_labels)

            with torch.no_grad(): # we dont need to build a graph for validation computation
                val_pred = self.forward(val_input)
                val_loss = self.loss_fn(val_pred, val_labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % interval_to_print == 0 or epoch == n_epochs - 1:
                # print(f"Params: {self.model.hidden_linear.weight, self.model.hidden_linear.bias}")
                # print(f"Grad: {self.model.weight.grad, self.model.bias.grad}")
                print(f'Epoch {epoch}, train loss {train_loss.item():.4f},'
                      f' val loss {val_loss.item(): .4f}')
