import torch


def loss_fn(predicted: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    squared_diffs = (predicted - labels) ** 2
    return squared_diffs.mean()


def grad_fn(weights: torch.Tensor,
            biases: torch.Tensor,
            input: torch.Tensor,
            labels: torch.Tensor,
            predicted: torch.Tensor) -> torch.Tensor:
    """
    :return: torch tensor containing updated weights and biases
    """
    def dloss_fn(t_p, t_c):
        dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)  # division is from derivative of mean
        return dsq_diffs

    def dmodel_dw(t_u, w, b):
        # model = w*t_u + b
        return t_u

    def dmodel_db(t_u, w, b):
        return 1.0

    dloss_dtp = dloss_fn(predicted, labels)
    dloss_dw = dloss_dtp * dmodel_dw(input, weights, biases)
    dloss_db = dloss_dtp * dmodel_db(input, weights, biases)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])  # why stack and sum?
