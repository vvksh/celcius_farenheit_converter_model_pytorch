# inputs
import torch
import mean_square_loss
import model


def run_model():
    # the t_c values are temperatures in Celsius, and the t_u values are our unknown units.
    # We can expect noise in both measurements,
    # coming from the devices them- selves and from our approximate readings.

    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    t_un = t_u * 0.1 #without normalization, gradient for weights and biases start out
    # drastically different

    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)
    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[: -n_val]
    val_indices = shuffled_indices[-n_val:]

    train_t_u = t_u[train_indices]
    train_t_c = t_c[train_indices]
    val_t_u = t_u[val_indices]
    val_t_c = t_c[val_indices]


    m = model.Model(loss_fn=mean_square_loss.loss_fn)
    m.training_loop(n_epochs=5000,
                    learning_rate=1e-1,
                    train_input=train_t_u,
                    train_labels=train_t_c,
                    val_input=val_t_u,
                    val_labels=val_t_c,
                    optimizer=torch.optim.Adam,
                    interval_to_print=500)
    print(f" weights: {m.params[0]}, biases: {m.params[1]}")


if __name__ == '__main__':
    run_model()