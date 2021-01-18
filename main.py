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

    m = model.Model(loss_fn=mean_square_loss.loss_fn, grad_fn=mean_square_loss.grad_fn)
    w, b = m.training_loop(n_epochs=100,
                           learning_rate=1e-4,
                           input_tensor=t_un,
                           label_tensor=t_c)
    print(f" weights: {w}, biases: {b}")


if __name__ == '__main__':
    run_model()