import torch
from torch import nn
from torch import distributions as dist
from typing import Tuple
import time

torch.manual_seed(11)


def polynomial_fun(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Returns output value of phi(x) @ weights where phi is polynomial mapping.

    :param weight: a vector of polynomial weights.
    :param x: vector of trianing samples.
    :return:
        y: torch.tensor
    """

    pows = torch.arange(0, weight.shape[0])
    x_map = torch.pow(x[:, None] * torch.ones_like(weight), pows)
    return torch.einsum("ij, j -> i ", x_map, weight)


def fit_polynomial_ls(x_train, t_train, degree):
    """Finds optimal weight vector for a train set using sq."""
    pows = torch.arange(0, degree + 1)
    phi = torch.pow(x_train[:, None] * torch.ones((x_train.shape[0], degree + 1)), pows)
    weight = torch.linalg.inv(phi.T @ phi) @ phi.T @ t_train
    return weight


def fit_polynomial_sgd(
    x_train, t_train, degree, lr=0.01, mini_batch_size=32, epochs=10000, verbose=True
):

    batch_size = t_train.shape[0]
    weight = torch.ones((degree + 1,), requires_grad=True)
    opt = torch.optim.SGD([weight], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    for epoch in range(epochs):
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            x_mb, t_mb = x_train[start:end], t_train[start:end]
            # pred = torch.zeros_like(x_mb)
            # for i in range(degree+1):
            # pred += weight[i] * x_mb**i
            # pred = pred.sum()
            pred = polynomial_fun(weight, x_mb)
            opt.zero_grad()
            loss = ((t_mb - pred) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_([weight], 1)
            opt.step()
        if (epoch + 1) % 1000 == 0 and verbose:
            print(f"Loss at epoch {epoch + 1}: {loss.item():.2}")
            scheduler.step()

    return weight


def generate_set(
    num_points: int, std: float, weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_dist = dist.Uniform(-20, 20)
    eps_dist = dist.Normal(0, std)
    x = x_dist.sample((num_points,))
    f = polynomial_fun(weights, x)
    y = f + eps_dist.sample((num_points,))

    return x, y


if __name__ == "__main__":
    gt_weights = torch.Tensor([1, 2, 3, 4, 5])
    train_x, train_y = generate_set(100, 0.2, gt_weights)
    # print("shapes", train_x.shape, train_y.shape)
    # print("\n first few rows", train_x[:5], train_y[:5])
    test_x, test_y = generate_set(50, 0.2, gt_weights)

    t0 = time.time_ns()
    weight_sgd = fit_polynomial_sgd(train_x, train_y, 5)
    t1 = time.time_ns()
    t_sgd = t1 - t0
    t0 = time.time_ns()
    weight_ls = fit_polynomial_ls(train_x, train_y, 5)
    t1 = time.time_ns()
    t_ls = t1 - t0
    print("----------------------------")
    pred_sgd = polynomial_fun(weight_sgd, test_x)
    pred_ls = polynomial_fun(weight_ls, test_x)

    print("RMS(y_pred_ls - y_true):", (((pred_ls - test_y) ** 2).mean() ** 0.5).item())
    print("STD(y_pred_ls - y_true)", (pred_ls - test_y).std().item())
    print(
        "\nRMS(y_pred_sgd - y_true):", (((pred_sgd - test_y) ** 2).mean() ** 0.5).item()
    )
    print("STD(y_pred_sgd - y_true):", (pred_sgd - test_y).std().item())
    print(
        "\nRMS(y_pred_ls - y_pred_sgd):",
        (((pred_sgd - pred_ls) ** 2).mean() ** 0.5).item(),
    )
    print("STD(y_pred_ls - y_pred_sgd):", (pred_sgd - pred_ls).std().item())
    print("\n Runtime Info")
    print(f"SGD fit time:{float(t_sgd):.2} ns")
    print(f"Least Squares fit time: {float(t_ls):.2} ns")
