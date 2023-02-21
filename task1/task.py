import torch
from torch import nn

def polynomial_fun(weight: torch.Tensor, x: torch.Tensor)-> torch.Tensor:
    """Returns output value of phi(x)@weights where phi is polynomial mapping.

    :param weight: a vector of polynomial weights.
    :param x: vector of trianing samples.
    :return:
        y: torch.tensor
    """

    pows = torch.arange(0, weight.shape[0])
    assert pows.shape == weight.shape,
    x_map = torch.pow(x[:, None]*torch.ones_like(weight), pows)
    assert x_map.shape[1] == weight.shape[0], f"Weights and X^pows are't the same size."
    return (weight @ x_map).squeeze()


def fit_polynomial_ls(x_train, t_train, degree):
    """Finds optimal weight vector for a train set using sq."""
    pows = torch.arange(0, degree + 1)
    phi = torch.pow(x_train*torch.ones((x_train.shape[0], degree + 1)), pows)
    assert phi[:,1]**2 == phi[:,2], "phi is not working"
    weight = torch.linalg.inv(phi.T@phi) @ phi.T @ t_train
    assert weight.shape[0] == degree, "lsq is not working"
    return weight


def fit_polynomial_sgd(x_train, t_train, degree, lr, mini_batch_size=16, epochs=10):
    weight = nn.Parameter(torch.randn(degree + 1), requires_grad=True)
    opt = torch.optim.SGD(weight, lr=lr)
    batch_index = torch.randperm(t_train.shape[0])
    for epoch in range(epochs):
        for start in range(0, t_train.shape[0], mini_batch_size):
            opt.zero_grad()
            end = start + mini_batch_size
            mbi = batch_index[start:end]
            x_mb, t_mb = x_train[mbi], t_train[mbi]
            preds = polynomial_fun(weight, x_mb)
            loss = ((preds - t_train)**2).mean()
            loss.backward()
            opt.step()
        print(f"Loss at Epoch{epoch+1}: {loss.item()}")
    return weight