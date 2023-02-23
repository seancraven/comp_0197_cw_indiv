"""
The script performs a line search to find the best degree for the fitted polynomial.
Where the polynomial is using SGD.
"""
import torch
from task import *

def line_search(train_x, train_y, val_x, val_f):
    """Performs a line search over polynomials order to optimise validation error
    on a polynomial model."""
    degrees = [i for i in range(15)]
    best_val = torch.inf
    print("Performing Line Search")
    print("----------------------")
    for degree in degrees:
        print("Current Degree: ", degree)
        weight_dg = fit_polynomial_sgd(train_x, train_y, degree, verbose=False)
        val_rmse = (((polynomial_fun(weight_dg, val_x) - val_f)**2).mean())**0.5
        val_std = (polynomial_fun(weight_dg, val_x) - val_f).std()
        if val_rmse < best_val:
            best_val = val_rmse
            best_std = val_std
            best_weight_dg = weight_dg
            best_degree = degree
    print("----------------------")
    print(f"Degree of best fit Polynomial: {best_degree}")
    print("Weights:\n", best_weight_dg.tolist())
    print(f"Best Model Validation RMS(y_pred - y_true): {best_val:.2}")
    print(f"Best Model Validation STD(y-pred - y_true): {best_std:.2}")


if __name__ == "__main__":
    weights_gt = torch.Tensor([1, 2, 3, 4, 5])
    print("Original weights for data generation: \n", weights_gt)
    train_x, train_y, train_f = generate_set(100, 0.2, weights_gt)
    print(f"Training on {train_y.shape[0]} samples of (x_train, y_train)")
    val_x, val_y, val_f = generate_set(50, 0.2, weights_gt)
    print(f"Validating on {train_y.shape[0]} samples of (x_val, y_val)")
    print("Notes:")
    print(f"Validation is performed on samples without noise. ie the true curve.")
    print(f"Model is trained on examples with noise epsilon ~ N(0, 0.2).")
    print("----------------------")
    line_search(train_x, train_y, val_x, val_f)
