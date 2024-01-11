import numpy as np


def compute_quantile_loss(y_true, y_pred, quantile: float):
    """
    URL: https://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html
    """
    residual = y_true - y_pred
    return 2 * np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
