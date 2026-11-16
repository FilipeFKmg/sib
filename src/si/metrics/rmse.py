import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the RMSE (root mean squared error): the typical size of the error,
    expressed in the same units as y.

        rmse = sqrt( mean( (y_true - y_pred)^2 ) )

    Parameters
    ----------
    y_true: np.ndarray - the real values of y
    y_pred: np.ndarray - the predicted values of y

    Returns
    -------
    float - the error between y_true and y_pred (0 means a perfect fit)
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # example with y_true = [1, 2, 3, 7] and y_pred = [1, 8, 4, 5]
    errors = y_true - y_pred          # -> [0, -6, -1, 2]
    squared = errors ** 2             # -> [0, 36, 1, 4]   removes the sign of each error
    mean_squared = np.mean(squared)   # -> 10.25           no axis: a single value for the whole array
    return np.sqrt(mean_squared)      # -> 3.2             back to the units of y


if __name__ == "__main__":
    y_true = np.array([1, 2, 3, 7])

    print("exact predictions     :", rmse(y_true, np.array([1, 2, 3, 7])))   # expected: 0.0
    print("constant error of 1   :", rmse(y_true, np.array([2, 3, 4, 8])))   # expected: 1.0
    print("single large error:", rmse(y_true, np.array([1, 8, 4, 5])))   # expected around: 3.2