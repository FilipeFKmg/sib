import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy: the portion of well classified samples.

        accuracy = (number of correct predictions) / (total number of predictions)

    Parameters
    ----------
    y_true: np.ndarray - the real label values
    y_pred: np.ndarray - the predicted label values

    Returns
    -------
    float - a value between 0 and 1
    """

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # example with y_true = ["a", "b", "a", "c"] and y_pred = ["a", "b", "c", "c"]
    bool_array = np.equal(y_true, y_pred)    # -> [True, True, False, True]   one comparison per position
    number_total_true = np.sum(bool_array)   # -> 3    in numpy, True counts as 1 and False as 0
    total = np.size(bool_array)              # -> 4

    return number_total_true / total         # -> 0.75


if __name__ == "__main__":
    true_y = np.array(["cancer", "normal tissue", "cancer", "cancer"])
    pred_y = np.array(["cancer", "normal tissue", "normal tissue", "cancer"])

    print("Accuracy of the model's predictions:", accuracy(y_true=true_y, y_pred=pred_y))
