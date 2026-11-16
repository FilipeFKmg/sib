import numpy as np

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the euclidean distance between one sample x and every sample in y.

        distance(x, y_i) = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xn - yn)^2)

    Parameters
    ----------
    x: np.ndarray (n_features,) - a single sample
    y: np.ndarray (n_samples, n_features) - several samples

    Returns
    -------
    np.ndarray (n_samples,) - one distance per sample in y
    """

    if x.ndim > 1 and x.shape[0] != 1:
        raise ValueError("x must be a single sample")
    
    # example with x = [4, 5] and y = [[3, 4], [1, 1]]
    
    substract = x - y                # -> [[4-3, 5-4], [4-1, 5-1]] = [[1, 1], [3, 4]]
    squ       = substract ** 2       # -> [[1, 1], [9, 16]]
    sum_line  = np.sum(squ, axis=1)  # -> [2, 25]   one value per row of squ
    dist      = np.sqrt(sum_line)    # -> [1.41, 5.0]

    return dist

if __name__ == "__main__":
    single_sample = np.array([6.3, 3.3, 6.0, 2.5])
    samples = np.array([[5.1, 3.5, 1.4, 0.2],
                      [7.0, 3.2, 4.7, 1.4],
                      [6.5, 3.0, 5.8, 2.2]])
    print("distances to the three samples of single_sample:",
          np.round(euclidean_distance(single_sample, samples), 2))