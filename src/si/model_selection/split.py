from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset into a training and a testing Dataset.

    Parameters
    ----------
    dataset: Dataset - the dataset to split
    test_size: float - the portion kept for testing (e.g. 0.2 for 20%)
    random_state: int - seed for generating the permutations

    Returns
    -------
    Tuple[Dataset, Dataset] - the train dataset and the test dataset
    """

    np.random.seed(random_state)          # same seed -> same shuffle, every time

    n_samples, _ = dataset.shape()        # the second value (the columns) is not needed here
    n_test = int(n_samples * test_size)   # must convert to int to work in the future slicing

    permutations = np.random.permutation(n_samples)   # the row numbers, shuffled

    test_idx = permutations[:n_test]      # the first n_test shuffled rows
    train_idx = permutations[n_test:]     # all the others

    X_test = dataset.X[test_idx]          # the same indices are used on X and on y,
    y_test = dataset.y[test_idx]          # so each row keeps its own answer

    X_train = dataset.X[train_idx]
    y_train = dataset.y[train_idx]

    test = Dataset(X_test, y_test, dataset.features, dataset.label)
    train = Dataset(X_train, y_train, dataset.features, dataset.label)

    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset keeping the proportion of each class in both halves.

    Instead of shuffling all the rows together, it shuffles the rows of each class
    separately and takes the same portion out of each one.

    Parameters
    ----------
    dataset: Dataset - the dataset to split
    test_size: float - the portion kept for testing (e.g. 0.2 for 20%)
    random_state: int - seed for generating the permutations

    Returns
    -------
    Tuple[Dataset, Dataset] - the train dataset and the test dataset
    """

    np.random.seed(random_state)

    classes = np.unique(dataset.y)        # e.g. ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    train_list = []                       # these two survive the loop and keep growing
    test_list = []

    for label in classes:
        class_idx = np.where(dataset.y == label)[0]   # the row numbers of this class only
        n_test = int(len(class_idx) * test_size)      # how many of THIS class go to the test set

        shuffled = np.random.permutation(class_idx)

        test_list.extend(shuffled[:n_test])           # extend, not append: one index at a time
        train_list.extend(shuffled[n_test:])

    train = Dataset(dataset.X[train_list], dataset.y[train_list], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_list], dataset.y[test_list], features=dataset.features, label=dataset.label)

    return train, test


if __name__ == "__main__":
    from si.io.csv_file import read_csv

    dataset = read_csv("datasets/iris/iris.csv", sep=",", features=True, label=True)

    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    print("normal     -> train:", train.shape(), "| test:", test.shape(),
          "| classes in the test set:", np.unique(test.y, return_counts=True)[1])

    train, test = stratified_train_test_split(dataset, test_size=0.2, random_state=42)
    print("stratified -> train:", train.shape(), "| test:", test.shape(),
          "| classes in the test set:", np.unique(test.y, return_counts=True)[1])
