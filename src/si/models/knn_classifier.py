from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    """
    The k-Nearest Neighbors classifier.
    It estimates the class of a sample based on the classes of the k most similar
    samples in the training dataset.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Parameters
        ----------
        k: int - the number of nearest examples to consider
        distance: Callable - function that computes the distance between a sample
                             and the training samples
        """
        super().__init__(**kwargs)
        self.k= k
        self.distance= distance
        self.dataset= None

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Stores the training dataset. The KNN learns nothing else.
        """
        self.dataset= dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class of every sample of the given dataset.
        """

        predictions= []

        for sample in dataset.X:
            all_distance= self.distance(sample, self.dataset.X)
            positions= np.argsort(all_distance)[:self.k]
            labels= self.dataset.y[positions]                     # labels of the k nearest samples, e.g. ["1", "1", "3"]
            classes, counts= np.unique(labels, return_counts=True)  # e.g. ["1", "3"] and [2, 1]
            prediction= classes[np.argmax(counts)]                # the most frequent class wins: "1"

            predictions.append(prediction)

        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the accuracy between the real classes and the predicted ones.
        """
        return accuracy(dataset.y, predictions)

if __name__ == "__main__":
    from si.io.csv_file import read_csv
    from si.io.data_file import read_data_file
    from si.model_selection.split import train_test_split

    # --- iris: 150 flowers, 4 measurements, 3 species ---
    iris = read_csv("datasets/iris/iris.csv", sep=",", features=True, label=True)
    train, test = train_test_split(iris, test_size=0.2, random_state=42)

    model = KNNClassifier(k=3).fit(train)
    predictions = model.predict(test)

    print("--- iris ---")
    print("train:", train.shape(), "| test:", test.shape())
    print("predicted:", predictions[:4])
    print("true     :", test.y[:4])
    print("accuracy :", model.score(test))

    # --- breast_bin: 699 tumours, 9 measurements, benign (0) or malignant (1) ---
    breast = read_data_file("datasets/breast_bin/breast-bin.data", sep=",", label=True)
    train, test = train_test_split(breast, test_size=0.2, random_state=42)

    model = KNNClassifier(k=3).fit(train)

    print()
    print("--- breast_bin ---")
    print("train:", train.shape(), "| test:", test.shape())
    print("accuracy :", model.score(test))

    # --- does k change the result? ---
    print()
    print("--- breast_bin, different values of k ---")
    for k in [1, 3, 5, 11, 51]:
        print(f"   k = {k:2} -> accuracy {KNNClassifier(k=k).fit(train).score(test):.3f}")
