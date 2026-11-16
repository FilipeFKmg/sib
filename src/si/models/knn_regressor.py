from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Model):
    """
    The k-Nearest Neighbors regressor.
    It estimates the value of a sample as the mean of the values of the k most
    similar samples in the training dataset.
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

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset. The KNN learns nothing else.
        """
        self.dataset= dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the value of every sample of the given dataset.
        """

        predictions= []

        for sample in dataset.X:
            all_distance= self.distance(sample, self.dataset.X)
            positions= np.argsort(all_distance)[:self.k]
            values= self.dataset.y[positions]                   # y values of the k nearest samples, e.g. [274, 30, 118]
            prediction= np.mean(values)                         # their mean: 140.67
            predictions.append(prediction)

        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the RMSE between the real values and the predicted ones (lower is better).
        """
        return rmse(dataset.y, predictions)

if __name__ == "__main__":
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    # --- cpu: 209 CPUs, 6 characteristics, y = relative performance (perf) ---
    cpu = read_csv("datasets/cpu/cpu.csv", sep=",", features=True, label=True)
    train, test = train_test_split(cpu, test_size=0.2, random_state=42)

    model = KNNRegressor(k=3).fit(train)
    predictions = model.predict(test)
    print(predictions)
    print(":::"*50)

    
    
    print("--- cpu ---")
    print("train:", train.shape(), "| test:", test.shape())
    print("predicted:", np.round(predictions[:5], 1))
    print("true     :", test.y[:5])
    print("rmse     :", round(model.score(test), 2))