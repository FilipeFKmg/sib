import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class VarianceThreshold(Transformer):
    """
    Feature selector that removes all features whose variance does not exceed
    a given threshold. It ignores the labels, so it is an unsupervised method:
    it can only detect features that barely change, not features that are
    irrelevant to the target.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        """
        Parameters
        ----------
        threshold: float - the minimum variance a feature must exceed to be kept
        """

        super().__init__(**kwargs)

        if threshold<0:
            raise ValueError("The variance threshold value must be greater than or equal to 0")
        
        self.threshold= threshold
        self.variance= None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Computes the variance of every feature and stores it in self.variance.
        """

        mean_values= np.mean(dataset.X, axis=0)

        lis=[]
    
        for i in range(len(mean_values)):
            varian= np.sum((dataset.X[:,i]- mean_values[i])**2)/len(dataset.X[:,0])
            lis.append(varian)

        # The same result can be obtained directly with dataset.get_variance().
        # The explicit computation is kept here to make the underlying formula clear.

        self.variance= lis

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Returns a new Dataset containing only the features whose variance is
        greater than the threshold. The labels are left unchanged.
        """
        lis=[]
        
        for j in range(len(self.variance)):
            if self.variance[j] <= self.threshold:
                lis.append(j)

        trans_dataset= np.delete(dataset.X, lis, axis=1)
        features= (np.delete(dataset.features, lis)).tolist()

        return Dataset(trans_dataset, dataset.y, features=features, label=dataset.label)

if __name__ == "__main__":
    from si.io.csv_file import read_csv

    # --- iris: 150 flowers, 4 measurements (features), 3 species ---
    iris = read_csv("datasets/iris/iris.csv", sep=",", features=True, label=True)

    selector = VarianceThreshold(threshold=0.5)
    selector.fit(iris)
    print("variance of each feature:", np.round(selector.variance, 3))
    print()

    # the same dataset with four different thresholds
    for threshold in (0.0, 0.5, 0.6, 1.0):
        reduced = VarianceThreshold(threshold=threshold).fit_transform(iris)
        print(f"threshold = {threshold} -> {reduced.shape()}  {reduced.features}")
    print()

    # a constant feature must be removed even with the default threshold
    random_dataset = Dataset.from_random(100, 10)
    random_dataset.X[:, 2] = 0           # feat_2 becomes constant: variance = 0
    reduced = VarianceThreshold().fit_transform(random_dataset)
    print("random dataset with a constant feat_2:")
    print("   before:", random_dataset.shape(), random_dataset.features)
    print("   after :", reduced.shape(), reduced.features)

    # a negative threshold is not a valid variance
    try:
        VarianceThreshold(threshold=-1)
    except ValueError as error:
        print()
        print("VarianceThreshold(threshold=-1) ->", error)
