import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename, sep=",", features=False, label=False):

    data=pd.read_csv(filename,sep=sep)

    if label:
        X = data.iloc[:, :-1]  # all rows, every column except the last one
        y = data.iloc[:, -1]   # all rows, only the last column (the labels)

        X_to_numpy=X.to_numpy() # transforms it into an ndarray
        y_to_numpy=y.to_numpy() # transforms it into an ndarray

        label_y= y.name

    else:
        X = data  # the whole table: there is no label column to remove
        X_to_numpy=X.to_numpy() # transforms it into an ndarray

        y_to_numpy= None
        label_y= None

    if features:
        features_X= X.columns.to_list() # the headers of the table X - the to_list converts to an list of strings

    else:
        features_X= None

    return Dataset(X=X_to_numpy, y=y_to_numpy, features=features_X, label=label_y) # Creates an Dataset class


def write_csv(filename, dataset, sep=",", features=False, label=False):
    """
    Writes a Dataset object to a csv file.

    Parameters
    ----------
    filename: str - path of the file
    dataset: Dataset - the dataset to write
    sep: str - the value separator
    features: bool - write the feature names in the header?
    label: bool - write the label column?
    """
    
    data = pd.DataFrame(dataset.X) # this transforms the array of numpy in Dataset object into an panda table
    
    if features:
        data.columns = dataset.features
    
    else:
        pos_collumn=[]
        line, collumn = dataset.shape()
        for i in range(0,collumn):

            pos_collumn.append(f"collumn{i}")
        
        data.columns = pos_collumn

    if label:
        data[dataset.label]= dataset.y

    data.to_csv(filename, sep=sep, index=False)


if __name__ == "__main__":
    import os

    # reading: the three cases
    d1 = read_csv("datasets/iris/iris.csv", sep=",", features=True, label=True)
    print("case 1 (features and label):", d1.shape(), d1.has_label(), d1.features, "|", d1.label)

    d2 = read_csv("datasets/iris/iris_no_label.csv", sep=",", features=True, label=False)
    print("case 2 (no label)          :", d2.shape(), d2.has_label(), d2.features, "|", d2.label)

    d3 = read_csv("datasets/iris/iris_no_features.csv", sep=",", features=False, label=True)
    print("case 3 (no features)       :", d3.shape(), d3.has_label(), d3.get_classes(), "|", d3.features, "|", d3.label)

    # writing: write case 1 to a file, read it back, nothing should be lost
    write_csv("test_write.csv", d1, sep=",", features=True, label=True)
    d4 = read_csv("test_write.csv", sep=",", features=True, label=True)
    print("round trip                 :", d4.shape(), d4.has_label(), d4.features, "|", d4.label)
    print("same classes as case 1?    :", d1.get_classes().tolist() == d4.get_classes().tolist())
    os.remove("test_write.csv")
