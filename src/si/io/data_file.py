import numpy as np
import os
from si.data.dataset import Dataset


def read_data_file(filename, sep=",", label=False):
    """
    Reads a data file into a Dataset object.
    A data file has no header: it is only numbers.

    Parameters
    ----------
    filename: str - path of the file
    sep: str - the value separator
    label: bool - does the file have y? (if yes, it is the last column)

    Returns
    -------
    Dataset
    """

    data=np.genfromtxt(filename, delimiter=sep)

    features_name= None
    label_name= None

    if label:
        data_X= data[:,:-1] # all rows and every column except the last one
        label_y= data[:,-1] # all rows and only the last column (the labels)

    else:
        data_X= data[:,:] # all rows and all columns: there is no label column to remove
        label_y= None
    
    return Dataset(X=data_X ,y=label_y, features=features_name ,label= label_name)

def write_data_file(filename, dataset, sep=",", label= False):
    """
    Writes a Dataset object to a data file.

    Parameters
    ----------
    filename: str - path of the file
    dataset: Dataset - the dataset to write
    sep: str - the value separator
    label: bool - write the label column?
    """

    if label:
        y_column = dataset.y.reshape(-1, 1)      # turn the 1D array into a single column (-1 = work out how many rows)
        data = np.hstack((dataset.X, y_column))  # glue them side by side, X first, so y ends up as the last column
    else:
        data = dataset.X                         # there is no label column, so only X gets written

    np.savetxt(filename, data, delimiter=sep)

if __name__ == "__main__":

    # reading: with and without the label column
    d1 = read_data_file("datasets/breast_bin/breast-bin.data", sep=",", label=True)
    print("with label   :", d1.shape(), d1.has_label(), d1.get_classes())

    d2 = read_data_file("datasets/breast_bin/breast-bin.data", sep=",", label=False)
    print("without label:", d2.shape(), d2.has_label())

    # writing: write it out, read it back, nothing should be lost
    write_data_file("test_write.data", d1, sep=",", label=True)
    d3 = read_data_file("test_write.data", sep=",", label=True)
    print("round trip   :", d3.shape(), d3.has_label(), d3.get_classes())
    os.remove("test_write.data")