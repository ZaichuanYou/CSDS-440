"""
Created on 11/3/22

Utility functions for reading / writing bag data

Datasets from http://www.multipleinstancelearning.com/datasets/

Each dataset file is a comma-separated value (CSV) formatted file which has number of instances many rows and number of
features many columns together with two additionally attached columns. The first attached column corresponds to the bag
class labels which are propagated to the instances. The second column is the bag ID column where each instance receives
the bag ID number of its owner bag. The remaining columns individually store the feature values of the instances.
"""

import os
from typing import Tuple, List

import numpy as np


def load_data(data_dir: str, data_file: str) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    filepath = os.path.join(data_dir, data_file)

    data = np.loadtxt(filepath, delimiter=",", dtype=float)

    # Second column is which bag each example belongs to. Extract the indices of the example for each bag
    # Dataset indexing starts at 1, and goes to the max index
    bag_indices = [np.where(data[:, 1] == i)[0] for i in range(1, int(1 + max(data[:, 1])))]

    # Labels are coloumn zero. The label is listed the same for all example so just extract one
    labels = np.array([int(data[examples_indices[0], 0]) for examples_indices in bag_indices])

    # The rest is the actual data
    X = data[:, 2:]

    return X, bag_indices, labels


def save_data(filename: str, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:
    new_labels = np.empty((X.shape[0], 1), dtype=int)

    # Expand labels so there is one label for each instance
    counter = 0
    for label, indices in zip(y, bag_indices):
        new_labels[counter: counter + len(indices)] = label
        counter += len(indices)

    # stack all the labels into one big column in 2D
    bag_indices = np.atleast_2d(np.hstack(bag_indices)).T

    data = np.hstack((new_labels, bag_indices, X))

    # This will save labels and indecis as floats but that's fine
    np.savetxt(filename, data, delimiter=",", fmt="%.4f")
