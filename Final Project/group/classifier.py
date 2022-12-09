"""
Created on 11/4/22

top level module for classifier abstractions
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class MILClassifier(ABC):
    """
    Abstract base class defining common classifier functions.
    Classifier implementations should inherit this class, not instantiate it directly.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:
        """
        fit the classifier given a set of examples X with shape (num_examples, num_features) and labels y with shape (num_examples,).

        Args:
            X (np.ndarray): the example set with shape (num_instances, num_features)
            bag_indices (List[np.ndarray]): lists of which instances belong to which bag
            y (np.ndarray): the labels with shape (num_examples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, bag_indices: List[np.ndarray]) -> np.ndarray:
        """
        produce a list of output labels for a set of examples X with shape (num_examples, num_features).

        Args:
            X (np.ndarray): instances belonging to bags with shape (num_instances, num_features)
            bag_indices (List[np.ndarray]): which instances belong to which bag for which outputs should be provided

        Returns:
            np.ndarray: the predicted outputs with shape (len(bag_indices),)
        """
        pass
