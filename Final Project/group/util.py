"""
Created on 11/4/22

Utility functions
"""

from typing import Tuple, List, Iterable
from time import time

from group.classifier import MILClassifier

import numpy as np


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates accuracy

    Args:
        y: True labels.
        y_hat: Predicted labels.
    """

    return (y == y_hat).sum() / y.size


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates precision

    Args:
        y: True labels.
        y_hat: Predicted labels.
    """

    # Precision = TP / (TP + FP)
    tp = ((y == 1) & (y_hat == 1)).sum()
    fp = ((y == 0) & (y_hat == 1)).sum()

    return tp / (tp + fp) if (tp + fp) != 0 else 0.0


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Calculates recall

    Args:
        y: True labels.
        y_hat: Predicted labels.
    """

    # Recall = TP / (TP + FN)
    tp = ((y == 1) & (y_hat == 1)).sum()
    fn = ((y == 1) & (y_hat == 0)).sum()

    return tp / (tp + fn)


def cv_split(bag_indices: List[np.ndarray], y: np.ndarray, folds: int, stratified: bool = False) \
      -> Tuple[Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray], ...]:
    """
    Uses arbitrary_cv_indices() to split bag_indices and labels into folds.
    """

    folds = arbitrary_cv_indices(y, folds, stratified=stratified)

    ret = []

    for train_indices, test_indices in folds:
        ret.append(
            (
                [bag_indices[i] for i in train_indices],
                y[train_indices],

                [bag_indices[i] for i in test_indices],
                y[test_indices],
            )
        )

    return tuple(ret)


def arbitrary_cv_indices(y: np.ndarray, folds: int, stratified: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Creates the indicies to do a cross validation split on given data

    Args:
        y: Labels of shape (n_examples,). Necessary if doing stratified
        folds: Number of CV folds
        stratified: Whether or not to keep proportions of constant among folds

    Returns: A tuple containing the training data indices, testing data indices for each fold

    For example, 5 fold cross validation would return the following:
    (
        (train_1, test_1),
        (train_2, test_2),
        (train_3, test_3),
        (train_4, test_4),
        (train_5, test_5)
    )

    """
    if folds < 2:
        raise ValueError(f"Cross validation requires at least 2 folds, not {folds}")

    if stratified:
        # To stratify, we divide the set into its classes, and split into folds each, then recombine.
        # This gives each resulting fold the same ratio of each class. We then shuffle.

        # Get indices where data corresponds to each class
        class_0_indices = np.nonzero(y == 0)[0]
        class_1_indices = np.nonzero(y == 1)[0]

        # Extract y for each class
        y_for_class_0 = y[class_0_indices]
        y_for_class_1 = y[class_1_indices]

        # Split into folds each
        split_for_class_0 = arbitrary_cv_indices(y_for_class_0, folds, stratified=False)
        split_for_class_1 = arbitrary_cv_indices(y_for_class_1, folds, stratified=False)

        # Merge the two classes with shuffling
        # Because we distribute extra elements to the first folds, by reversing one of these splits,
        # we get more even sizes of the results.
        split_data = []
        for class0s, class1s in zip(split_for_class_0, split_for_class_1):
            train_indices = np.append(class_0_indices[class0s[0]], class_1_indices[class1s[0]])
            test_indices = np.append(class_0_indices[class0s[1]], class_1_indices[class1s[1]])

            # Shuffle the indices
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

            split_data.append(
                (train_indices, test_indices)
            )

        return tuple(split_data)

    # Create indices to split into folds, allowing us to associate data array with label array.
    # Shuffle the indices so that the resulting data will also be shuffled
    N = y.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    # Find the size of each fold, as well as how many elements are left over
    split_size = N // folds
    extra_element_num = N - split_size * folds

    # Extract fold indices, adding in the extra elements as needed. Store the indices in an array
    fold_list = []
    fold_start_index = 0
    for i in range(folds):
        extra_element = 1 if i < extra_element_num else 0  # Will give the correct number of extra elements in total
        fold_list.append(indices[fold_start_index:fold_start_index+split_size+extra_element])
        fold_start_index += split_size+extra_element

    # We have our folds, create a tuple for each one of all the other folds for data/labels, and the test data/labels
    # np.delete returns all values except for fold_indices, without modifying existing array.
    split_indices = tuple(
        (
            np.delete(indices, fold_indices),
            indices[fold_indices],
        ) for fold_indices in fold_list
    )

    return split_indices


def print_evalutions(
        X: np.ndarray,
        splits: Tuple[Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray], ...],
        base_classifier: MILClassifier) -> None:
    """
    Print out evaluation metrics

    Args:
        X: Features matrix.
        splits: Train and test splits.
        base_classifier: Base MILClassifier model.
    """
    # Initial
    accs = []
    precs = []
    recs = []
    start_time = time()

    for bag_indices_train, y_train, bag_indices_test, y_test in splits:
        classifier = base_classifier
        classifier.fit(X, bag_indices_train, y_train)
        y_hat = classifier.predict(X, bag_indices_test)

        # Add evaluations
        accs.append(accuracy(y_test, y_hat))
        precs.append(precision(y_test, y_hat))
        recs.append(recall(y_test, y_hat))

    # Print out metrics
    print(f"Acc: {np.mean(accs):.2} ± {np.std(accs):.2}")
    print(f"Prec: {np.mean(precs):.2} ± {np.std(precs):.2}")
    print(f"Rec: {np.mean(recs):.2} ± {np.std(recs):.2}")
    print(f"Time: {time()-start_time:.3}s")
    print()

def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    """
    Finds the values of the ROC curve as in roc_curve_pairs, but in O(n) time.
    * From (James, Zaichuan, John)'s own project code.

    Args:
        y: the true labels for a set of examples X.
        p_y_hat: the confidences P(y = 1|X) produced by the model, given X.
    
    Returns: an iterable of tuples representing ROC points.
    """
    assert np.shape(y) == np.shape(p_y_hat), 'Arguments must be the same size'
    sorted_pairs = sorted(zip(p_y_hat, y), key=lambda x: x[0], reverse=True) # zip and sort
    p_y_hat, y = zip(*sorted_pairs) # unzip
    
    pairs = [(0, 0)]
    num_positives = sum(y)
    num_negatives = len(y) - num_positives
    tps, fps = 0, 0
    for label in y:
        if label == 1:
            # You've added a true positive
            tps += 1
        else:
            # You've added a false positive
            fps += 1
        pairs.append((fps/num_negatives, tps/num_positives))

    return pairs

def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    """
    Finds the area under the ROC curve essentially via a sum of Reimann rectangles.
    * From (James, Zaichuan, John)'s own project code.

    Args: 
        y: True labels.
        p_y_hat: Probabilities of the predicted labels.

    Returns the area under the ROC curve as a float.
    """
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    roc_pairs.sort(key = lambda x: x[0])
    
    # Debugging code to plot the ROC curve
    #plt.title("ROC Curve")
    #plt.plot([p[0] for p in roc_pairs], [p[1] for p in roc_pairs])
    #plt.show()

    area = 0
    last_pair = roc_pairs[0]
    for i in range(1, len(roc_pairs)):
        next_pair = roc_pairs[i]
        area += ((last_pair[1] + next_pair[1]) / 2) * abs(next_pair[0] - last_pair[0])
        last_pair = next_pair
    
    return area

def unbag_MIL_data(X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray):
    """
    Simply flattens the labels (y) so that they may correspond to the examples in X irrespective of bags (on the instance level).

    Args:
        X: examples matrix
        bag_indices: list of lists that represent bags.  Elements of sublists are indices of X.
        y: labels vector

    Returns:
        X: unmodified. see above ^
        new_y: flattened y.
    """
    new_y = np.zeros(np.shape(X)[0], dtype=int)
    for indices, bag_label in zip(bag_indices, y):
        for index in indices:
            new_y[index] = bag_label
    return X, new_y
