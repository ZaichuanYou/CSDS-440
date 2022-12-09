"""
Created on 11/27/22 by Ethan Frank

Runs all my algorithms on a data set(s). Prints results for further processing
"""

# I don't know why python won't just use the current working directory as part of the python path
import sys
sys.path.append("code")

from typing import List
import argparse
import os.path

import numpy as np
import matplotlib.pyplot as plt

from group.data_io import load_data
from group import util

from edf32.distance_metrics import DistanceMetric

from edf32.citation_knn import CitationKNN
from edf32.bayesian_knn import BayesianKNN
from edf32.naive_knn import NaiveKNN
from jpm221.em_dd import EMDDClassifier
from jkm100.apr import APRClassifier
from jkm100.sbMIL import sbMILClassifier
from zxy456.MI_SVM import SVM
from zxy456.MIL_dis import MIL_dis
# from zxy456.MIL_dis import //;

from matplotlib.font_manager import FontProperties

algo_list = 'MI_SVM(zxy456), em_dd, em_dd_extension, yards, yards_extension, yards_extension2 (jpm221), apr, sbMIL (jkm100), bayesian_KNN, citation_KNN, or naive_KNN (edf32)'


def display_data_in_table(results: dict):
    algorithms = list(results.keys())
    datasets = list(results[algorithms[0]].keys())

    fig, ax = plt.subplots()
    ax.set_axis_off()

    formated_values = []
    for algorithm in algorithms:
        values = []
        for dataset in datasets:
            values.append(f"{results[algorithm][dataset]['mean']:.2f}±{results[algorithm][dataset]['std']:.2f}")
        formated_values.append(values)

    table = ax.table(colLabels=datasets,
                     rowLabels=algorithms,
                     cellText=formated_values,
                     rowLoc='right',
                     loc='center')

    # Collect the values into an array to find the max and bold it
    value_array = np.empty((len(algorithms), len(datasets)))
    for i, algorithm in enumerate(algorithms):
        for j, dataset in enumerate(datasets):
            value_array[i, j] = results[algorithm][dataset]['mean']

    max_values = np.argmax(value_array, axis=0)
    for (row, col), cell in table.get_celld().items():
        if col != -1 and row == max_values[col] + 1:  # Apparently headers count towards the indices, so need to add 1
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    plt.show()


def main(data_folder: str, algorithm: str, datasets: List[str]):
    np.random.seed(0)

    for data_file in datasets:
        X, bag_indices, y = load_data(data_folder, data_file)
        splits = util.cv_split(bag_indices, y, folds=5, stratified=True)

        classifier = None
        # ETHAN'S (edf32)
        if algorithm.casefold() == 'citation_KNN'.casefold():
            classifier = CitationKNN(distance_metric=DistanceMetric.MIN_HAUSDORFF, r=3, c=5)
        elif algorithm.casefold() == 'bayesian_KNN'.casefold():
            classifier = BayesianKNN(DistanceMetric.MIN_HAUSDORFF, k=3)
        elif algorithm.casefold() == 'naive_KNN'.casefold():
            classifier = NaiveKNN(DistanceMetric.MIN_HAUSDORFF, k_neighbors=3)
        # JOHN'S (jkm100)
        elif algorithm.casefold() == 'apr'.casefold():
            classifier = APRClassifier()
        elif algorithm.casefold() == 'sbMIL'.casefold():
            classifier = sbMILClassifier()
        # ZAICHUAN'S (zxy456)
        elif algorithm.casefold() == 'MI_SVM'.casefold():
            classifier = SVM(0.01, 0.1)
        elif algorithm.casefold() == 'MIL_dis'.casefold():
            classifier = MIL_dis()
        # JAMES'S (jpm221)
            # all of James's classifiers!
        else:
            raise argparse.ArgumentError('you did not correctly specify an algorithm... try one of these: ' + algo_list)

        print(f"Running {type(classifier).__name__} on {data_file}")
        accs = []
        precs = []
        recs = []
        for bag_indices_train, y_train, bag_indices_test, y_test in splits:
            classifier.fit(X, bag_indices_train, y_train)

            y_hat = classifier.predict(X, bag_indices_test)
            acc = util.accuracy(y_test, y_hat)
            prec = util.precision(y_test, y_hat)
            rec = util.recall(y_test, y_hat)

            accs.append(acc)
            precs.append(prec)
            recs.append(rec)

            print(f"Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}")
        print(f"Acc: {np.mean(accs):.2}, {np.std(accs):.2}")
        print(f"Prec: {np.mean(precs):.2}, {np.std(precs):.2}")
        print(f"Rec: {np.mean(recs):.2}, {np.std(recs):.2}")

        print()


if __name__ == '__main__':
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description="Run all of the group\'s algorithms on various datasets.")
    parser.add_argument('folder', metavar='folder', type=str, help='The folder that stores the datasets.')
    parser.add_argument('algo', metavar='algo', type=str, help='Which algos to run... choose from these: ' + algo_list)
    parser.add_argument('datasets', metavar='datasets', type=str, help='Comma separated list of dataset names.')
    args = parser.parse_args()

    data_folder = os.path.expanduser(args.folder)
    algorithm = args.algo
    data_sets = args.datasets.split(",")

    main(data_folder, algorithm, data_sets)
