"""
Created on 11/22/2022

Implemented MIL based on bag dissimilarities proposed by Veronika Cheplygina, David M.J. Tax, Marco Loog

Bag dissimilarities is an extention of Citation-k NN proposed in 2014 by Veronika Cheplygina
Bag dissimilarities is an instance-space algorithm that classifies bags under the assumption that
- a bag is positive if it contains at least one 'positive instance', and
- additional distribution assumtion specific to each sample
In this approach I am using Muck dataset which is a point set distribution
So the approaches are specified for point set distribution
"""
import numpy as np
import pandas as pd
from group.util import cv_split, accuracy, precision, recall
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append(".")
#from group import util
from group import data_io
from typing import List, Tuple
from group.classifier import MILClassifier


def reshape_Euclid(data, method):
    new_set = np.zeros((len(data),len(data)))
    for a in range(0, len(data)):
        for b in range(0, len(data)):
            new_set[a][b]=method(data[a], data[b])
    return new_set

def Euclid_dis_mean_min(a,b):
    min = 999999999999999999
    for bi in b:
        temp_min = 0
        for ai in a:
            temp_min += np.sqrt(((bi - ai)**2).sum())
        temp_min = temp_min/len(a)
        if temp_min < min:
            min = temp_min
    return min

def Euclid_dis_min_min(a,b):
    min = 999999999999999999
    for bi in b:
        for ai in a:
            temp_min = np.sqrt(((bi - ai)**2).sum())
            if temp_min < min:
                min = temp_min
    return min

def Eucild_dis_mean_mean(a,b):
    temp_min=0
    for bi in b:
        for ai in a:
            temp_min += np.sqrt(((bi - ai)**2).sum())
    return temp_min/len(a)/len(b)


def load_data(data_dir):
    data = pd.read_csv(data_dir, index_col=False, header=None)

    new_data = {}
    for index, value in enumerate(data[1].value_counts()):
        temp = {}
        temp["class"] = data.loc[data[1]==index+1][0].mean()
        temp["data"] = data.loc[data[1]==index+1].drop([0,1], axis=1)
        new_data[index+1]=temp

    bags=[]
    labels=[]
    for a in new_data:
        bags.append(new_data[a]["data"])
        labels.append(new_data[a]["class"])
    
    return bags, labels

def validate():
    X, bags, labels = data_io.load_data("/content/drive/Othercomputers/My Laptop/Final Project/csds440-f22-p3/code/group", "elephant.csv")
    NUM_FOLDS = 5
    folds = cv_split(bags, labels, NUM_FOLDS, stratified=True)

    acc = []
    prec = []
    rec = []
    for i, fold in enumerate(folds):
        train_x, train_y, test_x, test_y = fold
        model = MIL_dis()
        model.fit(X, train_x, train_y)
        y_hat = model.predict(X, test_x)
        acc.append(accuracy(test_y, y_hat))
        prec.append(precision(test_y, y_hat))
        rec.append(recall(test_y, y_hat))

    print(f"Acc: {np.mean(acc):.2}, {np.std(acc):.2}")
    print(f"Prec: {np.mean(prec):.2}, {np.std(prec):.2}")
    print(f"Rec: {np.mean(rec):.2}, {np.std(rec):.2}")

class MIL_dis(MILClassifier):
    def fit(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:
        self.bag = []
        self.labels = []
        for a in range(0, len(bag_indices)):
            self.bag.append(X[bag_indices[a]])
            self.labels.append(y[a])
        

    def predict(self, X: np.ndarray, bag_indices: List[np.ndarray]) -> np.ndarray:
        count = 0
        for a in bag_indices:
            count +=1
            self.bag.append(X[a])
        bags = reshape_Euclid(self.bag, Euclid_dis_min_min)
        train_bags = bags[:len(self.labels)]
        labels = self.labels
        model = LogisticRegression(max_iter=5000).fit(train_bags, labels)
        test_bags = bags[len(self.labels):]
        y_hat = model.predict(test_bags)
        return y_hat