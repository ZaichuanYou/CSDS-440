"""
Created on 11/22/2022

Implemented Support Vector Machines for Multi ple-Instance Learning proposed by S. Andrews, I. Tsochantaridis and T Hofmann

This approach tries to learn the true label for each instance. 
To do that they used the imbalanced information provided by positive and negative bags to train the classifier. 
They will keep correcting the label of instance in positive bags using the instance from negative bag until there 
exist at least one instance in each positive bag which been classified as positive.
"""

import numpy as np
import pandas as pd

import sys
sys.path.append(".")
from group import util
from util import cv_split, accuracy, precision, recall
from group.data_io import load_data
from typing import List, Tuple
from group.classifier import MILClassifier
from sklearn.svm import SVC


class SVM(MILClassifier):
    
    def __init__(self, learning_rate, lamb) -> None:
        self.learning_rate = learning_rate
        self.lamb = lamb

    def set_weight(self, w):
        self.w = w

    def fit(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:
        for l in range(0, len(y)):
            if y[l] == 0:
                y[l]=-1

        y_label = {}
        for bag in range(0, len(y)):
            for example in bag_indices[bag]:
                y_label[example] = y[bag]
        
        self.w = np.zeros(len(X[0]))
        changed = True


        epoch = 0
        while changed and epoch<10000:
            changed = False
            epoch+=1
            
            for bag in bag_indices:
                for e in bag:
                    if (y_label[e] * np.dot(X[e], self.w)) < 1:
                        self.w = self.w + self.learning_rate * ((X[e] * y_label[e]) + (-2 * self.lamb * self.w))
                    else:
                        self.w = self.w + self.learning_rate * (-2 * self.lamb * self.w)

            for bag in range(0, len(bag_indices)):
                if y[bag]==1:
                    for example in bag_indices[bag]:
                        y_label[example] = np.sign(self.predict_instance(X[example]))


            for index in range(0, len(bag_indices)):
                if y[index]==-1:
                    continue
                y_hat = self.predict_instance(X[bag_indices[index]])
                
                y_hat = np.sign(y_hat)
       
                if (y_hat+1).sum() == 0:
                    max_ind = bag_indices[index][0]
                    max=self.predict_instance(X[bag_indices[index]])[0]

                    for example in range(0, len(bag_indices[index])):
                        result = self.predict_instance(X[bag_indices[index][example]])
                        if result > max:
                            max=result
                            max_ind=bag_indices[index][example]

                    if y_label[max_ind]!=1:
                        changed=True
                        y_label[max_ind]=1

    

    def fit_mi_Euclid(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:

        for l in range(0, len(y)):
            if y[l] == 0:
                y[l]=-1

        y_label = {}
        for bag in range(0, len(y)):
            for example in bag_indices[bag]:
                y_label[example] = y[bag]

        self.w = np.zeros(len(X[0]))
        changed = True

        maxs = {}
        for bag in range(0, len(bag_indices)):
            max_ind = -1
            max=0

            for example in bag_indices[bag]:
                result = 0
                current = X[example]
                for bag_i in range(0, len(bag_indices)):
                    if y[bag_i] == 1:
                        continue
                    for e in bag_indices[bag_i]:
                        result += np.sqrt(((e-current)**2).sum())

                if result >= max:
                    max=result
                    max_ind= example
            maxs[bag] = max_ind
        
        epoch = 0
        while changed and epoch<10000:
            changed = False
            epoch+=1
            
            for bag in bag_indices:
                for e in bag:
                    if (y_label[e] * np.dot(X[e], self.w)) < 1:
                        self.w = self.w + self.learning_rate * ((X[e] * y_label[e]) + (-2 * self.learning_rate * self.w))
                    else:
                        self.w = self.w + self.learning_rate * (-2 * self.learning_rate * self.w)
            for bag in range(0, len(bag_indices)):
                if y[bag]==1:
                    for example in bag_indices[bag]:
                        y_label[example] = np.sign(np.dot(X[example], self.w))

            for index in range(0, len(bag_indices)):
                if y[index]==-1:
                    continue
                y_hat = self.predict_instance(X[bag_indices[index]])

                y_hat = np.sign(y_hat)
                if (y_hat+1).sum() == 0:
                    max_ind = maxs[index]
                    if y_label[max_ind]!=1:
                        changed=True
                        y_label[max_ind]=1



    def fit_MI(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:

        for l in range(0, len(y)):
            if y[l] == 0:
                y[l]=-1

        y_label = {}
        for bag in range(0, len(y)):
            for example in bag_indices[bag]:
                y_label[example] = y[bag]
        
        x_pos = []
        y_pos = []
        for bag in range(0, len(bag_indices)):
            if y[bag] == 1:
                x_pos.append(np.mean(X[bag_indices[bag]], axis=0))
                y_pos.append(1)

        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)


        self.w = np.zeros(len(X[0]))
        changed = True

        epoch = 0
        while changed and epoch<10000:
            changed = False
            epoch+=1
            
            for bag in range(0, len(x_pos)):
                if (y_pos[bag] * np.dot(x_pos[bag], self.w)) < 1:
                    self.w = self.w + self.learning_rate * ((x_pos[bag] * y_pos[bag]) + (-2 * self.lamb * self.w))
                else:
                    self.w = self.w + self.learning_rate * (-2 * self.lamb * self.w)

    
    def fit_MI_mean(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> None:

        for l in range(0, len(y)):
            if y[l] == 0:
                y[l]=-1

        
        x_new = []
        y_new = []
        for bag in range(0, len(bag_indices)):
            if y[bag] == 1:
                x_new.append(np.mean(X[bag_indices[bag]], axis=0))
                y_new.append(1)
            else:
                for example in X[bag_indices[bag]]:
                    x_new.append(example)
                    y_new.append(-1)

        x_new = np.array(x_new)
        y_new = np.array(y_new)


        self.w = np.zeros(len(X[0]))

        epoch=0
        while epoch<10000:
            epoch += 1
            for bag in range(0, len(x_new)):
                if (y_new[bag] * np.dot(x_new[bag], self.w)) < 1:
                    self.w = self.w + self.learning_rate * ((x_new[bag] * y_new[bag]) + (-2 * self.lamb * self.w))
                else:
                    self.w = self.w + self.learning_rate * (-2 * self.lamb * self.w)


    def predict_instance(self, X):
        Y = np.dot(X, self.w)
        return Y


    def predict(self, X, bag_indices):
        Y = np.zeros(len(bag_indices))
        for bag in range(0, len(bag_indices)):
            result = self.predict_instance(X[bag_indices[bag]])
            for a in result:
                if a > 0:
                    Y[bag] = 1
        return Y
    
    def find_index(self, bag_indices, y, index):
        count = 0
        for bag in range(0, len(bag_indices)):
            if bag == index:
                return count
            elif y[bag]==1:
                count += 1


def new_val():
    X, bags, labels= load_data("/content/drive/Othercomputers/My Laptop/Final Project/csds440-f22-p3/code/group", "musk1.csv")

    NUM_FOLDS = 10
    folds = cv_split(bags, labels, NUM_FOLDS, stratified=True)

    acc = []
    prec = []
    rec = []
    for i, fold in enumerate(folds):
        train_x, train_y, test_x, test_y = fold
        model = SVM(0.01, 0.1)
        model.fit_MI_mean(X, train_x, train_y)
        y_hat = model.predict(X, test_x)
        acc.append(accuracy(test_y, y_hat))
        prec.append(precision(test_y, y_hat))
        rec.append(recall(test_y, y_hat))
    print(f"Acc: {np.mean(acc):.2}, {np.std(acc):.2}")
    print(f"Prec: {np.mean(prec):.2}, {np.std(prec):.2}")
    print(f"Rec: {np.mean(rec):.2}, {np.std(rec):.2}")


