#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import time
import random
import sklearn
from sys import argv
# from numpy import *
from pandas import DataFrame
import os
import math
from operator import itemgetter, attrgetter, methodcaller
type = sys.getfilesystemencoding()
from sklearn.model_selection import KFold
import random
from sklearn import metrics
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import graphviz
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

from numpy import interp

from sklearn.inspection import permutation_importance
from sklearn.datasets import fetch_olivetti_faces

from sklearn.metrics import RocCurveDisplay

from sklearn.model_selection import RepeatedStratifiedKFold

import keras
import seaborn as sns
#kf = KFold(n_splits=10, shuffle=True, random_state=0)
for a in [1,2,3]:
     writer_test = pd.ExcelWriter("..//data/new/pH_result_test_1" + "_" + "%d" %a + ".xlsx")
     writer_train = pd.ExcelWriter("..//data/new/pH_result_train_1" + "_" + "%d" %a + ".xlsx")
     writer_E_v = pd.ExcelWriter("..//data/new/pH_result_E_v" + "_" + "%d" %a + ".xlsx")

     writer_test_F = pd.ExcelWriter("..//data/new/pH_result_test_F_1" + "_" + "%d" % a + ".xlsx")
     writer_train_F = pd.ExcelWriter("..//data/new/pH_result_train_F_1" + "_" + "%d" % a + ".xlsx")
     writer_E_v_F = pd.ExcelWriter("..//data/new/pH_result_E_v_F" + "_" + "%d" % a + ".xlsx")
     #writer_follow_up_2 = pd.ExcelWriter("..//data/pH_result_follow-up-2_1" + "_" + "%d" %a + ".xlsx")
     #writer_follow_up_3 = pd.ExcelWriter("..//data/pH_result_follow-up-3_1" + "_" + "%d" %a + ".xlsx")
     for b in [1,2]:
        pH = pd.read_excel("..//data/pH_data_split_" + "%d"%a+".xlsx",sheet_name="pH_"+"%d"%a+"_"+"%d"%b,header=0,names=None,index_col=0)
        #pH = pd.read_excel("..//data/pH_sample" + ".xlsx", sheet_name="pH_" + "%d" % a + "_" + "%d" % b,header=0, names=None, index_col=0)
        data = np.array(pH)
        X = np.array(pH.iloc[:, 0:-1])
        y = np.array(pH.iloc[:, -1])
        pH_E_v = pd.read_excel("..//data/pH_data_v.xlsx",sheet_name="pH_" + "%d"%a + "_" + "%d"%b, header=0,names=None, index_col=0)
        E_v = np.array(pH_E_v.iloc[0:,1:])

        # print(follow_up_1)
        # follow_up_2 = np.array(pH_follow_up.iloc[6:,1:])
        #follow_up_3 = np.array(pH_follow_up.iloc[6:, 1:])
#
#         rL_pH.dropna(inplace=True)
#         data_4 = np.array(rL_pH)
#         #print(np.isnan(rL_pH).any())
#
#         X_2 = data_4[:, 0:-1]
#         y_2 = data_4[:, -1]
#
#         #rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0)
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        train_A_DT_100 = []
        train_A_NB_100 = []
        train_A_kNN_100 = []
        train_A_SVM_100 = []
        train_A_RF_100 = []
        train_A_ANN_100 = []

        train_F_DT_100 = []
        train_F_NB_100 = []
        train_F_kNN_100 = []
        train_F_SVM_100 = []
        train_F_RF_100 = []
        train_F_ANN_100 = []

        train_P_DT_100 = []
        train_P_NB_100 = []
        train_P_kNN_100 = []
        train_P_SVM_100 = []
        train_P_RF_100 = []
        train_P_ANN_100 = []

        train_R_DT_100 = []
        train_R_NB_100 = []
        train_R_kNN_100 = []
        train_R_SVM_100 = []
        train_R_RF_100 = []
        train_R_ANN_100 = []

        train_ROC_DT_100 = []
        train_ROC_NB_100 = []
        train_ROC_kNN_100 = []
        train_ROC_SVM_100 = []
        train_ROC_RF_100 = []
        train_ROC_ANN_100 = []

        E_v_A_DT_100 = []
        E_v_A_NB_100 = []
        E_v_A_kNN_100 = []
        E_v_A_SVM_100 = []
        E_v_A_RF_100 = []
        E_v_A_ANN_100 = []

        E_v_F_DT_100 = []
        E_v_F_NB_100 = []
        E_v_F_kNN_100 = []
        E_v_F_SVM_100 = []
        E_v_F_RF_100 = []
        E_v_F_ANN_100 = []

        E_v_P_DT_100 = []
        E_v_P_NB_100 = []
        E_v_P_kNN_100 = []
        E_v_P_SVM_100 = []
        E_v_P_RF_100 = []
        E_v_P_ANN_100 = []

        E_v_R_DT_100 = []
        E_v_R_NB_100 = []
        E_v_R_kNN_100 = []
        E_v_R_SVM_100 = []
        E_v_R_RF_100 = []
        E_v_R_ANN_100 = []

        E_v_ROC_DT_100 = []
        E_v_ROC_NB_100 = []
        E_v_ROC_kNN_100 = []
        E_v_ROC_SVM_100 = []
        E_v_ROC_RF_100 = []
        E_v_ROC_ANN_100 = []

        E_v_F_A_DT_100 = []
        E_v_F_A_NB_100 = []
        E_v_F_A_kNN_100 = []
        E_v_F_A_SVM_100 = []
        E_v_F_A_RF_100 = []
        E_v_F_A_ANN_100 = []

        E_v_F_F_DT_100 = []
        E_v_F_F_NB_100 = []
        E_v_F_F_kNN_100 = []
        E_v_F_F_SVM_100 = []
        E_v_F_F_RF_100 = []
        E_v_F_F_ANN_100 = []

        E_v_F_P_DT_100 = []
        E_v_F_P_NB_100 = []
        E_v_F_P_kNN_100 = []
        E_v_F_P_SVM_100 = []
        E_v_F_P_RF_100 = []
        E_v_F_P_ANN_100 = []

        E_v_F_R_DT_100 = []
        E_v_F_R_NB_100 = []
        E_v_F_R_kNN_100 = []
        E_v_F_R_SVM_100 = []
        E_v_F_R_RF_100 = []
        E_v_F_R_ANN_100 = []

        E_v_F_ROC_DT_100 = []
        E_v_F_ROC_NB_100 = []
        E_v_F_ROC_kNN_100 = []
        E_v_F_ROC_SVM_100 = []
        E_v_F_ROC_RF_100 = []
        E_v_F_ROC_ANN_100 = []

        # follow_up_2_A_DT_100 = []
        # follow_up_2_A_NB_100 = []
        # follow_up_2_A_kNN_100 = []
        # follow_up_2_A_SVM_100 = []
        # follow_up_2_A_RF_100 = []
        # follow_up_2_A_ANN_100 = []
        #
        # follow_up_2_F_DT_100 = []
        # follow_up_2_F_NB_100 = []
        # follow_up_2_F_kNN_100 = []
        # follow_up_2_F_SVM_100 = []
        # follow_up_2_F_RF_100 = []
        # follow_up_2_F_ANN_100 = []
        #
        # follow_up_2_P_DT_100 = []
        # follow_up_2_P_NB_100 = []
        # follow_up_2_P_kNN_100 = []
        # follow_up_2_P_SVM_100 = []
        # follow_up_2_P_RF_100 = []
        # follow_up_2_P_ANN_100 = []
        #
        # follow_up_2_R_DT_100 = []
        # follow_up_2_R_NB_100 = []
        # follow_up_2_R_kNN_100 = []
        # follow_up_2_R_SVM_100 = []
        # follow_up_2_R_RF_100 = []
        # follow_up_2_R_ANN_100 = []
        #
        # follow_up_2_ROC_DT_100 = []
        # follow_up_2_ROC_NB_100 = []
        # follow_up_2_ROC_kNN_100 = []
        # follow_up_2_ROC_SVM_100 = []
        # follow_up_2_ROC_RF_100 = []
        # follow_up_2_ROC_ANN_100 = []

        # follow_up_3_A_DT_100 = []
        # follow_up_3_A_NB_100 = []
        # follow_up_3_A_kNN_100 = []
        # follow_up_3_A_SVM_100 = []
        # follow_up_3_A_RF_100 = []
        # follow_up_3_A_ANN_100 = []
        #
        # follow_up_3_F_DT_100 = []
        # follow_up_3_F_NB_100 = []
        # follow_up_3_F_kNN_100 = []
        # follow_up_3_F_SVM_100 = []
        # follow_up_3_F_RF_100 = []
        # follow_up_3_F_ANN_100 = []
        #
        # follow_up_3_P_DT_100 = []
        # follow_up_3_P_NB_100 = []
        # follow_up_3_P_kNN_100 = []
        # follow_up_3_P_SVM_100 = []
        # follow_up_3_P_RF_100 = []
        # follow_up_3_P_ANN_100 = []
        #
        # follow_up_3_R_DT_100 = []
        # follow_up_3_R_NB_100 = []
        # follow_up_3_R_kNN_100 = []
        # follow_up_3_R_SVM_100 = []
        # follow_up_3_R_RF_100 = []
        # follow_up_3_R_ANN_100 = []
        #
        # follow_up_3_ROC_DT_100 = []
        # follow_up_3_ROC_NB_100 = []
        # follow_up_3_ROC_kNN_100 = []
        # follow_up_3_ROC_SVM_100 = []
        # follow_up_3_ROC_RF_100 = []
        # follow_up_3_ROC_ANN_100 = []

        test_A_DT_100 = []
        test_A_NB_100 = []
        test_A_kNN_100 = []
        test_A_SVM_100 = []
        test_A_RF_100 = []
        test_A_ANN_100 = []

        test_F_DT_100 = []
        test_F_NB_100 = []
        test_F_kNN_100 = []
        test_F_SVM_100 = []
        test_F_RF_100 = []
        test_F_ANN_100 = []

        test_P_DT_100 = []
        test_P_NB_100 = []
        test_P_kNN_100 = []
        test_P_SVM_100 = []
        test_P_RF_100 = []
        test_P_ANN_100 = []

        test_R_DT_100 = []
        test_R_NB_100 = []
        test_R_kNN_100 = []
        test_R_SVM_100 = []
        test_R_RF_100 = []
        test_R_ANN_100 = []

        test_ROC_DT_100 = []
        test_ROC_NB_100 = []
        test_ROC_kNN_100 = []
        test_ROC_SVM_100 = []
        test_ROC_RF_100 = []
        test_ROC_ANN_100 = []

        test_F_A_DT_100 = []
        test_F_A_NB_100 = []
        test_F_A_kNN_100 = []
        test_F_A_SVM_100 = []
        test_F_A_RF_100 = []
        test_F_A_ANN_100 = []

        test_F_F_DT_100 = []
        test_F_F_NB_100 = []
        test_F_F_kNN_100 = []
        test_F_F_SVM_100 = []
        test_F_F_RF_100 = []
        test_F_F_ANN_100 = []

        test_F_P_DT_100 = []
        test_F_P_NB_100 = []
        test_F_P_kNN_100 = []
        test_F_P_SVM_100 = []
        test_F_P_RF_100 = []
        test_F_P_ANN_100 = []

        test_F_R_DT_100 = []
        test_F_R_NB_100 = []
        test_F_R_kNN_100 = []
        test_F_R_SVM_100 = []
        test_F_R_RF_100 = []
        test_F_R_ANN_100 = []

        test_F_ROC_DT_100 = []
        test_F_ROC_NB_100 = []
        test_F_ROC_kNN_100 = []
        test_F_ROC_SVM_100 = []
        test_F_ROC_RF_100 = []
        test_F_ROC_ANN_100 = []

        train_F_A_DT_100 = []
        train_F_A_NB_100 = []
        train_F_A_kNN_100 = []
        train_F_A_SVM_100 = []
        train_F_A_RF_100 = []
        train_F_A_ANN_100 = []

        train_F_F_DT_100 = []
        train_F_F_NB_100 = []
        train_F_F_kNN_100 = []
        train_F_F_SVM_100 = []
        train_F_F_RF_100 = []
        train_F_F_ANN_100 = []

        train_F_P_DT_100 = []
        train_F_P_NB_100 = []
        train_F_P_kNN_100 = []
        train_F_P_SVM_100 = []
        train_F_P_RF_100 = []
        train_F_P_ANN_100 = []

        train_F_R_DT_100 = []
        train_F_R_NB_100 = []
        train_F_R_kNN_100 = []
        train_F_R_SVM_100 = []
        train_F_R_RF_100 = []
        train_F_R_ANN_100 = []

        train_F_ROC_DT_100 = []
        train_F_ROC_NB_100 = []
        train_F_ROC_kNN_100 = []
        train_F_ROC_SVM_100 = []
        train_F_ROC_RF_100 = []
        train_F_ROC_ANN_100 = []

        result_matrix_DT = np.zeros((3, 3))
        result_matrix_NB = np.zeros((3, 3))
        result_matrix_kNN = np.zeros((3, 3))
        result_matrix_SVM = np.zeros((3, 3))
        result_matrix_RF = np.zeros((3, 3))
        result_matrix_ANN = np.zeros((3, 3))

        result_matrix_DT_E_v = np.zeros((3, 3))
        result_matrix_NB_E_v = np.zeros((3, 3))
        result_matrix_kNN_E_v = np.zeros((3, 3))
        result_matrix_SVM_E_v = np.zeros((3, 3))
        result_matrix_RF_E_v = np.zeros((3, 3))
        result_matrix_ANN_E_v = np.zeros((3, 3))

        result_matrix_DT_E_v_F = np.zeros((3, 3))
        result_matrix_NB_E_v_F = np.zeros((3, 3))
        result_matrix_kNN_E_v_F = np.zeros((3, 3))
        result_matrix_SVM_E_v_F = np.zeros((3, 3))
        result_matrix_RF_E_v_F = np.zeros((3, 3))
        result_matrix_ANN_E_v_F = np.zeros((3, 3))

        result_matrix_DT_test_F = np.zeros((3, 3))
        result_matrix_NB_test_F = np.zeros((3, 3))
        result_matrix_kNN_test_F = np.zeros((3, 3))
        result_matrix_SVM_test_F = np.zeros((3, 3))
        result_matrix_RF_test_F = np.zeros((3, 3))
        result_matrix_ANN_test_F = np.zeros((3, 3))

        result_matrix_DT_train_F = np.zeros((3, 3))
        result_matrix_NB_train_F = np.zeros((3, 3))
        result_matrix_kNN_train_F = np.zeros((3, 3))
        result_matrix_SVM_train_F = np.zeros((3, 3))
        result_matrix_RF_train_F = np.zeros((3, 3))
        result_matrix_ANN_train_F = np.zeros((3, 3))


        n = 0
        for train_index, test_index in kf.split(X, y):
            # print(train_index)
            n = n + 1
            print("迭代次数:", n)
            #
            trainset = data[train_index]
            # print(trainset)
            testset = data[test_index]
            X_train = trainset[:, 0:-1]
            y_train = trainset[:, -1]
            X_test = testset[:, 0:-1]
            y_test = testset[:, -1]
            # X_train = X[train_index]
            # y_train = y[train_index]
            # X_test = X[test_index]
            # y_test = y[test_index]

            X_E_v = E_v[:,0:-1]
            y_E_v = E_v[:,-1]
            # X_follow_up_2 = follow_up_2[:,0:-1]
            # y_follow_up_2 = follow_up_2[:,-1]
            # X_follow_up_3 = follow_up_3[:, 0:-1]
            # y_follow_up_3 = follow_up_3[:, -1]
            #
            if a==3 and b==2:
                #E_v_F = np.array(pH_E_v.iloc[0:, [1, 2, 6, 7, 8, 9]])
                X_E_v_F = E_v[:, [0, 1, 5, 6, 7, 8]]
                y_E_v_F = E_v[:, -1]
                X_train_F = trainset[:, [0, 1, 5, 6, 7, 8]]
                y_train_F = trainset[:, -1]
                X_test_F = testset[:, [0, 1, 5, 6, 7, 8]]
                y_test_F = testset[:, -1]
            else:
                X_E_v_F = X_E_v
                y_E_v_F = y_E_v
                X_train_F = X_train
                y_train_F = y_train
                X_test_F = X_test
                y_test_F = y_test
            #
            if a==1 and b==1:
              classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0003, min_samples_leaf=3).fit(X_train, y_train)
              classifier_NB = GaussianNB().fit(X_train, y_train)
              classifier_kNN = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)
              classifier_SVM = SVC(coef0=0.0546, C=28, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
              classifier_RF = RandomForestClassifier(n_estimators=14, max_depth=6, min_samples_leaf=22).fit(X_train, y_train)
              classifier_ANN = MLPClassifier(learning_rate_init=0.0250, batch_size=32).fit(X_train, y_train)

              classifier_DT_F = DecisionTreeClassifier(ccp_alpha=0.0003, min_samples_leaf=3).fit(X_train_F, y_train_F)
              classifier_NB_F = GaussianNB().fit(X_train_F, y_train_F)
              classifier_kNN_F = KNeighborsClassifier(n_neighbors=7).fit(X_train_F, y_train_F)
              classifier_SVM_F = SVC(coef0=0.0546, C=28, decision_function_shape='ovr', probability=True).fit(X_train_F, y_train_F)
              classifier_RF_F = RandomForestClassifier(n_estimators=14, max_depth=6, min_samples_leaf=22).fit(X_train_F, y_train_F)
              classifier_ANN_F = MLPClassifier(learning_rate_init=0.0250, batch_size=32).fit(X_train_F, y_train_F)

            if a==1 and b==2:
              classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0055, min_samples_leaf=11).fit(X_train, y_train)
              classifier_NB = GaussianNB().fit(X_train, y_train)
              classifier_kNN = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
              classifier_SVM = SVC(coef0=8.1031, C=15, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
              classifier_RF = RandomForestClassifier(n_estimators=40, max_depth=9, min_samples_leaf=10).fit(X_train, y_train)
              classifier_ANN = MLPClassifier(learning_rate_init=0.0250, batch_size=22).fit(X_train, y_train)

              classifier_DT_F = DecisionTreeClassifier(ccp_alpha=0.0055, min_samples_leaf=11).fit(X_train_F, y_train_F)
              classifier_NB_F = GaussianNB().fit(X_train_F, y_train_F)
              classifier_kNN_F = KNeighborsClassifier(n_neighbors=9).fit(X_train_F, y_train_F)
              classifier_SVM_F = SVC(coef0=8.1031, C=15, decision_function_shape='ovr', probability=True).fit(X_train_F, y_train_F)
              classifier_RF_F = RandomForestClassifier(n_estimators=40, max_depth=9, min_samples_leaf=10).fit(X_train_F, y_train_F)
              classifier_ANN_F = MLPClassifier(learning_rate_init=0.0250, batch_size=22).fit(X_train_F, y_train_F)

            if a==2 and b==1:
              classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0020, min_samples_leaf=7).fit(X_train, y_train)
              classifier_NB = GaussianNB().fit(X_train, y_train)
              classifier_kNN = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
              classifier_SVM = SVC(coef0=0.0201, C=10, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
              classifier_RF = RandomForestClassifier(n_estimators=46, max_depth=8, min_samples_leaf=8).fit(X_train, y_train)
              classifier_ANN = MLPClassifier(learning_rate_init=0.0250, batch_size=31).fit(X_train, y_train)

              classifier_DT_F = DecisionTreeClassifier(ccp_alpha=0.0020, min_samples_leaf=7).fit(X_train_F, y_train_F)
              classifier_NB_F = GaussianNB().fit(X_train_F, y_train_F)
              classifier_kNN_F = KNeighborsClassifier(n_neighbors=2).fit(X_train_F, y_train_F)
              classifier_SVM_F = SVC(coef0=0.0201, C=10, decision_function_shape='ovr', probability=True).fit(X_train_F, y_train_F)
              classifier_RF_F = RandomForestClassifier(n_estimators=46, max_depth=8, min_samples_leaf=8).fit(X_train_F, y_train_F)
              classifier_ANN_F = MLPClassifier(learning_rate_init=0.0250, batch_size=31).fit(X_train_F, y_train_F)

            if a==2 and b==2:
              classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0403, min_samples_leaf=9).fit(X_train, y_train)
              classifier_NB = GaussianNB().fit(X_train, y_train)
              classifier_kNN = KNeighborsClassifier(n_neighbors=34).fit(X_train, y_train)
              classifier_SVM = SVC(coef0=0.0074, C=2, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
              classifier_RF = RandomForestClassifier(n_estimators=14, max_depth=2, min_samples_leaf=13).fit(X_train, y_train)
              classifier_ANN = MLPClassifier(learning_rate_init=0.1847, batch_size=25).fit(X_train, y_train)

              classifier_DT_F = DecisionTreeClassifier(ccp_alpha=0.0403, min_samples_leaf=9).fit(X_train_F, y_train_F)
              classifier_NB_F = GaussianNB().fit(X_train_F, y_train_F)
              classifier_kNN_F = KNeighborsClassifier(n_neighbors=34).fit(X_train_F, y_train_F)
              classifier_SVM_F = SVC(coef0=0.0074, C=2, decision_function_shape='ovr', probability=True).fit(X_train_F, y_train_F)
              classifier_RF_F = RandomForestClassifier(n_estimators=14, max_depth=2, min_samples_leaf=13).fit(X_train_F, y_train_F)
              classifier_ANN_F = MLPClassifier(learning_rate_init=0.1847, batch_size=25).fit(X_train_F, y_train_F)

            if a==3 and b==1:
              classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0403, min_samples_leaf=15).fit(X_train, y_train)
              classifier_NB = GaussianNB().fit(X_train, y_train)
              classifier_kNN = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)
              classifier_SVM = SVC(coef0=0.0027, C=34, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
              classifier_RF = RandomForestClassifier(n_estimators=15, max_depth=7, min_samples_leaf=14).fit(X_train, y_train)
              classifier_ANN = MLPClassifier(learning_rate_init=0.0250, batch_size=31).fit(X_train, y_train)

              classifier_DT_F = DecisionTreeClassifier(ccp_alpha=0.0403, min_samples_leaf=15).fit(X_train_F, y_train_F)
              classifier_NB_F = GaussianNB().fit(X_train_F, y_train_F)
              classifier_kNN_F = KNeighborsClassifier(n_neighbors=11).fit(X_train_F, y_train_F)
              classifier_SVM_F = SVC(coef0=0.0027, C=34, decision_function_shape='ovr', probability=True).fit(X_train_F, y_train_F)
              classifier_RF_F = RandomForestClassifier(n_estimators=15, max_depth=7, min_samples_leaf=14).fit(X_train_F, y_train_F)
              classifier_ANN_F = MLPClassifier(learning_rate_init=0.0250, batch_size=31).fit(X_train_F, y_train_F)


            if a==3 and b==2:
              classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0003, min_samples_leaf=22).fit(X_train, y_train)
              classifier_NB = GaussianNB().fit(X_train, y_train)
              classifier_kNN = KNeighborsClassifier(n_neighbors=15).fit(X_train, y_train)
              classifier_SVM = SVC(coef0=0.0201, C=5, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
              classifier_RF = RandomForestClassifier(n_estimators=29, max_depth=5, min_samples_leaf=2).fit(X_train, y_train)
              classifier_ANN = MLPClassifier(learning_rate_init=0.0250, batch_size=27).fit(X_train, y_train)

              classifier_DT_F = DecisionTreeClassifier(ccp_alpha=0.0003, min_samples_leaf=22).fit(X_train_F, y_train_F)
              classifier_NB_F = GaussianNB().fit(X_train_F, y_train_F)
              classifier_kNN_F = KNeighborsClassifier(n_neighbors=15).fit(X_train_F, y_train_F)
              classifier_SVM_F = SVC(coef0=0.0201, C=5, decision_function_shape='ovr', probability=True).fit(X_train_F, y_train_F)
              classifier_RF_F = RandomForestClassifier(n_estimators=29, max_depth=5, min_samples_leaf=2).fit(X_train_F, y_train_F)
              classifier_ANN_F = MLPClassifier(learning_rate_init=0.0250, batch_size=27).fit(X_train_F, y_train_F)

            # if a==4 and b==1:
            #   classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0020, min_samples_leaf=7).fit(X_train, y_train)
            #   classifier_NB = GaussianNB().fit(X_train, y_train)
            #   classifier_kNN = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)
            #   classifier_SVM = SVC(coef0=8.1031, C=32, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
            #   classifier_RF = RandomForestClassifier(n_estimators=40, max_depth=5, min_samples_leaf=7).fit(X_train, y_train)
            #   classifier_ANN = MLPClassifier(learning_rate_init=0.0680, batch_size=29).fit(X_train, y_train)
            #
            # if a==4 and b==2:
            #   classifier_DT = DecisionTreeClassifier(ccp_alpha=0.0020, min_samples_leaf=9).fit(X_train, y_train)
            #   classifier_NB = GaussianNB().fit(X_train, y_train)
            #   classifier_kNN = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
            #   classifier_SVM = SVC(coef0=0.1484, C=16, decision_function_shape='ovr', probability=True).fit(X_train, y_train)
            #   classifier_RF = RandomForestClassifier(n_estimators=41, max_depth=7, min_samples_leaf=2).fit(X_train, y_train)
            #   classifier_ANN = MLPClassifier(learning_rate_init=0.0680, batch_size=11).fit(X_train, y_train)


            #训练集预测
            train_y_pred_DT = classifier_DT.predict(X_train)
            train_y_pred_NB = classifier_NB.predict(X_train)
            train_y_pred_kNN = classifier_kNN.predict(X_train)
            train_y_pred_SVM = classifier_SVM.predict(X_train)
            train_y_pred_RF = classifier_RF.predict(X_train)
            train_y_pred_ANN = classifier_ANN.predict(X_train)

            train_y_F_pred_DT = classifier_DT_F.predict(X_train_F)
            train_y_F_pred_NB = classifier_NB_F.predict(X_train_F)
            train_y_F_pred_kNN = classifier_kNN_F.predict(X_train_F)
            train_y_F_pred_SVM = classifier_SVM_F.predict(X_train_F)
            train_y_F_pred_RF = classifier_RF_F.predict(X_train_F)
            train_y_F_pred_ANN = classifier_ANN_F.predict(X_train_F)

            train_y_pred_proba_DT = classifier_DT.predict_proba(X_train)
            train_y_pred_proba_NB = classifier_NB.predict_proba(X_train)
            train_y_pred_proba_kNN = classifier_kNN.predict_proba(X_train)
            train_y_pred_proba_SVM = classifier_SVM.predict_proba(X_train)
            train_y_pred_proba_RF = classifier_RF.predict_proba(X_train)
            train_y_pred_proba_ANN = classifier_ANN.predict_proba(X_train)

            train_y_F_pred_proba_DT = classifier_DT_F.predict_proba(X_train_F)
            train_y_F_pred_proba_NB = classifier_NB_F.predict_proba(X_train_F)
            train_y_F_pred_proba_kNN = classifier_kNN_F.predict_proba(X_train_F)
            train_y_F_pred_proba_SVM = classifier_SVM_F.predict_proba(X_train_F)
            train_y_F_pred_proba_RF = classifier_RF_F.predict_proba(X_train_F)
            train_y_F_pred_proba_ANN = classifier_ANN_F.predict_proba(X_train_F)

            train_A_DT = accuracy_score(y_train, train_y_pred_DT)
            train_A_NB = accuracy_score(y_train, train_y_pred_NB)
            train_A_kNN = accuracy_score(y_train, train_y_pred_kNN)
            train_A_SVM = accuracy_score(y_train, train_y_pred_SVM)
            train_A_RF = accuracy_score(y_train, train_y_pred_RF)
            train_A_ANN = accuracy_score(y_train, train_y_pred_ANN)

            train_A_DT_100.append(train_A_DT)
            train_A_NB_100.append(train_A_NB)
            train_A_kNN_100.append(train_A_kNN)
            train_A_SVM_100.append(train_A_SVM)
            train_A_RF_100.append(train_A_RF)
            train_A_ANN_100.append(train_A_ANN)

            train_F_A_DT = accuracy_score(y_train_F, train_y_F_pred_DT)
            train_F_A_NB = accuracy_score(y_train_F, train_y_F_pred_NB)
            train_F_A_kNN = accuracy_score(y_train_F, train_y_F_pred_kNN)
            train_F_A_SVM = accuracy_score(y_train_F, train_y_F_pred_SVM)
            train_F_A_RF = accuracy_score(y_train_F, train_y_F_pred_RF)
            train_F_A_ANN = accuracy_score(y_train_F, train_y_F_pred_ANN)

            train_F_A_DT_100.append(train_F_A_DT)
            train_F_A_NB_100.append(train_F_A_NB)
            train_F_A_kNN_100.append(train_F_A_kNN)
            train_F_A_SVM_100.append(train_F_A_SVM)
            train_F_A_RF_100.append(train_F_A_RF)
            train_F_A_ANN_100.append(train_F_A_ANN)

            train_F_DT = f1_score(y_train, train_y_pred_DT, average='macro')
            train_F_NB = f1_score(y_train, train_y_pred_NB, average='macro')
            train_F_kNN = f1_score(y_train, train_y_pred_kNN, average='macro')
            train_F_SVM = f1_score(y_train, train_y_pred_SVM, average='macro')
            train_F_RF = f1_score(y_train, train_y_pred_RF, average='macro')
            train_F_ANN = f1_score(y_train, train_y_pred_ANN, average='macro')

            train_F_DT_100.append(train_F_DT)
            train_F_NB_100.append(train_F_NB)
            train_F_kNN_100.append(train_F_kNN)
            train_F_SVM_100.append(train_F_SVM)
            train_F_RF_100.append(train_F_RF)
            train_F_ANN_100.append(train_F_ANN)

            train_F_F_DT = f1_score(y_train_F, train_y_F_pred_DT, average='macro')
            train_F_F_NB = f1_score(y_train_F, train_y_F_pred_NB, average='macro')
            train_F_F_kNN = f1_score(y_train_F, train_y_F_pred_kNN, average='macro')
            train_F_F_SVM = f1_score(y_train_F, train_y_F_pred_SVM, average='macro')
            train_F_F_RF = f1_score(y_train_F, train_y_F_pred_RF, average='macro')
            train_F_F_ANN = f1_score(y_train_F, train_y_F_pred_ANN, average='macro')

            train_F_F_DT_100.append(train_F_F_DT)
            train_F_F_NB_100.append(train_F_F_NB)
            train_F_F_kNN_100.append(train_F_F_kNN)
            train_F_F_SVM_100.append(train_F_F_SVM)
            train_F_F_RF_100.append(train_F_F_RF)
            train_F_F_ANN_100.append(train_F_F_ANN)

            train_P_DT = precision_score(y_train, train_y_pred_DT, average='macro')
            train_P_NB = precision_score(y_train, train_y_pred_NB, average='macro')
            train_P_kNN = precision_score(y_train, train_y_pred_kNN, average='macro')
            train_P_SVM = precision_score(y_train, train_y_pred_SVM, average='macro')
            train_P_RF = precision_score(y_train, train_y_pred_RF, average='macro')
            train_P_ANN = precision_score(y_train, train_y_pred_ANN, average='macro')

            train_P_DT_100.append(train_P_DT)
            train_P_NB_100.append(train_P_NB)
            train_P_kNN_100.append(train_P_kNN)
            train_P_SVM_100.append(train_P_SVM)
            train_P_RF_100.append(train_P_RF)
            train_P_ANN_100.append(train_P_ANN)

            train_F_P_DT = precision_score(y_train_F, train_y_F_pred_DT, average='macro')
            train_F_P_NB = precision_score(y_train_F, train_y_F_pred_NB, average='macro')
            train_F_P_kNN = precision_score(y_train_F, train_y_F_pred_kNN, average='macro')
            train_F_P_SVM = precision_score(y_train_F, train_y_F_pred_SVM, average='macro')
            train_F_P_RF = precision_score(y_train_F, train_y_F_pred_RF, average='macro')
            train_F_P_ANN = precision_score(y_train_F, train_y_F_pred_ANN, average='macro')

            train_F_P_DT_100.append(train_F_P_DT)
            train_F_P_NB_100.append(train_F_P_NB)
            train_F_P_kNN_100.append(train_F_P_kNN)
            train_F_P_SVM_100.append(train_F_P_SVM)
            train_F_P_RF_100.append(train_F_P_RF)
            train_F_P_ANN_100.append(train_F_P_ANN)

            train_R_DT = recall_score(y_train, train_y_pred_DT, average='macro')
            train_R_NB = recall_score(y_train, train_y_pred_NB, average='macro')
            train_R_kNN = recall_score(y_train, train_y_pred_kNN, average='macro')
            train_R_SVM = recall_score(y_train, train_y_pred_SVM, average='macro')
            train_R_RF = recall_score(y_train, train_y_pred_RF, average='macro')
            train_R_ANN = recall_score(y_train, train_y_pred_ANN, average='macro')

            train_R_DT_100.append(train_R_DT)
            train_R_NB_100.append(train_R_NB)
            train_R_kNN_100.append(train_R_kNN)
            train_R_SVM_100.append(train_R_SVM)
            train_R_RF_100.append(train_R_RF)
            train_R_ANN_100.append(train_R_ANN)

            train_F_R_DT = recall_score(y_train_F, train_y_F_pred_DT, average='macro')
            train_F_R_NB = recall_score(y_train_F, train_y_F_pred_NB, average='macro')
            train_F_R_kNN = recall_score(y_train_F, train_y_F_pred_kNN, average='macro')
            train_F_R_SVM = recall_score(y_train_F, train_y_F_pred_SVM, average='macro')
            train_F_R_RF = recall_score(y_train_F, train_y_F_pred_RF, average='macro')
            train_F_R_ANN = recall_score(y_train_F, train_y_F_pred_ANN, average='macro')

            train_F_R_DT_100.append(train_F_R_DT)
            train_F_R_NB_100.append(train_F_R_NB)
            train_F_R_kNN_100.append(train_F_R_kNN)
            train_F_R_SVM_100.append(train_F_R_SVM)
            train_F_R_RF_100.append(train_F_R_RF)
            train_F_R_ANN_100.append(train_F_R_ANN)

            train_ROC_DT = roc_auc_score(y_train, train_y_pred_proba_DT, multi_class='ovr')
            train_ROC_NB = roc_auc_score(y_train, train_y_pred_proba_NB, multi_class='ovr')
            train_ROC_kNN = roc_auc_score(y_train, train_y_pred_proba_kNN, multi_class='ovr')
            train_ROC_SVM = roc_auc_score(y_train, train_y_pred_proba_SVM, multi_class='ovr')
            train_ROC_RF = roc_auc_score(y_train, train_y_pred_proba_RF, multi_class='ovr')
            train_ROC_ANN = roc_auc_score(y_train, train_y_pred_proba_ANN, multi_class='ovr')

            train_ROC_DT_100.append(train_ROC_DT)
            train_ROC_NB_100.append(train_ROC_NB)
            train_ROC_kNN_100.append(train_ROC_kNN)
            train_ROC_SVM_100.append(train_ROC_SVM)
            train_ROC_RF_100.append(train_ROC_RF)
            train_ROC_ANN_100.append(train_ROC_ANN)

            train_F_ROC_DT = roc_auc_score(y_train_F, train_y_F_pred_proba_DT, multi_class='ovr')
            train_F_ROC_NB = roc_auc_score(y_train_F, train_y_F_pred_proba_NB, multi_class='ovr')
            train_F_ROC_kNN = roc_auc_score(y_train_F, train_y_F_pred_proba_kNN, multi_class='ovr')
            train_F_ROC_SVM = roc_auc_score(y_train_F, train_y_F_pred_proba_SVM, multi_class='ovr')
            train_F_ROC_RF = roc_auc_score(y_train_F, train_y_F_pred_proba_RF, multi_class='ovr')
            train_F_ROC_ANN = roc_auc_score(y_train_F, train_y_F_pred_proba_ANN, multi_class='ovr')

            train_F_ROC_DT_100.append(train_F_ROC_DT)
            train_F_ROC_NB_100.append(train_F_ROC_NB)
            train_F_ROC_kNN_100.append(train_F_ROC_kNN)
            train_F_ROC_SVM_100.append(train_F_ROC_SVM)
            train_F_ROC_RF_100.append(train_F_ROC_RF)
            train_F_ROC_ANN_100.append(train_F_ROC_ANN)

            #第一次随访

            E_v_y_pred_DT = classifier_DT.predict(X_E_v)
            E_v_y_pred_NB = classifier_NB.predict(X_E_v)
            E_v_y_pred_kNN = classifier_kNN.predict(X_E_v)
            E_v_y_pred_SVM = classifier_SVM.predict(X_E_v)
            E_v_y_pred_RF = classifier_RF.predict(X_E_v)
            E_v_y_pred_ANN = classifier_ANN.predict(X_E_v)

            E_v_y_F_pred_DT = classifier_DT_F.predict(X_E_v_F)
            E_v_y_F_pred_NB = classifier_NB_F.predict(X_E_v_F)
            E_v_y_F_pred_kNN = classifier_kNN_F.predict(X_E_v_F)
            E_v_y_F_pred_SVM = classifier_SVM_F.predict(X_E_v_F)
            E_v_y_F_pred_RF = classifier_RF_F.predict(X_E_v_F)
            E_v_y_F_pred_ANN = classifier_ANN_F.predict(X_E_v_F)

            E_v_y_pred_proba_DT = classifier_DT.predict_proba(X_E_v)
            E_v_y_pred_proba_NB = classifier_NB.predict_proba(X_E_v)
            E_v_y_pred_proba_kNN = classifier_kNN.predict_proba(X_E_v)
            E_v_y_pred_proba_SVM = classifier_SVM.predict_proba(X_E_v)
            E_v_y_pred_proba_RF = classifier_RF.predict_proba(X_E_v)
            E_v_y_pred_proba_ANN = classifier_ANN.predict_proba(X_E_v)

            E_v_y_F_pred_proba_DT = classifier_DT_F.predict_proba(X_E_v_F)
            E_v_y_F_pred_proba_NB = classifier_NB_F.predict_proba(X_E_v_F)
            E_v_y_F_pred_proba_kNN = classifier_kNN_F.predict_proba(X_E_v_F)
            E_v_y_F_pred_proba_SVM = classifier_SVM_F.predict_proba(X_E_v_F)
            E_v_y_F_pred_proba_RF = classifier_RF_F.predict_proba(X_E_v_F)
            E_v_y_F_pred_proba_ANN = classifier_ANN_F.predict_proba(X_E_v_F)

            E_v_A_DT = accuracy_score(y_E_v, E_v_y_pred_DT)
            E_v_A_NB = accuracy_score(y_E_v, E_v_y_pred_NB)
            E_v_A_kNN = accuracy_score(y_E_v, E_v_y_pred_kNN)
            E_v_A_SVM = accuracy_score(y_E_v, E_v_y_pred_SVM)
            E_v_A_RF = accuracy_score(y_E_v, E_v_y_pred_RF)
            E_v_A_ANN = accuracy_score(y_E_v, E_v_y_pred_ANN)

            E_v_A_DT_100.append(E_v_A_DT)
            E_v_A_NB_100.append(E_v_A_NB)
            E_v_A_kNN_100.append(E_v_A_kNN)
            E_v_A_SVM_100.append(E_v_A_SVM)
            E_v_A_RF_100.append(E_v_A_RF)
            E_v_A_ANN_100.append(E_v_A_ANN)

            E_v_F_A_DT = accuracy_score(y_E_v_F, E_v_y_F_pred_DT)
            E_v_F_A_NB = accuracy_score(y_E_v_F, E_v_y_F_pred_NB)
            E_v_F_A_kNN = accuracy_score(y_E_v_F, E_v_y_F_pred_kNN)
            E_v_F_A_SVM = accuracy_score(y_E_v_F, E_v_y_F_pred_SVM)
            E_v_F_A_RF = accuracy_score(y_E_v_F, E_v_y_F_pred_RF)
            E_v_F_A_ANN = accuracy_score(y_E_v_F, E_v_y_F_pred_ANN)

            E_v_F_A_DT_100.append(E_v_F_A_DT)
            E_v_F_A_NB_100.append(E_v_F_A_NB)
            E_v_F_A_kNN_100.append(E_v_F_A_kNN)
            E_v_F_A_SVM_100.append(E_v_F_A_SVM)
            E_v_F_A_RF_100.append(E_v_F_A_RF)
            E_v_F_A_ANN_100.append(E_v_F_A_ANN)

            E_v_F_DT = f1_score(y_E_v, E_v_y_pred_DT, average='macro')
            E_v_F_NB = f1_score(y_E_v, E_v_y_pred_NB, average='macro')
            E_v_F_kNN = f1_score(y_E_v, E_v_y_pred_kNN, average='macro')
            E_v_F_SVM = f1_score(y_E_v, E_v_y_pred_SVM, average='macro')
            E_v_F_RF = f1_score(y_E_v, E_v_y_pred_RF, average='macro')
            E_v_F_ANN = f1_score(y_E_v, E_v_y_pred_ANN, average='macro')

            E_v_F_DT_100.append(E_v_F_DT)
            E_v_F_NB_100.append(E_v_F_NB)
            E_v_F_kNN_100.append(E_v_F_kNN)
            E_v_F_SVM_100.append(E_v_F_SVM)
            E_v_F_RF_100.append(E_v_F_RF)
            E_v_F_ANN_100.append(E_v_F_ANN)

            E_v_F_F_DT = f1_score(y_E_v_F, E_v_y_F_pred_DT, average='macro')
            E_v_F_F_NB = f1_score(y_E_v_F, E_v_y_F_pred_NB, average='macro')
            E_v_F_F_kNN = f1_score(y_E_v_F, E_v_y_F_pred_kNN, average='macro')
            E_v_F_F_SVM = f1_score(y_E_v_F, E_v_y_F_pred_SVM, average='macro')
            E_v_F_F_RF = f1_score(y_E_v_F, E_v_y_F_pred_RF, average='macro')
            E_v_F_F_ANN = f1_score(y_E_v_F, E_v_y_F_pred_ANN, average='macro')

            E_v_F_F_DT_100.append(E_v_F_F_DT)
            E_v_F_F_NB_100.append(E_v_F_F_NB)
            E_v_F_F_kNN_100.append(E_v_F_F_kNN)
            E_v_F_F_SVM_100.append(E_v_F_F_SVM)
            E_v_F_F_RF_100.append(E_v_F_F_RF)
            E_v_F_F_ANN_100.append(E_v_F_F_ANN)

            E_v_P_DT = precision_score(y_E_v, E_v_y_pred_DT, average='macro')
            E_v_P_NB = precision_score(y_E_v, E_v_y_pred_NB, average='macro')
            E_v_P_kNN = precision_score(y_E_v, E_v_y_pred_kNN, average='macro')
            E_v_P_SVM = precision_score(y_E_v, E_v_y_pred_SVM, average='macro')
            E_v_P_RF = precision_score(y_E_v, E_v_y_pred_RF, average='macro')
            E_v_P_ANN = precision_score(y_E_v, E_v_y_pred_ANN, average='macro')

            E_v_P_DT_100.append(E_v_P_DT)
            E_v_P_NB_100.append(E_v_P_NB)
            E_v_P_kNN_100.append(E_v_P_kNN)
            E_v_P_SVM_100.append(E_v_P_SVM)
            E_v_P_RF_100.append(E_v_P_RF)
            E_v_P_ANN_100.append(E_v_P_ANN)

            E_v_F_P_DT = precision_score(y_E_v_F, E_v_y_F_pred_DT, average='macro')
            E_v_F_P_NB = precision_score(y_E_v_F, E_v_y_F_pred_NB, average='macro')
            E_v_F_P_kNN = precision_score(y_E_v_F, E_v_y_F_pred_kNN, average='macro')
            E_v_F_P_SVM = precision_score(y_E_v_F, E_v_y_F_pred_SVM, average='macro')
            E_v_F_P_RF = precision_score(y_E_v_F, E_v_y_F_pred_RF, average='macro')
            E_v_F_P_ANN = precision_score(y_E_v_F, E_v_y_F_pred_ANN, average='macro')

            E_v_F_P_DT_100.append(E_v_F_P_DT)
            E_v_F_P_NB_100.append(E_v_F_P_NB)
            E_v_F_P_kNN_100.append(E_v_F_P_kNN)
            E_v_F_P_SVM_100.append(E_v_F_P_SVM)
            E_v_F_P_RF_100.append(E_v_F_P_RF)
            E_v_F_P_ANN_100.append(E_v_F_P_ANN)

            E_v_R_DT = recall_score(y_E_v, E_v_y_pred_DT, average='macro')
            E_v_R_NB = recall_score(y_E_v, E_v_y_pred_NB, average='macro')
            E_v_R_kNN = recall_score(y_E_v, E_v_y_pred_kNN, average='macro')
            E_v_R_SVM = recall_score(y_E_v, E_v_y_pred_SVM, average='macro')
            E_v_R_RF = recall_score(y_E_v, E_v_y_pred_RF, average='macro')
            E_v_R_ANN = recall_score(y_E_v, E_v_y_pred_ANN, average='macro')

            E_v_R_DT_100.append(E_v_R_DT)
            E_v_R_NB_100.append(E_v_R_NB)
            E_v_R_kNN_100.append(E_v_R_kNN)
            E_v_R_SVM_100.append(E_v_R_SVM)
            E_v_R_RF_100.append(E_v_R_RF)
            E_v_R_ANN_100.append(E_v_R_ANN)

            E_v_F_R_DT = recall_score(y_E_v_F, E_v_y_F_pred_DT, average='macro')
            E_v_F_R_NB = recall_score(y_E_v_F, E_v_y_F_pred_NB, average='macro')
            E_v_F_R_kNN = recall_score(y_E_v_F, E_v_y_F_pred_kNN, average='macro')
            E_v_F_R_SVM = recall_score(y_E_v_F, E_v_y_F_pred_SVM, average='macro')
            E_v_F_R_RF = recall_score(y_E_v_F, E_v_y_F_pred_RF, average='macro')
            E_v_F_R_ANN = recall_score(y_E_v_F, E_v_y_F_pred_ANN, average='macro')

            E_v_F_R_DT_100.append(E_v_F_R_DT)
            E_v_F_R_NB_100.append(E_v_F_R_NB)
            E_v_F_R_kNN_100.append(E_v_F_R_kNN)
            E_v_F_R_SVM_100.append(E_v_F_R_SVM)
            E_v_F_R_RF_100.append(E_v_F_R_RF)
            E_v_F_R_ANN_100.append(E_v_F_R_ANN)

            E_v_ROC_DT = roc_auc_score(y_E_v, E_v_y_pred_proba_DT, multi_class='ovr')
            E_v_ROC_NB = roc_auc_score(y_E_v, E_v_y_pred_proba_NB, multi_class='ovr')
            E_v_ROC_kNN = roc_auc_score(y_E_v, E_v_y_pred_proba_kNN, multi_class='ovr')
            E_v_ROC_SVM = roc_auc_score(y_E_v, E_v_y_pred_proba_SVM, multi_class='ovr')
            E_v_ROC_RF = roc_auc_score(y_E_v, E_v_y_pred_proba_RF, multi_class='ovr')
            E_v_ROC_ANN = roc_auc_score(y_E_v, E_v_y_pred_proba_ANN, multi_class='ovr')

            E_v_ROC_DT_100.append(E_v_ROC_DT)
            E_v_ROC_NB_100.append(E_v_ROC_NB)
            E_v_ROC_kNN_100.append(E_v_ROC_kNN)
            E_v_ROC_SVM_100.append(E_v_ROC_SVM)
            E_v_ROC_RF_100.append(E_v_ROC_RF)
            E_v_ROC_ANN_100.append(E_v_ROC_ANN)

            E_v_F_ROC_DT = roc_auc_score(y_E_v_F, E_v_y_F_pred_proba_DT, multi_class='ovr')
            E_v_F_ROC_NB = roc_auc_score(y_E_v_F, E_v_y_F_pred_proba_NB, multi_class='ovr')
            E_v_F_ROC_kNN = roc_auc_score(y_E_v_F, E_v_y_F_pred_proba_kNN, multi_class='ovr')
            E_v_F_ROC_SVM = roc_auc_score(y_E_v_F, E_v_y_F_pred_proba_SVM, multi_class='ovr')
            E_v_F_ROC_RF = roc_auc_score(y_E_v_F, E_v_y_F_pred_proba_RF, multi_class='ovr')
            E_v_F_ROC_ANN = roc_auc_score(y_E_v_F, E_v_y_F_pred_proba_ANN, multi_class='ovr')

            E_v_F_ROC_DT_100.append(E_v_F_ROC_DT)
            E_v_F_ROC_NB_100.append(E_v_F_ROC_NB)
            E_v_F_ROC_kNN_100.append(E_v_F_ROC_kNN)
            E_v_F_ROC_SVM_100.append(E_v_F_ROC_SVM)
            E_v_F_ROC_RF_100.append(E_v_F_ROC_RF)
            E_v_F_ROC_ANN_100.append(E_v_F_ROC_ANN)


            #第二次随访
            # follow_up_2_y_pred_DT = classifier_DT.predict(X_follow_up_2)
            # follow_up_2_y_pred_NB = classifier_NB.predict(X_follow_up_2)
            # follow_up_2_y_pred_kNN = classifier_kNN.predict(X_follow_up_2)
            # follow_up_2_y_pred_SVM = classifier_SVM.predict(X_follow_up_2)
            # follow_up_2_y_pred_RF = classifier_RF.predict(X_follow_up_2)
            # follow_up_2_y_pred_ANN = classifier_ANN.predict(X_follow_up_2)
            #
            # follow_up_2_y_pred_proba_DT = classifier_DT.predict_proba(X_follow_up_2)
            # follow_up_2_y_pred_proba_NB = classifier_NB.predict_proba(X_follow_up_2)
            # follow_up_2_y_pred_proba_kNN = classifier_kNN.predict_proba(X_follow_up_2)
            # follow_up_2_y_pred_proba_SVM = classifier_SVM.predict_proba(X_follow_up_2)
            # follow_up_2_y_pred_proba_RF = classifier_RF.predict_proba(X_follow_up_2)
            # follow_up_2_y_pred_proba_ANN = classifier_ANN.predict_proba(X_follow_up_2)
            #
            # follow_up_2_A_DT = accuracy_score(y_follow_up_2, follow_up_2_y_pred_DT)
            # follow_up_2_A_NB = accuracy_score(y_follow_up_2, follow_up_2_y_pred_NB)
            # follow_up_2_A_kNN = accuracy_score(y_follow_up_2, follow_up_2_y_pred_kNN)
            # follow_up_2_A_SVM = accuracy_score(y_follow_up_2, follow_up_2_y_pred_SVM)
            # follow_up_2_A_RF = accuracy_score(y_follow_up_2, follow_up_2_y_pred_RF)
            # follow_up_2_A_ANN = accuracy_score(y_follow_up_2, follow_up_2_y_pred_ANN)
            #
            # follow_up_2_A_DT_100.append(follow_up_2_A_DT)
            # follow_up_2_A_NB_100.append(follow_up_2_A_NB)
            # follow_up_2_A_kNN_100.append(follow_up_2_A_kNN)
            # follow_up_2_A_SVM_100.append(follow_up_2_A_SVM)
            # follow_up_2_A_RF_100.append(follow_up_2_A_RF)
            # follow_up_2_A_ANN_100.append(follow_up_2_A_ANN)
            #
            # follow_up_2_F_DT = f1_score(y_follow_up_2, follow_up_2_y_pred_DT, average='macro')
            # follow_up_2_F_NB = f1_score(y_follow_up_2, follow_up_2_y_pred_NB, average='macro')
            # follow_up_2_F_kNN = f1_score(y_follow_up_2, follow_up_2_y_pred_kNN, average='macro')
            # follow_up_2_F_SVM = f1_score(y_follow_up_2, follow_up_2_y_pred_SVM, average='macro')
            # follow_up_2_F_RF = f1_score(y_follow_up_2, follow_up_2_y_pred_RF, average='macro')
            # follow_up_2_F_ANN = f1_score(y_follow_up_2, follow_up_2_y_pred_ANN, average='macro')
            #
            # follow_up_2_F_DT_100.append(follow_up_2_F_DT)
            # follow_up_2_F_NB_100.append(follow_up_2_F_NB)
            # follow_up_2_F_kNN_100.append(follow_up_2_F_kNN)
            # follow_up_2_F_SVM_100.append(follow_up_2_F_SVM)
            # follow_up_2_F_RF_100.append(follow_up_2_F_RF)
            # follow_up_2_F_ANN_100.append(follow_up_2_F_ANN)
            #
            # follow_up_2_P_DT = precision_score(y_follow_up_2, follow_up_2_y_pred_DT, average='macro')
            # follow_up_2_P_NB = precision_score(y_follow_up_2, follow_up_2_y_pred_NB, average='macro')
            # follow_up_2_P_kNN = precision_score(y_follow_up_2, follow_up_2_y_pred_kNN, average='macro')
            # follow_up_2_P_SVM = precision_score(y_follow_up_2, follow_up_2_y_pred_SVM, average='macro')
            # follow_up_2_P_RF = precision_score(y_follow_up_2, follow_up_2_y_pred_RF, average='macro')
            # follow_up_2_P_ANN = precision_score(y_follow_up_2, follow_up_2_y_pred_ANN, average='macro')
            #
            # follow_up_2_P_DT_100.append(follow_up_2_P_DT)
            # follow_up_2_P_NB_100.append(follow_up_2_P_NB)
            # follow_up_2_P_kNN_100.append(follow_up_2_P_kNN)
            # follow_up_2_P_SVM_100.append(follow_up_2_P_SVM)
            # follow_up_2_P_RF_100.append(follow_up_2_P_RF)
            # follow_up_2_P_ANN_100.append(follow_up_2_P_ANN)
            #
            # follow_up_2_R_DT = recall_score(y_follow_up_2, follow_up_2_y_pred_DT, average='macro')
            # follow_up_2_R_NB = recall_score(y_follow_up_2, follow_up_2_y_pred_NB, average='macro')
            # follow_up_2_R_kNN = recall_score(y_follow_up_2, follow_up_2_y_pred_kNN, average='macro')
            # follow_up_2_R_SVM = recall_score(y_follow_up_2, follow_up_2_y_pred_SVM, average='macro')
            # follow_up_2_R_RF = recall_score(y_follow_up_2, follow_up_2_y_pred_RF, average='macro')
            # follow_up_2_R_ANN = recall_score(y_follow_up_2, follow_up_2_y_pred_ANN, average='macro')
            #
            # follow_up_2_R_DT_100.append(follow_up_2_R_DT)
            # follow_up_2_R_NB_100.append(follow_up_2_R_NB)
            # follow_up_2_R_kNN_100.append(follow_up_2_R_kNN)
            # follow_up_2_R_SVM_100.append(follow_up_2_R_SVM)
            # follow_up_2_R_RF_100.append(follow_up_2_R_RF)
            # follow_up_2_R_ANN_100.append(follow_up_2_R_ANN)
            #
            # follow_up_2_ROC_DT = roc_auc_score(y_follow_up_2, follow_up_2_y_pred_proba_DT, multi_class='ovr')
            # follow_up_2_ROC_NB = roc_auc_score(y_follow_up_2, follow_up_2_y_pred_proba_NB, multi_class='ovr')
            # follow_up_2_ROC_kNN = roc_auc_score(y_follow_up_2, follow_up_2_y_pred_proba_kNN, multi_class='ovr')
            # follow_up_2_ROC_SVM = roc_auc_score(y_follow_up_2, follow_up_2_y_pred_proba_SVM, multi_class='ovr')
            # follow_up_2_ROC_RF = roc_auc_score(y_follow_up_2, follow_up_2_y_pred_proba_RF, multi_class='ovr')
            # follow_up_2_ROC_ANN = roc_auc_score(y_follow_up_2, follow_up_2_y_pred_proba_ANN, multi_class='ovr')
            #
            # follow_up_2_ROC_DT_100.append(follow_up_2_ROC_DT)
            # follow_up_2_ROC_NB_100.append(follow_up_2_ROC_NB)
            # follow_up_2_ROC_kNN_100.append(follow_up_2_ROC_kNN)
            # follow_up_2_ROC_SVM_100.append(follow_up_2_ROC_SVM)
            # follow_up_2_ROC_RF_100.append(follow_up_2_ROC_RF)
            # follow_up_2_ROC_ANN_100.append(follow_up_2_ROC_ANN)
            #

            # #第三次随访
            # follow_up_3_y_pred_DT = classifier_DT.predict(X_follow_up_3)
            # follow_up_3_y_pred_NB = classifier_NB.predict(X_follow_up_3)
            # follow_up_3_y_pred_kNN = classifier_kNN.predict(X_follow_up_3)
            # follow_up_3_y_pred_SVM = classifier_SVM.predict(X_follow_up_3)
            # follow_up_3_y_pred_RF = classifier_RF.predict(X_follow_up_3)
            # follow_up_3_y_pred_ANN = classifier_ANN.predict(X_follow_up_3)
            #
            # follow_up_3_y_pred_proba_DT = classifier_DT.predict_proba(X_follow_up_3)
            # follow_up_3_y_pred_proba_NB = classifier_NB.predict_proba(X_follow_up_3)
            # follow_up_3_y_pred_proba_kNN = classifier_kNN.predict_proba(X_follow_up_3)
            # follow_up_3_y_pred_proba_SVM = classifier_SVM.predict_proba(X_follow_up_3)
            # follow_up_3_y_pred_proba_RF = classifier_RF.predict_proba(X_follow_up_3)
            # follow_up_3_y_pred_proba_ANN = classifier_ANN.predict_proba(X_follow_up_3)
            #
            # follow_up_3_A_DT = accuracy_score(y_follow_up_3, follow_up_3_y_pred_DT)
            # follow_up_3_A_NB = accuracy_score(y_follow_up_3, follow_up_3_y_pred_NB)
            # follow_up_3_A_kNN = accuracy_score(y_follow_up_3, follow_up_3_y_pred_kNN)
            # follow_up_3_A_SVM = accuracy_score(y_follow_up_3, follow_up_3_y_pred_SVM)
            # follow_up_3_A_RF = accuracy_score(y_follow_up_3, follow_up_3_y_pred_RF)
            # follow_up_3_A_ANN = accuracy_score(y_follow_up_3, follow_up_3_y_pred_ANN)
            #
            # follow_up_3_A_DT_100.append(follow_up_3_A_DT)
            # follow_up_3_A_NB_100.append(follow_up_3_A_NB)
            # follow_up_3_A_kNN_100.append(follow_up_3_A_kNN)
            # follow_up_3_A_SVM_100.append(follow_up_3_A_SVM)
            # follow_up_3_A_RF_100.append(follow_up_3_A_RF)
            # follow_up_3_A_ANN_100.append(follow_up_3_A_ANN)
            #
            # follow_up_3_F_DT = f1_score(y_follow_up_3, follow_up_3_y_pred_DT, average='macro')
            # follow_up_3_F_NB = f1_score(y_follow_up_3, follow_up_3_y_pred_NB, average='macro')
            # follow_up_3_F_kNN = f1_score(y_follow_up_3, follow_up_3_y_pred_kNN, average='macro')
            # follow_up_3_F_SVM = f1_score(y_follow_up_3, follow_up_3_y_pred_SVM, average='macro')
            # follow_up_3_F_RF = f1_score(y_follow_up_3, follow_up_3_y_pred_RF, average='macro')
            # follow_up_3_F_ANN = f1_score(y_follow_up_3, follow_up_3_y_pred_ANN, average='macro')
            #
            # follow_up_3_F_DT_100.append(follow_up_3_F_DT)
            # follow_up_3_F_NB_100.append(follow_up_3_F_NB)
            # follow_up_3_F_kNN_100.append(follow_up_3_F_kNN)
            # follow_up_3_F_SVM_100.append(follow_up_3_F_SVM)
            # follow_up_3_F_RF_100.append(follow_up_3_F_RF)
            # follow_up_3_F_ANN_100.append(follow_up_3_F_ANN)
            #
            # follow_up_3_P_DT = precision_score(y_follow_up_3, follow_up_3_y_pred_DT, average='macro')
            # follow_up_3_P_NB = precision_score(y_follow_up_3, follow_up_3_y_pred_NB, average='macro')
            # follow_up_3_P_kNN = precision_score(y_follow_up_3, follow_up_3_y_pred_kNN, average='macro')
            # follow_up_3_P_SVM = precision_score(y_follow_up_3, follow_up_3_y_pred_SVM, average='macro')
            # follow_up_3_P_RF = precision_score(y_follow_up_3, follow_up_3_y_pred_RF, average='macro')
            # follow_up_3_P_ANN = precision_score(y_follow_up_3, follow_up_3_y_pred_ANN, average='macro')
            #
            # follow_up_3_P_DT_100.append(follow_up_3_P_DT)
            # follow_up_3_P_NB_100.append(follow_up_3_P_NB)
            # follow_up_3_P_kNN_100.append(follow_up_3_P_kNN)
            # follow_up_3_P_SVM_100.append(follow_up_3_P_SVM)
            # follow_up_3_P_RF_100.append(follow_up_3_P_RF)
            # follow_up_3_P_ANN_100.append(follow_up_3_P_ANN)
            #
            # follow_up_3_R_DT = recall_score(y_follow_up_3, follow_up_3_y_pred_DT, average='macro')
            # follow_up_3_R_NB = recall_score(y_follow_up_3, follow_up_3_y_pred_NB, average='macro')
            # follow_up_3_R_kNN = recall_score(y_follow_up_3, follow_up_3_y_pred_kNN, average='macro')
            # follow_up_3_R_SVM = recall_score(y_follow_up_3, follow_up_3_y_pred_SVM, average='macro')
            # follow_up_3_R_RF = recall_score(y_follow_up_3, follow_up_3_y_pred_RF, average='macro')
            # follow_up_3_R_ANN = recall_score(y_follow_up_3, follow_up_3_y_pred_ANN, average='macro')
            #
            # follow_up_3_R_DT_100.append(follow_up_3_R_DT)
            # follow_up_3_R_NB_100.append(follow_up_3_R_NB)
            # follow_up_3_R_kNN_100.append(follow_up_3_R_kNN)
            # follow_up_3_R_SVM_100.append(follow_up_3_R_SVM)
            # follow_up_3_R_RF_100.append(follow_up_3_R_RF)
            # follow_up_3_R_ANN_100.append(follow_up_3_R_ANN)
            #
            # follow_up_3_ROC_DT = roc_auc_score(y_follow_up_3, follow_up_3_y_pred_proba_DT, multi_class='ovr')
            # follow_up_3_ROC_NB = roc_auc_score(y_follow_up_3, follow_up_3_y_pred_proba_NB, multi_class='ovr')
            # follow_up_3_ROC_kNN = roc_auc_score(y_follow_up_3, follow_up_3_y_pred_proba_kNN, multi_class='ovr')
            # follow_up_3_ROC_SVM = roc_auc_score(y_follow_up_3, follow_up_3_y_pred_proba_SVM, multi_class='ovr')
            # follow_up_3_ROC_RF = roc_auc_score(y_follow_up_3, follow_up_3_y_pred_proba_RF, multi_class='ovr')
            # follow_up_3_ROC_ANN = roc_auc_score(y_follow_up_3, follow_up_3_y_pred_proba_ANN, multi_class='ovr')
            #
            # follow_up_3_ROC_DT_100.append(follow_up_3_ROC_DT)
            # follow_up_3_ROC_NB_100.append(follow_up_3_ROC_NB)
            # follow_up_3_ROC_kNN_100.append(follow_up_3_ROC_kNN)
            # follow_up_3_ROC_SVM_100.append(follow_up_3_ROC_SVM)
            # follow_up_3_ROC_RF_100.append(follow_up_3_ROC_RF)
            # follow_up_3_ROC_ANN_100.append(follow_up_3_ROC_ANN)
            #

            #测试集预测
            test_y_pred_DT = classifier_DT.predict(X_test)
            test_y_pred_NB = classifier_NB.predict(X_test)
            test_y_pred_kNN = classifier_kNN.predict(X_test)
            test_y_pred_SVM = classifier_SVM.predict(X_test)
            test_y_pred_RF = classifier_RF.predict(X_test)
            test_y_pred_ANN = classifier_ANN.predict(X_test)

            test_y_F_pred_DT = classifier_DT_F.predict(X_test_F)
            test_y_F_pred_NB = classifier_NB_F.predict(X_test_F)
            test_y_F_pred_kNN = classifier_kNN_F.predict(X_test_F)
            test_y_F_pred_SVM = classifier_SVM_F.predict(X_test_F)
            test_y_F_pred_RF = classifier_RF_F.predict(X_test_F)
            test_y_F_pred_ANN = classifier_ANN_F.predict(X_test_F)

            test_y_pred_proba_DT = classifier_DT.predict_proba(X_test)
            test_y_pred_proba_NB = classifier_NB.predict_proba(X_test)
            test_y_pred_proba_kNN = classifier_kNN.predict_proba(X_test)
            test_y_pred_proba_SVM = classifier_SVM.predict_proba(X_test)
            test_y_pred_proba_RF = classifier_RF.predict_proba(X_test)
            test_y_pred_proba_ANN = classifier_ANN.predict_proba(X_test)

            test_y_F_pred_proba_DT = classifier_DT_F.predict_proba(X_test_F)
            test_y_F_pred_proba_NB = classifier_NB_F.predict_proba(X_test_F)
            test_y_F_pred_proba_kNN = classifier_kNN_F.predict_proba(X_test_F)
            test_y_F_pred_proba_SVM = classifier_SVM_F.predict_proba(X_test_F)
            test_y_F_pred_proba_RF = classifier_RF_F.predict_proba(X_test_F)
            test_y_F_pred_proba_ANN = classifier_ANN_F.predict_proba(X_test_F)

            test_A_DT = accuracy_score(y_test, test_y_pred_DT)
            test_A_NB = accuracy_score(y_test, test_y_pred_NB)
            test_A_kNN = accuracy_score(y_test, test_y_pred_kNN)
            test_A_SVM = accuracy_score(y_test, test_y_pred_SVM)
            test_A_RF = accuracy_score(y_test, test_y_pred_RF)
            test_A_ANN = accuracy_score(y_test, test_y_pred_ANN)

            test_A_DT_100.append(test_A_DT)
            test_A_NB_100.append(test_A_NB)
            test_A_kNN_100.append(test_A_kNN)
            test_A_SVM_100.append(test_A_SVM)
            test_A_RF_100.append(test_A_RF)
            test_A_ANN_100.append(test_A_ANN)

            test_F_A_DT = accuracy_score(y_test_F, test_y_F_pred_DT)
            test_F_A_NB = accuracy_score(y_test_F, test_y_F_pred_NB)
            test_F_A_kNN = accuracy_score(y_test_F, test_y_F_pred_kNN)
            test_F_A_SVM = accuracy_score(y_test_F, test_y_F_pred_SVM)
            test_F_A_RF = accuracy_score(y_test_F, test_y_F_pred_RF)
            test_F_A_ANN = accuracy_score(y_test_F, test_y_F_pred_ANN)

            test_F_A_DT_100.append(test_F_A_DT)
            test_F_A_NB_100.append(test_F_A_NB)
            test_F_A_kNN_100.append(test_F_A_kNN)
            test_F_A_SVM_100.append(test_F_A_SVM)
            test_F_A_RF_100.append(test_F_A_RF)
            test_F_A_ANN_100.append(test_F_A_ANN)

            test_F_DT = f1_score(y_test, test_y_pred_DT, average='macro')
            test_F_NB = f1_score(y_test, test_y_pred_NB, average='macro')
            test_F_kNN = f1_score(y_test, test_y_pred_kNN, average='macro')
            test_F_SVM = f1_score(y_test, test_y_pred_SVM, average='macro')
            test_F_RF = f1_score(y_test, test_y_pred_RF, average='macro')
            test_F_ANN = f1_score(y_test, test_y_pred_ANN, average='macro')

            test_F_DT_100.append(test_F_DT)
            test_F_NB_100.append(test_F_NB)
            test_F_kNN_100.append(test_F_kNN)
            test_F_SVM_100.append(test_F_SVM)
            test_F_RF_100.append(test_F_RF)
            test_F_ANN_100.append(test_F_ANN)

            test_F_F_DT = f1_score(y_test_F, test_y_F_pred_DT, average='macro')
            test_F_F_NB = f1_score(y_test_F, test_y_F_pred_NB, average='macro')
            test_F_F_kNN = f1_score(y_test_F, test_y_F_pred_kNN, average='macro')
            test_F_F_SVM = f1_score(y_test_F, test_y_F_pred_SVM, average='macro')
            test_F_F_RF = f1_score(y_test_F, test_y_F_pred_RF, average='macro')
            test_F_F_ANN = f1_score(y_test_F, test_y_F_pred_ANN, average='macro')

            test_F_F_DT_100.append(test_F_F_DT)
            test_F_F_NB_100.append(test_F_F_NB)
            test_F_F_kNN_100.append(test_F_F_kNN)
            test_F_F_SVM_100.append(test_F_F_SVM)
            test_F_F_RF_100.append(test_F_F_RF)
            test_F_F_ANN_100.append(test_F_F_ANN)

            test_P_DT = precision_score(y_test, test_y_pred_DT, average='macro')
            test_P_NB = precision_score(y_test, test_y_pred_NB, average='macro')
            test_P_kNN = precision_score(y_test, test_y_pred_kNN, average='macro')
            test_P_SVM = precision_score(y_test, test_y_pred_SVM, average='macro')
            test_P_RF = precision_score(y_test, test_y_pred_RF, average='macro')
            test_P_ANN = precision_score(y_test, test_y_pred_ANN, average='macro')

            test_P_DT_100.append(test_P_DT)
            test_P_NB_100.append(test_P_NB)
            test_P_kNN_100.append(test_P_kNN)
            test_P_SVM_100.append(test_P_SVM)
            test_P_RF_100.append(test_P_RF)
            test_P_ANN_100.append(test_P_ANN)

            test_F_P_DT = precision_score(y_test_F, test_y_F_pred_DT, average='macro')
            test_F_P_NB = precision_score(y_test_F, test_y_F_pred_NB, average='macro')
            test_F_P_kNN = precision_score(y_test_F, test_y_F_pred_kNN, average='macro')
            test_F_P_SVM = precision_score(y_test_F, test_y_F_pred_SVM, average='macro')
            test_F_P_RF = precision_score(y_test_F, test_y_F_pred_RF, average='macro')
            test_F_P_ANN = precision_score(y_test_F, test_y_F_pred_ANN, average='macro')

            test_F_P_DT_100.append(test_F_P_DT)
            test_F_P_NB_100.append(test_F_P_NB)
            test_F_P_kNN_100.append(test_F_P_kNN)
            test_F_P_SVM_100.append(test_F_P_SVM)
            test_F_P_RF_100.append(test_F_P_RF)
            test_F_P_ANN_100.append(test_F_P_ANN)

            test_R_DT = recall_score(y_test, test_y_pred_DT, average='macro')
            test_R_NB = recall_score(y_test, test_y_pred_NB, average='macro')
            test_R_kNN = recall_score(y_test, test_y_pred_kNN, average='macro')
            test_R_SVM = recall_score(y_test, test_y_pred_SVM, average='macro')
            test_R_RF = recall_score(y_test, test_y_pred_RF, average='macro')
            test_R_ANN = recall_score(y_test, test_y_pred_ANN, average='macro')

            test_R_DT_100.append(test_R_DT)
            test_R_NB_100.append(test_R_NB)
            test_R_kNN_100.append(test_R_kNN)
            test_R_SVM_100.append(test_R_SVM)
            test_R_RF_100.append(test_R_RF)
            test_R_ANN_100.append(test_R_ANN)

            test_F_R_DT = recall_score(y_test_F, test_y_F_pred_DT, average='macro')
            test_F_R_NB = recall_score(y_test_F, test_y_F_pred_NB, average='macro')
            test_F_R_kNN = recall_score(y_test_F, test_y_F_pred_kNN, average='macro')
            test_F_R_SVM = recall_score(y_test_F, test_y_F_pred_SVM, average='macro')
            test_F_R_RF = recall_score(y_test_F, test_y_F_pred_RF, average='macro')
            test_F_R_ANN = recall_score(y_test_F, test_y_F_pred_ANN, average='macro')

            test_F_R_DT_100.append(test_F_R_DT)
            test_F_R_NB_100.append(test_F_R_NB)
            test_F_R_kNN_100.append(test_F_R_kNN)
            test_F_R_SVM_100.append(test_F_R_SVM)
            test_F_R_RF_100.append(test_F_R_RF)
            test_F_R_ANN_100.append(test_F_R_ANN)


            test_ROC_DT = roc_auc_score(y_test, test_y_pred_proba_DT, multi_class='ovr')
            test_ROC_NB = roc_auc_score(y_test, test_y_pred_proba_NB, multi_class='ovr')
            test_ROC_kNN = roc_auc_score(y_test, test_y_pred_proba_kNN, multi_class='ovr')
            test_ROC_SVM = roc_auc_score(y_test, test_y_pred_proba_SVM, multi_class='ovr')
            test_ROC_RF = roc_auc_score(y_test, test_y_pred_proba_RF, multi_class='ovr')
            test_ROC_ANN = roc_auc_score(y_test, test_y_pred_proba_ANN, multi_class='ovr')

            test_ROC_DT_100.append(test_ROC_DT)
            test_ROC_NB_100.append(test_ROC_NB)
            test_ROC_kNN_100.append(test_ROC_kNN)
            test_ROC_SVM_100.append(test_ROC_SVM)
            test_ROC_RF_100.append(test_ROC_RF)
            test_ROC_ANN_100.append(test_ROC_ANN)

            test_F_ROC_DT = roc_auc_score(y_test_F, test_y_F_pred_proba_DT, multi_class='ovr')
            test_F_ROC_NB = roc_auc_score(y_test_F, test_y_F_pred_proba_NB, multi_class='ovr')
            test_F_ROC_kNN = roc_auc_score(y_test_F, test_y_F_pred_proba_kNN, multi_class='ovr')
            test_F_ROC_SVM = roc_auc_score(y_test_F, test_y_F_pred_proba_SVM, multi_class='ovr')
            test_F_ROC_RF = roc_auc_score(y_test_F, test_y_F_pred_proba_RF, multi_class='ovr')
            test_F_ROC_ANN = roc_auc_score(y_test_F, test_y_F_pred_proba_ANN, multi_class='ovr')

            test_F_ROC_DT_100.append(test_F_ROC_DT)
            test_F_ROC_NB_100.append(test_F_ROC_NB)
            test_F_ROC_kNN_100.append(test_F_ROC_kNN)
            test_F_ROC_SVM_100.append(test_F_ROC_SVM)
            test_F_ROC_RF_100.append(test_F_ROC_RF)
            test_F_ROC_ANN_100.append(test_F_ROC_ANN)

            y = label_binarize(y_train, classes=[1, 2, 3])
            test_y = label_binarize(y_test, classes=[1, 2, 3])
            E_v_y = label_binarize(y_E_v, classes=[1, 2, 3])
            E_v_F_y = label_binarize(y_E_v_F, classes=[1, 2, 3])
            train_y = label_binarize(y_train, classes=[1, 2, 3])
            train_F_y = label_binarize(y_train_F, classes=[1, 2, 3])
            test_F_y = label_binarize(y_test_F, classes=[1, 2, 3])

            n_classes = y.shape[1]

            cm_DT = confusion_matrix(y_test, test_y_pred_DT, labels=classifier_DT.classes_)
            cm_NB = confusion_matrix(y_test, test_y_pred_NB, labels=classifier_NB.classes_)
            cm_kNN = confusion_matrix(y_test, test_y_pred_kNN, labels=classifier_kNN.classes_)
            cm_SVM = confusion_matrix(y_test, test_y_pred_SVM, labels=classifier_SVM.classes_)
            cm_RF = confusion_matrix(y_test, test_y_pred_RF, labels=classifier_RF.classes_)
            cm_ANN = confusion_matrix(y_test, test_y_pred_ANN, labels=classifier_ANN.classes_)

            cm_DT_E_v = confusion_matrix(y_E_v, E_v_y_pred_DT, labels=classifier_DT.classes_)
            cm_DT_E_v_F = confusion_matrix(y_E_v_F, E_v_y_F_pred_DT, labels=classifier_DT.classes_)
            cm_NB_E_v = confusion_matrix(y_E_v, E_v_y_pred_NB, labels=classifier_NB.classes_)
            cm_NB_E_v_F = confusion_matrix(y_E_v_F, E_v_y_F_pred_NB, labels=classifier_NB.classes_)
            cm_kNN_E_v = confusion_matrix(y_E_v, E_v_y_pred_kNN, labels=classifier_kNN.classes_)
            cm_kNN_E_v_F = confusion_matrix(y_E_v_F, E_v_y_F_pred_kNN, labels=classifier_kNN.classes_)
            cm_SVM_E_v = confusion_matrix(y_E_v, E_v_y_pred_SVM, labels=classifier_SVM.classes_)
            cm_SVM_E_v_F = confusion_matrix(y_E_v_F, E_v_y_F_pred_SVM, labels=classifier_SVM.classes_)
            cm_RF_E_v = confusion_matrix(y_E_v, E_v_y_pred_RF, labels=classifier_RF.classes_)
            cm_RF_E_v_F = confusion_matrix(y_E_v_F, E_v_y_F_pred_RF, labels=classifier_RF.classes_)
            cm_ANN_E_v = confusion_matrix(y_E_v, E_v_y_pred_ANN, labels=classifier_ANN.classes_)
            cm_ANN_E_v_F = confusion_matrix(y_E_v_F, E_v_y_F_pred_ANN, labels=classifier_ANN.classes_)

            cm_DT_test_F = confusion_matrix(y_test_F, test_y_F_pred_DT, labels=classifier_DT.classes_)
            cm_NB_test_F = confusion_matrix(y_test_F, test_y_F_pred_NB, labels=classifier_NB.classes_)
            cm_kNN_test_F = confusion_matrix(y_test_F, test_y_F_pred_kNN, labels=classifier_kNN.classes_)
            cm_SVM_test_F = confusion_matrix(y_test_F, test_y_F_pred_SVM, labels=classifier_SVM.classes_)
            cm_RF_test_F = confusion_matrix(y_test_F, test_y_F_pred_RF, labels=classifier_RF.classes_)
            cm_ANN_test_F = confusion_matrix(y_test_F, test_y_F_pred_ANN, labels=classifier_ANN.classes_)

            cm_DT_train_F = confusion_matrix(y_train_F, train_y_F_pred_DT, labels=classifier_DT.classes_)
            cm_NB_train_F = confusion_matrix(y_train_F, train_y_F_pred_NB, labels=classifier_NB.classes_)
            cm_kNN_train_F = confusion_matrix(y_train_F, train_y_F_pred_kNN, labels=classifier_kNN.classes_)
            cm_SVM_train_F = confusion_matrix(y_train_F, train_y_F_pred_SVM, labels=classifier_SVM.classes_)
            cm_RF_train_F = confusion_matrix(y_train_F, train_y_F_pred_RF, labels=classifier_RF.classes_)
            cm_ANN_train_F = confusion_matrix(y_train_F, train_y_F_pred_ANN, labels=classifier_ANN.classes_)


            #print(classifier_DT.classes_)

            cm_DT = cm_DT.astype('float') / cm_DT.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_DT = np.around(cm_DT, decimals=2)

            cm_NB = cm_NB.astype('float') / cm_NB.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_NB = np.around(cm_NB, decimals=2)

            cm_kNN = cm_kNN.astype('float') / cm_kNN.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_kNN = np.around(cm_kNN, decimals=2)

            cm_SVM = cm_SVM.astype('float') / cm_SVM.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_SVM = np.around(cm_SVM, decimals=2)

            cm_RF = cm_RF.astype('float') / cm_RF.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_RF = np.around(cm_RF, decimals=2)

            cm_ANN = cm_ANN.astype('float') / cm_ANN.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_ANN = np.around(cm_ANN, decimals=2)



            cm_DT_E_v = cm_DT_E_v.astype('float') / cm_DT_E_v.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_DT_E_v = np.around(cm_DT_E_v, decimals=2)

            cm_NB_E_v = cm_NB_E_v.astype('float') / cm_NB_E_v.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_NB_E_v = np.around(cm_NB_E_v, decimals=2)

            cm_kNN_E_v = cm_kNN_E_v.astype('float') / cm_kNN_E_v.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_kNN_E_v = np.around(cm_kNN_E_v, decimals=2)

            cm_SVM_E_v = cm_SVM_E_v.astype('float') / cm_SVM_E_v.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_SVM_E_v = np.around(cm_SVM_E_v, decimals=2)

            cm_RF_E_v = cm_RF_E_v.astype('float') / cm_RF_E_v.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_RF_E_v = np.around(cm_RF_E_v, decimals=2)

            cm_ANN_E_v = cm_ANN_E_v.astype('float') / cm_ANN_E_v.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_ANN_E_v = np.around(cm_ANN_E_v, decimals=2)



            cm_DT_E_v_F = cm_DT_E_v_F.astype('float') / cm_DT_E_v_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_DT_E_v_F = np.around(cm_DT_E_v_F, decimals=2)

            cm_NB_E_v_F = cm_NB_E_v_F.astype('float') / cm_NB_E_v_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_NB_E_v_F = np.around(cm_NB_E_v_F, decimals=2)

            cm_kNN_E_v_F = cm_kNN_E_v_F.astype('float') / cm_kNN_E_v_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_kNN_E_v_F = np.around(cm_kNN_E_v_F, decimals=2)

            cm_SVM_E_v_F = cm_SVM_E_v_F.astype('float') / cm_SVM_E_v_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_SVM_E_v_F = np.around(cm_SVM_E_v_F, decimals=2)

            cm_RF_E_v_F = cm_RF_E_v_F.astype('float') / cm_RF_E_v_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_RF_E_v_F = np.around(cm_RF_E_v_F, decimals=2)

            cm_ANN_E_v_F = cm_ANN_E_v_F.astype('float') / cm_ANN_E_v_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_ANN_E_v_F = np.around(cm_ANN_E_v_F, decimals=2)



            cm_DT_test_F = cm_DT_test_F.astype('float') / cm_DT_test_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_DT_test_F = np.around(cm_DT_test_F, decimals=2)

            cm_NB_test_F = cm_NB_test_F.astype('float') / cm_NB_test_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_NB_test_F = np.around(cm_NB_test_F, decimals=2)

            cm_kNN_test_F = cm_kNN_test_F.astype('float') / cm_kNN_test_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_kNN_test_F = np.around(cm_kNN_test_F, decimals=2)

            cm_SVM_test_F = cm_SVM_test_F.astype('float') / cm_SVM_test_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_SVM_test_F = np.around(cm_SVM_test_F, decimals=2)

            cm_RF_test_F = cm_RF_test_F.astype('float') / cm_RF_test_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_RF_test_F = np.around(cm_RF_test_F, decimals=2)

            cm_ANN_test_F = cm_ANN_test_F.astype('float') / cm_ANN_test_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_ANN_test_F = np.around(cm_ANN_test_F, decimals=2)



            cm_DT_train_F = cm_DT_train_F.astype('float') / cm_DT_train_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_DT_train_F = np.around(cm_DT_train_F, decimals=2)

            cm_NB_train_F = cm_NB_train_F.astype('float') / cm_NB_train_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_NB_train_F = np.around(cm_NB_train_F, decimals=2)

            cm_kNN_train_F = cm_kNN_train_F.astype('float') / cm_kNN_train_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_kNN_train_F = np.around(cm_kNN_train_F, decimals=2)

            cm_SVM_train_F = cm_SVM_train_F.astype('float') / cm_SVM_train_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_SVM_train_F = np.around(cm_SVM_train_F, decimals=2)

            cm_RF_train_F = cm_RF_train_F.astype('float') / cm_RF_train_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_RF_train_F = np.around(cm_RF_train_F, decimals=2)

            cm_ANN_train_F = cm_ANN_train_F.astype('float') / cm_ANN_train_F.sum(axis=1)[:, np.newaxis]  # 归一化
            cm_ANN_train_F = np.around(cm_ANN_train_F, decimals=2)
#
#
            for k in range(len(cm_DT)):
              for t in range(len(cm_DT[0])):
                 result_matrix_DT[k][t] = result_matrix_DT[k][t] + cm_DT[k][t]

            for k in range(len(cm_NB)):
              for t in range(len(cm_NB[0])):
                 result_matrix_NB[k][t] = result_matrix_NB[k][t] + cm_NB[k][t]

            for k in range(len(cm_kNN)):
              for t in range(len(cm_kNN[0])):
                 result_matrix_kNN[k][t] = result_matrix_kNN[k][t] + cm_kNN[k][t]

            for k in range(len(cm_SVM)):
              for t in range(len(cm_SVM[0])):
                 result_matrix_SVM[k][t] = result_matrix_SVM[k][t] + cm_SVM[k][t]

            for k in range(len(cm_RF)):
              for t in range(len(cm_RF[0])):
                 result_matrix_RF[k][t] = result_matrix_RF[k][t] + cm_RF[k][t]

            for k in range(len(cm_ANN)):
              for t in range(len(cm_ANN[0])):
                 result_matrix_ANN[k][t] = result_matrix_ANN[k][t] + cm_ANN[k][t]



            for k in range(len(cm_DT_E_v)):
              for t in range(len(cm_DT_E_v[0])):
                 result_matrix_DT_E_v[k][t] = result_matrix_DT_E_v[k][t] + cm_DT_E_v[k][t]

            for k in range(len(cm_NB_E_v)):
              for t in range(len(cm_NB_E_v[0])):
                 result_matrix_NB_E_v[k][t] = result_matrix_NB_E_v[k][t] + cm_NB_E_v[k][t]

            for k in range(len(cm_kNN_E_v)):
              for t in range(len(cm_kNN_E_v[0])):
                 result_matrix_kNN_E_v[k][t] = result_matrix_kNN_E_v[k][t] + cm_kNN_E_v[k][t]

            for k in range(len(cm_SVM_E_v)):
              for t in range(len(cm_SVM_E_v[0])):
                 result_matrix_SVM_E_v[k][t] = result_matrix_SVM_E_v[k][t] + cm_SVM_E_v[k][t]

            for k in range(len(cm_RF_E_v)):
              for t in range(len(cm_RF_E_v[0])):
                 result_matrix_RF_E_v[k][t] = result_matrix_RF_E_v[k][t] + cm_RF_E_v[k][t]

            for k in range(len(cm_ANN_E_v)):
              for t in range(len(cm_ANN_E_v[0])):
                 result_matrix_ANN_E_v[k][t] = result_matrix_ANN_E_v[k][t] + cm_ANN_E_v[k][t]




            for k in range(len(cm_DT_E_v_F)):
                for t in range(len(cm_DT_E_v_F[0])):
                    result_matrix_DT_E_v_F[k][t] = result_matrix_DT_E_v_F[k][t] + cm_DT_E_v_F[k][t]

            for k in range(len(cm_NB_E_v_F)):
                for t in range(len(cm_NB_E_v_F[0])):
                    result_matrix_NB_E_v_F[k][t] = result_matrix_NB_E_v_F[k][t] + cm_NB_E_v_F[k][t]

            for k in range(len(cm_kNN_E_v_F)):
                for t in range(len(cm_kNN_E_v_F[0])):
                    result_matrix_kNN_E_v_F[k][t] = result_matrix_kNN_E_v_F[k][t] + cm_kNN_E_v_F[k][t]

            for k in range(len(cm_SVM_E_v_F)):
                for t in range(len(cm_SVM_E_v_F[0])):
                    result_matrix_SVM_E_v_F[k][t] = result_matrix_SVM_E_v_F[k][t] + cm_SVM_E_v_F[k][t]

            for k in range(len(cm_RF_E_v_F)):
                for t in range(len(cm_RF_E_v_F[0])):
                    result_matrix_RF_E_v_F[k][t] = result_matrix_RF_E_v_F[k][t] + cm_RF_E_v_F[k][t]

            for k in range(len(cm_ANN_E_v_F)):
                for t in range(len(cm_ANN_E_v_F[0])):
                    result_matrix_ANN_E_v_F[k][t] = result_matrix_ANN_E_v_F[k][t] + cm_ANN_E_v_F[k][t]




            for k in range(len(cm_DT_test_F)):
                for t in range(len(cm_DT_test_F[0])):
                    result_matrix_DT_test_F[k][t] = result_matrix_DT_test_F[k][t] + cm_DT_test_F[k][t]

            for k in range(len(cm_NB_test_F)):
                for t in range(len(cm_NB_test_F[0])):
                    result_matrix_NB_test_F[k][t] = result_matrix_NB_test_F[k][t] + cm_NB_test_F[k][t]

            for k in range(len(cm_kNN_test_F)):
                for t in range(len(cm_kNN_test_F[0])):
                    result_matrix_kNN_test_F[k][t] = result_matrix_kNN_test_F[k][t] + cm_kNN_test_F[k][t]

            for k in range(len(cm_SVM_test_F)):
                for t in range(len(cm_SVM_test_F[0])):
                    result_matrix_SVM_test_F[k][t] = result_matrix_SVM_test_F[k][t] + cm_SVM_test_F[k][t]

            for k in range(len(cm_RF_test_F)):
                for t in range(len(cm_RF_test_F[0])):
                    result_matrix_RF_test_F[k][t] = result_matrix_RF_test_F[k][t] + cm_RF_test_F[k][t]

            for k in range(len(cm_ANN_test_F)):
                for t in range(len(cm_ANN_test_F[0])):
                    result_matrix_ANN_test_F[k][t] = result_matrix_ANN_test_F[k][t] + cm_ANN_test_F[k][t]



            for k in range(len(cm_DT_train_F)):
                for t in range(len(cm_DT_train_F[0])):
                    result_matrix_DT_train_F[k][t] = result_matrix_DT_train_F[k][t] + cm_DT_train_F[k][t]

            for k in range(len(cm_NB_train_F)):
                for t in range(len(cm_NB_train_F[0])):
                    result_matrix_NB_train_F[k][t] = result_matrix_NB_train_F[k][t] + cm_NB_train_F[k][t]

            for k in range(len(cm_kNN_train_F)):
                for t in range(len(cm_kNN_train_F[0])):
                    result_matrix_kNN_train_F[k][t] = result_matrix_kNN_train_F[k][t] + cm_kNN_train_F[k][t]

            for k in range(len(cm_SVM_train_F)):
                for t in range(len(cm_SVM_train_F[0])):
                    result_matrix_SVM_train_F[k][t] = result_matrix_SVM_train_F[k][t] + cm_SVM_train_F[k][t]

            for k in range(len(cm_RF_train_F)):
                for t in range(len(cm_RF_train_F[0])):
                    result_matrix_RF_train_F[k][t] = result_matrix_RF_train_F[k][t] + cm_RF_train_F[k][t]

            for k in range(len(cm_ANN_train_F)):
                for t in range(len(cm_ANN_train_F[0])):
                    result_matrix_ANN_train_F[k][t] = result_matrix_ANN_train_F[k][t] + cm_ANN_train_F[k][t]


        # Compute ROC curve and ROC area for each class
            fpr_DT = dict()
            tpr_DT = dict()
            roc_auc_DT = dict()
            for i in range(n_classes):
                fpr_DT[i], tpr_DT[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_DT[:, i])
                roc_auc_DT[i] = auc(fpr_DT[i], tpr_DT[i])

          # Compute micro-average ROC curve and ROC area
            fpr_DT["micro"], tpr_DT["micro"], _ = roc_curve(test_y.ravel(), test_y_pred_proba_DT.ravel())
            roc_auc_DT["micro"] = auc(fpr_DT["micro"], tpr_DT["micro"])


            fpr_NB = dict()
            tpr_NB = dict()
            roc_auc_NB = dict()
            for i in range(n_classes):
                fpr_NB[i], tpr_NB[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_NB[:, i])
                roc_auc_NB[i] = auc(fpr_NB[i], tpr_NB[i])

          # Compute micro-average ROC curve and ROC area
            fpr_NB["micro"], tpr_NB["micro"], _ = roc_curve(test_y.ravel(), test_y_pred_proba_NB.ravel())
            roc_auc_NB["micro"] = auc(fpr_NB["micro"], tpr_NB["micro"])

            fpr_kNN = dict()
            tpr_kNN = dict()
            roc_auc_kNN = dict()
            for i in range(n_classes):
                fpr_kNN[i], tpr_kNN[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_kNN[:, i])
                roc_auc_kNN[i] = auc(fpr_kNN[i], tpr_kNN[i])

          # Compute micro-average ROC curve and ROC area
            fpr_kNN["micro"], tpr_kNN["micro"], _ = roc_curve(test_y.ravel(), test_y_pred_proba_kNN.ravel())
            roc_auc_kNN["micro"] = auc(fpr_kNN["micro"], tpr_kNN["micro"])

            fpr_SVM = dict()
            tpr_SVM = dict()
            roc_auc_SVM = dict()
            for i in range(n_classes):
                fpr_SVM[i], tpr_SVM[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_SVM[:, i])
                roc_auc_SVM[i] = auc(fpr_SVM[i], tpr_SVM[i])

          # Compute micro-average ROC curve and ROC area
            fpr_SVM["micro"], tpr_SVM["micro"], _ = roc_curve(test_y.ravel(), test_y_pred_proba_SVM.ravel())
            roc_auc_SVM["micro"] = auc(fpr_SVM["micro"], tpr_SVM["micro"])

            fpr_RF = dict()
            tpr_RF = dict()
            roc_auc_RF = dict()
            for i in range(n_classes):
                fpr_RF[i], tpr_RF[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_RF[:, i])
                roc_auc_RF[i] = auc(fpr_RF[i], tpr_RF[i])

          # Compute micro-average ROC curve and ROC area
            fpr_RF["micro"], tpr_RF["micro"], _ = roc_curve(test_y.ravel(), test_y_pred_proba_RF.ravel())
            roc_auc_RF["micro"] = auc(fpr_RF["micro"], tpr_RF["micro"])

            fpr_ANN = dict()
            tpr_ANN = dict()
            roc_auc_ANN = dict()
            for i in range(n_classes):
                fpr_ANN[i], tpr_ANN[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_ANN[:, i])
                roc_auc_ANN[i] = auc(fpr_ANN[i], tpr_ANN[i])

          # Compute micro-average ROC curve and ROC area
            fpr_ANN["micro"], tpr_ANN["micro"], _ = roc_curve(test_y.ravel(), test_y_pred_proba_ANN.ravel())
            roc_auc_ANN["micro"] = auc(fpr_ANN["micro"], tpr_ANN["micro"])

            # Compute ROC curve and ROC area for each class
            fpr_DT_E_v = dict()
            tpr_DT_E_v = dict()
            roc_auc_DT_E_v = dict()
            for i in range(n_classes):
                fpr_DT_E_v[i], tpr_DT_E_v[i], _ = roc_curve(E_v_y[:, i], E_v_y_pred_proba_DT[:, i])
                roc_auc_DT_E_v[i] = auc(fpr_DT_E_v[i], tpr_DT_E_v[i])

            # Compute micro-average ROC curve and ROC area
            fpr_DT_E_v["micro"], tpr_DT_E_v["micro"], _ = roc_curve(E_v_y.ravel(), E_v_y_pred_proba_DT.ravel())
            roc_auc_DT_E_v["micro"] = auc(fpr_DT_E_v["micro"], tpr_DT_E_v["micro"])

            fpr_NB_E_v = dict()
            tpr_NB_E_v = dict()
            roc_auc_NB_E_v = dict()
            for i in range(n_classes):
                fpr_NB_E_v[i], tpr_NB_E_v[i], _ = roc_curve(E_v_y[:, i], E_v_y_pred_proba_NB[:, i])
                roc_auc_NB_E_v[i] = auc(fpr_NB_E_v[i], tpr_NB_E_v[i])

            # Compute micro-average ROC curve and ROC area
            fpr_NB_E_v["micro"], tpr_NB_E_v["micro"], _ = roc_curve(E_v_y.ravel(), E_v_y_pred_proba_NB.ravel())
            roc_auc_NB_E_v["micro"] = auc(fpr_NB_E_v["micro"], tpr_NB_E_v["micro"])

            fpr_kNN_E_v = dict()
            tpr_kNN_E_v = dict()
            roc_auc_kNN_E_v = dict()
            for i in range(n_classes):
                fpr_kNN_E_v[i], tpr_kNN_E_v[i], _ = roc_curve(E_v_y[:, i], E_v_y_pred_proba_kNN[:, i])
                roc_auc_kNN_E_v[i] = auc(fpr_kNN_E_v[i], tpr_kNN_E_v[i])

            # Compute micro-average ROC curve and ROC area
            fpr_kNN_E_v["micro"], tpr_kNN_E_v["micro"], _ = roc_curve(E_v_y.ravel(), E_v_y_pred_proba_kNN.ravel())
            roc_auc_kNN_E_v["micro"] = auc(fpr_kNN_E_v["micro"], tpr_kNN_E_v["micro"])

            fpr_SVM_E_v = dict()
            tpr_SVM_E_v = dict()
            roc_auc_SVM_E_v = dict()
            for i in range(n_classes):
                fpr_SVM_E_v[i], tpr_SVM_E_v[i], _ = roc_curve(E_v_y[:, i], E_v_y_pred_proba_SVM[:, i])
                roc_auc_SVM_E_v[i] = auc(fpr_SVM_E_v[i], tpr_SVM_E_v[i])

            # Compute micro-average ROC curve and ROC area
            fpr_SVM_E_v["micro"], tpr_SVM_E_v["micro"], _ = roc_curve(E_v_y.ravel(), E_v_y_pred_proba_SVM.ravel())
            roc_auc_SVM_E_v["micro"] = auc(fpr_SVM_E_v["micro"], tpr_SVM_E_v["micro"])

            fpr_RF_E_v = dict()
            tpr_RF_E_v = dict()
            roc_auc_RF_E_v = dict()
            for i in range(n_classes):
                fpr_RF_E_v[i], tpr_RF_E_v[i], _ = roc_curve(E_v_y[:, i], E_v_y_pred_proba_RF[:, i])
                roc_auc_RF_E_v[i] = auc(fpr_RF_E_v[i], tpr_RF_E_v[i])

            # Compute micro-average ROC curve and ROC area
            fpr_RF_E_v["micro"], tpr_RF_E_v["micro"], _ = roc_curve(E_v_y.ravel(), E_v_y_pred_proba_RF.ravel())
            roc_auc_RF_E_v["micro"] = auc(fpr_RF_E_v["micro"], tpr_RF_E_v["micro"])

            fpr_ANN_E_v = dict()
            tpr_ANN_E_v = dict()
            roc_auc_ANN_E_v = dict()
            for i in range(n_classes):
                fpr_ANN_E_v[i], tpr_ANN_E_v[i], _ = roc_curve(E_v_y[:, i], E_v_y_pred_proba_ANN[:, i])
                roc_auc_ANN_E_v[i] = auc(fpr_ANN_E_v[i], tpr_ANN_E_v[i])

            # Compute micro-average ROC curve and ROC area
            fpr_ANN_E_v["micro"], tpr_ANN_E_v["micro"], _ = roc_curve(E_v_y.ravel(), E_v_y_pred_proba_ANN.ravel())
            roc_auc_ANN_E_v["micro"] = auc(fpr_ANN_E_v["micro"], tpr_ANN_E_v["micro"])

            # Compute ROC curve and ROC area for each class
            fpr_DT_E_v_F = dict()
            tpr_DT_E_v_F = dict()
            roc_auc_DT_E_v_F = dict()
            for i in range(n_classes):
                fpr_DT_E_v_F[i], tpr_DT_E_v_F[i], _ = roc_curve(E_v_F_y[:, i], E_v_y_F_pred_proba_DT[:, i])
                roc_auc_DT_E_v_F[i] = auc(fpr_DT_E_v_F[i], tpr_DT_E_v_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_DT_E_v_F["micro"], tpr_DT_E_v_F["micro"], _ = roc_curve(E_v_F_y.ravel(), E_v_y_F_pred_proba_DT.ravel())
            roc_auc_DT_E_v_F["micro"] = auc(fpr_DT_E_v_F["micro"], tpr_DT_E_v_F["micro"])

            fpr_NB_E_v_F = dict()
            tpr_NB_E_v_F = dict()
            roc_auc_NB_E_v_F = dict()
            for i in range(n_classes):
                fpr_NB_E_v_F[i], tpr_NB_E_v_F[i], _ = roc_curve(E_v_F_y[:, i], E_v_y_F_pred_proba_NB[:, i])
                roc_auc_NB_E_v_F[i] = auc(fpr_NB_E_v_F[i], tpr_NB_E_v_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_NB_E_v_F["micro"], tpr_NB_E_v_F["micro"], _ = roc_curve(E_v_F_y.ravel(), E_v_y_F_pred_proba_NB.ravel())
            roc_auc_NB_E_v_F["micro"] = auc(fpr_NB_E_v_F["micro"], tpr_NB_E_v_F["micro"])

            fpr_kNN_E_v_F = dict()
            tpr_kNN_E_v_F = dict()
            roc_auc_kNN_E_v_F = dict()
            for i in range(n_classes):
                fpr_kNN_E_v_F[i], tpr_kNN_E_v_F[i], _ = roc_curve(E_v_F_y[:, i], E_v_y_F_pred_proba_kNN[:, i])
                roc_auc_kNN_E_v_F[i] = auc(fpr_kNN_E_v_F[i], tpr_kNN_E_v_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_kNN_E_v_F["micro"], tpr_kNN_E_v_F["micro"], _ = roc_curve(E_v_F_y.ravel(), E_v_y_F_pred_proba_kNN.ravel())
            roc_auc_kNN_E_v_F["micro"] = auc(fpr_kNN_E_v_F["micro"], tpr_kNN_E_v_F["micro"])

            fpr_SVM_E_v_F = dict()
            tpr_SVM_E_v_F = dict()
            roc_auc_SVM_E_v_F = dict()
            for i in range(n_classes):
                fpr_SVM_E_v_F[i], tpr_SVM_E_v_F[i], _ = roc_curve(E_v_F_y[:, i], E_v_y_F_pred_proba_SVM[:, i])
                roc_auc_SVM_E_v_F[i] = auc(fpr_SVM_E_v_F[i], tpr_SVM_E_v_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_SVM_E_v_F["micro"], tpr_SVM_E_v_F["micro"], _ = roc_curve(E_v_F_y.ravel(), E_v_y_F_pred_proba_SVM.ravel())
            roc_auc_SVM_E_v_F["micro"] = auc(fpr_SVM_E_v_F["micro"], tpr_SVM_E_v_F["micro"])

            fpr_RF_E_v_F = dict()
            tpr_RF_E_v_F = dict()
            roc_auc_RF_E_v_F = dict()
            for i in range(n_classes):
                fpr_RF_E_v_F[i], tpr_RF_E_v_F[i], _ = roc_curve(E_v_F_y[:, i], E_v_y_F_pred_proba_RF[:, i])
                roc_auc_RF_E_v_F[i] = auc(fpr_RF_E_v_F[i], tpr_RF_E_v_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_RF_E_v_F["micro"], tpr_RF_E_v_F["micro"], _ = roc_curve(E_v_F_y.ravel(), E_v_y_F_pred_proba_RF.ravel())
            roc_auc_RF_E_v_F["micro"] = auc(fpr_RF_E_v_F["micro"], tpr_RF_E_v_F["micro"])

            fpr_ANN_E_v_F = dict()
            tpr_ANN_E_v_F = dict()
            roc_auc_ANN_E_v_F = dict()
            for i in range(n_classes):
                fpr_ANN_E_v_F[i], tpr_ANN_E_v_F[i], _ = roc_curve(E_v_F_y[:, i], E_v_y_F_pred_proba_ANN[:, i])
                roc_auc_ANN_E_v_F[i] = auc(fpr_ANN_E_v_F[i], tpr_ANN_E_v_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_ANN_E_v_F["micro"], tpr_ANN_E_v_F["micro"], _ = roc_curve(E_v_F_y.ravel(), E_v_y_F_pred_proba_ANN.ravel())
            roc_auc_ANN_E_v_F["micro"] = auc(fpr_ANN_E_v_F["micro"], tpr_ANN_E_v_F["micro"])

            fpr_DT_train_F = dict()
            tpr_DT_train_F = dict()
            roc_auc_DT_train_F = dict()
            for i in range(n_classes):
                fpr_DT_train_F[i], tpr_DT_train_F[i], _ = roc_curve(train_F_y[:, i], train_y_F_pred_proba_DT[:, i])
                roc_auc_DT_train_F[i] = auc(fpr_DT_train_F[i], tpr_DT_train_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_DT_train_F["micro"], tpr_DT_train_F["micro"], _ = roc_curve(train_F_y.ravel(), train_y_F_pred_proba_DT.ravel())
            roc_auc_DT_train_F["micro"] = auc(fpr_DT_train_F["micro"], tpr_DT_train_F["micro"])

            fpr_NB_train_F = dict()
            tpr_NB_train_F = dict()
            roc_auc_NB_train_F = dict()
            for i in range(n_classes):
                fpr_NB_train_F[i], tpr_NB_train_F[i], _ = roc_curve(train_F_y[:, i], train_y_F_pred_proba_NB[:, i])
                roc_auc_NB_train_F[i] = auc(fpr_NB_train_F[i], tpr_NB_train_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_NB_train_F["micro"], tpr_NB_train_F["micro"], _ = roc_curve(train_F_y.ravel(), train_y_F_pred_proba_NB.ravel())
            roc_auc_NB_train_F["micro"] = auc(fpr_NB_train_F["micro"], tpr_NB_train_F["micro"])

            fpr_kNN_train_F = dict()
            tpr_kNN_train_F = dict()
            roc_auc_kNN_train_F = dict()
            for i in range(n_classes):
                fpr_kNN_train_F[i], tpr_kNN_train_F[i], _ = roc_curve(train_F_y[:, i], train_y_F_pred_proba_kNN[:, i])
                roc_auc_kNN_train_F[i] = auc(fpr_kNN_train_F[i], tpr_kNN_train_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_kNN_train_F["micro"], tpr_kNN_train_F["micro"], _ = roc_curve(train_F_y.ravel(),train_y_F_pred_proba_kNN.ravel())
            roc_auc_kNN_train_F["micro"] = auc(fpr_kNN_train_F["micro"], tpr_kNN_train_F["micro"])

            fpr_SVM_train_F = dict()
            tpr_SVM_train_F = dict()
            roc_auc_SVM_train_F = dict()
            for i in range(n_classes):
                fpr_SVM_train_F[i], tpr_SVM_train_F[i], _ = roc_curve(train_F_y[:, i], train_y_F_pred_proba_SVM[:, i])
                roc_auc_SVM_train_F[i] = auc(fpr_SVM_train_F[i], tpr_SVM_train_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_SVM_train_F["micro"], tpr_SVM_train_F["micro"], _ = roc_curve(train_F_y.ravel(),train_y_F_pred_proba_SVM.ravel())
            roc_auc_SVM_train_F["micro"] = auc(fpr_SVM_train_F["micro"], tpr_SVM_train_F["micro"])

            fpr_RF_train_F = dict()
            tpr_RF_train_F = dict()
            roc_auc_RF_train_F = dict()
            for i in range(n_classes):
                fpr_RF_train_F[i], tpr_RF_train_F[i], _ = roc_curve(train_F_y[:, i], train_y_F_pred_proba_RF[:, i])
                roc_auc_RF_train_F[i] = auc(fpr_RF_train_F[i], tpr_RF_train_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_RF_train_F["micro"], tpr_RF_train_F["micro"], _ = roc_curve(train_F_y.ravel(), train_y_F_pred_proba_RF.ravel())
            roc_auc_RF_train_F["micro"] = auc(fpr_RF_train_F["micro"], tpr_RF_train_F["micro"])

            fpr_ANN_train_F = dict()
            tpr_ANN_train_F = dict()
            roc_auc_ANN_train_F = dict()
            for i in range(n_classes):
                fpr_ANN_train_F[i], tpr_ANN_train_F[i], _ = roc_curve(train_F_y[:, i], train_y_F_pred_proba_ANN[:, i])
                roc_auc_ANN_train_F[i] = auc(fpr_ANN_train_F[i], tpr_ANN_train_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_ANN_train_F["micro"], tpr_ANN_train_F["micro"], _ = roc_curve(train_F_y.ravel(),train_y_F_pred_proba_ANN.ravel())
            roc_auc_ANN_train_F["micro"] = auc(fpr_ANN_train_F["micro"], tpr_ANN_train_F["micro"])


            fpr_DT_train = dict()
            tpr_DT_train = dict()
            roc_auc_DT_train = dict()
            for i in range(n_classes):
                fpr_DT_train[i], tpr_DT_train[i], _ = roc_curve(train_y[:, i], train_y_pred_proba_DT[:, i])
                roc_auc_DT_train[i] = auc(fpr_DT_train[i], tpr_DT_train[i])

            # Compute micro-average ROC curve and ROC area
            fpr_DT_train["micro"], tpr_DT_train["micro"], _ = roc_curve(train_y.ravel(),train_y_pred_proba_DT.ravel())
            roc_auc_DT_train["micro"] = auc(fpr_DT_train["micro"], tpr_DT_train["micro"])

            fpr_NB_train = dict()
            tpr_NB_train = dict()
            roc_auc_NB_train = dict()
            for i in range(n_classes):
                fpr_NB_train[i], tpr_NB_train[i], _ = roc_curve(train_y[:, i], train_y_pred_proba_NB[:, i])
                roc_auc_NB_train[i] = auc(fpr_NB_train[i], tpr_NB_train[i])

            # Compute micro-average ROC curve and ROC area
            fpr_NB_train["micro"], tpr_NB_train["micro"], _ = roc_curve(train_y.ravel(),train_y_pred_proba_NB.ravel())
            roc_auc_NB_train["micro"] = auc(fpr_NB_train["micro"], tpr_NB_train["micro"])

            fpr_kNN_train = dict()
            tpr_kNN_train = dict()
            roc_auc_kNN_train = dict()
            for i in range(n_classes):
                fpr_kNN_train[i], tpr_kNN_train[i], _ = roc_curve(train_y[:, i], train_y_pred_proba_kNN[:, i])
                roc_auc_kNN_train[i] = auc(fpr_kNN_train[i], tpr_kNN_train[i])

            # Compute micro-average ROC curve and ROC area
            fpr_kNN_train["micro"], tpr_kNN_train["micro"], _ = roc_curve(train_y.ravel(),train_y_pred_proba_kNN.ravel())
            roc_auc_kNN_train["micro"] = auc(fpr_kNN_train["micro"], tpr_kNN_train["micro"])

            fpr_SVM_train = dict()
            tpr_SVM_train = dict()
            roc_auc_SVM_train = dict()
            for i in range(n_classes):
                fpr_SVM_train[i], tpr_SVM_train[i], _ = roc_curve(train_y[:, i], train_y_pred_proba_SVM[:, i])
                roc_auc_SVM_train[i] = auc(fpr_SVM_train[i], tpr_SVM_train[i])

            # Compute micro-average ROC curve and ROC area
            fpr_SVM_train["micro"], tpr_SVM_train["micro"], _ = roc_curve(train_y.ravel(),train_y_pred_proba_SVM.ravel())
            roc_auc_SVM_train["micro"] = auc(fpr_SVM_train["micro"], tpr_SVM_train["micro"])

            fpr_RF_train = dict()
            tpr_RF_train = dict()
            roc_auc_RF_train = dict()
            for i in range(n_classes):
                fpr_RF_train[i], tpr_RF_train[i], _ = roc_curve(train_y[:, i], train_y_pred_proba_RF[:, i])
                roc_auc_RF_train[i] = auc(fpr_RF_train[i], tpr_RF_train[i])

            # Compute micro-average ROC curve and ROC area
            fpr_RF_train["micro"], tpr_RF_train["micro"], _ = roc_curve(train_y.ravel(),train_y_pred_proba_RF.ravel())
            roc_auc_RF_train["micro"] = auc(fpr_RF_train["micro"], tpr_RF_train["micro"])

            fpr_ANN_train = dict()
            tpr_ANN_train = dict()
            roc_auc_ANN_train = dict()
            for i in range(n_classes):
                fpr_ANN_train[i], tpr_ANN_train[i], _ = roc_curve(train_y[:, i], train_y_pred_proba_ANN[:, i])
                roc_auc_ANN_train[i] = auc(fpr_ANN_train[i], tpr_ANN_train[i])

            # Compute micro-average ROC curve and ROC area
            fpr_ANN_train["micro"], tpr_ANN_train["micro"], _ = roc_curve(train_y.ravel(),train_y_pred_proba_ANN.ravel())
            roc_auc_ANN_train["micro"] = auc(fpr_ANN_train["micro"], tpr_ANN_train["micro"])

            fpr_DT_test_F = dict()
            tpr_DT_test_F = dict()
            roc_auc_DT_test_F = dict()
            for i in range(n_classes):
                fpr_DT_test_F[i], tpr_DT_test_F[i], _ = roc_curve(test_F_y[:, i], test_y_F_pred_proba_DT[:, i])
                roc_auc_DT_test_F[i] = auc(fpr_DT_test_F[i], tpr_DT_test_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_DT_test_F["micro"], tpr_DT_test_F["micro"], _ = roc_curve(test_F_y.ravel(), test_y_F_pred_proba_DT.ravel())
            roc_auc_DT_test_F["micro"] = auc(fpr_DT_test_F["micro"], tpr_DT_test_F["micro"])

            fpr_NB_test_F = dict()
            tpr_NB_test_F = dict()
            roc_auc_NB_test_F = dict()
            for i in range(n_classes):
                fpr_NB_test_F[i], tpr_NB_test_F[i], _ = roc_curve(test_F_y[:, i], test_y_F_pred_proba_NB[:, i])
                roc_auc_NB_test_F[i] = auc(fpr_NB_test_F[i], tpr_NB_test_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_NB_test_F["micro"], tpr_NB_test_F["micro"], _ = roc_curve(test_F_y.ravel(), test_y_F_pred_proba_NB.ravel())
            roc_auc_NB_test_F["micro"] = auc(fpr_NB_test_F["micro"], tpr_NB_test_F["micro"])

            fpr_kNN_test_F = dict()
            tpr_kNN_test_F = dict()
            roc_auc_kNN_test_F = dict()
            for i in range(n_classes):
                fpr_kNN_test_F[i], tpr_kNN_test_F[i], _ = roc_curve(test_F_y[:, i], test_y_F_pred_proba_kNN[:, i])
                roc_auc_kNN_test_F[i] = auc(fpr_kNN_test_F[i], tpr_kNN_test_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_kNN_test_F["micro"], tpr_kNN_test_F["micro"], _ = roc_curve(test_F_y.ravel(),test_y_F_pred_proba_kNN.ravel())
            roc_auc_kNN_test_F["micro"] = auc(fpr_kNN_test_F["micro"], tpr_kNN_test_F["micro"])

            fpr_SVM_test_F = dict()
            tpr_SVM_test_F = dict()
            roc_auc_SVM_test_F = dict()
            for i in range(n_classes):
                fpr_SVM_test_F[i], tpr_SVM_test_F[i], _ = roc_curve(test_F_y[:, i], test_y_F_pred_proba_SVM[:, i])
                roc_auc_SVM_test_F[i] = auc(fpr_SVM_test_F[i], tpr_SVM_test_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_SVM_test_F["micro"], tpr_SVM_test_F["micro"], _ = roc_curve(test_F_y.ravel(),test_y_F_pred_proba_SVM.ravel())
            roc_auc_SVM_test_F["micro"] = auc(fpr_SVM_test_F["micro"], tpr_SVM_test_F["micro"])

            fpr_RF_test_F = dict()
            tpr_RF_test_F = dict()
            roc_auc_RF_test_F = dict()
            for i in range(n_classes):
                fpr_RF_test_F[i], tpr_RF_test_F[i], _ = roc_curve(test_F_y[:, i], test_y_F_pred_proba_RF[:, i])
                roc_auc_RF_test_F[i] = auc(fpr_RF_test_F[i], tpr_RF_test_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_RF_test_F["micro"], tpr_RF_test_F["micro"], _ = roc_curve(test_F_y.ravel(), test_y_F_pred_proba_RF.ravel())
            roc_auc_RF_test_F["micro"] = auc(fpr_RF_test_F["micro"], tpr_RF_test_F["micro"])

            fpr_ANN_test_F = dict()
            tpr_ANN_test_F = dict()
            roc_auc_ANN_test_F = dict()
            for i in range(n_classes):
                fpr_ANN_test_F[i], tpr_ANN_test_F[i], _ = roc_curve(test_F_y[:, i], test_y_F_pred_proba_ANN[:, i])
                roc_auc_ANN_test_F[i] = auc(fpr_ANN_test_F[i], tpr_ANN_test_F[i])

            # Compute micro-average ROC curve and ROC area
            fpr_ANN_test_F["micro"], tpr_ANN_test_F["micro"], _ = roc_curve(test_F_y.ravel(),test_y_F_pred_proba_ANN.ravel())
            roc_auc_ANN_test_F["micro"] = auc(fpr_ANN_test_F["micro"], tpr_ANN_test_F["micro"])


            fpr_DT_test= dict()
            tpr_DT_test = dict()
            roc_auc_DT_test = dict()
            for i in range(n_classes):
                fpr_DT_test[i], tpr_DT_test[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_DT[:, i])
                roc_auc_DT_test[i] = auc(fpr_DT_test[i], tpr_DT_test[i])

            # Compute micro-average ROC curve and ROC area
            fpr_DT_test["micro"], tpr_DT_test["micro"], _ = roc_curve(test_y.ravel(),test_y_pred_proba_DT.ravel())
            roc_auc_DT_test["micro"] = auc(fpr_DT_test["micro"], tpr_DT_test["micro"])

            fpr_NB_test = dict()
            tpr_NB_test = dict()
            roc_auc_NB_test = dict()
            for i in range(n_classes):
                fpr_NB_test[i], tpr_NB_test[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_NB[:, i])
                roc_auc_NB_test[i] = auc(fpr_NB_test[i], tpr_NB_test[i])

            # Compute micro-average ROC curve and ROC area
            fpr_NB_test["micro"], tpr_NB_test["micro"], _ = roc_curve(test_y.ravel(),test_y_pred_proba_NB.ravel())
            roc_auc_NB_test["micro"] = auc(fpr_NB_test["micro"], tpr_NB_test["micro"])

            fpr_kNN_test = dict()
            tpr_kNN_test = dict()
            roc_auc_kNN_test = dict()
            for i in range(n_classes):
                fpr_kNN_test[i], tpr_kNN_test[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_kNN[:, i])
                roc_auc_kNN_test[i] = auc(fpr_kNN_test[i], tpr_kNN_test[i])

            # Compute micro-average ROC curve and ROC area
            fpr_kNN_test["micro"], tpr_kNN_test["micro"], _ = roc_curve(test_y.ravel(),test_y_pred_proba_kNN.ravel())
            roc_auc_kNN_test["micro"] = auc(fpr_kNN_test["micro"], tpr_kNN_test["micro"])

            fpr_SVM_test = dict()
            tpr_SVM_test = dict()
            roc_auc_SVM_test = dict()
            for i in range(n_classes):
                fpr_SVM_test[i], tpr_SVM_test[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_SVM[:, i])
                roc_auc_SVM_test[i] = auc(fpr_SVM_test[i], tpr_SVM_test[i])

            # Compute micro-average ROC curve and ROC area
            fpr_SVM_test["micro"], tpr_SVM_test["micro"], _ = roc_curve(test_y.ravel(),test_y_pred_proba_SVM.ravel())
            roc_auc_SVM_test["micro"] = auc(fpr_SVM_test["micro"], tpr_SVM_test["micro"])

            fpr_RF_test = dict()
            tpr_RF_test = dict()
            roc_auc_RF_test = dict()
            for i in range(n_classes):
                fpr_RF_test[i], tpr_RF_test[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_RF[:, i])
                roc_auc_RF_test[i] = auc(fpr_RF_test[i], tpr_RF_test[i])

            # Compute micro-average ROC curve and ROC area
            fpr_RF_test["micro"], tpr_RF_test["micro"], _ = roc_curve(test_y.ravel(),test_y_pred_proba_RF.ravel())
            roc_auc_RF_test["micro"] = auc(fpr_RF_test["micro"], tpr_RF_test["micro"])

            fpr_ANN_test = dict()
            tpr_ANN_test = dict()
            roc_auc_ANN_test = dict()
            for i in range(n_classes):
                fpr_ANN_test[i], tpr_ANN_test[i], _ = roc_curve(test_y[:, i], test_y_pred_proba_ANN[:, i])
                roc_auc_ANN_test[i] = auc(fpr_ANN_test[i], tpr_ANN_test[i])

            # Compute micro-average ROC curve and ROC area
            fpr_ANN_test["micro"], tpr_ANN_test["micro"], _ = roc_curve(test_y.ravel(),test_y_pred_proba_ANN.ravel())
            roc_auc_ANN_test["micro"] = auc(fpr_ANN_test["micro"], tpr_ANN_test["micro"])

            # First aggregate all false positive rates
            all_fpr_DT = np.unique(np.concatenate([fpr_DT[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT = np.zeros_like(all_fpr_DT)
            for i in range(n_classes):
              mean_tpr_DT += np.interp(all_fpr_DT, fpr_DT[i], tpr_DT[i])

            # Finally average it and compute AUC
            mean_tpr_DT /= n_classes

            fpr_DT["macro"] = all_fpr_DT
            tpr_DT["macro"] = mean_tpr_DT
            roc_auc_DT["macro"] = auc(fpr_DT["macro"], tpr_DT["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
              fpr_DT["micro"],
              tpr_DT["micro"],
              label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT["micro"]),
              color="deeppink",
              linestyle=":",
              linewidth=4,
            )

            plt.plot(
              fpr_DT["macro"],
              tpr_DT["macro"],
              label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT["macro"]),
              color="navy",
              linestyle=":",
              linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
              plt.plot(
                  fpr_DT[i],
                  tpr_DT[i],
                  color=color,
                  lw=lw,
                  label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc_DT[i]),
              )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            #plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}DT_ROC_1_".format(n) + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")
#
            # First aggregate all false positive rates
            all_fpr_NB = np.unique(np.concatenate([fpr_NB[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB = np.zeros_like(all_fpr_NB)
            for i in range(n_classes):
              mean_tpr_NB += np.interp(all_fpr_NB, fpr_NB[i], tpr_NB[i])

            # Finally average it and compute AUC
            mean_tpr_NB /= n_classes

            fpr_NB["macro"] = all_fpr_NB
            tpr_NB["macro"] = mean_tpr_NB
            roc_auc_NB["macro"] = auc(fpr_NB["macro"], tpr_NB["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
              fpr_NB["micro"],
              tpr_NB["micro"],
              label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB["micro"]),
              color="deeppink",
              linestyle=":",
              linewidth=4,
            )

            plt.plot(
              fpr_NB["macro"],
              tpr_NB["macro"],
              label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB["macro"]),
              color="navy",
              linestyle=":",
              linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
              plt.plot(
                  fpr_NB[i],
                  tpr_NB[i],
                  color=color,
                  lw=lw,
                  label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc_NB[i]),
              )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            #plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}NB_ROC_1_".format(n) + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")
#
            # First aggregate all false positive rates
            all_fpr_kNN = np.unique(np.concatenate([fpr_kNN[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN = np.zeros_like(all_fpr_kNN)
            for i in range(n_classes):
              mean_tpr_kNN += np.interp(all_fpr_kNN, fpr_kNN[i], tpr_kNN[i])

            # Finally average it and compute AUC
            mean_tpr_kNN /= n_classes

            fpr_kNN["macro"] = all_fpr_kNN
            tpr_kNN["macro"] = mean_tpr_kNN
            roc_auc_kNN["macro"] = auc(fpr_kNN["macro"], tpr_kNN["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
              fpr_kNN["micro"],
              tpr_kNN["micro"],
              label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN["micro"]),
              color="deeppink",
              linestyle=":",
              linewidth=4,
            )

            plt.plot(
              fpr_kNN["macro"],
              tpr_kNN["macro"],
              label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN["macro"]),
              color="navy",
              linestyle=":",
              linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
              plt.plot(
                  fpr_kNN[i],
                  tpr_kNN[i],
                  color=color,
                  lw=lw,
                  label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc_kNN[i]),
              )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            #plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}kNN_ROC_1_".format(n) + "%d"%a + "_" + "%d"%b + ".pdf".format(n),dpi = 600,bbox_inches = "tight")
#
            # First aggregate all false positive rates
            all_fpr_SVM = np.unique(np.concatenate([fpr_SVM[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM = np.zeros_like(all_fpr_SVM)
            for i in range(n_classes):
              mean_tpr_SVM += np.interp(all_fpr_SVM, fpr_SVM[i], tpr_SVM[i])

            # Finally average it and compute AUC
            mean_tpr_SVM /= n_classes

            fpr_SVM["macro"] = all_fpr_SVM
            tpr_SVM["macro"] = mean_tpr_SVM
            roc_auc_SVM["macro"] = auc(fpr_SVM["macro"], tpr_SVM["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
              fpr_SVM["micro"],
              tpr_SVM["micro"],
              label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM["micro"]),
              color="deeppink",
              linestyle=":",
              linewidth=4,
            )

            plt.plot(
              fpr_SVM["macro"],
              tpr_SVM["macro"],
              label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM["macro"]),
              color="navy",
              linestyle=":",
              linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
              plt.plot(
                  fpr_SVM[i],
                  tpr_SVM[i],
                  color=color,
                  lw=lw,
                  label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc_SVM[i]),
              )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            #plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}SVM_1_".format(n) + "%d"%a + "_" + "%d"%b + ".pdf".format(n),dpi = 600,bbox_inches = "tight")

            # First aggregate all false positive rates
            all_fpr_RF = np.unique(np.concatenate([fpr_RF[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF = np.zeros_like(all_fpr_RF)
            for i in range(n_classes):
              mean_tpr_RF += np.interp(all_fpr_RF, fpr_RF[i], tpr_RF[i])

            # Finally average it and compute AUC
            mean_tpr_RF /= n_classes

            fpr_RF["macro"] = all_fpr_RF
            tpr_RF["macro"] = mean_tpr_RF
            roc_auc_RF["macro"] = auc(fpr_RF["macro"], tpr_RF["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
              fpr_RF["micro"],
              tpr_RF["micro"],
              label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF["micro"]),
              color="deeppink",
              linestyle=":",
              linewidth=4,
            )

            plt.plot(
              fpr_RF["macro"],
              tpr_RF["macro"],
              label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF["macro"]),
              color="navy",
              linestyle=":",
              linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
              plt.plot(
                  fpr_RF[i],
                  tpr_RF[i],
                  color=color,
                  lw=lw,
                  label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc_RF[i]),
              )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            #plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}RF_ROC_1_".format(n) + "%d"%a + "_" + "%d"%b + ".pdf".format(n),dpi = 600,bbox_inches = "tight")
#
            # First aggregate all false positive rates
            all_fpr_ANN = np.unique(np.concatenate([fpr_ANN[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_ANN = np.zeros_like(all_fpr_ANN)
            for i in range(n_classes):
              mean_tpr_ANN += np.interp(all_fpr_ANN, fpr_ANN[i], tpr_ANN[i])

            # Finally average it and compute AUC
            mean_tpr_ANN /= n_classes

            fpr_ANN["macro"] = all_fpr_ANN
            tpr_ANN["macro"] = mean_tpr_ANN
            roc_auc_ANN["macro"] = auc(fpr_ANN["macro"], tpr_ANN["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
              fpr_ANN["micro"],
              tpr_ANN["micro"],
              label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN["micro"]),
              color="deeppink",
              linestyle=":",
              linewidth=4,
            )

            plt.plot(
              fpr_ANN["macro"],
              tpr_ANN["macro"],
              label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN["macro"]),
              color="navy",
              linestyle=":",
              linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
              plt.plot(
                  fpr_ANN[i],
                  tpr_ANN[i],
                  color=color,
                  lw=lw,
                  label="ROC curve of class {0} (area = {1:0.2f})".format(i+1, roc_auc_ANN[i]),
              )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            #plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}ANN_ROC_1_".format(n) + "%d"%a + "_" + "%d"%b + ".pdf".format(n),dpi = 600,bbox_inches = "tight")

            # First aggregate all false positive rates
            all_fpr_DT_E_v = np.unique(np.concatenate([fpr_DT_E_v[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT_E_v = np.zeros_like(all_fpr_DT_E_v)
            for i in range(n_classes):
                mean_tpr_DT_E_v += np.interp(all_fpr_DT_E_v, fpr_DT_E_v[i], tpr_DT_E_v[i])

            # Finally average it and compute AUC
            mean_tpr_DT_E_v /= n_classes

            fpr_DT_E_v["macro"] = all_fpr_DT_E_v
            tpr_DT_E_v["macro"] = mean_tpr_DT_E_v
            roc_auc_DT_E_v["macro"] = auc(fpr_DT_E_v["macro"], tpr_DT_E_v["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_DT_E_v["micro"],
                tpr_DT_E_v["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_E_v["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_DT_E_v["macro"],
                tpr_DT_E_v["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_E_v["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_DT_E_v[i],
                    tpr_DT_E_v[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_DT_E_v[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_DT_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")



            # First aggregate all false positive rates
            all_fpr_NB_E_v = np.unique(np.concatenate([fpr_NB_E_v[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB_E_v = np.zeros_like(all_fpr_NB_E_v)
            for i in range(n_classes):
                mean_tpr_NB_E_v += np.interp(all_fpr_NB_E_v, fpr_NB_E_v[i], tpr_NB_E_v[i])

            # Finally average it and compute AUC
            mean_tpr_NB_E_v /= n_classes

            fpr_NB_E_v["macro"] = all_fpr_NB_E_v
            tpr_NB_E_v["macro"] = mean_tpr_NB_E_v
            roc_auc_NB_E_v["macro"] = auc(fpr_NB_E_v["macro"], tpr_NB_E_v["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_NB_E_v["micro"],
                tpr_NB_E_v["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_E_v["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_NB_E_v["macro"],
                tpr_NB_E_v["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_E_v["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_NB_E_v[i],
                    tpr_NB_E_v[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_NB_E_v[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_NB_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_kNN_E_v = np.unique(np.concatenate([fpr_kNN_E_v[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN_E_v = np.zeros_like(all_fpr_kNN_E_v)
            for i in range(n_classes):
                mean_tpr_kNN_E_v += np.interp(all_fpr_kNN_E_v, fpr_kNN_E_v[i], tpr_kNN_E_v[i])

            # Finally average it and compute AUC
            mean_tpr_kNN_E_v /= n_classes

            fpr_kNN_E_v["macro"] = all_fpr_kNN_E_v
            tpr_kNN_E_v["macro"] = mean_tpr_kNN_E_v
            roc_auc_kNN_E_v["macro"] = auc(fpr_kNN_E_v["macro"], tpr_kNN_E_v["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_kNN_E_v["micro"],
                tpr_kNN_E_v["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_E_v["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_kNN_E_v["macro"],
                tpr_kNN_E_v["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_E_v["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_kNN_E_v[i],
                    tpr_kNN_E_v[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_kNN_E_v[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_kNN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_SVM_E_v = np.unique(np.concatenate([fpr_SVM_E_v[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM_E_v = np.zeros_like(all_fpr_SVM_E_v)
            for i in range(n_classes):
                mean_tpr_SVM_E_v += np.interp(all_fpr_SVM_E_v, fpr_SVM_E_v[i], tpr_SVM_E_v[i])

            # Finally average it and compute AUC
            mean_tpr_SVM_E_v /= n_classes

            fpr_SVM_E_v["macro"] = all_fpr_SVM_E_v
            tpr_SVM_E_v["macro"] = mean_tpr_SVM_E_v
            roc_auc_SVM_E_v["macro"] = auc(fpr_SVM_E_v["macro"], tpr_SVM_E_v["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_SVM_E_v["micro"],
                tpr_SVM_E_v["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_E_v["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_SVM_E_v["macro"],
                tpr_SVM_E_v["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_E_v["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_SVM_E_v[i],
                    tpr_SVM_E_v[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_SVM_E_v[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_SVM_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_RF_E_v = np.unique(np.concatenate([fpr_RF_E_v[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF_E_v = np.zeros_like(all_fpr_RF_E_v)
            for i in range(n_classes):
                mean_tpr_RF_E_v += np.interp(all_fpr_RF_E_v, fpr_RF_E_v[i], tpr_RF_E_v[i])

            # Finally average it and compute AUC
            mean_tpr_RF_E_v /= n_classes

            fpr_RF_E_v["macro"] = all_fpr_RF_E_v
            tpr_RF_E_v["macro"] = mean_tpr_RF_E_v
            roc_auc_RF_E_v["macro"] = auc(fpr_RF_E_v["macro"], tpr_RF_E_v["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_RF_E_v["micro"],
                tpr_RF_E_v["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_E_v["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_RF_E_v["macro"],
                tpr_RF_E_v["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_E_v["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_RF_E_v[i],
                    tpr_RF_E_v[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_RF_E_v[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_RF_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_ANN_E_v = np.unique(np.concatenate([fpr_ANN_E_v[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_ANN_E_v = np.zeros_like(all_fpr_ANN_E_v)
            for i in range(n_classes):
                mean_tpr_ANN_E_v += np.interp(all_fpr_ANN_E_v, fpr_ANN_E_v[i], tpr_ANN_E_v[i])

            # Finally average it and compute AUC
            mean_tpr_ANN_E_v /= n_classes

            fpr_ANN_E_v["macro"] = all_fpr_ANN_E_v
            tpr_ANN_E_v["macro"] = mean_tpr_ANN_E_v
            roc_auc_ANN_E_v["macro"] = auc(fpr_ANN_E_v["macro"], tpr_ANN_E_v["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_ANN_E_v["micro"],
                tpr_ANN_E_v["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_E_v["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_ANN_E_v["macro"],
                tpr_ANN_E_v["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_E_v["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_ANN_E_v[i],
                    tpr_ANN_E_v[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_ANN_E_v[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_ANN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_DT_E_v_F = np.unique(np.concatenate([fpr_DT_E_v_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT_E_v_F = np.zeros_like(all_fpr_DT_E_v_F)
            for i in range(n_classes):
                mean_tpr_DT_E_v_F += np.interp(all_fpr_DT_E_v_F, fpr_DT_E_v_F[i], tpr_DT_E_v_F[i])

            # Finally average it and compute AUC
            mean_tpr_DT_E_v_F /= n_classes

            fpr_DT_E_v_F["macro"] = all_fpr_DT_E_v_F
            tpr_DT_E_v_F["macro"] = mean_tpr_DT_E_v_F
            roc_auc_DT_E_v_F["macro"] = auc(fpr_DT_E_v_F["macro"], tpr_DT_E_v_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_DT_E_v_F["micro"],
                tpr_DT_E_v_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_E_v_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_DT_E_v_F["macro"],
                tpr_DT_E_v_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_E_v_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_DT_E_v_F[i],
                    tpr_DT_E_v_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_DT_E_v_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_F_DT_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_NB_E_v_F = np.unique(np.concatenate([fpr_NB_E_v_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB_E_v_F = np.zeros_like(all_fpr_NB_E_v_F)
            for i in range(n_classes):
                mean_tpr_NB_E_v_F += np.interp(all_fpr_NB_E_v_F, fpr_NB_E_v_F[i], tpr_NB_E_v_F[i])

            # Finally average it and compute AUC
            mean_tpr_NB_E_v_F /= n_classes

            fpr_NB_E_v_F["macro"] = all_fpr_NB_E_v_F
            tpr_NB_E_v_F["macro"] = mean_tpr_NB_E_v_F
            roc_auc_NB_E_v_F["macro"] = auc(fpr_NB_E_v_F["macro"], tpr_NB_E_v_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_NB_E_v_F["micro"],
                tpr_NB_E_v_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_E_v_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_NB_E_v_F["macro"],
                tpr_NB_E_v_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_E_v_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_NB_E_v_F[i],
                    tpr_NB_E_v_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_NB_E_v_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_F_NB_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_kNN_E_v_F = np.unique(np.concatenate([fpr_kNN_E_v_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN_E_v_F = np.zeros_like(all_fpr_kNN_E_v_F)
            for i in range(n_classes):
                mean_tpr_kNN_E_v_F += np.interp(all_fpr_kNN_E_v_F, fpr_kNN_E_v_F[i], tpr_kNN_E_v_F[i])

            # Finally average it and compute AUC
            mean_tpr_kNN_E_v_F /= n_classes

            fpr_kNN_E_v_F["macro"] = all_fpr_kNN_E_v_F
            tpr_kNN_E_v_F["macro"] = mean_tpr_kNN_E_v_F
            roc_auc_kNN_E_v_F["macro"] = auc(fpr_kNN_E_v_F["macro"], tpr_kNN_E_v_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_kNN_E_v_F["micro"],
                tpr_kNN_E_v_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_E_v_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_kNN_E_v_F["macro"],
                tpr_kNN_E_v_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_E_v_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_kNN_E_v_F[i],
                    tpr_kNN_E_v_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_kNN_E_v_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_F_kNN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_SVM_E_v_F = np.unique(np.concatenate([fpr_SVM_E_v_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM_E_v_F = np.zeros_like(all_fpr_SVM_E_v_F)
            for i in range(n_classes):
                mean_tpr_SVM_E_v_F += np.interp(all_fpr_SVM_E_v_F, fpr_SVM_E_v_F[i], tpr_SVM_E_v_F[i])

            # Finally average it and compute AUC
            mean_tpr_SVM_E_v_F /= n_classes

            fpr_SVM_E_v_F["macro"] = all_fpr_SVM_E_v_F
            tpr_SVM_E_v_F["macro"] = mean_tpr_SVM_E_v_F
            roc_auc_SVM_E_v_F["macro"] = auc(fpr_SVM_E_v_F["macro"], tpr_SVM_E_v_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_SVM_E_v_F["micro"],
                tpr_SVM_E_v_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_E_v_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_SVM_E_v_F["macro"],
                tpr_SVM_E_v_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_E_v_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_SVM_E_v_F[i],
                    tpr_SVM_E_v_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_SVM_E_v_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_F_SVM_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_RF_E_v_F = np.unique(np.concatenate([fpr_RF_E_v_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF_E_v_F = np.zeros_like(all_fpr_RF_E_v_F)
            for i in range(n_classes):
                mean_tpr_RF_E_v_F += np.interp(all_fpr_RF_E_v_F, fpr_RF_E_v_F[i], tpr_RF_E_v_F[i])

            # Finally average it and compute AUC
            mean_tpr_RF_E_v_F /= n_classes

            fpr_RF_E_v_F["macro"] = all_fpr_RF_E_v_F
            tpr_RF_E_v_F["macro"] = mean_tpr_RF_E_v_F
            roc_auc_RF_E_v_F["macro"] = auc(fpr_RF_E_v_F["macro"], tpr_RF_E_v_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_RF_E_v_F["micro"],
                tpr_RF_E_v_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_E_v_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_RF_E_v_F["macro"],
                tpr_RF_E_v_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_E_v_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_RF_E_v_F[i],
                    tpr_RF_E_v_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_RF_E_v_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_F_RF_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_ANN_E_v_F = np.unique(np.concatenate([fpr_ANN_E_v_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_ANN_E_v_F = np.zeros_like(all_fpr_ANN_E_v_F)
            for i in range(n_classes):
                mean_tpr_ANN_E_v_F += np.interp(all_fpr_ANN_E_v_F, fpr_ANN_E_v_F[i], tpr_ANN_E_v_F[i])

            # Finally average it and compute AUC
            mean_tpr_ANN_E_v_F /= n_classes

            fpr_ANN_E_v_F["macro"] = all_fpr_ANN_E_v_F
            tpr_ANN_E_v_F["macro"] = mean_tpr_ANN_E_v_F
            roc_auc_ANN_E_v_F["macro"] = auc(fpr_ANN_E_v_F["macro"], tpr_ANN_E_v_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_ANN_E_v_F["micro"],
                tpr_ANN_E_v_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_E_v_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_ANN_E_v_F["macro"],
                tpr_ANN_E_v_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_E_v_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_ANN_E_v_F[i],
                    tpr_ANN_E_v_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_ANN_E_v_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}E_v_F_ANN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_DT_train_F = np.unique(np.concatenate([fpr_DT_train_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT_train_F = np.zeros_like(all_fpr_DT_train_F)
            for i in range(n_classes):
                mean_tpr_DT_train_F += np.interp(all_fpr_DT_train_F, fpr_DT_train_F[i], tpr_DT_train_F[i])

            # Finally average it and compute AUC
            mean_tpr_DT_train_F /= n_classes

            fpr_DT_train_F["macro"] = all_fpr_DT_train_F
            tpr_DT_train_F["macro"] = mean_tpr_DT_train_F
            roc_auc_DT_train_F["macro"] = auc(fpr_DT_train_F["macro"], tpr_DT_train_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_DT_train_F["micro"],
                tpr_DT_train_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_train_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_DT_train_F["macro"],
                tpr_DT_train_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_train_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_DT_train_F[i],
                    tpr_DT_train_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_DT_train_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_F_DT_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_NB_train_F = np.unique(np.concatenate([fpr_NB_train_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB_train_F = np.zeros_like(all_fpr_NB_train_F)
            for i in range(n_classes):
                mean_tpr_NB_train_F += np.interp(all_fpr_NB_train_F, fpr_NB_train_F[i], tpr_NB_train_F[i])

            # Finally average it and compute AUC
            mean_tpr_NB_train_F /= n_classes

            fpr_NB_train_F["macro"] = all_fpr_NB_train_F
            tpr_NB_train_F["macro"] = mean_tpr_NB_train_F
            roc_auc_NB_train_F["macro"] = auc(fpr_NB_train_F["macro"], tpr_NB_train_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_NB_train_F["micro"],
                tpr_NB_train_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_train_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_NB_train_F["macro"],
                tpr_NB_train_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_train_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_NB_train_F[i],
                    tpr_NB_train_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_NB_train_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_F_NB_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_kNN_train_F = np.unique(np.concatenate([fpr_kNN_train_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN_train_F = np.zeros_like(all_fpr_kNN_train_F)
            for i in range(n_classes):
                mean_tpr_kNN_train_F += np.interp(all_fpr_kNN_train_F, fpr_kNN_train_F[i], tpr_kNN_train_F[i])

            # Finally average it and compute AUC
            mean_tpr_kNN_train_F /= n_classes

            fpr_kNN_train_F["macro"] = all_fpr_kNN_train_F
            tpr_kNN_train_F["macro"] = mean_tpr_kNN_train_F
            roc_auc_kNN_train_F["macro"] = auc(fpr_kNN_train_F["macro"], tpr_kNN_train_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_kNN_train_F["micro"],
                tpr_kNN_train_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_train_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_kNN_train_F["macro"],
                tpr_kNN_train_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_train_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_kNN_train_F[i],
                    tpr_kNN_train_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_kNN_train_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_F_kNN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_SVM_train_F = np.unique(np.concatenate([fpr_SVM_train_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM_train_F = np.zeros_like(all_fpr_SVM_train_F)
            for i in range(n_classes):
                mean_tpr_SVM_train_F += np.interp(all_fpr_SVM_train_F, fpr_SVM_train_F[i], tpr_SVM_train_F[i])

            # Finally average it and compute AUC
            mean_tpr_SVM_train_F /= n_classes

            fpr_SVM_train_F["macro"] = all_fpr_SVM_train_F
            tpr_SVM_train_F["macro"] = mean_tpr_SVM_train_F
            roc_auc_SVM_train_F["macro"] = auc(fpr_SVM_train_F["macro"], tpr_SVM_train_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_SVM_train_F["micro"],
                tpr_SVM_train_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_train_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_SVM_train_F["macro"],
                tpr_SVM_train_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_train_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_SVM_train_F[i],
                    tpr_SVM_train_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_SVM_train_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_F_SVM_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_RF_train_F = np.unique(np.concatenate([fpr_RF_train_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF_train_F = np.zeros_like(all_fpr_RF_train_F)
            for i in range(n_classes):
                mean_tpr_RF_train_F += np.interp(all_fpr_RF_train_F, fpr_RF_train_F[i], tpr_RF_train_F[i])

            # Finally average it and compute AUC
            mean_tpr_RF_train_F /= n_classes

            fpr_RF_train_F["macro"] = all_fpr_RF_train_F
            tpr_RF_train_F["macro"] = mean_tpr_RF_train_F
            roc_auc_RF_train_F["macro"] = auc(fpr_RF_train_F["macro"], tpr_RF_train_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_RF_train_F["micro"],
                tpr_RF_train_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_train_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_RF_train_F["macro"],
                tpr_RF_train_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_train_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_RF_train_F[i],
                    tpr_RF_train_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_RF_train_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_F_RF_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_ANN_train_F = np.unique(np.concatenate([fpr_ANN_train_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_ANN_train_F = np.zeros_like(all_fpr_ANN_train_F)
            for i in range(n_classes):
                mean_tpr_ANN_train_F += np.interp(all_fpr_ANN_train_F, fpr_ANN_train_F[i], tpr_ANN_train_F[i])

            # Finally average it and compute AUC
            mean_tpr_ANN_train_F /= n_classes

            fpr_ANN_train_F["macro"] = all_fpr_ANN_train_F
            tpr_ANN_train_F["macro"] = mean_tpr_ANN_train_F
            roc_auc_ANN_train_F["macro"] = auc(fpr_ANN_train_F["macro"], tpr_ANN_train_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_ANN_train_F["micro"],
                tpr_ANN_train_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_train_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_ANN_train_F["macro"],
                tpr_ANN_train_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_train_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_ANN_train_F[i],
                    tpr_ANN_train_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_ANN_train_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_F_ANN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")


            # First aggregate all false positive rates
            all_fpr_DT_train = np.unique(np.concatenate([fpr_DT_train[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT_train = np.zeros_like(all_fpr_DT_train)
            for i in range(n_classes):
                mean_tpr_DT_train += np.interp(all_fpr_DT_train, fpr_DT_train[i], tpr_DT_train[i])

            # Finally average it and compute AUC
            mean_tpr_DT_train /= n_classes

            fpr_DT_train["macro"] = all_fpr_DT_train
            tpr_DT_train["macro"] = mean_tpr_DT_train
            roc_auc_DT_train["macro"] = auc(fpr_DT_train["macro"], tpr_DT_train["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_DT_train["micro"],
                tpr_DT_train["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_train["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_DT_train["macro"],
                tpr_DT_train["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_train["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_DT_train[i],
                    tpr_DT_train[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_DT_train[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_DT_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_NB_train = np.unique(np.concatenate([fpr_DT_train[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB_train = np.zeros_like(all_fpr_NB_train)
            for i in range(n_classes):
                mean_tpr_NB_train += np.interp(all_fpr_NB_train, fpr_NB_train[i], tpr_NB_train[i])

            # Finally average it and compute AUC
            mean_tpr_NB_train /= n_classes

            fpr_NB_train["macro"] = all_fpr_NB_train
            tpr_NB_train["macro"] = mean_tpr_NB_train
            roc_auc_NB_train["macro"] = auc(fpr_NB_train["macro"], tpr_NB_train["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_NB_train["micro"],
                tpr_NB_train["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_train["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_NB_train["macro"],
                tpr_NB_train["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_train["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_NB_train[i],
                    tpr_NB_train[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_NB_train[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_NB_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_kNN_train = np.unique(np.concatenate([fpr_kNN_train[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN_train = np.zeros_like(all_fpr_kNN_train)
            for i in range(n_classes):
                mean_tpr_kNN_train += np.interp(all_fpr_kNN_train, fpr_kNN_train[i], tpr_kNN_train[i])

            # Finally average it and compute AUC
            mean_tpr_kNN_train /= n_classes

            fpr_kNN_train["macro"] = all_fpr_kNN_train
            tpr_kNN_train["macro"] = mean_tpr_kNN_train
            roc_auc_kNN_train["macro"] = auc(fpr_kNN_train["macro"], tpr_kNN_train["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_kNN_train["micro"],
                tpr_kNN_train["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_train["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_kNN_train["macro"],
                tpr_kNN_train["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_train["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_kNN_train[i],
                    tpr_kNN_train[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_kNN_train[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_kNN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_SVM_train = np.unique(np.concatenate([fpr_SVM_train[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM_train = np.zeros_like(all_fpr_SVM_train)
            for i in range(n_classes):
                mean_tpr_SVM_train += np.interp(all_fpr_SVM_train, fpr_SVM_train[i], tpr_SVM_train[i])

            # Finally average it and compute AUC
            mean_tpr_SVM_train /= n_classes

            fpr_SVM_train["macro"] = all_fpr_SVM_train
            tpr_SVM_train["macro"] = mean_tpr_SVM_train
            roc_auc_SVM_train["macro"] = auc(fpr_SVM_train["macro"], tpr_SVM_train["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_SVM_train["micro"],
                tpr_SVM_train["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_train["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_SVM_train["macro"],
                tpr_SVM_train["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_train["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_SVM_train[i],
                    tpr_SVM_train[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_SVM_train[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_SVM_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            # First aggregate all false positive rates
            all_fpr_RF_train = np.unique(np.concatenate([fpr_RF_train[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF_train = np.zeros_like(all_fpr_RF_train)
            for i in range(n_classes):
                mean_tpr_RF_train += np.interp(all_fpr_RF_train, fpr_RF_train[i], tpr_RF_train[i])

            # Finally average it and compute AUC
            mean_tpr_RF_train /= n_classes

            fpr_RF_train["macro"] = all_fpr_RF_train
            tpr_RF_train["macro"] = mean_tpr_RF_train
            roc_auc_RF_train["macro"] = auc(fpr_RF_train["macro"], tpr_RF_train["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_RF_train["micro"],
                tpr_RF_train["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_train["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_RF_train["macro"],
                tpr_RF_train["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_train["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_RF_train[i],
                    tpr_RF_train[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_RF_train[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_RF_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")


            # First aggregate all false positive rates
            all_fpr_ANN_train = np.unique(np.concatenate([fpr_ANN_train[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_ANN_train = np.zeros_like(all_fpr_ANN_train)
            for i in range(n_classes):
                mean_tpr_ANN_train += np.interp(all_fpr_ANN_train, fpr_ANN_train[i], tpr_ANN_train[i])

            # Finally average it and compute AUC
            mean_tpr_ANN_train /= n_classes

            fpr_ANN_train["macro"] = all_fpr_ANN_train
            tpr_ANN_train["macro"] = mean_tpr_ANN_train
            roc_auc_ANN_train["macro"] = auc(fpr_ANN_train["macro"], tpr_ANN_train["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_ANN_train["micro"],
                tpr_ANN_train["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_train["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_ANN_train["macro"],
                tpr_ANN_train["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_train["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_ANN_train[i],
                    tpr_ANN_train[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_ANN_train[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}train_ANN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")


            all_fpr_DT_test = np.unique(np.concatenate([fpr_DT_test[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT_test = np.zeros_like(all_fpr_DT_test)
            for i in range(n_classes):
                mean_tpr_DT_test += np.interp(all_fpr_DT_test, fpr_DT_test[i], tpr_DT_test[i])

            # Finally average it and compute AUC
            mean_tpr_DT_test /= n_classes

            fpr_DT_test["macro"] = all_fpr_DT_test
            tpr_DT_test["macro"] = mean_tpr_DT_test
            roc_auc_DT_test["macro"] = auc(fpr_DT_test["macro"], tpr_DT_test["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_DT_test["micro"],
                tpr_DT_test["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_test["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_DT_test["macro"],
                tpr_DT_test["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_test["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_DT_test[i],
                    tpr_DT_test[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_DT_test[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_DT_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),dpi=600, bbox_inches="tight")

            all_fpr_NB_test = np.unique(np.concatenate([fpr_NB_test[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB_test = np.zeros_like(all_fpr_NB_test)
            for i in range(n_classes):
                mean_tpr_NB_test += np.interp(all_fpr_NB_test, fpr_NB_test[i], tpr_NB_test[i])

            # Finally average it and compute AUC
            mean_tpr_NB_test /= n_classes

            fpr_NB_test["macro"] = all_fpr_NB_test
            tpr_NB_test["macro"] = mean_tpr_NB_test
            roc_auc_NB_test["macro"] = auc(fpr_NB_test["macro"], tpr_NB_test["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_NB_test["micro"],
                tpr_NB_test["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_test["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_NB_test["macro"],
                tpr_NB_test["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_test["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_NB_test[i],
                    tpr_NB_test[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_NB_test[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_NB_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            all_fpr_kNN_test = np.unique(np.concatenate([fpr_kNN_test[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN_test = np.zeros_like(all_fpr_kNN_test)
            for i in range(n_classes):
                mean_tpr_kNN_test += np.interp(all_fpr_kNN_test, fpr_kNN_test[i], tpr_kNN_test[i])

            # Finally average it and compute AUC
            mean_tpr_kNN_test /= n_classes

            fpr_kNN_test["macro"] = all_fpr_kNN_test
            tpr_kNN_test["macro"] = mean_tpr_kNN_test
            roc_auc_kNN_test["macro"] = auc(fpr_kNN_test["macro"], tpr_kNN_test["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_kNN_test["micro"],
                tpr_kNN_test["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_test["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_kNN_test["macro"],
                tpr_kNN_test["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_test["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_kNN_test[i],
                    tpr_kNN_test[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_kNN_test[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_kNN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            all_fpr_SVM_test = np.unique(np.concatenate([fpr_SVM_test[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM_test = np.zeros_like(all_fpr_SVM_test)
            for i in range(n_classes):
                mean_tpr_SVM_test += np.interp(all_fpr_SVM_test, fpr_SVM_test[i], tpr_SVM_test[i])

            # Finally average it and compute AUC
            mean_tpr_SVM_test /= n_classes

            fpr_SVM_test["macro"] = all_fpr_SVM_test
            tpr_SVM_test["macro"] = mean_tpr_SVM_test
            roc_auc_SVM_test["macro"] = auc(fpr_SVM_test["macro"], tpr_SVM_test["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_SVM_test["micro"],
                tpr_SVM_test["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_test["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_SVM_test["macro"],
                tpr_SVM_test["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_test["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_SVM_test[i],
                    tpr_SVM_test[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_SVM_test[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_SVM_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n), dpi=600, bbox_inches="tight")

            all_fpr_RF_test = np.unique(np.concatenate([fpr_RF_test[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF_test = np.zeros_like(all_fpr_RF_test)
            for i in range(n_classes):
                mean_tpr_RF_test += np.interp(all_fpr_RF_test, fpr_RF_test[i], tpr_RF_test[i])

            # Finally average it and compute AUC
            mean_tpr_RF_test /= n_classes

            fpr_RF_test["macro"] = all_fpr_RF_test
            tpr_RF_test["macro"] = mean_tpr_RF_test
            roc_auc_RF_test["macro"] = auc(fpr_RF_test["macro"], tpr_RF_test["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_RF_test["micro"],
                tpr_RF_test["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_test["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_RF_test["macro"],
                tpr_RF_test["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_test["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_RF_test[i],
                    tpr_RF_test[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_RF_test[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_RF_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            all_fpr_ANN_test = np.unique(np.concatenate([fpr_ANN_test[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_ANN_test = np.zeros_like(all_fpr_ANN_test)
            for i in range(n_classes):
                mean_tpr_ANN_test += np.interp(all_fpr_ANN_test, fpr_ANN_test[i], tpr_ANN_test[i])

            # Finally average it and compute AUC
            mean_tpr_ANN_test /= n_classes

            fpr_ANN_test["macro"] = all_fpr_ANN_test
            tpr_ANN_test["macro"] = mean_tpr_ANN_test
            roc_auc_ANN_test["macro"] = auc(fpr_ANN_test["macro"], tpr_ANN_test["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_ANN_test["micro"],
                tpr_ANN_test["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_test["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_ANN_test["macro"],
                tpr_ANN_test["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_ANN_test["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_ANN_test[i],
                    tpr_ANN_test[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_ANN_test[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_ANN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")



            all_fpr_DT_test_F = np.unique(np.concatenate([fpr_DT_test_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_DT_test_F = np.zeros_like(all_fpr_DT_test_F)
            for i in range(n_classes):
                mean_tpr_DT_test_F += np.interp(all_fpr_DT_test_F, fpr_DT_test_F[i], tpr_DT_test_F[i])

            # Finally average it and compute AUC
            mean_tpr_DT_test_F /= n_classes

            fpr_DT_test_F["macro"] = all_fpr_DT_test_F
            tpr_DT_test_F["macro"] = mean_tpr_DT_test_F
            roc_auc_DT_test_F["macro"] = auc(fpr_DT_test_F["macro"], tpr_DT_test_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_DT_test_F["micro"],
                tpr_DT_test_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_test_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_DT_test_F["macro"],
                tpr_DT_test_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_DT_test_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_DT_test_F[i],
                    tpr_DT_test_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_DT_test_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_F_DT_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),dpi=600, bbox_inches="tight")

            all_fpr_NB_test_F = np.unique(np.concatenate([fpr_NB_test_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_NB_test_F = np.zeros_like(all_fpr_NB_test_F)
            for i in range(n_classes):
                mean_tpr_NB_test_F += np.interp(all_fpr_NB_test_F, fpr_NB_test_F[i], tpr_NB_test_F[i])

            # Finally average it and compute AUC
            mean_tpr_NB_test_F /= n_classes

            fpr_NB_test_F["macro"] = all_fpr_NB_test_F
            tpr_NB_test_F["macro"] = mean_tpr_NB_test_F
            roc_auc_NB_test_F["macro"] = auc(fpr_NB_test_F["macro"], tpr_NB_test_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_NB_test_F["micro"],
                tpr_NB_test_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_test_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_NB_test_F["macro"],
                tpr_NB_test_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_NB_test_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_NB_test_F[i],
                    tpr_NB_test_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_NB_test_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_F_NB_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                dpi=600, bbox_inches="tight")

            all_fpr_kNN_test_F = np.unique(np.concatenate([fpr_kNN_test_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_kNN_test_F = np.zeros_like(all_fpr_kNN_test_F)
            for i in range(n_classes):
                mean_tpr_kNN_test_F += np.interp(all_fpr_kNN_test_F, fpr_kNN_test_F[i], tpr_kNN_test_F[i])

            # Finally average it and compute AUC
            mean_tpr_kNN_test_F /= n_classes

            fpr_kNN_test_F["macro"] = all_fpr_kNN_test_F
            tpr_kNN_test_F["macro"] = mean_tpr_kNN_test_F
            roc_auc_kNN_test_F["macro"] = auc(fpr_kNN_test_F["macro"], tpr_kNN_test_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_kNN_test_F["micro"],
                tpr_kNN_test_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_test_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_kNN_test_F["macro"],
                tpr_kNN_test_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_kNN_test_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_kNN_test_F[i],
                    tpr_kNN_test_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_kNN_test_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_F_kNN_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            all_fpr_SVM_test_F = np.unique(np.concatenate([fpr_SVM_test_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_SVM_test_F = np.zeros_like(all_fpr_SVM_test_F)
            for i in range(n_classes):
                mean_tpr_SVM_test_F += np.interp(all_fpr_SVM_test_F, fpr_SVM_test_F[i], tpr_SVM_test_F[i])

            # Finally average it and compute AUC
            mean_tpr_SVM_test_F /= n_classes

            fpr_SVM_test_F["macro"] = all_fpr_SVM_test_F
            tpr_SVM_test_F["macro"] = mean_tpr_SVM_test_F
            roc_auc_SVM_test_F["macro"] = auc(fpr_SVM_test_F["macro"], tpr_SVM_test_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_SVM_test_F["micro"],
                tpr_SVM_test_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_test_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_SVM_test_F["macro"],
                tpr_SVM_test_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_SVM_test_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_SVM_test_F[i],
                    tpr_SVM_test_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_SVM_test_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_F_SVM_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

            all_fpr_RF_test_F = np.unique(np.concatenate([fpr_RF_test_F[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr_RF_test_F = np.zeros_like(all_fpr_RF_test_F)
            for i in range(n_classes):
                mean_tpr_RF_test_F += np.interp(all_fpr_RF_test_F, fpr_RF_test_F[i], tpr_RF_test_F[i])

            # Finally average it and compute AUC
            mean_tpr_RF_test_F /= n_classes

            fpr_RF_test_F["macro"] = all_fpr_RF_test_F
            tpr_RF_test_F["macro"] = mean_tpr_RF_test_F
            roc_auc_RF_test_F["macro"] = auc(fpr_RF_test_F["macro"], tpr_RF_test_F["macro"])

            # Plot all ROC curves
            plt.figure()
            plt.plot(
                fpr_RF_test_F["micro"],
                tpr_RF_test_F["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_test_F["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr_RF_test_F["macro"],
                tpr_RF_test_F["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc_RF_test_F["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )
            lw = 2
            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr_RF_test_F[i],
                    tpr_RF_test_F[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i + 1, roc_auc_RF_test_F[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            # plt.title("Some extension of Receiver operating characteristic to multiclass")
            plt.legend(loc="lower right")
            # plt.show()
            plt.savefig("..//data/new/" + "{0}test_F_RF_ROC_1_".format(n) + "%d" % a + "_" + "%d" % b + ".pdf".format(n),
                        dpi=600, bbox_inches="tight")

        #训练集
        train_A_DT_100_mean = np.mean(train_A_DT_100)
        train_A_DT_100_std = np.std(train_A_DT_100)
        train_A_NB_100_mean = np.mean(train_A_NB_100)
        train_A_NB_100_std = np.std(train_A_NB_100)
        train_A_kNN_100_mean = np.mean(train_A_kNN_100)
        train_A_kNN_100_std = np.std(train_A_kNN_100)
        train_A_SVM_100_mean = np.mean(train_A_SVM_100)
        train_A_SVM_100_std = np.std(train_A_SVM_100)
        train_A_RF_100_mean = np.mean(train_A_RF_100)
        train_A_RF_100_std = np.std(train_A_RF_100)
        train_A_ANN_100_mean = np.mean(train_A_ANN_100)
        train_A_ANN_100_std = np.std(train_A_ANN_100)

        train_F_DT_100_mean = np.mean(train_F_DT_100)
        train_F_DT_100_std = np.std(train_F_DT_100)
        train_F_NB_100_mean = np.mean(train_F_NB_100)
        train_F_NB_100_std = np.std(train_F_NB_100)
        train_F_kNN_100_mean = np.mean(train_F_kNN_100)
        train_F_kNN_100_std = np.std(train_F_kNN_100)
        train_F_SVM_100_mean = np.mean(train_F_SVM_100)
        train_F_SVM_100_std = np.std(train_F_SVM_100)
        train_F_RF_100_mean = np.mean(train_F_RF_100)
        train_F_RF_100_std = np.std(train_F_RF_100)
        train_F_ANN_100_mean = np.mean(train_F_ANN_100)
        train_F_ANN_100_std = np.std(train_F_ANN_100)

        train_P_DT_100_mean = np.mean(train_P_DT_100)
        train_P_DT_100_std = np.std(train_P_DT_100)
        train_P_NB_100_mean = np.mean(train_P_NB_100)
        train_P_NB_100_std = np.std(train_P_NB_100)
        train_P_kNN_100_mean = np.mean(train_P_kNN_100)
        train_P_kNN_100_std = np.std(train_P_kNN_100)
        train_P_SVM_100_mean = np.mean(train_P_SVM_100)
        train_P_SVM_100_std = np.std(train_P_SVM_100)
        train_P_RF_100_mean = np.mean(train_P_RF_100)
        train_P_RF_100_std = np.std(train_P_RF_100)
        train_P_ANN_100_mean = np.mean(train_P_ANN_100)
        train_P_ANN_100_std = np.std(train_P_ANN_100)

        train_R_DT_100_mean = np.mean(train_R_DT_100)
        train_R_DT_100_std = np.std(train_R_DT_100)
        train_R_NB_100_mean = np.mean(train_R_NB_100)
        train_R_NB_100_std = np.std(train_R_NB_100)
        train_R_kNN_100_mean = np.mean(train_R_kNN_100)
        train_R_kNN_100_std = np.std(train_R_kNN_100)
        train_R_SVM_100_mean = np.mean(train_R_SVM_100)
        train_R_SVM_100_std = np.std(train_R_SVM_100)
        train_R_RF_100_mean = np.mean(train_R_RF_100)
        train_R_RF_100_std = np.std(train_R_RF_100)
        train_R_ANN_100_mean = np.mean(train_R_ANN_100)
        train_R_ANN_100_std = np.std(train_R_ANN_100)

        train_ROC_DT_100_mean = np.mean(train_ROC_DT_100)
        train_ROC_DT_100_std = np.std(train_ROC_DT_100)
        train_ROC_NB_100_mean = np.mean(train_ROC_NB_100)
        train_ROC_NB_100_std = np.std(train_ROC_NB_100)
        train_ROC_kNN_100_mean = np.mean(train_ROC_kNN_100)
        train_ROC_kNN_100_std = np.std(train_ROC_kNN_100)
        train_ROC_SVM_100_mean = np.mean(train_ROC_SVM_100)
        train_ROC_SVM_100_std = np.std(train_ROC_SVM_100)
        train_ROC_RF_100_mean = np.mean(train_ROC_RF_100)
        train_ROC_RF_100_std = np.std(train_ROC_RF_100)
        train_ROC_ANN_100_mean = np.mean(train_ROC_ANN_100)
        train_ROC_ANN_100_std = np.std(train_ROC_ANN_100)

        train_F_A_DT_100_mean = np.mean(train_F_A_DT_100)
        train_F_A_DT_100_std = np.std(train_F_A_DT_100)
        train_F_A_NB_100_mean = np.mean(train_F_A_NB_100)
        train_F_A_NB_100_std = np.std(train_F_A_NB_100)
        train_F_A_kNN_100_mean = np.mean(train_F_A_kNN_100)
        train_F_A_kNN_100_std = np.std(train_F_A_kNN_100)
        train_F_A_SVM_100_mean = np.mean(train_F_A_SVM_100)
        train_F_A_SVM_100_std = np.std(train_F_A_SVM_100)
        train_F_A_RF_100_mean = np.mean(train_F_A_RF_100)
        train_F_A_RF_100_std = np.std(train_F_A_RF_100)
        train_F_A_ANN_100_mean = np.mean(train_F_A_ANN_100)
        train_F_A_ANN_100_std = np.std(train_F_A_ANN_100)

        train_F_F_DT_100_mean = np.mean(train_F_F_DT_100)
        train_F_F_DT_100_std = np.std(train_F_F_DT_100)
        train_F_F_NB_100_mean = np.mean(train_F_F_NB_100)
        train_F_F_NB_100_std = np.std(train_F_F_NB_100)
        train_F_F_kNN_100_mean = np.mean(train_F_F_kNN_100)
        train_F_F_kNN_100_std = np.std(train_F_F_kNN_100)
        train_F_F_SVM_100_mean = np.mean(train_F_F_SVM_100)
        train_F_F_SVM_100_std = np.std(train_F_F_SVM_100)
        train_F_F_RF_100_mean = np.mean(train_F_F_RF_100)
        train_F_F_RF_100_std = np.std(train_F_F_RF_100)
        train_F_F_ANN_100_mean = np.mean(train_F_F_ANN_100)
        train_F_F_ANN_100_std = np.std(train_F_F_ANN_100)

        train_F_P_DT_100_mean = np.mean(train_F_P_DT_100)
        train_F_P_DT_100_std = np.std(train_F_P_DT_100)
        train_F_P_NB_100_mean = np.mean(train_F_P_NB_100)
        train_F_P_NB_100_std = np.std(train_F_P_NB_100)
        train_F_P_kNN_100_mean = np.mean(train_F_P_kNN_100)
        train_F_P_kNN_100_std = np.std(train_F_P_kNN_100)
        train_F_P_SVM_100_mean = np.mean(train_F_P_SVM_100)
        train_F_P_SVM_100_std = np.std(train_F_P_SVM_100)
        train_F_P_RF_100_mean = np.mean(train_F_P_RF_100)
        train_F_P_RF_100_std = np.std(train_F_P_RF_100)
        train_F_P_ANN_100_mean = np.mean(train_F_P_ANN_100)
        train_F_P_ANN_100_std = np.std(train_F_P_ANN_100)

        train_F_R_DT_100_mean = np.mean(train_F_R_DT_100)
        train_F_R_DT_100_std = np.std(train_F_R_DT_100)
        train_F_R_NB_100_mean = np.mean(train_F_R_NB_100)
        train_F_R_NB_100_std = np.std(train_F_R_NB_100)
        train_F_R_kNN_100_mean = np.mean(train_F_R_kNN_100)
        train_F_R_kNN_100_std = np.std(train_F_R_kNN_100)
        train_F_R_SVM_100_mean = np.mean(train_F_R_SVM_100)
        train_F_R_SVM_100_std = np.std(train_F_R_SVM_100)
        train_F_R_RF_100_mean = np.mean(train_F_R_RF_100)
        train_F_R_RF_100_std = np.std(train_F_R_RF_100)
        train_F_R_ANN_100_mean = np.mean(train_F_R_ANN_100)
        train_F_R_ANN_100_std = np.std(train_F_R_ANN_100)

        train_F_ROC_DT_100_mean = np.mean(train_F_ROC_DT_100)
        train_F_ROC_DT_100_std = np.std(train_F_ROC_DT_100)
        train_F_ROC_NB_100_mean = np.mean(train_F_ROC_NB_100)
        train_F_ROC_NB_100_std = np.std(train_F_ROC_NB_100)
        train_F_ROC_kNN_100_mean = np.mean(train_F_ROC_kNN_100)
        train_F_ROC_kNN_100_std = np.std(train_F_ROC_kNN_100)
        train_F_ROC_SVM_100_mean = np.mean(train_F_ROC_SVM_100)
        train_F_ROC_SVM_100_std = np.std(train_F_ROC_SVM_100)
        train_F_ROC_RF_100_mean = np.mean(train_F_ROC_RF_100)
        train_F_ROC_RF_100_std = np.std(train_F_ROC_RF_100)
        train_F_ROC_ANN_100_mean = np.mean(train_F_ROC_ANN_100)
        train_F_ROC_ANN_100_std = np.std(train_F_ROC_ANN_100)
        #第一次随访
        E_v_A_DT_100_mean = np.mean(E_v_A_DT_100)
        E_v_A_DT_100_std = np.std(E_v_A_DT_100)
        E_v_A_NB_100_mean = np.mean(E_v_A_NB_100)
        E_v_A_NB_100_std = np.std(E_v_A_NB_100)
        E_v_A_kNN_100_mean = np.mean(E_v_A_kNN_100)
        E_v_A_kNN_100_std = np.std(E_v_A_kNN_100)
        E_v_A_SVM_100_mean = np.mean(E_v_A_SVM_100)
        E_v_A_SVM_100_std = np.std(E_v_A_SVM_100)
        E_v_A_RF_100_mean = np.mean(E_v_A_RF_100)
        E_v_A_RF_100_std = np.std(E_v_A_RF_100)
        E_v_A_ANN_100_mean = np.mean(E_v_A_ANN_100)
        E_v_A_ANN_100_std = np.std(E_v_A_ANN_100)

        E_v_F_DT_100_mean = np.mean(E_v_F_DT_100)
        E_v_F_DT_100_std = np.std(E_v_F_DT_100)
        E_v_F_NB_100_mean = np.mean(E_v_F_NB_100)
        E_v_F_NB_100_std = np.std(E_v_F_NB_100)
        E_v_F_kNN_100_mean = np.mean(E_v_F_kNN_100)
        E_v_F_kNN_100_std = np.std(E_v_F_kNN_100)
        E_v_F_SVM_100_mean = np.mean(E_v_F_SVM_100)
        E_v_F_SVM_100_std = np.std(E_v_F_SVM_100)
        E_v_F_RF_100_mean = np.mean(E_v_F_RF_100)
        E_v_F_RF_100_std = np.std(E_v_F_RF_100)
        E_v_F_ANN_100_mean = np.mean(E_v_F_ANN_100)
        E_v_F_ANN_100_std = np.std(E_v_F_ANN_100)

        E_v_P_DT_100_mean = np.mean(E_v_P_DT_100)
        E_v_P_DT_100_std = np.std(E_v_P_DT_100)
        E_v_P_NB_100_mean = np.mean(E_v_P_NB_100)
        E_v_P_NB_100_std = np.std(E_v_P_NB_100)
        E_v_P_kNN_100_mean = np.mean(E_v_P_kNN_100)
        E_v_P_kNN_100_std = np.std(E_v_P_kNN_100)
        E_v_P_SVM_100_mean = np.mean(E_v_P_SVM_100)
        E_v_P_SVM_100_std = np.std(E_v_P_SVM_100)
        E_v_P_RF_100_mean = np.mean(E_v_P_RF_100)
        E_v_P_RF_100_std = np.std(E_v_P_RF_100)
        E_v_P_ANN_100_mean = np.mean(E_v_P_ANN_100)
        E_v_P_ANN_100_std = np.std(E_v_P_ANN_100)

        E_v_R_DT_100_mean = np.mean(E_v_R_DT_100)
        E_v_R_DT_100_std = np.std(E_v_R_DT_100)
        E_v_R_NB_100_mean = np.mean(E_v_R_NB_100)
        E_v_R_NB_100_std = np.std(E_v_R_NB_100)
        E_v_R_kNN_100_mean = np.mean(E_v_R_kNN_100)
        E_v_R_kNN_100_std = np.std(E_v_R_kNN_100)
        E_v_R_SVM_100_mean = np.mean(E_v_R_SVM_100)
        E_v_R_SVM_100_std = np.std(E_v_R_SVM_100)
        E_v_R_RF_100_mean = np.mean(E_v_R_RF_100)
        E_v_R_RF_100_std = np.std(E_v_R_RF_100)
        E_v_R_ANN_100_mean = np.mean(E_v_R_ANN_100)
        E_v_R_ANN_100_std = np.std(E_v_R_ANN_100)

        E_v_ROC_DT_100_mean = np.mean(E_v_ROC_DT_100)
        E_v_ROC_DT_100_std = np.std(E_v_ROC_DT_100)
        E_v_ROC_NB_100_mean = np.mean(E_v_ROC_NB_100)
        E_v_ROC_NB_100_std = np.std(E_v_ROC_NB_100)
        E_v_ROC_kNN_100_mean = np.mean(E_v_ROC_kNN_100)
        E_v_ROC_kNN_100_std = np.std(E_v_ROC_kNN_100)
        E_v_ROC_SVM_100_mean = np.mean(E_v_ROC_SVM_100)
        E_v_ROC_SVM_100_std = np.std(E_v_ROC_SVM_100)
        E_v_ROC_RF_100_mean = np.mean(E_v_ROC_RF_100)
        E_v_ROC_RF_100_std = np.std(E_v_ROC_RF_100)
        E_v_ROC_ANN_100_mean = np.mean(E_v_ROC_ANN_100)
        E_v_ROC_ANN_100_std = np.std(E_v_ROC_ANN_100)

        E_v_F_A_DT_100_mean = np.mean(E_v_F_A_DT_100)
        E_v_F_A_DT_100_std = np.std(E_v_F_A_DT_100)
        E_v_F_A_NB_100_mean = np.mean(E_v_F_A_NB_100)
        E_v_F_A_NB_100_std = np.std(E_v_F_A_NB_100)
        E_v_F_A_kNN_100_mean = np.mean(E_v_F_A_kNN_100)
        E_v_F_A_kNN_100_std = np.std(E_v_F_A_kNN_100)
        E_v_F_A_SVM_100_mean = np.mean(E_v_F_A_SVM_100)
        E_v_F_A_SVM_100_std = np.std(E_v_F_A_SVM_100)
        E_v_F_A_RF_100_mean = np.mean(E_v_F_A_RF_100)
        E_v_F_A_RF_100_std = np.std(E_v_F_A_RF_100)
        E_v_F_A_ANN_100_mean = np.mean(E_v_F_A_ANN_100)
        E_v_F_A_ANN_100_std = np.std(E_v_F_A_ANN_100)

        E_v_F_F_DT_100_mean = np.mean(E_v_F_F_DT_100)
        E_v_F_F_DT_100_std = np.std(E_v_F_F_DT_100)
        E_v_F_F_NB_100_mean = np.mean(E_v_F_F_NB_100)
        E_v_F_F_NB_100_std = np.std(E_v_F_F_NB_100)
        E_v_F_F_kNN_100_mean = np.mean(E_v_F_F_kNN_100)
        E_v_F_F_kNN_100_std = np.std(E_v_F_F_kNN_100)
        E_v_F_F_SVM_100_mean = np.mean(E_v_F_F_SVM_100)
        E_v_F_F_SVM_100_std = np.std(E_v_F_F_SVM_100)
        E_v_F_F_RF_100_mean = np.mean(E_v_F_F_RF_100)
        E_v_F_F_RF_100_std = np.std(E_v_F_F_RF_100)
        E_v_F_F_ANN_100_mean = np.mean(E_v_F_F_ANN_100)
        E_v_F_F_ANN_100_std = np.std(E_v_F_F_ANN_100)

        E_v_F_P_DT_100_mean = np.mean(E_v_F_P_DT_100)
        E_v_F_P_DT_100_std = np.std(E_v_F_P_DT_100)
        E_v_F_P_NB_100_mean = np.mean(E_v_F_P_NB_100)
        E_v_F_P_NB_100_std = np.std(E_v_F_P_NB_100)
        E_v_F_P_kNN_100_mean = np.mean(E_v_F_P_kNN_100)
        E_v_F_P_kNN_100_std = np.std(E_v_F_P_kNN_100)
        E_v_F_P_SVM_100_mean = np.mean(E_v_F_P_SVM_100)
        E_v_F_P_SVM_100_std = np.std(E_v_F_P_SVM_100)
        E_v_F_P_RF_100_mean = np.mean(E_v_F_P_RF_100)
        E_v_F_P_RF_100_std = np.std(E_v_F_P_RF_100)
        E_v_F_P_ANN_100_mean = np.mean(E_v_F_P_ANN_100)
        E_v_F_P_ANN_100_std = np.std(E_v_F_P_ANN_100)

        E_v_F_R_DT_100_mean = np.mean(E_v_F_R_DT_100)
        E_v_F_R_DT_100_std = np.std(E_v_F_R_DT_100)
        E_v_F_R_NB_100_mean = np.mean(E_v_F_R_NB_100)
        E_v_F_R_NB_100_std = np.std(E_v_F_R_NB_100)
        E_v_F_R_kNN_100_mean = np.mean(E_v_F_R_kNN_100)
        E_v_F_R_kNN_100_std = np.std(E_v_F_R_kNN_100)
        E_v_F_R_SVM_100_mean = np.mean(E_v_F_R_SVM_100)
        E_v_F_R_SVM_100_std = np.std(E_v_F_R_SVM_100)
        E_v_F_R_RF_100_mean = np.mean(E_v_F_R_RF_100)
        E_v_F_R_RF_100_std = np.std(E_v_F_R_RF_100)
        E_v_F_R_ANN_100_mean = np.mean(E_v_F_R_ANN_100)
        E_v_F_R_ANN_100_std = np.std(E_v_F_R_ANN_100)

        E_v_F_ROC_DT_100_mean = np.mean(E_v_F_ROC_DT_100)
        E_v_F_ROC_DT_100_std = np.std(E_v_F_ROC_DT_100)
        E_v_F_ROC_NB_100_mean = np.mean(E_v_F_ROC_NB_100)
        E_v_F_ROC_NB_100_std = np.std(E_v_F_ROC_NB_100)
        E_v_F_ROC_kNN_100_mean = np.mean(E_v_F_ROC_kNN_100)
        E_v_F_ROC_kNN_100_std = np.std(E_v_F_ROC_kNN_100)
        E_v_F_ROC_SVM_100_mean = np.mean(E_v_F_ROC_SVM_100)
        E_v_F_ROC_SVM_100_std = np.std(E_v_F_ROC_SVM_100)
        E_v_F_ROC_RF_100_mean = np.mean(E_v_F_ROC_RF_100)
        E_v_F_ROC_RF_100_std = np.std(E_v_F_ROC_RF_100)
        E_v_F_ROC_ANN_100_mean = np.mean(E_v_F_ROC_ANN_100)
        E_v_F_ROC_ANN_100_std = np.std(E_v_F_ROC_ANN_100)
        #第二次随访
        # follow_up_2_A_DT_100_mean = np.mean(follow_up_2_A_DT_100)
        # follow_up_2_A_DT_100_std = np.std(follow_up_2_A_DT_100)
        # follow_up_2_A_NB_100_mean = np.mean(follow_up_2_A_NB_100)
        # follow_up_2_A_NB_100_std = np.std(follow_up_2_A_NB_100)
        # follow_up_2_A_kNN_100_mean = np.mean(follow_up_2_A_kNN_100)
        # follow_up_2_A_kNN_100_std = np.std(follow_up_2_A_kNN_100)
        # follow_up_2_A_SVM_100_mean = np.mean(follow_up_2_A_SVM_100)
        # follow_up_2_A_SVM_100_std = np.std(follow_up_2_A_SVM_100)
        # follow_up_2_A_RF_100_mean = np.mean(follow_up_2_A_RF_100)
        # follow_up_2_A_RF_100_std = np.std(follow_up_2_A_RF_100)
        # follow_up_2_A_ANN_100_mean = np.mean(follow_up_2_A_ANN_100)
        # follow_up_2_A_ANN_100_std = np.std(follow_up_2_A_ANN_100)
        #
        # follow_up_2_F_DT_100_mean = np.mean(follow_up_2_F_DT_100)
        # follow_up_2_F_DT_100_std = np.std(follow_up_2_F_DT_100)
        # follow_up_2_F_NB_100_mean = np.mean(follow_up_2_F_NB_100)
        # follow_up_2_F_NB_100_std = np.std(follow_up_2_F_NB_100)
        # follow_up_2_F_kNN_100_mean = np.mean(follow_up_2_F_kNN_100)
        # follow_up_2_F_kNN_100_std = np.std(follow_up_2_F_kNN_100)
        # follow_up_2_F_SVM_100_mean = np.mean(follow_up_2_F_SVM_100)
        # follow_up_2_F_SVM_100_std = np.std(follow_up_2_F_SVM_100)
        # follow_up_2_F_RF_100_mean = np.mean(follow_up_2_F_RF_100)
        # follow_up_2_F_RF_100_std = np.std(follow_up_2_F_RF_100)
        # follow_up_2_F_ANN_100_mean = np.mean(follow_up_2_F_ANN_100)
        # follow_up_2_F_ANN_100_std = np.std(follow_up_2_F_ANN_100)
        #
        # follow_up_2_P_DT_100_mean = np.mean(follow_up_2_P_DT_100)
        # follow_up_2_P_DT_100_std = np.std(follow_up_2_P_DT_100)
        # follow_up_2_P_NB_100_mean = np.mean(follow_up_2_P_NB_100)
        # follow_up_2_P_NB_100_std = np.std(follow_up_2_P_NB_100)
        # follow_up_2_P_kNN_100_mean = np.mean(follow_up_2_P_DT_100)
        # follow_up_2_P_kNN_100_std = np.std(follow_up_2_P_DT_100)
        # follow_up_2_P_SVM_100_mean = np.mean(follow_up_2_P_DT_100)
        # follow_up_2_P_SVM_100_std = np.std(follow_up_2_P_DT_100)
        # follow_up_2_P_RF_100_mean = np.mean(follow_up_2_P_DT_100)
        # follow_up_2_P_RF_100_std = np.std(follow_up_2_P_DT_100)
        # follow_up_2_P_ANN_100_mean = np.mean(follow_up_2_P_DT_100)
        # follow_up_2_P_ANN_100_std = np.std(follow_up_2_P_DT_100)
        #
        # follow_up_2_R_DT_100_mean = np.mean(follow_up_2_R_DT_100)
        # follow_up_2_R_DT_100_std = np.std(follow_up_2_R_DT_100)
        # follow_up_2_R_NB_100_mean = np.mean(follow_up_2_R_NB_100)
        # follow_up_2_R_NB_100_std = np.std(follow_up_2_R_NB_100)
        # follow_up_2_R_kNN_100_mean = np.mean(follow_up_2_R_kNN_100)
        # follow_up_2_R_kNN_100_std = np.std(follow_up_2_R_kNN_100)
        # follow_up_2_R_SVM_100_mean = np.mean(follow_up_2_R_SVM_100)
        # follow_up_2_R_SVM_100_std = np.std(follow_up_2_R_SVM_100)
        # follow_up_2_R_RF_100_mean = np.mean(follow_up_2_R_RF_100)
        # follow_up_2_R_RF_100_std = np.std(follow_up_2_R_RF_100)
        # follow_up_2_R_ANN_100_mean = np.mean(follow_up_2_R_ANN_100)
        # follow_up_2_R_ANN_100_std = np.std(follow_up_2_R_ANN_100)
        #
        # follow_up_2_ROC_DT_100_mean = np.mean(follow_up_2_ROC_DT_100)
        # follow_up_2_ROC_DT_100_std = np.std(follow_up_2_ROC_DT_100)
        # follow_up_2_ROC_NB_100_mean = np.mean(follow_up_2_ROC_NB_100)
        # follow_up_2_ROC_NB_100_std = np.std(follow_up_2_ROC_NB_100)
        # follow_up_2_ROC_kNN_100_mean = np.mean(follow_up_2_ROC_kNN_100)
        # follow_up_2_ROC_kNN_100_std = np.std(follow_up_2_ROC_kNN_100)
        # follow_up_2_ROC_SVM_100_mean = np.mean(follow_up_2_ROC_SVM_100)
        # follow_up_2_ROC_SVM_100_std = np.std(follow_up_2_ROC_SVM_100)
        # follow_up_2_ROC_RF_100_mean = np.mean(follow_up_2_ROC_RF_100)
        # follow_up_2_ROC_RF_100_std = np.std(follow_up_2_ROC_RF_100)
        # follow_up_2_ROC_ANN_100_mean = np.mean(follow_up_2_ROC_ANN_100)
        # follow_up_2_ROC_ANN_100_std = np.std(follow_up_2_ROC_ANN_100)
        #第三次随访
        # follow_up_3_A_DT_100_mean = np.mean(follow_up_3_A_DT_100)
        # follow_up_3_A_DT_100_std = np.std(follow_up_3_A_DT_100)
        # follow_up_3_A_NB_100_mean = np.mean(follow_up_3_A_NB_100)
        # follow_up_3_A_NB_100_std = np.std(follow_up_3_A_NB_100)
        # follow_up_3_A_kNN_100_mean = np.mean(follow_up_3_A_kNN_100)
        # follow_up_3_A_kNN_100_std = np.std(follow_up_3_A_kNN_100)
        # follow_up_3_A_SVM_100_mean = np.mean(follow_up_3_A_SVM_100)
        # follow_up_3_A_SVM_100_std = np.std(follow_up_3_A_SVM_100)
        # follow_up_3_A_RF_100_mean = np.mean(follow_up_3_A_RF_100)
        # follow_up_3_A_RF_100_std = np.std(follow_up_3_A_RF_100)
        # follow_up_3_A_ANN_100_mean = np.mean(follow_up_3_A_ANN_100)
        # follow_up_3_A_ANN_100_std = np.std(follow_up_3_A_ANN_100)
        #
        # follow_up_3_F_DT_100_mean = np.mean(follow_up_3_F_DT_100)
        # follow_up_3_F_DT_100_std = np.std(follow_up_3_F_DT_100)
        # follow_up_3_F_NB_100_mean = np.mean(follow_up_3_F_NB_100)
        # follow_up_3_F_NB_100_std = np.std(follow_up_3_F_NB_100)
        # follow_up_3_F_kNN_100_mean = np.mean(follow_up_3_F_kNN_100)
        # follow_up_3_F_kNN_100_std = np.std(follow_up_3_F_kNN_100)
        # follow_up_3_F_SVM_100_mean = np.mean(follow_up_3_F_SVM_100)
        # follow_up_3_F_SVM_100_std = np.std(follow_up_3_F_SVM_100)
        # follow_up_3_F_RF_100_mean = np.mean(follow_up_3_F_RF_100)
        # follow_up_3_F_RF_100_std = np.std(follow_up_3_F_RF_100)
        # follow_up_3_F_ANN_100_mean = np.mean(follow_up_3_F_ANN_100)
        # follow_up_3_F_ANN_100_std = np.std(follow_up_3_F_ANN_100)

        # follow_up_3_P_DT_100_mean = np.mean(follow_up_3_P_DT_100)
        # follow_up_3_P_DT_100_std = np.std(follow_up_3_P_DT_100)
        # follow_up_3_P_NB_100_mean = np.mean(follow_up_3_P_NB_100)
        # follow_up_3_P_NB_100_std = np.std(follow_up_3_P_NB_100)
        # follow_up_3_P_kNN_100_mean = np.mean(follow_up_3_P_DT_100)
        # follow_up_3_P_kNN_100_std = np.std(follow_up_3_P_DT_100)
        # follow_up_3_P_SVM_100_mean = np.mean(follow_up_3_P_DT_100)
        # follow_up_3_P_SVM_100_std = np.std(follow_up_3_P_DT_100)
        # follow_up_3_P_RF_100_mean = np.mean(follow_up_3_P_DT_100)
        # follow_up_3_P_RF_100_std = np.std(follow_up_3_P_DT_100)
        # follow_up_3_P_ANN_100_mean = np.mean(follow_up_3_P_DT_100)
        # follow_up_3_P_ANN_100_std = np.std(follow_up_3_P_DT_100)
        #
        # follow_up_3_R_DT_100_mean = np.mean(follow_up_3_R_DT_100)
        # follow_up_3_R_DT_100_std = np.std(follow_up_3_R_DT_100)
        # follow_up_3_R_NB_100_mean = np.mean(follow_up_3_R_NB_100)
        # follow_up_3_R_NB_100_std = np.std(follow_up_3_R_NB_100)
        # follow_up_3_R_kNN_100_mean = np.mean(follow_up_3_R_kNN_100)
        # follow_up_3_R_kNN_100_std = np.std(follow_up_3_R_kNN_100)
        # follow_up_3_R_SVM_100_mean = np.mean(follow_up_3_R_SVM_100)
        # follow_up_3_R_SVM_100_std = np.std(follow_up_3_R_SVM_100)
        # follow_up_3_R_RF_100_mean = np.mean(follow_up_3_R_RF_100)
        # follow_up_3_R_RF_100_std = np.std(follow_up_3_R_RF_100)
        # follow_up_3_R_ANN_100_mean = np.mean(follow_up_3_R_ANN_100)
        # follow_up_3_R_ANN_100_std = np.std(follow_up_3_R_ANN_100)
        #
        # follow_up_3_ROC_DT_100_mean = np.mean(follow_up_3_ROC_DT_100)
        # follow_up_3_ROC_DT_100_std = np.std(follow_up_3_ROC_DT_100)
        # follow_up_3_ROC_NB_100_mean = np.mean(follow_up_3_ROC_NB_100)
        # follow_up_3_ROC_NB_100_std = np.std(follow_up_3_ROC_NB_100)
        # follow_up_3_ROC_kNN_100_mean = np.mean(follow_up_3_ROC_kNN_100)
        # follow_up_3_ROC_kNN_100_std = np.std(follow_up_3_ROC_kNN_100)
        # follow_up_3_ROC_SVM_100_mean = np.mean(follow_up_3_ROC_SVM_100)
        # follow_up_3_ROC_SVM_100_std = np.std(follow_up_3_ROC_SVM_100)
        # follow_up_3_ROC_RF_100_mean = np.mean(follow_up_3_ROC_RF_100)
        # follow_up_3_ROC_RF_100_std = np.std(follow_up_3_ROC_RF_100)
        # follow_up_3_ROC_ANN_100_mean = np.mean(follow_up_3_ROC_ANN_100)
        # follow_up_3_ROC_ANN_100_std = np.std(follow_up_3_ROC_ANN_100)

        #测试集
        test_A_DT_100_mean = np.mean(test_A_DT_100)
        test_A_DT_100_std = np.std(test_A_DT_100)
        test_A_NB_100_mean = np.mean(test_A_NB_100)
        test_A_NB_100_std = np.std(test_A_NB_100)
        test_A_kNN_100_mean = np.mean(test_A_kNN_100)
        test_A_kNN_100_std = np.std(test_A_kNN_100)
        test_A_SVM_100_mean = np.mean(test_A_SVM_100)
        test_A_SVM_100_std = np.std(test_A_SVM_100)
        test_A_RF_100_mean = np.mean(test_A_RF_100)
        test_A_RF_100_std = np.std(test_A_RF_100)
        test_A_ANN_100_mean = np.mean(test_A_ANN_100)
        test_A_ANN_100_std = np.std(test_A_ANN_100)

        test_F_DT_100_mean = np.mean(test_F_DT_100)
        test_F_DT_100_std = np.std(test_F_DT_100)
        test_F_NB_100_mean = np.mean(test_F_NB_100)
        test_F_NB_100_std = np.std(test_F_NB_100)
        test_F_kNN_100_mean = np.mean(test_F_kNN_100)
        test_F_kNN_100_std = np.std(test_F_kNN_100)
        test_F_SVM_100_mean = np.mean(test_F_SVM_100)
        test_F_SVM_100_std = np.std(test_F_SVM_100)
        test_F_RF_100_mean = np.mean(test_F_RF_100)
        test_F_RF_100_std = np.std(test_F_RF_100)
        test_F_ANN_100_mean = np.mean(test_F_ANN_100)
        test_F_ANN_100_std = np.std(test_F_ANN_100)

        test_P_DT_100_mean = np.mean(test_P_DT_100)
        test_P_DT_100_std = np.std(test_P_DT_100)
        test_P_NB_100_mean = np.mean(test_P_NB_100)
        test_P_NB_100_std = np.std(test_P_NB_100)
        test_P_kNN_100_mean = np.mean(test_P_kNN_100)
        test_P_kNN_100_std = np.std(test_P_kNN_100)
        test_P_SVM_100_mean = np.mean(test_P_SVM_100)
        test_P_SVM_100_std = np.std(test_P_SVM_100)
        test_P_RF_100_mean = np.mean(test_P_RF_100)
        test_P_RF_100_std = np.std(test_P_RF_100)
        test_P_ANN_100_mean = np.mean(test_P_ANN_100)
        test_P_ANN_100_std = np.std(test_P_ANN_100)

        test_R_DT_100_mean = np.mean(test_R_DT_100)
        test_R_DT_100_std = np.std(test_R_DT_100)
        test_R_NB_100_mean = np.mean(test_R_NB_100)
        test_R_NB_100_std = np.std(test_R_NB_100)
        test_R_kNN_100_mean = np.mean(test_R_kNN_100)
        test_R_kNN_100_std = np.std(test_R_kNN_100)
        test_R_SVM_100_mean = np.mean(test_R_SVM_100)
        test_R_SVM_100_std = np.std(test_R_SVM_100)
        test_R_RF_100_mean = np.mean(test_R_RF_100)
        test_R_RF_100_std = np.std(test_R_RF_100)
        test_R_ANN_100_mean = np.mean(test_R_ANN_100)
        test_R_ANN_100_std = np.std(test_R_ANN_100)

        test_ROC_DT_100_mean = np.mean(test_ROC_DT_100)
        test_ROC_DT_100_std = np.std(test_ROC_DT_100)
        test_ROC_NB_100_mean = np.mean(test_ROC_NB_100)
        test_ROC_NB_100_std = np.std(test_ROC_NB_100)
        test_ROC_kNN_100_mean = np.mean(test_ROC_kNN_100)
        test_ROC_kNN_100_std = np.std(test_ROC_kNN_100)
        test_ROC_SVM_100_mean = np.mean(test_ROC_SVM_100)
        test_ROC_SVM_100_std = np.std(test_ROC_SVM_100)
        test_ROC_RF_100_mean = np.mean(test_ROC_RF_100)
        test_ROC_RF_100_std = np.std(test_ROC_RF_100)
        test_ROC_ANN_100_mean = np.mean(test_ROC_ANN_100)
        test_ROC_ANN_100_std = np.std(test_ROC_ANN_100)

        # 测试集
        test_F_A_DT_100_mean = np.mean(test_F_A_DT_100)
        test_F_A_DT_100_std = np.std(test_F_A_DT_100)
        test_F_A_NB_100_mean = np.mean(test_F_A_NB_100)
        test_F_A_NB_100_std = np.std(test_F_A_NB_100)
        test_F_A_kNN_100_mean = np.mean(test_F_A_kNN_100)
        test_F_A_kNN_100_std = np.std(test_F_A_kNN_100)
        test_F_A_SVM_100_mean = np.mean(test_F_A_SVM_100)
        test_F_A_SVM_100_std = np.std(test_F_A_SVM_100)
        test_F_A_RF_100_mean = np.mean(test_F_A_RF_100)
        test_F_A_RF_100_std = np.std(test_F_A_RF_100)
        test_F_A_ANN_100_mean = np.mean(test_F_A_ANN_100)
        test_F_A_ANN_100_std = np.std(test_F_A_ANN_100)

        test_F_F_DT_100_mean = np.mean(test_F_F_DT_100)
        test_F_F_DT_100_std = np.std(test_F_F_DT_100)
        test_F_F_NB_100_mean = np.mean(test_F_F_NB_100)
        test_F_F_NB_100_std = np.std(test_F_F_NB_100)
        test_F_F_kNN_100_mean = np.mean(test_F_F_kNN_100)
        test_F_F_kNN_100_std = np.std(test_F_F_kNN_100)
        test_F_F_SVM_100_mean = np.mean(test_F_F_SVM_100)
        test_F_F_SVM_100_std = np.std(test_F_F_SVM_100)
        test_F_F_RF_100_mean = np.mean(test_F_F_RF_100)
        test_F_F_RF_100_std = np.std(test_F_F_RF_100)
        test_F_F_ANN_100_mean = np.mean(test_F_F_ANN_100)
        test_F_F_ANN_100_std = np.std(test_F_F_ANN_100)

        test_F_P_DT_100_mean = np.mean(test_F_P_DT_100)
        test_F_P_DT_100_std = np.std(test_F_P_DT_100)
        test_F_P_NB_100_mean = np.mean(test_F_P_NB_100)
        test_F_P_NB_100_std = np.std(test_F_P_NB_100)
        test_F_P_kNN_100_mean = np.mean(test_F_P_kNN_100)
        test_F_P_kNN_100_std = np.std(test_F_P_kNN_100)
        test_F_P_SVM_100_mean = np.mean(test_F_P_SVM_100)
        test_F_P_SVM_100_std = np.std(test_F_P_SVM_100)
        test_F_P_RF_100_mean = np.mean(test_F_P_RF_100)
        test_F_P_RF_100_std = np.std(test_F_P_RF_100)
        test_F_P_ANN_100_mean = np.mean(test_F_P_ANN_100)
        test_F_P_ANN_100_std = np.std(test_F_P_ANN_100)

        test_F_R_DT_100_mean = np.mean(test_F_R_DT_100)
        test_F_R_DT_100_std = np.std(test_F_R_DT_100)
        test_F_R_NB_100_mean = np.mean(test_F_R_NB_100)
        test_F_R_NB_100_std = np.std(test_F_R_NB_100)
        test_F_R_kNN_100_mean = np.mean(test_F_R_kNN_100)
        test_F_R_kNN_100_std = np.std(test_F_R_kNN_100)
        test_F_R_SVM_100_mean = np.mean(test_F_R_SVM_100)
        test_F_R_SVM_100_std = np.std(test_F_R_SVM_100)
        test_F_R_RF_100_mean = np.mean(test_F_R_RF_100)
        test_F_R_RF_100_std = np.std(test_F_R_RF_100)
        test_F_R_ANN_100_mean = np.mean(test_F_R_ANN_100)
        test_F_R_ANN_100_std = np.std(test_F_R_ANN_100)

        test_F_ROC_DT_100_mean = np.mean(test_F_ROC_DT_100)
        test_F_ROC_DT_100_std = np.std(test_F_ROC_DT_100)
        test_F_ROC_NB_100_mean = np.mean(test_F_ROC_NB_100)
        test_F_ROC_NB_100_std = np.std(test_F_ROC_NB_100)
        test_F_ROC_kNN_100_mean = np.mean(test_F_ROC_kNN_100)
        test_F_ROC_kNN_100_std = np.std(test_F_ROC_kNN_100)
        test_F_ROC_SVM_100_mean = np.mean(test_F_ROC_SVM_100)
        test_F_ROC_SVM_100_std = np.std(test_F_ROC_SVM_100)
        test_F_ROC_RF_100_mean = np.mean(test_F_ROC_RF_100)
        test_F_ROC_RF_100_std = np.std(test_F_ROC_RF_100)
        test_F_ROC_ANN_100_mean = np.mean(test_F_ROC_ANN_100)
        test_F_ROC_ANN_100_std = np.std(test_F_ROC_ANN_100)

        #训练集
        train_data_result = [["%f" % train_A_DT_100_mean + "-" + "%f" % train_A_DT_100_std,
                             "%f" % train_A_NB_100_mean + "-" + "%f" % train_A_NB_100_std,
                             "%f" % train_A_kNN_100_mean + "-" + "%f" % train_A_kNN_100_std,
                             "%f" % train_A_SVM_100_mean + "-" + "%f" % train_A_SVM_100_std,
                             "%f" % train_A_RF_100_mean + "-" + "%f" % train_A_RF_100_std,
                             "%f" % train_A_ANN_100_mean + "-" + "%f" % train_A_ANN_100_std],
                            ["%f" % train_F_DT_100_mean + "-" + "%f" % train_F_DT_100_std,
                             "%f" % train_F_NB_100_mean + "-" + "%f" % train_F_NB_100_std,
                             "%f" % train_F_kNN_100_mean + "-" + "%f" % train_F_kNN_100_std,
                             "%f" % train_F_SVM_100_mean + "-" + "%f" % train_F_SVM_100_std,
                             "%f" % train_F_RF_100_mean + "-" + "%f" % train_F_RF_100_std,
                             "%f" % train_F_ANN_100_mean + "-" + "%f" % train_F_ANN_100_std],
                            ["%f" % train_P_DT_100_mean + "-" + "%f" % train_P_DT_100_std,
                             "%f" % train_P_NB_100_mean + "-" + "%f" % train_P_NB_100_std,
                             "%f" % train_P_kNN_100_mean + "-" + "%f" % train_P_kNN_100_std,
                             "%f" % train_P_SVM_100_mean + "-" + "%f" % train_P_SVM_100_std,
                             "%f" % train_P_RF_100_mean + "-" + "%f" % train_P_RF_100_std,
                             "%f" % train_P_ANN_100_mean + "-" + "%f" % train_P_ANN_100_std],
                            ["%f" % train_R_DT_100_mean + "-" + "%f" % train_R_DT_100_std,
                             "%f" % train_R_NB_100_mean + "-" + "%f" % train_R_NB_100_std,
                             "%f" % train_R_kNN_100_mean + "-" + "%f" % train_R_kNN_100_std,
                             "%f" % train_R_SVM_100_mean + "-" + "%f" % train_R_SVM_100_std,
                             "%f" % train_R_RF_100_mean + "-" + "%f" % train_R_RF_100_std,
                             "%f" % train_R_ANN_100_mean + "-" + "%f" % train_R_ANN_100_std],
                            ["%f" % train_ROC_DT_100_mean + "-" + "%f" % train_ROC_DT_100_std,
                             "%f" % train_ROC_NB_100_mean + "-" + "%f" % train_ROC_NB_100_std,
                             "%f" % train_ROC_kNN_100_mean + "-" + "%f" % train_ROC_kNN_100_std,
                             "%f" % train_ROC_SVM_100_mean + "-" + "%f" % train_ROC_SVM_100_std,
                             "%f" % train_ROC_RF_100_mean + "-" + "%f" % train_ROC_RF_100_std,
                             "%f" % train_ROC_ANN_100_mean + "-" + "%f" % train_ROC_ANN_100_std]]
        train_df_result = pd.DataFrame(train_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
                                      columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
                                      dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        train_df_result.to_excel(writer_train, sheet_name="train_pH_" + "%d" % a + "_" + "%d" % b)

        # 训练集
        train_F_data_result = [["%f" % train_F_A_DT_100_mean + "-" + "%f" % train_F_A_DT_100_std,
                              "%f" % train_F_A_NB_100_mean + "-" + "%f" % train_F_A_NB_100_std,
                              "%f" % train_F_A_kNN_100_mean + "-" + "%f" % train_F_A_kNN_100_std,
                              "%f" % train_F_A_SVM_100_mean + "-" + "%f" % train_F_A_SVM_100_std,
                              "%f" % train_F_A_RF_100_mean + "-" + "%f" % train_F_A_RF_100_std,
                              "%f" % train_F_A_ANN_100_mean + "-" + "%f" % train_F_A_ANN_100_std],
                             ["%f" % train_F_F_DT_100_mean + "-" + "%f" % train_F_F_DT_100_std,
                              "%f" % train_F_F_NB_100_mean + "-" + "%f" % train_F_F_NB_100_std,
                              "%f" % train_F_F_kNN_100_mean + "-" + "%f" % train_F_F_kNN_100_std,
                              "%f" % train_F_F_SVM_100_mean + "-" + "%f" % train_F_F_SVM_100_std,
                              "%f" % train_F_F_RF_100_mean + "-" + "%f" % train_F_F_RF_100_std,
                              "%f" % train_F_F_ANN_100_mean + "-" + "%f" % train_F_F_ANN_100_std],
                             ["%f" % train_F_P_DT_100_mean + "-" + "%f" % train_F_P_DT_100_std,
                              "%f" % train_F_P_NB_100_mean + "-" + "%f" % train_F_P_NB_100_std,
                              "%f" % train_F_P_kNN_100_mean + "-" + "%f" % train_F_P_kNN_100_std,
                              "%f" % train_F_P_SVM_100_mean + "-" + "%f" % train_F_P_SVM_100_std,
                              "%f" % train_F_P_RF_100_mean + "-" + "%f" % train_F_P_RF_100_std,
                              "%f" % train_F_P_ANN_100_mean + "-" + "%f" % train_F_P_ANN_100_std],
                             ["%f" % train_F_R_DT_100_mean + "-" + "%f" % train_F_R_DT_100_std,
                              "%f" % train_F_R_NB_100_mean + "-" + "%f" % train_F_R_NB_100_std,
                              "%f" % train_F_R_kNN_100_mean + "-" + "%f" % train_F_R_kNN_100_std,
                              "%f" % train_F_R_SVM_100_mean + "-" + "%f" % train_F_R_SVM_100_std,
                              "%f" % train_F_R_RF_100_mean + "-" + "%f" % train_F_R_RF_100_std,
                              "%f" % train_F_R_ANN_100_mean + "-" + "%f" % train_F_R_ANN_100_std],
                             ["%f" % train_F_ROC_DT_100_mean + "-" + "%f" % train_F_ROC_DT_100_std,
                              "%f" % train_F_ROC_NB_100_mean + "-" + "%f" % train_F_ROC_NB_100_std,
                              "%f" % train_F_ROC_kNN_100_mean + "-" + "%f" % train_F_ROC_kNN_100_std,
                              "%f" % train_F_ROC_SVM_100_mean + "-" + "%f" % train_F_ROC_SVM_100_std,
                              "%f" % train_F_ROC_RF_100_mean + "-" + "%f" % train_F_ROC_RF_100_std,
                              "%f" % train_F_ROC_ANN_100_mean + "-" + "%f" % train_F_ROC_ANN_100_std]]
        train_F_df_result = pd.DataFrame(train_F_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
                                       columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
                                       dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        train_F_df_result.to_excel(writer_train_F, sheet_name="train_F_pH_" + "%d" % a + "_" + "%d" % b)
        #第一次随访
        E_v_data_result = [["%f" % E_v_A_DT_100_mean + "-" + "%f" % E_v_A_DT_100_std,
                             "%f" % E_v_A_NB_100_mean + "-" + "%f" % E_v_A_NB_100_std,
                             "%f" % E_v_A_kNN_100_mean + "-" + "%f" % E_v_A_kNN_100_std,
                             "%f" % E_v_A_SVM_100_mean + "-" + "%f" % E_v_A_SVM_100_std,
                             "%f" % E_v_A_RF_100_mean + "-" + "%f" % E_v_A_RF_100_std,
                             "%f" % E_v_A_ANN_100_mean + "-" + "%f" % E_v_A_ANN_100_std],
                            ["%f" % E_v_F_DT_100_mean + "-" + "%f" % E_v_F_DT_100_std,
                             "%f" % E_v_F_NB_100_mean + "-" + "%f" % E_v_F_NB_100_std,
                             "%f" % E_v_F_kNN_100_mean + "-" + "%f" % E_v_F_kNN_100_std,
                             "%f" % E_v_F_SVM_100_mean + "-" + "%f" % E_v_F_SVM_100_std,
                             "%f" % E_v_F_RF_100_mean + "-" + "%f" % E_v_F_RF_100_std,
                             "%f" % E_v_F_ANN_100_mean + "-" + "%f" % E_v_F_ANN_100_std],
                            ["%f" % E_v_P_DT_100_mean + "-" + "%f" % E_v_P_DT_100_std,
                             "%f" % E_v_P_NB_100_mean + "-" + "%f" % E_v_P_NB_100_std,
                             "%f" % E_v_P_kNN_100_mean + "-" + "%f" % E_v_P_kNN_100_std,
                             "%f" % E_v_P_SVM_100_mean + "-" + "%f" % E_v_P_SVM_100_std,
                             "%f" % E_v_P_RF_100_mean + "-" + "%f" % E_v_P_RF_100_std,
                             "%f" % E_v_P_ANN_100_mean + "-" + "%f" % E_v_P_ANN_100_std],
                            ["%f" % E_v_R_DT_100_mean + "-" + "%f" % E_v_R_DT_100_std,
                             "%f" % E_v_R_NB_100_mean + "-" + "%f" % E_v_R_NB_100_std,
                             "%f" % E_v_R_kNN_100_mean + "-" + "%f" % E_v_R_kNN_100_std,
                             "%f" % E_v_R_SVM_100_mean + "-" + "%f" % E_v_R_SVM_100_std,
                             "%f" % E_v_R_RF_100_mean + "-" + "%f" % E_v_R_RF_100_std,
                             "%f" % E_v_R_ANN_100_mean + "-" + "%f" % E_v_R_ANN_100_std],
                            ["%f" % E_v_ROC_DT_100_mean + "-" + "%f" % E_v_ROC_DT_100_std,
                             "%f" % E_v_ROC_NB_100_mean + "-" + "%f" % E_v_ROC_NB_100_std,
                             "%f" % E_v_ROC_kNN_100_mean + "-" + "%f" % E_v_ROC_kNN_100_std,
                             "%f" % E_v_ROC_SVM_100_mean + "-" + "%f" % E_v_ROC_SVM_100_std,
                             "%f" % E_v_ROC_RF_100_mean + "-" + "%f" % E_v_ROC_RF_100_std,
                             "%f" % E_v_ROC_ANN_100_mean + "-" + "%f" % E_v_ROC_ANN_100_std]]
        E_v_df_result = pd.DataFrame(E_v_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
                                      columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
                                      dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        E_v_df_result.to_excel(writer_E_v, sheet_name="follow_up_1_pH_" + "%d" % a + "_" + "%d" % b)

        E_v_F_data_result = [["%f" % E_v_F_A_DT_100_mean + "-" + "%f" % E_v_F_A_DT_100_std,
                            "%f" % E_v_F_A_NB_100_mean + "-" + "%f" % E_v_F_A_NB_100_std,
                            "%f" % E_v_F_A_kNN_100_mean + "-" + "%f" % E_v_F_A_kNN_100_std,
                            "%f" % E_v_F_A_SVM_100_mean + "-" + "%f" % E_v_F_A_SVM_100_std,
                            "%f" % E_v_F_A_RF_100_mean + "-" + "%f" % E_v_F_A_RF_100_std,
                            "%f" % E_v_F_A_ANN_100_mean + "-" + "%f" % E_v_F_A_ANN_100_std],
                           ["%f" % E_v_F_F_DT_100_mean + "-" + "%f" % E_v_F_F_DT_100_std,
                            "%f" % E_v_F_F_NB_100_mean + "-" + "%f" % E_v_F_F_NB_100_std,
                            "%f" % E_v_F_F_kNN_100_mean + "-" + "%f" % E_v_F_F_kNN_100_std,
                            "%f" % E_v_F_F_SVM_100_mean + "-" + "%f" % E_v_F_F_SVM_100_std,
                            "%f" % E_v_F_F_RF_100_mean + "-" + "%f" % E_v_F_F_RF_100_std,
                            "%f" % E_v_F_F_ANN_100_mean + "-" + "%f" % E_v_F_F_ANN_100_std],
                           ["%f" % E_v_F_P_DT_100_mean + "-" + "%f" % E_v_F_P_DT_100_std,
                            "%f" % E_v_F_P_NB_100_mean + "-" + "%f" % E_v_F_P_NB_100_std,
                            "%f" % E_v_F_P_kNN_100_mean + "-" + "%f" % E_v_F_P_kNN_100_std,
                            "%f" % E_v_F_P_SVM_100_mean + "-" + "%f" % E_v_F_P_SVM_100_std,
                            "%f" % E_v_F_P_RF_100_mean + "-" + "%f" % E_v_F_P_RF_100_std,
                            "%f" % E_v_F_P_ANN_100_mean + "-" + "%f" % E_v_F_P_ANN_100_std],
                           ["%f" % E_v_F_R_DT_100_mean + "-" + "%f" % E_v_F_R_DT_100_std,
                            "%f" % E_v_F_R_NB_100_mean + "-" + "%f" % E_v_F_R_NB_100_std,
                            "%f" % E_v_F_R_kNN_100_mean + "-" + "%f" % E_v_F_R_kNN_100_std,
                            "%f" % E_v_F_R_SVM_100_mean + "-" + "%f" % E_v_F_R_SVM_100_std,
                            "%f" % E_v_F_R_RF_100_mean + "-" + "%f" % E_v_F_R_RF_100_std,
                            "%f" % E_v_F_R_ANN_100_mean + "-" + "%f" % E_v_F_R_ANN_100_std],
                           ["%f" % E_v_F_ROC_DT_100_mean + "-" + "%f" % E_v_F_ROC_DT_100_std,
                            "%f" % E_v_F_ROC_NB_100_mean + "-" + "%f" % E_v_F_ROC_NB_100_std,
                            "%f" % E_v_F_ROC_kNN_100_mean + "-" + "%f" % E_v_F_ROC_kNN_100_std,
                            "%f" % E_v_F_ROC_SVM_100_mean + "-" + "%f" % E_v_F_ROC_SVM_100_std,
                            "%f" % E_v_F_ROC_RF_100_mean + "-" + "%f" % E_v_F_ROC_RF_100_std,
                            "%f" % E_v_F_ROC_ANN_100_mean + "-" + "%f" % E_v_F_ROC_ANN_100_std]]
        E_v_F_df_result = pd.DataFrame(E_v_F_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
                                     columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
                                     dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        E_v_F_df_result.to_excel(writer_E_v_F, sheet_name="follow_up_1_pH_" + "%d" % a + "_" + "%d" % b)

        #第二次随访
        # follow_up_2_data_result = [["%f" % follow_up_2_A_DT_100_mean + "-" + "%f" % follow_up_2_A_DT_100_std,
        #                      "%f" % follow_up_2_A_NB_100_mean + "-" + "%f" % follow_up_2_A_NB_100_std,
        #                      "%f" % follow_up_2_A_kNN_100_mean + "-" + "%f" % follow_up_2_A_kNN_100_std,
        #                      "%f" % follow_up_2_A_SVM_100_mean + "-" + "%f" % follow_up_2_A_SVM_100_std,
        #                      "%f" % follow_up_2_A_RF_100_mean + "-" + "%f" % follow_up_2_A_RF_100_std,
        #                      "%f" % follow_up_2_A_ANN_100_mean + "-" + "%f" % follow_up_2_A_ANN_100_std],
        #                     ["%f" % follow_up_2_F_DT_100_mean + "-" + "%f" % follow_up_2_F_DT_100_std,
        #                      "%f" % follow_up_2_F_NB_100_mean + "-" + "%f" % follow_up_2_F_NB_100_std,
        #                      "%f" % follow_up_2_F_kNN_100_mean + "-" + "%f" % follow_up_2_F_kNN_100_std,
        #                      "%f" % follow_up_2_F_SVM_100_mean + "-" + "%f" % follow_up_2_F_SVM_100_std,
        #                      "%f" % follow_up_2_F_RF_100_mean + "-" + "%f" % follow_up_2_F_RF_100_std,
        #                      "%f" % follow_up_2_F_ANN_100_mean + "-" + "%f" % follow_up_2_F_ANN_100_std],
        #                     ["%f" % follow_up_2_P_DT_100_mean + "-" + "%f" % follow_up_2_P_DT_100_std,
        #                      "%f" % follow_up_2_P_NB_100_mean + "-" + "%f" % follow_up_2_P_NB_100_std,
        #                      "%f" % follow_up_2_P_kNN_100_mean + "-" + "%f" % follow_up_2_P_kNN_100_std,
        #                      "%f" % follow_up_2_P_SVM_100_mean + "-" + "%f" % follow_up_2_P_SVM_100_std,
        #                      "%f" % follow_up_2_P_RF_100_mean + "-" + "%f" % follow_up_2_P_RF_100_std,
        #                      "%f" % follow_up_2_P_ANN_100_mean + "-" + "%f" % follow_up_2_P_ANN_100_std],
        #                     ["%f" % follow_up_2_R_DT_100_mean + "-" + "%f" % follow_up_2_R_DT_100_std,
        #                      "%f" % follow_up_2_R_NB_100_mean + "-" + "%f" % follow_up_2_R_NB_100_std,
        #                      "%f" % follow_up_2_R_kNN_100_mean + "-" + "%f" % follow_up_2_R_kNN_100_std,
        #                      "%f" % follow_up_2_R_SVM_100_mean + "-" + "%f" % follow_up_2_R_SVM_100_std,
        #                      "%f" % follow_up_2_R_RF_100_mean + "-" + "%f" % follow_up_2_R_RF_100_std,
        #                      "%f" % follow_up_2_R_ANN_100_mean + "-" + "%f" % follow_up_2_R_ANN_100_std],
        #                     ["%f" % follow_up_2_ROC_DT_100_mean + "-" + "%f" % follow_up_2_ROC_DT_100_std,
        #                      "%f" % follow_up_2_ROC_NB_100_mean + "-" + "%f" % follow_up_2_ROC_NB_100_std,
        #                      "%f" % follow_up_2_ROC_kNN_100_mean + "-" + "%f" % follow_up_2_ROC_kNN_100_std,
        #                      "%f" % follow_up_2_ROC_SVM_100_mean + "-" + "%f" % follow_up_2_ROC_SVM_100_std,
        #                      "%f" % follow_up_2_ROC_RF_100_mean + "-" + "%f" % follow_up_2_ROC_RF_100_std,
        #                      "%f" % follow_up_2_ROC_ANN_100_mean + "-" + "%f" % follow_up_2_ROC_ANN_100_std]]
        # follow_up_2_df_result = pd.DataFrame(follow_up_2_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
        #                               columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
        #                               dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        # follow_up_2_df_result.to_excel(writer_follow_up_2, sheet_name="follow_up_2_pH_" + "%d" % a + "_" + "%d" % b)
        #
        #第三次随访
        # follow_up_3_data_result = [["%f" % follow_up_3_A_DT_100_mean + "-" + "%f" % follow_up_3_A_DT_100_std,
        #                      "%f" % follow_up_3_A_NB_100_mean + "-" + "%f" % follow_up_3_A_NB_100_std,
        #                      "%f" % follow_up_3_A_kNN_100_mean + "-" + "%f" % follow_up_3_A_kNN_100_std,
        #                      "%f" % follow_up_3_A_SVM_100_mean + "-" + "%f" % follow_up_3_A_SVM_100_std,
        #                      "%f" % follow_up_3_A_RF_100_mean + "-" + "%f" % follow_up_3_A_RF_100_std,
        #                      "%f" % follow_up_3_A_ANN_100_mean + "-" + "%f" % follow_up_3_A_ANN_100_std],
        #                     ["%f" % follow_up_3_F_DT_100_mean + "-" + "%f" % follow_up_3_F_DT_100_std,
        #                      "%f" % follow_up_3_F_NB_100_mean + "-" + "%f" % follow_up_3_F_NB_100_std,
        #                      "%f" % follow_up_3_F_kNN_100_mean + "-" + "%f" % follow_up_3_F_kNN_100_std,
        #                      "%f" % follow_up_3_F_SVM_100_mean + "-" + "%f" % follow_up_3_F_SVM_100_std,
        #                      "%f" % follow_up_3_F_RF_100_mean + "-" + "%f" % follow_up_3_F_RF_100_std,
        #                      "%f" % follow_up_3_F_ANN_100_mean + "-" + "%f" % follow_up_3_F_ANN_100_std],
        #                     ["%f" % follow_up_3_P_DT_100_mean + "-" + "%f" % follow_up_3_P_DT_100_std,
        #                      "%f" % follow_up_3_P_NB_100_mean + "-" + "%f" % follow_up_3_P_NB_100_std,
        #                      "%f" % follow_up_3_P_kNN_100_mean + "-" + "%f" % follow_up_3_P_kNN_100_std,
        #                      "%f" % follow_up_3_P_SVM_100_mean + "-" + "%f" % follow_up_3_P_SVM_100_std,
        #                      "%f" % follow_up_3_P_RF_100_mean + "-" + "%f" % follow_up_3_P_RF_100_std,
        #                      "%f" % follow_up_3_P_ANN_100_mean + "-" + "%f" % follow_up_3_P_ANN_100_std],
        #                     ["%f" % follow_up_3_R_DT_100_mean + "-" + "%f" % follow_up_3_R_DT_100_std,
        #                      "%f" % follow_up_3_R_NB_100_mean + "-" + "%f" % follow_up_3_R_NB_100_std,
        #                      "%f" % follow_up_3_R_kNN_100_mean + "-" + "%f" % follow_up_3_R_kNN_100_std,
        #                      "%f" % follow_up_3_R_SVM_100_mean + "-" + "%f" % follow_up_3_R_SVM_100_std,
        #                      "%f" % follow_up_3_R_RF_100_mean + "-" + "%f" % follow_up_3_R_RF_100_std,
        #                      "%f" % follow_up_3_R_ANN_100_mean + "-" + "%f" % follow_up_3_R_ANN_100_std],
        #                     ["%f" % follow_up_3_ROC_DT_100_mean + "-" + "%f" % follow_up_3_ROC_DT_100_std,
        #                      "%f" % follow_up_3_ROC_NB_100_mean + "-" + "%f" % follow_up_3_ROC_NB_100_std,
        #                      "%f" % follow_up_3_ROC_kNN_100_mean + "-" + "%f" % follow_up_3_ROC_kNN_100_std,
        #                      "%f" % follow_up_3_ROC_SVM_100_mean + "-" + "%f" % follow_up_3_ROC_SVM_100_std,
        #                      "%f" % follow_up_3_ROC_RF_100_mean + "-" + "%f" % follow_up_3_ROC_RF_100_std,
        #                      "%f" % follow_up_3_ROC_ANN_100_mean + "-" + "%f" % follow_up_3_ROC_ANN_100_std]]
        # follow_up_3_df_result = pd.DataFrame(follow_up_3_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
        #                               columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
        #                               dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        # follow_up_3_df_result.to_excel(writer_follow_up_3, sheet_name="follow_up_3_pH_" + "%d" % a + "_" + "%d" % b)
        #测试集
        test_data_result = [["%f"%test_A_DT_100_mean + "-" + "%f"%test_A_DT_100_std,"%f"%test_A_NB_100_mean + "-" + "%f"%test_A_NB_100_std,"%f"%test_A_kNN_100_mean+"-"+"%f"%test_A_kNN_100_std,"%f"%test_A_SVM_100_mean+"-"+"%f"%test_A_SVM_100_std,"%f"%test_A_RF_100_mean+"-"+"%f"%test_A_RF_100_std,"%f"%test_A_ANN_100_mean+"-"+"%f"%test_A_ANN_100_std],
                       ["%f"%test_F_DT_100_mean + "-" + "%f"%test_F_DT_100_std,"%f"%test_F_NB_100_mean + "-" + "%f"%test_F_NB_100_std,"%f"%test_F_kNN_100_mean+"-"+"%f"%test_F_kNN_100_std,"%f"%test_F_SVM_100_mean+"-"+"%f"%test_F_SVM_100_std,"%f"%test_F_RF_100_mean+"-"+"%f"%test_F_RF_100_std,"%f"%test_F_ANN_100_mean+"-"+"%f"%test_F_ANN_100_std],
                       ["%f"%test_P_DT_100_mean + "-" + "%f"%test_P_DT_100_std,"%f"%test_P_NB_100_mean + "-" + "%f"%test_P_NB_100_std,"%f"%test_P_kNN_100_mean+"-"+"%f"%test_P_kNN_100_std,"%f"%test_P_SVM_100_mean+"-"+"%f"%test_P_SVM_100_std,"%f"%test_P_RF_100_mean+"-"+"%f"%test_P_RF_100_std,"%f"%test_P_ANN_100_mean+"-"+"%f"%test_P_ANN_100_std],
                       ["%f"%test_R_DT_100_mean + "-" + "%f"%test_R_DT_100_std,"%f"%test_R_NB_100_mean + "-" + "%f"%test_R_NB_100_std,"%f"%test_R_kNN_100_mean+"-"+"%f"%test_R_kNN_100_std,"%f"%test_R_SVM_100_mean+"-"+"%f"%test_R_SVM_100_std,"%f"%test_R_RF_100_mean+"-"+"%f"%test_R_RF_100_std,"%f"%test_R_ANN_100_mean+"-"+"%f"%test_R_ANN_100_std],
                       ["%f"%test_ROC_DT_100_mean + "-" + "%f"%test_ROC_DT_100_std,"%f"%test_ROC_NB_100_mean + "-" + "%f"%test_ROC_NB_100_std,"%f"%test_ROC_kNN_100_mean+"-"+"%f"%test_ROC_kNN_100_std,"%f"%test_ROC_SVM_100_mean+"-"+"%f"%test_ROC_SVM_100_std,"%f"%test_ROC_RF_100_mean+"-"+"%f"%test_ROC_RF_100_std,"%f"%test_ROC_ANN_100_mean+"-"+"%f"%test_ROC_ANN_100_std]]
        test_df_result = pd.DataFrame(test_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'], columns=['DT','NB','kNN','SVM','RF','ANN'],dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        test_df_result.to_excel(writer_test, sheet_name="test_pH_"+ "%d"% a + "_" + "%d"% b)

        test_F_data_result = [["%f" % test_F_A_DT_100_mean + "-" + "%f" % test_F_A_DT_100_std,
                             "%f" % test_F_A_NB_100_mean + "-" + "%f" % test_F_A_NB_100_std,
                             "%f" % test_F_A_kNN_100_mean + "-" + "%f" % test_F_A_kNN_100_std,
                             "%f" % test_F_A_SVM_100_mean + "-" + "%f" % test_F_A_SVM_100_std,
                             "%f" % test_F_A_RF_100_mean + "-" + "%f" % test_F_A_RF_100_std,
                             "%f" % test_F_A_ANN_100_mean + "-" + "%f" % test_F_A_ANN_100_std],
                            ["%f" % test_F_F_DT_100_mean + "-" + "%f" % test_F_F_DT_100_std,
                             "%f" % test_F_F_NB_100_mean + "-" + "%f" % test_F_F_NB_100_std,
                             "%f" % test_F_F_kNN_100_mean + "-" + "%f" % test_F_F_kNN_100_std,
                             "%f" % test_F_F_SVM_100_mean + "-" + "%f" % test_F_F_SVM_100_std,
                             "%f" % test_F_F_RF_100_mean + "-" + "%f" % test_F_F_RF_100_std,
                             "%f" % test_F_F_ANN_100_mean + "-" + "%f" % test_F_F_ANN_100_std],
                            ["%f" % test_F_P_DT_100_mean + "-" + "%f" % test_F_P_DT_100_std,
                             "%f" % test_F_P_NB_100_mean + "-" + "%f" % test_F_P_NB_100_std,
                             "%f" % test_F_P_kNN_100_mean + "-" + "%f" % test_F_P_kNN_100_std,
                             "%f" % test_F_P_SVM_100_mean + "-" + "%f" % test_F_P_SVM_100_std,
                             "%f" % test_F_P_RF_100_mean + "-" + "%f" % test_F_P_RF_100_std,
                             "%f" % test_F_P_ANN_100_mean + "-" + "%f" % test_F_P_ANN_100_std],
                            ["%f" % test_F_R_DT_100_mean + "-" + "%f" % test_F_R_DT_100_std,
                             "%f" % test_F_R_NB_100_mean + "-" + "%f" % test_F_R_NB_100_std,
                             "%f" % test_F_R_kNN_100_mean + "-" + "%f" % test_F_R_kNN_100_std,
                             "%f" % test_F_R_SVM_100_mean + "-" + "%f" % test_F_R_SVM_100_std,
                             "%f" % test_F_R_RF_100_mean + "-" + "%f" % test_F_R_RF_100_std,
                             "%f" % test_F_R_ANN_100_mean + "-" + "%f" % test_F_R_ANN_100_std],
                            ["%f" % test_F_ROC_DT_100_mean + "-" + "%f" % test_F_ROC_DT_100_std,
                             "%f" % test_F_ROC_NB_100_mean + "-" + "%f" % test_F_ROC_NB_100_std,
                             "%f" % test_F_ROC_kNN_100_mean + "-" + "%f" % test_F_ROC_kNN_100_std,
                             "%f" % test_F_ROC_SVM_100_mean + "-" + "%f" % test_F_ROC_SVM_100_std,
                             "%f" % test_F_ROC_RF_100_mean + "-" + "%f" % test_F_ROC_RF_100_std,
                             "%f" % test_F_ROC_ANN_100_mean + "-" + "%f" % test_F_ROC_ANN_100_std]]
        test_F_df_result = pd.DataFrame(test_F_data_result, index=['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
                                      columns=['DT', 'NB', 'kNN', 'SVM', 'RF', 'ANN'],
                                      dtype=float)  # 将第一维度数据转为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        test_F_df_result.to_excel(writer_test_F, sheet_name="test_pH_" + "%d" % a + "_" + "%d" % b)

        cm_DT = np.around(result_matrix_DT / 10, decimals=2)


        cm_NB = np.around(result_matrix_NB / 10, decimals=2)


        cm_kNN = np.around(result_matrix_kNN / 10, decimals=2)


        cm_SVM = np.around(result_matrix_SVM / 10, decimals=2)


        cm_RF = np.around(result_matrix_RF / 10, decimals=2)

        cm_ANN = np.around(result_matrix_ANN / 10, decimals=2)


        cm_DT_E_v = np.around(result_matrix_DT_E_v / 10, decimals=2)

        cm_NB_E_v = np.around(result_matrix_NB_E_v / 10, decimals=2)

        cm_kNN_E_v = np.around(result_matrix_kNN_E_v / 10, decimals=2)

        cm_SVM_E_v = np.around(result_matrix_SVM_E_v / 10, decimals=2)

        cm_RF_E_v = np.around(result_matrix_RF_E_v / 10, decimals=2)

        cm_ANN_E_v = np.around(result_matrix_ANN_E_v / 10, decimals=2)



        cm_DT_test_F = np.around(result_matrix_DT_test_F / 10, decimals=2)

        cm_NB_test_F = np.around(result_matrix_NB_test_F / 10, decimals=2)

        cm_kNN_test_F = np.around(result_matrix_kNN_test_F / 10, decimals=2)

        cm_SVM_test_F = np.around(result_matrix_SVM_test_F / 10, decimals=2)

        cm_RF_test_F = np.around(result_matrix_RF_test_F / 10, decimals=2)

        cm_ANN_test_F = np.around(result_matrix_ANN_test_F / 10, decimals=2)


        cm_DT_train_F = np.around(result_matrix_DT_train_F / 10, decimals=2)

        cm_NB_train_F = np.around(result_matrix_NB_train_F / 10, decimals=2)

        cm_kNN_train_F = np.around(result_matrix_kNN_train_F / 10, decimals=2)

        cm_SVM_train_F = np.around(result_matrix_SVM_train_F / 10, decimals=2)

        cm_RF_train_F = np.around(result_matrix_RF_train_F / 10, decimals=2)

        cm_ANN_train_F = np.around(result_matrix_ANN_train_F / 10, decimals=2)


        cm_DT_E_v_F = np.around(result_matrix_DT_E_v_F / 10, decimals=2)

        cm_NB_E_v_F = np.around(result_matrix_NB_E_v_F / 10, decimals=2)

        cm_kNN_E_v_F = np.around(result_matrix_kNN_E_v_F / 10, decimals=2)

        cm_SVM_E_v_F = np.around(result_matrix_SVM_E_v_F / 10, decimals=2)

        cm_RF_E_v_F = np.around(result_matrix_RF_E_v_F / 10, decimals=2)

        cm_ANN_E_v_F = np.around(result_matrix_ANN_E_v_F / 10, decimals=2)




        disp_DT = ConfusionMatrixDisplay(confusion_matrix=(cm_DT),
                                          display_labels=['1', '2', '3'])
        disp_NB = ConfusionMatrixDisplay(confusion_matrix=(cm_NB),
                                         display_labels=['1', '2', '3'])
        disp_kNN = ConfusionMatrixDisplay(confusion_matrix=(cm_kNN),
                                          display_labels=['1', '2', '3'])
        disp_SVM = ConfusionMatrixDisplay(confusion_matrix=(cm_SVM),
                                          display_labels=['1', '2', '3'])
        disp_RF = ConfusionMatrixDisplay(confusion_matrix=(cm_RF),
                                         display_labels=['1', '2', '3'])
        disp_ANN = ConfusionMatrixDisplay(confusion_matrix=(cm_ANN),
                                          display_labels=['1', '2', '3'])

        disp_DT_E_v = ConfusionMatrixDisplay(confusion_matrix=(cm_DT_E_v),
                                         display_labels=['1', '2', '3'])
        disp_NB_E_v = ConfusionMatrixDisplay(confusion_matrix=(cm_NB_E_v),
                                         display_labels=['1', '2', '3'])
        disp_kNN_E_v = ConfusionMatrixDisplay(confusion_matrix=(cm_kNN_E_v),
                                          display_labels=['1', '2', '3'])
        disp_SVM_E_v = ConfusionMatrixDisplay(confusion_matrix=(cm_SVM_E_v),
                                          display_labels=['1', '2', '3'])
        disp_RF_E_v = ConfusionMatrixDisplay(confusion_matrix=(cm_RF_E_v),
                                         display_labels=['1', '2', '3'])
        disp_ANN_E_v = ConfusionMatrixDisplay(confusion_matrix=(cm_ANN_E_v),
                                         display_labels=['1', '2', '3'])

        disp_DT_E_v_F = ConfusionMatrixDisplay(confusion_matrix=(cm_DT_E_v_F),
                                             display_labels=['1', '2', '3'])
        disp_NB_E_v_F = ConfusionMatrixDisplay(confusion_matrix=(cm_NB_E_v_F),
                                             display_labels=['1', '2', '3'])
        disp_kNN_E_v_F = ConfusionMatrixDisplay(confusion_matrix=(cm_kNN_E_v_F),
                                              display_labels=['1', '2', '3'])
        disp_SVM_E_v_F = ConfusionMatrixDisplay(confusion_matrix=(cm_SVM_E_v_F),
                                              display_labels=['1', '2', '3'])
        disp_RF_E_v_F = ConfusionMatrixDisplay(confusion_matrix=(cm_RF_E_v_F),
                                             display_labels=['1', '2', '3'])
        disp_ANN_E_v_F = ConfusionMatrixDisplay(confusion_matrix=(cm_ANN_E_v_F),
                                              display_labels=['1', '2', '3'])

        disp_DT_test_F = ConfusionMatrixDisplay(confusion_matrix=(cm_DT_test_F),
                                               display_labels=['1', '2', '3'])
        disp_NB_test_F = ConfusionMatrixDisplay(confusion_matrix=(cm_NB_test_F),
                                               display_labels=['1', '2', '3'])
        disp_kNN_test_F = ConfusionMatrixDisplay(confusion_matrix=(cm_kNN_test_F),
                                                display_labels=['1', '2', '3'])
        disp_SVM_test_F = ConfusionMatrixDisplay(confusion_matrix=(cm_SVM_test_F),
                                                display_labels=['1', '2', '3'])
        disp_RF_test_F = ConfusionMatrixDisplay(confusion_matrix=(cm_RF_test_F),
                                               display_labels=['1', '2', '3'])
        disp_ANN_test_F = ConfusionMatrixDisplay(confusion_matrix=(cm_ANN_test_F),
                                                display_labels=['1', '2', '3'])

        disp_DT_train_F = ConfusionMatrixDisplay(confusion_matrix=(cm_DT_train_F),
                                                display_labels=['1', '2', '3'])
        disp_NB_train_F = ConfusionMatrixDisplay(confusion_matrix=(cm_NB_train_F),
                                                display_labels=['1', '2', '3'])
        disp_kNN_train_F = ConfusionMatrixDisplay(confusion_matrix=(cm_kNN_train_F),
                                                 display_labels=['1', '2', '3'])
        disp_SVM_train_F = ConfusionMatrixDisplay(confusion_matrix=(cm_SVM_train_F),
                                                 display_labels=['1', '2', '3'])
        disp_RF_train_F = ConfusionMatrixDisplay(confusion_matrix=(cm_RF_train_F),
                                                display_labels=['1', '2', '3'])
        disp_ANN_train_F = ConfusionMatrixDisplay(confusion_matrix=(cm_ANN_train_F),
                                                 display_labels=['1', '2', '3'])

        # === plot ===
        disp_DT.plot(cmap='Blues')
        #plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "DT_1_" + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")

        disp_NB.plot(cmap='Blues')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "NB_1_" + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")

        disp_kNN.plot(cmap='Blues')
        #plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "kNN_1_" + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")

        disp_SVM.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "SVM_1_" + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")

        disp_RF.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "RF_1_" + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")

        disp_ANN.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "ANN_1_" + "%d"%a + "_" + "%d"%b + ".pdf",dpi = 600,bbox_inches = "tight")

        # === plot ===
        disp_DT_E_v.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_DT_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_NB_E_v.plot(cmap='Blues')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_NB_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_kNN_E_v.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_kNN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_SVM_E_v.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_SVM_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_RF_E_v.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_RF_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_ANN_E_v.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_ANN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        # === plot ===
        disp_DT_E_v_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_F_DT_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_NB_E_v_F.plot(cmap='Blues')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_F_NB_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_kNN_E_v_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_F_kNN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_SVM_E_v_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_F_SVM_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_RF_E_v_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_F_RF_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_ANN_E_v_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "E_v_F_ANN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        # === plot ===
        disp_DT_test_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "test_F_DT_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_NB_test_F.plot(cmap='Blues')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "test_F_NB_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_kNN_test_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "test_F_kNN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_SVM_test_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "test_F_SVM_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_RF_test_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "test_F_RF_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_ANN_test_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "test_F_ANN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        # === plot ===
        disp_DT_train_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "train_F_DT_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_NB_train_F.plot(cmap='Blues')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "train_F_NB_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_kNN_train_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "train_F_kNN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600,
                    bbox_inches="tight")

        disp_SVM_train_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "train_F_SVM_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600,
                    bbox_inches="tight")

        disp_RF_train_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "train_F_RF_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600, bbox_inches="tight")

        disp_ANN_train_F.plot(cmap='Blues')
        # plt.xlabel('Predicted labels')
        plt.ylabel('True label')
        plt.savefig("..//data/new/" + "train_F_ANN_1_" + "%d" % a + "_" + "%d" % b + ".pdf", dpi=600,
                    bbox_inches="tight")

     writer_train.save()
     writer_E_v.save()
     #writer_follow_up_2.save()
     # writer_follow_up_3.save()
     writer_test.save()

     writer_train_F.save()
     writer_E_v_F.save()
     # writer_follow_up_2.save()
     # writer_follow_up_3.save()
     writer_test_F.save()


