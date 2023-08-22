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

#数据集划分
from sklearn.model_selection import train_test_split

for a in [1,2,3]:
    writer = pd.ExcelWriter("..//data/pH_data_split" + "_" + "%d" % a + ".xlsx")
    for b in [1,2]:
        rL_pH = pd.read_excel("..//data/pH_sample" + ".xlsx",sheet_name="pH_"+"%a"%a+"_"+"%d"%b,header=0,names=None,index_col=0)

        X = rL_pH.iloc[:,0:-1].values
        y = rL_pH.iloc[:,-1].values

    #train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=0)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, stratify=y, shuffle=True, random_state=0)

        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        print(test_y.shape)

        train_x=pd.DataFrame(train_x)
        train_y=pd.DataFrame(train_y)
        test_x=pd.DataFrame(test_x)
        test_y=pd.DataFrame(test_y)

        data_train_test = pd.concat([train_x,train_y], axis=1, ignore_index=True)
        data_validation = pd.concat([test_x,test_y], axis=1, ignore_index=True)

        data_train_test.to_excel(writer, sheet_name="pH_" + "%d" % a + "_" + "%d" % b)
        data_validation.to_excel(writer, sheet_name="pH_" + "%d" % a + "_" + "%d" % (b+2))
    writer.save()

# 参数更新
data_train_test.to_csv("..//data/data_train_test.csv")
data_validation.to_csv("..//data/data_validation.csv")

for a in [1,2,3]:
     writer = pd.ExcelWriter("..//data/pH_para" + "_" + "%d" % a + ".xlsx")
     for b in [3,4]:
        rL_pH = pd.read_excel("..//data/pH_data_split_" + "%d"%a + ".xlsx",sheet_name="pH_" + "%d"%a + "_" + "%d"%b,header=0,names=None,index_col=0)

        X_1 = rL_pH.iloc[:,0:-1]
        y_1 = rL_pH.iloc[:,-1]

        clf_DT = DecisionTreeClassifier()
        #clf_NB = GaussianNB()
        clf_kNN = KNeighborsClassifier()
        clf_SVM = SVC()
        clf_RF = RandomForestClassifier()
        clf_ANN = MLPClassifier()

        param_DT = {"ccp_alpha" : math.e**np.arange(math.log(0.0001),math.log(0.1)),"min_samples_leaf" : np.arange(1,36)}

        param_kNN = {"n_neighbors" : np.arange(1,36)}

        param_SVM = {"coef0" : math.e**np.arange(math.log(0.001),math.log(10)),"C" : np.arange(1,36)}

        param_RF = {"n_estimators" : np.arange(0,50), "max_depth" : np.arange(1,10), "min_samples_leaf" : np.arange(1,36)}

        param_ANN = {"learning_rate_init" : math.e**np.arange(math.log(0.025),math.log(2.5)),"batch_size": np.arange(1, 36)}
        # "batch_size": np.arange(10, 151)
        rscv_DT = RandomizedSearchCV(clf_DT,param_DT,scoring='accuracy',n_jobs=1,cv=3)
        rscv_kNN = RandomizedSearchCV(clf_kNN,param_kNN,scoring='accuracy',n_jobs=1,cv=3)
        rscv_SVM = RandomizedSearchCV(clf_SVM,param_SVM,scoring='accuracy',n_jobs=1,cv=3)
        rscv_RF = RandomizedSearchCV(clf_RF,param_RF,scoring='accuracy',n_jobs=1,cv=3)
        rscv_ANN = RandomizedSearchCV(clf_ANN,param_ANN,scoring='accuracy',n_jobs=1,cv=3)

        rscv_DT.fit(X_1,y_1)
        rscv_kNN.fit(X_1,y_1)
        rscv_SVM.fit(X_1,y_1)
        rscv_RF.fit(X_1,y_1)
        rscv_ANN.fit(X_1,y_1)

        # print('rscv_DT',rscv_DT.best_params_)
        # print('rscv_kNN',rscv_kNN.best_params_)
        # print('rscv_SVM',rscv_SVM.best_params_)
        # print('rscv_RF',rscv_RF.best_params_)
        # print('rscv_ANN',rscv_ANN.best_params_)

        data_result = [[rscv_DT.best_params_], [rscv_kNN.best_params_], [rscv_SVM.best_params_], [rscv_RF.best_params_], [rscv_ANN.best_params_]]
        df_result = pd.DataFrame(data_result, index=['rscv_DT', 'rscv_kNN', 'rscv_SVM', 'rscv_RF', 'rscv_ANN'], columns=['para'],dtype=float)  # 将第一维度数据转为为行，第二维度数据转化为列，即 6 行 2 列，并设置列标签
        df_result.to_excel(writer, sheet_name="pH_para_" + "%d"%a + "_" + "%d"%b)

     writer.save()