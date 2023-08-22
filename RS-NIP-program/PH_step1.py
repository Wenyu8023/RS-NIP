#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.unicode.east_asian_width",True)
pH_1 = pd.read_excel("..//data/ph_data_20230206.xlsx",sheet_name="pH_1",header=0,names=None,index_col=False)
# print(pH_1)
pH_1.index = range(len(pH_1))
#print(pH_1.columns)
#归一化
X = np.array(pH_1.iloc[:,2:7])
# print(X)
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(X)
# print(X_minMax)
F_scaler = pd.DataFrame(X_minMax)
F_scaler.columns = ['$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$']
# print(F_scaler)
# print(pH_1.iloc[:,[0,6]])
# pH_1_1_number = pd.concat([pH_1.iloc[:,0],F_scaler], axis=1, ignore_index=True)
pH_1_1 = pd.concat([pH_1.iloc[:,0],F_scaler,pH_1.iloc[:,[1,7]]], axis=1, ignore_index=True)
pH_1_1.columns = ['$number$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$WHO$ $FC$','$RS$']
print(pH_1_1)

#离散化，按照标准离散化
pH_1_2 = pd.read_excel("..//data/pH_data_20230206.xlsx",sheet_name='pH_1',header=0,names=None,index_col=False)
pH_1_2.index = range(len(pH_1_2))
#print(pH_1_2.columns)
for i in range(len(pH_1_2.iloc[:,2])):
    if pH_1_2.iloc[i,2] > 440:
        pH_1_2.iloc[i,2] = 1
    elif 165 <= pH_1_2.iloc[i,2] <= 440:
        pH_1_2.iloc[i,2] = 2
    elif pH_1_2.iloc[i,2] < 165:
        pH_1_2.iloc[i,2] = 3

for i in range(len(pH_1_2.iloc[:,3])):
    if pH_1_2.iloc[i,3] < 300:
        pH_1_2.iloc[i,3] = 1
    elif 300 <= pH_1_2.iloc[i,3] <= 1400:
        pH_1_2.iloc[i,3] = 2
    elif pH_1_2.iloc[i,3]>1400:
        pH_1_2.iloc[i,3] = 3

for i in range(len(pH_1_2.iloc[:,4])):
    if pH_1_2.iloc[i,4] < 8:
        pH_1_2.iloc[i,4] = 1
    elif 8 <= pH_1_2.iloc[i,4] <= 14:
        pH_1_2.iloc[i,4] = 2
    elif 14 < pH_1_2.iloc[i,4]:
         pH_1_2.iloc[i,4] = 3

for i in range(len(pH_1_2.iloc[:,5])):
    if pH_1_2.iloc[i,5] > 65:
        pH_1_2.iloc[i,5] = 1
    elif 60 <= pH_1_2.iloc[i,5] <= 65:
        pH_1_2.iloc[i,5] = 2
    elif pH_1_2.iloc[i,5]<60:
        pH_1_2.iloc[i,5] = 3

for i in range(len(pH_1_2.iloc[:,6])):
    if pH_1_2.iloc[i,6] >= 2.5:
        pH_1_2.iloc[i,6] = 1
    elif 2.0 <= pH_1_2.iloc[i,6] <2.5:
        pH_1_2.iloc[i,6] = 2
    elif pH_1_2.iloc[i,6] < 2.0:
        pH_1_2.iloc[i,6] = 3

for i in range(len(pH_1_2.iloc[:,1])):
    if pH_1_2.iloc[i,1] == 1 or pH_1_2.iloc[i,1] == 2:
        pH_1_2.iloc[i,1] = 1
    elif pH_1_2.iloc[i,1] == 3:
        pH_1_2.iloc[i,1] = 2
    elif pH_1_2.iloc[i,1]== 4:
        pH_1_2.iloc[i,1] = 3

#print(pH_1_2)


pd.set_option("display.unicode.east_asian_width",True)
pH_2 = pd.read_excel("..//data/pH_data_20230206.xlsx",sheet_name="pH_2",header=0,names=None,index_col=False)
#print(pH_2.columns)
pH_2.index = range(len(pH_2))
#print(pH_2)
#归一化
X = np.array(pH_2.iloc[:,2:4])
# print(X)
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(X)
# print(X_minMax)
F_scaler = pd.DataFrame(X_minMax)
F_scaler.columns = ['$6MWD$','$NT$-$proBNP$']
# print(F_scaler)
# print(pH_2.iloc[:,[0,6]])
pH_2_1 = pd.concat([pH_2.iloc[:,0],F_scaler,pH_2.iloc[:,[1,4]]], axis=1, ignore_index=True)
pH_2_1.columns = ['$number$','$6MWD$','$NT$-$proBNP$','$WHO$ $FC$','$RS$']
print(pH_2_1)

#离散化，按照标准离散化
pH_2_2 = pd.read_excel("..//data/pH_data_20230206.xlsx",sheet_name='pH_2',header=0,names=None,index_col=False)
pH_2_2.index = range(len(pH_2_2))
#print(pH_2_2.columns)
for i in range(len(pH_2_2.iloc[:,2])):
    if pH_2_2.iloc[i,2] > 440:
        pH_2_2.iloc[i,2] = 1
    elif 165 <= pH_2_2.iloc[i,2] <= 440:
        pH_2_2.iloc[i,2] = 2
    elif pH_2_2.iloc[i,2] < 165:
        pH_2_2.iloc[i,2] = 3

for i in range(len(pH_2_2.iloc[:,3])):
    if pH_2_2.iloc[i,3] < 300:
        pH_2_2.iloc[i,3] = 1
    elif 300 <= pH_2_2.iloc[i,3] <= 1400:
        pH_2_2.iloc[i,3] = 2
    elif pH_2_2.iloc[i,3]>1400:
        pH_2_2.iloc[i,3] = 3

# for i in range(len(pH_1_2.iloc[:,3])):
#     if pH_1_2.iloc[i,3] < 8:
#         pH_1_2.iloc[i,3] = 1
#     elif 8 <= pH_1_2.iloc[i,3] <= 14:
#         pH_1_2.iloc[i,3] = 2
#     elif 14 < pH_1_2.iloc[i,3]:
#          pH_1_2.iloc[i,3] = 3
#
# for i in range(len(pH_1_2.iloc[:,4])):
#     if pH_1_2.iloc[i,4] > 65:
#         pH_1_2.iloc[i,4] = 1
#     elif 60 <= pH_1_2.iloc[i,4] <= 65:
#         pH_1_2.iloc[i,4] = 2
#     elif pH_1_2.iloc[i,4]<60:
#         pH_1_2.iloc[i,4] = 3
#
# for i in range(len(pH_1_2.iloc[:,5])):
#     if pH_1_2.iloc[i,5] >= 2.5:
#         pH_1_2.iloc[i,5] = 1
#     elif 2.0 <= pH_1_2.iloc[i,5] <2.5:
#         pH_1_2.iloc[i,5] = 2
#     elif pH_1_2.iloc[i,5] < 2.0:
#         pH_1_2.iloc[i,5] = 3
#
for i in range(len(pH_2_2.iloc[:,1])):
    if pH_2_2.iloc[i,1] == 1 or pH_2_2.iloc[i,1] == 2:
        pH_2_2.iloc[i,1] = 1
    elif pH_2_2.iloc[i,1] == 3:
        pH_2_2.iloc[i,1] = 2
    elif pH_2_2.iloc[i,1]== 4:
        pH_2_2.iloc[i,1] = 3
#
#print(pH_2_2)
#
#
#
pd.set_option("display.unicode.east_asian_width",True)
pH_3 = pd.read_excel("..//data/pH_data_20230206.xlsx",sheet_name="pH_3",header=0,names=None,index_col=False)
#print(pH_3.columns)
pH_3.index = range(len(pH_3))
#print(type(pH_3.iloc[3,9]))
# for i in range(len(pH_3.iloc[:,10])):
#     if pH_3.iloc[i,10] == '＞50%':
#         pH_3.iloc[i,10] = 1
#     elif pH_3.iloc[i,10] == '＜50%':
#         pH_3.iloc[i,10] = 2
# #print(pH_3.iloc[:,9])
#归一化
X = np.array(pH_3.iloc[:,[2,3,4,5,6,7,9,10]])
# print(X)
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(X)
# print(X_minMax)
F_scaler = pd.DataFrame(X_minMax)
F_scaler.columns = ['$6MWD$','$NT$-$proBNP$','$sPAP_{echo}$','$TRV_{max}$','$TAPSE$','TAPSE/sPAP','$RAA$','$RVD$']
# print(F_scaler)
# print(pH_2.iloc[:,[0,6]])
pH_3_1 = pd.concat([pH_3.iloc[:,0],F_scaler,pH_3.iloc[:,[1,8,11,12]]], axis=1, ignore_index=True)
pH_3_1.columns = ['$number$','$6MWD$','$NT$-$proBNP$','$sPAP_{echo}$','$TRV_{max}$','$TAPSE$','TAPSE/sPAP','$RAA$','$RVD$','$WHO$ $FC$','$IVC-CI$','$CE$','$RS$']
print(pH_3_1)
#
#离散化，按照标准离散化
pH_3_2 = pd.read_excel("..//data/pH_data_20230206.xlsx",sheet_name='pH_3',header=0,names=None,index_col=False)
pH_3_2.index = range(len(pH_3_2))
# for i in range(len(pH_3_2.iloc[:,10])):
#     if pH_3_2.iloc[i,10] == '＞50%':
#         pH_3_2.iloc[i,10] = 1
#     elif pH_3_2.iloc[i,10] == '＜50%':
#         pH_3_2.iloc[i,10] = 2
#print(pH_3_2.columns)
for i in range(len(pH_3_2.iloc[:,2])):
    if pH_3_2.iloc[i,2] > 440:
        pH_3_2.iloc[i,2] = 1
    elif 165 <= pH_3_2.iloc[i,2] <= 440:
        pH_3_2.iloc[i,2] = 2
    elif pH_3_2.iloc[i,2] < 165:
        pH_3_2.iloc[i,2] = 3

for i in range(len(pH_3_2.iloc[:,3])):
    if pH_3_2.iloc[i,3] < 300:
        pH_3_2.iloc[i,3] = 1
    elif 300 <= pH_3_2.iloc[i,3] <= 1400:
        pH_3_2.iloc[i,3] = 2
    elif pH_3_2.iloc[i,3]>1400:
        pH_3_2.iloc[i,3] = 3

# for i in range(len(pH_3_2.iloc[:,4])):
#     if pH_3_2.iloc[i,4] < 8:
#         pH_3_2.iloc[i,4] = 1
#     elif 8 <= pH_3_2.iloc[i,4] <= 14:
#         pH_3_2.iloc[i,4] = 2
#     elif 14 < pH_3_2.iloc[i,4]:
#          pH_3_2.iloc[i,4] = 3

# for i in range(len(pH_3_2.iloc[:,5])):
#     if pH_3_2.iloc[i,5] > 65:
#         pH_3_2.iloc[i,5] = 1
#     elif 60 <= pH_3_2.iloc[i,5] <= 65:
#         pH_3_2.iloc[i,5] = 2
#     elif pH_3_2.iloc[i,5]<60:
#         pH_3_2.iloc[i,5] = 3

# for i in range(len(pH_3_2.iloc[:,6])):
#     if pH_3_2.iloc[i,6] >= 2.5:
#         pH_3_2.iloc[i,6] = 1
#     elif 2.0 <= pH_3_2.iloc[i,6] <2.5:
#         pH_3_2.iloc[i,6] = 2
#     elif pH_3_2.iloc[i,6] < 2.0:
#         pH_3_2.iloc[i,6] = 3

for i in range(len(pH_3_2.iloc[:,1])):
    if pH_3_2.iloc[i,1] == 1 or pH_3_2.iloc[i,1] == 2:
        pH_3_2.iloc[i,1] = 1
    elif pH_3_2.iloc[i,1] == 3:
        pH_3_2.iloc[i,1] = 2
    elif pH_3_2.iloc[i,1]== 4:
        pH_3_2.iloc[i,1] = 3
#
X = np.array(pH_3_2.iloc[:,[4,5,6,7,9,10]])
# print(X)
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(X)
# print(X_minMax)
F_scaler = pd.DataFrame(X_minMax)
F_scaler.columns = ['$sPAP_{echo}$','$TRV_{max}$','$TAPSE$','TAPSE/sPAP','$RAA$','$RVD$']
# print(F_scaler)
# print(pH_2.iloc[:,[0,6]])
pH_3_3 = pd.concat([pH_3_2.iloc[:,0],pH_3_2.iloc[:,[2,3]],F_scaler,pH_3_2.iloc[:,[1,8,11,12]]], axis=1, ignore_index=True)
pH_3_3.columns = ['$number$','$6MWD$','$NT$-$proBNP$','$sPAP_{echo}$','$TRV_{max}$','$TAPSE$','TAPSE/sPAP','$RAA$','$RVD$','$WHO$ $FC$','$IVC-CI$','$CE$','$RS$']
# print(pH_3_1)
#print(pH_3_2)

# pd.set_option("display.unicode.east_asian_width",True)
# pH_4 = pd.read_excel("..//data/pH_risk_level_5.xlsx",sheet_name="pH_4",header=0,names=None,index_col=False)
# #print(pH_4.columns)
# pH_4.index = range(len(pH_4))
#
# #归一化
# X = np.array(pH_4.iloc[:,[2,3,4,5,6,7,8,9,10,11,12]])
# # print(X)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_minMax = min_max_scaler.fit_transform(X)
# # print(X_minMax)
# F_scaler = pd.DataFrame(X_minMax)
# F_scaler.columns = ['$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']
# # print(F_scaler)
# # print(pH_2.iloc[:,[0,6]])
# pH_4_1 = pd.concat([pH_4.iloc[:,0],F_scaler,pH_4.iloc[:,[1,13]]], axis=1, ignore_index=True)
# pH_4_1.columns = ['$number$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$','$WHO$ $FC$','$RS$']
# #print(pH_4_1)
# #
# #离散化，按照标准离散化
# pH_4_2 = pd.read_excel("..//data/pH_risk_level_5.xlsx",sheet_name='pH_4',header=0,names=None,index_col=False)
# pH_4_2.index = range(len(pH_4_2))
# #print(pH_4_2.columns)
# for i in range(len(pH_4_2.iloc[:,2])):
#     if pH_4_2.iloc[i,2] > 440:
#         pH_4_2.iloc[i,2] = 1
#     elif 165 <= pH_4_2.iloc[i,2] <= 440:
#         pH_4_2.iloc[i,2] = 2
#     elif pH_4_2.iloc[i,2] < 165:
#         pH_4_2.iloc[i,2] = 3
#
# for i in range(len(pH_4_2.iloc[:,3])):
#     if pH_4_2.iloc[i,3] < 300:
#         pH_4_2.iloc[i,3] = 1
#     elif 300 <= pH_4_2.iloc[i,3] <= 1400:
#         pH_4_2.iloc[i,3] = 2
#     elif pH_4_2.iloc[i,3]>1400:
#         pH_4_2.iloc[i,3] = 3
#
# for i in range(len(pH_4_2.iloc[:,4])):
#     if pH_4_2.iloc[i,4] < 8:
#         pH_4_2.iloc[i,4] = 1
#     elif 8 <= pH_4_2.iloc[i,4] <= 14:
#         pH_4_2.iloc[i,4] = 2
#     elif 14 < pH_4_2.iloc[i,4]:
#          pH_4_2.iloc[i,4] = 3
#
# for i in range(len(pH_4_2.iloc[:,5])):
#     if pH_4_2.iloc[i,5] > 65:
#         pH_4_2.iloc[i,5] = 1
#     elif 60 <= pH_4_2.iloc[i,5] <= 65:
#         pH_4_2.iloc[i,5] = 2
#     elif pH_4_2.iloc[i,5]<60:
#         pH_4_2.iloc[i,5] = 3
#
# for i in range(len(pH_4_2.iloc[:,6])):
#     if pH_4_2.iloc[i,6] >= 2.5:
#         pH_4_2.iloc[i,6] = 1
#     elif 2.0 <= pH_4_2.iloc[i,6] <2.5:
#         pH_4_2.iloc[i,6] = 2
#     elif pH_4_2.iloc[i,6] < 2.0:
#         pH_4_2.iloc[i,6] = 3
#
# for i in range(len(pH_4_2.iloc[:,1])):
#     if pH_4_2.iloc[i,1] == 1 or pH_4_2.iloc[i,1] == 2:
#         pH_4_2.iloc[i,1] = 1
#     elif pH_4_2.iloc[i,1] == 3:
#         pH_4_2.iloc[i,1] = 2
#     elif pH_4_2.iloc[i,1]== 4:
#         pH_4_2.iloc[i,1] = 3
# #
# X = np.array(pH_4_2.iloc[:,[7,8,9,10,11,12]])
# # print(X)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_minMax = min_max_scaler.fit_transform(X)
# # print(X_minMax)
# F_scaler = pd.DataFrame(X_minMax)
# F_scaler.columns = ['$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']
# # print(F_scaler)
# # print(pH_2.iloc[:,[0,6]])
# pH_4_3 = pd.concat([pH_4_2.iloc[:,0],F_scaler,pH_4_2.iloc[:,[1,2,3,4,5,6,13]]], axis=1, ignore_index=True)
# pH_4_3.columns = ['$number$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$','$WHO$ $FC$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$RS$']
# #print(pH_4_1)
# #print(pH_4_2)
#
# #
writer = pd.ExcelWriter('..//data/pH_data_2.xlsx')
pH_1_1.to_excel(writer,sheet_name='pH_1_1')
pH_1_2.to_excel(writer,sheet_name='pH_1_2')
pH_2_1.to_excel(writer,sheet_name='pH_2_1')
pH_2_2.to_excel(writer,sheet_name='pH_2_2')
pH_3_1.to_excel(writer,sheet_name='pH_3_1')
pH_3_3.to_excel(writer,sheet_name='pH_3_2')
# pH_4_1.to_excel(writer,sheet_name='pH_4_1')
# pH_4_3.to_excel(writer,sheet_name='pH_4_2')
writer.save()
