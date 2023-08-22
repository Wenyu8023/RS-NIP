#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler  # 随机重复采样
#from imblearn.over_sampling import SMOTE  # 选取少数类样本插值采样
#from imblearn.over_sampling import BorderlineSMOTE  # 边界类样本采样
#from imblearn.over_sampling import ADASYN  # 自适应合成抽样
# for a in [1,2,3]:
#     writer = pd.ExcelWriter("..//data/pH_sample_" + "%d" % a + ".xlsx")
#     for b in [1,2]:
#         rL_pH = pd.read_excel("..//data/pH_risk_level_"+"%d"%a+".xlsx",sheet_name="rL_pH_"+"%a"%a+"_"+"%d"%b,header=0,names=None,index_col=0)
#         #print(rL_pH)
#         # data_4 = rL_pH.iloc[:,0:-1]
#         #print(data_4)
#
#         X = np.array(rL_pH.iloc[:,0:-1])
#         y = np.array(rL_pH.iloc[:,-1])
#
#         #ros = RandomOverSampler(sampling_strategy={0: 700,1:200,2:150 },random_state=0)
#         ros = RandomOverSampler(random_state=0)
#         X_resampled, y_resampled = ros.fit_resample(X, y)
#         X_1 = pd.DataFrame(X_resampled)
#         y_1 = pd.DataFrame(y_resampled)
#         data_5 = pd.concat([X_1,y_1], axis=1, ignore_index=True)
#         #print(data_5)
#         data_5.to_excel(writer,sheet_name="rL_pH_"+"%d"%a+"_"+"%d"%b)
#         # rL_pH_1_2.to_excel(writer,sheet_name='rL_pH_1_2')
#     writer.save()
        #print(data_5.iloc[:,-1].value_counts())


writer = pd.ExcelWriter("..//data/pH_sample" + ".xlsx")
for i in [1,2,3]:
    for j in [1,2]:
        rL_pH = pd.read_excel("..//data/pH_data_2" + ".xlsx",sheet_name="pH_" + "%d"%i + "_" + "%d"%j, header=0,names=None,index_col=0)
        rl = rL_pH.iloc[:,1:]
        X = np.array(rL_pH.iloc[:,1:-1])
        y = np.array(rL_pH.iloc[:,-1])
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        X_1 = pd.DataFrame(X_resampled)
        y_1 = pd.DataFrame(y_resampled)
        data = pd.concat([X_1,y_1], axis=1, ignore_index=True)
        data.columns = rl.columns
        data.to_excel(writer,sheet_name="pH_" + "%d"%i + "_" + "%d"%j)
writer.save()
