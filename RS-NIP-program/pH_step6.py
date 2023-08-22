#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

def Friedman(n, k, data_matrix):
    #     Friedman检验
    #     参数：数据集个数n, 算法种数k, 排序矩阵data_matrix
    #     返回值是Tf

    # 计算每个算法的平均序值，即求每一列的排序均值
    hang, lie = data_matrix.shape  # 查看数据形状
    XuZhi_mean = list()
    for i in range(lie):  # 计算平均序值
        XuZhi_mean.append(data_matrix[:, i].mean())

    sum_mean = np.array(XuZhi_mean)  # 转换数据结构方便下面运算
    ## 计算总的排序和即西伽马ri^2
    sum_ri2_mean = (sum_mean ** 2).sum()
    # 计算Tf
    result_Tx2 = (12 * n) * (sum_ri2_mean - ((k * (k + 1) ** 2) / 4)) / (k * (k + 1))
    result_Tf = (n - 1) * result_Tx2 / (n * (k - 1) - result_Tx2)

    return result_Tf

rL_Accuracy = pd.read_excel("..//data/new/data_result_train.xlsx",sheet_name='Accuracy',header=0,names=None,index_col=0)
rL_Fscore = pd.read_excel("..//data/new/data_result_train.xlsx",sheet_name='F-score',header=0,names=None,index_col=0)
rL_Precision = pd.read_excel("..//data/new/data_result_train.xlsx",sheet_name='Precision',header=0,names=None,index_col=0)
rL_Recall = pd.read_excel("..//data/new/data_result_train.xlsx",sheet_name='Recall',header=0,names=None,index_col=0)

#rL_Accuracy = pd.read_excel("..//data/new/data_result_test.xlsx",sheet_name='Accuracy',header=0,names=None,index_col=0)
#rL_Fscore = pd.read_excel("..//data/new/data_result_test.xlsx",sheet_name='F-score',header=0,names=None,index_col=0)
#rL_Precision = pd.read_excel("..//data/new/data_result_test.xlsx",sheet_name='Precision',header=0,names=None,index_col=0)
#rL_Recall = pd.read_excel("..//data/new/data_result_test.xlsx",sheet_name='Recall',header=0,names=None,index_col=0)

#rL_Accuracy = pd.read_excel("..//data/new/data_result_E_v.xlsx",sheet_name='Accuracy',header=0,names=None,index_col=0)
#rL_Fscore = pd.read_excel("..//data/new/data_result_E_v.xlsx",sheet_name='F-score',header=0,names=None,index_col=0)
#rL_Precision = pd.read_excel("..//data/new/data_result_E_v.xlsx",sheet_name='Precision',header=0,names=None,index_col=0)
#rL_Recall = pd.read_excel("..//data/new/data_result_E_v.xlsx",sheet_name='Recall',header=0,names=None,index_col=0)

#rL_Accuracy = pd.read_excel("..//data/new/data_result_train_F.xlsx",sheet_name='Accuracy',header=0,names=None,index_col=0)
#rL_Fscore = pd.read_excel("..//data/new/data_result_train_F.xlsx",sheet_name='F-score',header=0,names=None,index_col=0)
#rL_Precision = pd.read_excel("..//data/new/data_result_train_F.xlsx",sheet_name='Precision',header=0,names=None,index_col=0)
#rL_Recall = pd.read_excel("..//data/new/data_result_train_F.xlsx",sheet_name='Recall',header=0,names=None,index_col=0)

#rL_Accuracy = pd.read_excel("..//data/new/data_result_test_F.xlsx",sheet_name='Accuracy',header=0,names=None,index_col=0)
#rL_Fscore = pd.read_excel("..//data/new/data_result_test_F.xlsx",sheet_name='F-score',header=0,names=None,index_col=0)
#rL_Precision = pd.read_excel("..//data/new/data_result_test_F.xlsx",sheet_name='Precision',header=0,names=None,index_col=0)
#rL_Recall = pd.read_excel("..//data/new/data_result_test_F.xlsx",sheet_name='Recall',header=0,names=None,index_col=0)

#rL_Accuracy = pd.read_excel("..//data/new/data_result_E_v_F.xlsx",sheet_name='Accuracy',header=0,names=None,index_col=0)
#rL_Fscore = pd.read_excel("..//data/new/data_result_E_v_F.xlsx",sheet_name='F-score',header=0,names=None,index_col=0)
#rL_Precision = pd.read_excel("..//data/new/data_result_E_v_F.xlsx",sheet_name='Precision',header=0,names=None,index_col=0)
#rL_Recall = pd.read_excel("..//data/new/data_result_E_v_F.xlsx",sheet_name='Recall',header=0,names=None,index_col=0)



# #Acuracy
# rL_Accuracy = rL_Accuracy.drop(["DS 4.1"], axis=1)
# rL_Accuracy = rL_Accuracy.drop(["DS 4.2"], axis=1)
# #print(rL_Accuracy)

rL_Accuracy_1_1 = rL_Accuracy["DS 1.1"].str.split('-', expand=True)
rL_Accuracy_1_1.columns = ["DS 1.1","DS 1.1_"]
rL_Accuracy_1_2 = rL_Accuracy["DS 1.2"].str.split('-', expand=True)
rL_Accuracy_1_2.columns = ["DS 1.2","DS 1.2_"]
rL_Accuracy_2_1 = rL_Accuracy["DS 2.1"].str.split('-', expand=True)
rL_Accuracy_2_1.columns = ["DS 2.1","DS 2.1_"]
rL_Accuracy_2_2 = rL_Accuracy["DS 2.2"].str.split('-', expand=True)
rL_Accuracy_2_2.columns = ["DS 2.2","DS 2.2_"]
rL_Accuracy_3_1 = rL_Accuracy["DS 3.1"].str.split('-', expand=True)
rL_Accuracy_3_1.columns = ["DS 3.1","DS 3.1_"]
rL_Accuracy_3_2 = rL_Accuracy["DS 3.2"].str.split('-', expand=True)
rL_Accuracy_3_2.columns = ["DS 3.2","DS 3.2_"]
#print(rL_Accuracy_1_1)
rL_Accuracy = rL_Accuracy.drop(["DS 1.1"], axis=1).join(rL_Accuracy_1_1)
rL_Accuracy = rL_Accuracy.drop(["DS 1.2"], axis=1).join(rL_Accuracy_1_2)
rL_Accuracy = rL_Accuracy.drop(["DS 2.1"], axis=1).join(rL_Accuracy_2_1)
rL_Accuracy = rL_Accuracy.drop(["DS 2.2"], axis=1).join(rL_Accuracy_2_2)
rL_Accuracy = rL_Accuracy.drop(["DS 3.1"], axis=1).join(rL_Accuracy_3_1)
rL_Accuracy = rL_Accuracy.drop(["DS 3.2"], axis=1).join(rL_Accuracy_3_2)
#print(rL_Accuracy)

rL_Accuracy_R = rL_Accuracy.iloc[:,[0,2,4,6,8,10]].copy(deep=True)
#print(rL_Accuracy_R)
rL_Accuracy_R.iloc[0,:] = rL_Accuracy.iloc[0,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Accuracy_R.iloc[1,:] = rL_Accuracy.iloc[1,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Accuracy_R.iloc[2,:] = rL_Accuracy.iloc[2,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Accuracy_R.iloc[3,:] = rL_Accuracy.iloc[3,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Accuracy_R.iloc[4,:] = rL_Accuracy.iloc[4,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Accuracy_R.iloc[5,:] = rL_Accuracy.iloc[5,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)

#print(rL_Accuracy_R)
for col in rL_Accuracy:
    rL_Accuracy[col] = pd.to_numeric(rL_Accuracy[col],errors='coerce')

rL_Accuracy.loc['Average'] = 0

#print(rL_Accuracy)
for i in range(len(rL_Accuracy.columns)):
    rL_Accuracy.iloc[6,i] = np.mean([rL_Accuracy.iloc[:-1,i]],1)
#print(rL_Accuracy)

rL_Accuracy_R.loc['Average'] = 0

print(rL_Accuracy_R)
for i in range(len(rL_Accuracy_R.columns)):
    rL_Accuracy_R.iloc[6,i] = np.mean([rL_Accuracy_R.iloc[:-1,i]],1)
    rL_Accuracy_R.iloc[6, i] = rL_Accuracy_R.iloc[6,i][0]
print(rL_Accuracy_R)
# rL_Accuracy['DS 1.1']=rL_Accuracy['DS 1.1']+" ± "+rL_Accuracy['DS 1.1_']
rL_Accuracy_R=rL_Accuracy_R.astype(float)
rL_Accuracy_R=rL_Accuracy_R.apply(lambda x:round(x,2))
print(rL_Accuracy_R)

print(rL_Accuracy_R)
rL_Accuracy=rL_Accuracy.apply(lambda x:round(x,4))
#print(rL_Accuracy)
rL_Accuracy=rL_Accuracy.astype(str)
#print(type(rL_Accuracy.iloc[0,0]))
print(rL_Accuracy)
rL_Accuracy_C = rL_Accuracy_R.iloc[:,:].copy(deep=True)
print(rL_Accuracy_C)
rL_Accuracy_C["DS 1.1"]=rL_Accuracy.apply(lambda x: x["DS 1.1"] + "±" + x["DS 1.1_"],axis=1)
rL_Accuracy_C["DS 1.2"]=rL_Accuracy.apply(lambda x: x["DS 1.2"] + "±" + x["DS 1.2_"],axis=1)
rL_Accuracy_C["DS 2.1"]=rL_Accuracy.apply(lambda x: x["DS 2.1"] + "±" + x["DS 2.1_"],axis=1)
rL_Accuracy_C["DS 2.2"]=rL_Accuracy.apply(lambda x: x["DS 2.2"] + "±" + x["DS 2.2_"],axis=1)
rL_Accuracy_C["DS 3.1"]=rL_Accuracy.apply(lambda x: x["DS 3.1"] + "±" + x["DS 3.1_"],axis=1)
rL_Accuracy_C["DS 3.2"]=rL_Accuracy.apply(lambda x: x["DS 3.2"] + "±" + x["DS 3.2_"],axis=1)
print(rL_Accuracy_C)
rL_Accuracy_R.columns = ["R 1.1","R 1.2","R 2.1","R 2.2","R 3.1","R 3.2"]
rL_Accuracy_CC = pd.concat([rL_Accuracy_C.iloc[:,0],rL_Accuracy_R.iloc[:,0],rL_Accuracy_C.iloc[:,1],rL_Accuracy_R.iloc[:,1],rL_Accuracy_C.iloc[:,2],rL_Accuracy_R.iloc[:,2],rL_Accuracy_C.iloc[:,3],rL_Accuracy_R.iloc[:,3],rL_Accuracy_C.iloc[:,4],rL_Accuracy_R.iloc[:,4],rL_Accuracy_C.iloc[:,5],rL_Accuracy_R.iloc[:,5]],axis=1)
print(rL_Accuracy_CC)

#rL_Accuracy_C.iloc[:,0]=rL_Accuracy.iloc[:,0] + "±" + rL_Accuracy.iloc[:,1]
rL_Accuracy_w_t_l = pd.DataFrame(index=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"),columns=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"))
print(rL_Accuracy_w_t_l)
rL_Accuracy_RR = rL_Accuracy_R.drop(["Average"], axis=0).copy(deep=True)
print(rL_Accuracy_RR)
for m in range(len(rL_Accuracy_RR.columns)):
    for n in range(len(rL_Accuracy_RR.columns)):
        win = 0
        tie = 0
        los = 0
        for t in range(len(rL_Accuracy_RR)):
            if rL_Accuracy_RR.iloc[t,m] < rL_Accuracy_RR.iloc[t,n]:
                win = win + 1
            elif rL_Accuracy_RR.iloc[t,m] == rL_Accuracy_RR.iloc[t,n]:
                tie = tie + 1
            else:
                los = los + 1
        rL_Accuracy_w_t_l.iloc[m,n] = "{0}/{1}/{2}".format(win,tie,los)
#
print(rL_Accuracy_w_t_l)

Accuracy_data = np.array([list(rL_Accuracy_RR.iloc[0,:]), list(rL_Accuracy_RR.iloc[1,:]), list(rL_Accuracy_RR.iloc[2,:]), list(rL_Accuracy_RR.iloc[3,:]), list(rL_Accuracy_RR.iloc[4,:]),list(rL_Accuracy_RR.iloc[5,:])])
Accuracy_Tf = Friedman(6,6,Accuracy_data)
print(Accuracy_Tf)

#F-score

# rL_Fscore = rL_Fscore.drop(["DS 4.1"], axis=1)
# rL_Fscore = rL_Fscore.drop(["DS 4.2"], axis=1)
#print(rL_Fscore)

rL_Fscore_1_1 = rL_Fscore["DS 1.1"].str.split('-', expand=True)
rL_Fscore_1_1.columns = ["DS 1.1","DS 1.1_"]
rL_Fscore_1_2 = rL_Fscore["DS 1.2"].str.split('-', expand=True)
rL_Fscore_1_2.columns = ["DS 1.2","DS 1.2_"]
rL_Fscore_2_1 = rL_Fscore["DS 2.1"].str.split('-', expand=True)
rL_Fscore_2_1.columns = ["DS 2.1","DS 2.1_"]
rL_Fscore_2_2 = rL_Fscore["DS 2.2"].str.split('-', expand=True)
rL_Fscore_2_2.columns = ["DS 2.2","DS 2.2_"]
rL_Fscore_3_1 = rL_Fscore["DS 3.1"].str.split('-', expand=True)
rL_Fscore_3_1.columns = ["DS 3.1","DS 3.1_"]
rL_Fscore_3_2 = rL_Fscore["DS 3.2"].str.split('-', expand=True)
rL_Fscore_3_2.columns = ["DS 3.2","DS 3.2_"]
#print(rL_Fscore_1_1)
rL_Fscore = rL_Fscore.drop(["DS 1.1"], axis=1).join(rL_Fscore_1_1)
rL_Fscore = rL_Fscore.drop(["DS 1.2"], axis=1).join(rL_Fscore_1_2)
rL_Fscore = rL_Fscore.drop(["DS 2.1"], axis=1).join(rL_Fscore_2_1)
rL_Fscore = rL_Fscore.drop(["DS 2.2"], axis=1).join(rL_Fscore_2_2)
rL_Fscore = rL_Fscore.drop(["DS 3.1"], axis=1).join(rL_Fscore_3_1)
rL_Fscore = rL_Fscore.drop(["DS 3.2"], axis=1).join(rL_Fscore_3_2)
#print(rL_Fscore)

rL_Fscore_R = rL_Fscore.iloc[:,[0,2,4,6,8,10]].copy(deep=True)
#print(rL_Fscore_R)
rL_Fscore_R.iloc[0,:] = rL_Fscore.iloc[0,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Fscore_R.iloc[1,:] = rL_Fscore.iloc[1,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Fscore_R.iloc[2,:] = rL_Fscore.iloc[2,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Fscore_R.iloc[3,:] = rL_Fscore.iloc[3,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Fscore_R.iloc[4,:] = rL_Fscore.iloc[4,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Fscore_R.iloc[5,:] = rL_Fscore.iloc[5,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)

#print(rL_Fscore_R)
for col in rL_Fscore:
    rL_Fscore[col] = pd.to_numeric(rL_Fscore[col],errors='coerce')

rL_Fscore.loc['Average'] = 0

#print(rL_Fscore)
for i in range(len(rL_Fscore.columns)):
    rL_Fscore.iloc[6,i] = np.mean([rL_Fscore.iloc[:-1,i]],1)
#print(rL_Fscore)

rL_Fscore_R.loc['Average'] = 0

print(rL_Fscore_R)
for i in range(len(rL_Fscore_R.columns)):
    rL_Fscore_R.iloc[6,i] = np.mean([rL_Fscore_R.iloc[:-1,i]],1)
    rL_Fscore_R.iloc[6, i] = rL_Fscore_R.iloc[6,i][0]
print(rL_Fscore_R)
# rL_Fscore['DS 1.1']=rL_Fscore['DS 1.1']+" ± "+rL_Fscore['DS 1.1_']
rL_Fscore_R=rL_Fscore_R.astype(float)
rL_Fscore_R=rL_Fscore_R.apply(lambda x:round(x,2))
print(rL_Fscore_R)

print(rL_Fscore_R)
rL_Fscore=rL_Fscore.apply(lambda x:round(x,4))
#print(rL_Fscore)
rL_Fscore=rL_Fscore.astype(str)
#print(type(rL_Fscore.iloc[0,0]))
print(rL_Fscore)
rL_Fscore_C = rL_Fscore_R.iloc[:,:].copy(deep=True)
print(rL_Fscore_C)
rL_Fscore_C["DS 1.1"]=rL_Fscore.apply(lambda x: x["DS 1.1"] + "±" + x["DS 1.1_"],axis=1)
rL_Fscore_C["DS 1.2"]=rL_Fscore.apply(lambda x: x["DS 1.2"] + "±" + x["DS 1.2_"],axis=1)
rL_Fscore_C["DS 2.1"]=rL_Fscore.apply(lambda x: x["DS 2.1"] + "±" + x["DS 2.1_"],axis=1)
rL_Fscore_C["DS 2.2"]=rL_Fscore.apply(lambda x: x["DS 2.2"] + "±" + x["DS 2.2_"],axis=1)
rL_Fscore_C["DS 3.1"]=rL_Fscore.apply(lambda x: x["DS 3.1"] + "±" + x["DS 3.1_"],axis=1)
rL_Fscore_C["DS 3.2"]=rL_Fscore.apply(lambda x: x["DS 3.2"] + "±" + x["DS 3.2_"],axis=1)
print(rL_Fscore_C)
rL_Fscore_R.columns = ["R 1.1","R 1.2","R 2.1","R 2.2","R 3.1","R 3.2"]
rL_Fscore_CC = pd.concat([rL_Fscore_C.iloc[:,0],rL_Fscore_R.iloc[:,0],rL_Fscore_C.iloc[:,1],rL_Fscore_R.iloc[:,1],rL_Fscore_C.iloc[:,2],rL_Fscore_R.iloc[:,2],rL_Fscore_C.iloc[:,3],rL_Fscore_R.iloc[:,3],rL_Fscore_C.iloc[:,4],rL_Fscore_R.iloc[:,4],rL_Fscore_C.iloc[:,5],rL_Fscore_R.iloc[:,5]],axis=1)
print(rL_Fscore_CC)

#rL_Fscore_C.iloc[:,0]=rL_Fscore.iloc[:,0] + "±" + rL_Fscore.iloc[:,1]
rL_Fscore_w_t_l = pd.DataFrame(index=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"),columns=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"))
print(rL_Fscore_w_t_l)
rL_Fscore_RR = rL_Fscore_R.drop(["Average"], axis=0).copy(deep=True)
print(rL_Fscore_RR)
for m in range(len(rL_Fscore_RR.columns)):
    for n in range(len(rL_Fscore_RR.columns)):
        win = 0
        tie = 0
        los = 0
        for t in range(len(rL_Fscore_RR)):
            if rL_Fscore_RR.iloc[t,m] < rL_Fscore_RR.iloc[t,n]:
                win = win + 1
            elif rL_Fscore_RR.iloc[t,m] == rL_Fscore_RR.iloc[t,n]:
                tie = tie + 1
            else:
                los = los + 1
        rL_Fscore_w_t_l.iloc[m,n] = "{0}/{1}/{2}".format(win,tie,los)
#
print(rL_Fscore_w_t_l)

Fscore_data = np.array([list(rL_Fscore_RR.iloc[0,:]), list(rL_Fscore_RR.iloc[1,:]), list(rL_Fscore_RR.iloc[2,:]), list(rL_Fscore_RR.iloc[3,:]), list(rL_Fscore_RR.iloc[4,:]),list(rL_Fscore_RR.iloc[5,:])])
Fscore_Tf = Friedman(6,6,Fscore_data)
print(Fscore_Tf)

#Precision

# rL_Precision = rL_Precision.drop(["DS 4.1"], axis=1)
# rL_Precision = rL_Precision.drop(["DS 4.2"], axis=1)
#print(rL_Precision)

rL_Precision_1_1 = rL_Precision["DS 1.1"].str.split('-', expand=True)
rL_Precision_1_1.columns = ["DS 1.1","DS 1.1_"]
rL_Precision_1_2 = rL_Precision["DS 1.2"].str.split('-', expand=True)
rL_Precision_1_2.columns = ["DS 1.2","DS 1.2_"]
rL_Precision_2_1 = rL_Precision["DS 2.1"].str.split('-', expand=True)
rL_Precision_2_1.columns = ["DS 2.1","DS 2.1_"]
rL_Precision_2_2 = rL_Precision["DS 2.2"].str.split('-', expand=True)
rL_Precision_2_2.columns = ["DS 2.2","DS 2.2_"]
rL_Precision_3_1 = rL_Precision["DS 3.1"].str.split('-', expand=True)
rL_Precision_3_1.columns = ["DS 3.1","DS 3.1_"]
rL_Precision_3_2 = rL_Precision["DS 3.2"].str.split('-', expand=True)
rL_Precision_3_2.columns = ["DS 3.2","DS 3.2_"]
#print(rL_Precision_1_1)
rL_Precision = rL_Precision.drop(["DS 1.1"], axis=1).join(rL_Precision_1_1)
rL_Precision = rL_Precision.drop(["DS 1.2"], axis=1).join(rL_Precision_1_2)
rL_Precision = rL_Precision.drop(["DS 2.1"], axis=1).join(rL_Precision_2_1)
rL_Precision = rL_Precision.drop(["DS 2.2"], axis=1).join(rL_Precision_2_2)
rL_Precision = rL_Precision.drop(["DS 3.1"], axis=1).join(rL_Precision_3_1)
rL_Precision = rL_Precision.drop(["DS 3.2"], axis=1).join(rL_Precision_3_2)
#print(rL_Precision)

rL_Precision_R = rL_Precision.iloc[:,[0,2,4,6,8,10]].copy(deep=True)
#print(rL_Precision_R)
rL_Precision_R.iloc[0,:] = rL_Precision.iloc[0,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Precision_R.iloc[1,:] = rL_Precision.iloc[1,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Precision_R.iloc[2,:] = rL_Precision.iloc[2,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Precision_R.iloc[3,:] = rL_Precision.iloc[3,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Precision_R.iloc[4,:] = rL_Precision.iloc[4,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Precision_R.iloc[5,:] = rL_Precision.iloc[5,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)

#print(rL_Precision_R)
for col in rL_Precision:
    rL_Precision[col] = pd.to_numeric(rL_Precision[col],errors='coerce')

rL_Precision.loc['Average'] = 0

#print(rL_Precision)
for i in range(len(rL_Precision.columns)):
    rL_Precision.iloc[6,i] = np.mean([rL_Precision.iloc[:-1,i]],1)
#print(rL_Precision)

rL_Precision_R.loc['Average'] = 0

print(rL_Precision_R)
for i in range(len(rL_Precision_R.columns)):
    rL_Precision_R.iloc[6,i] = np.mean([rL_Precision_R.iloc[:-1,i]],1)
    rL_Precision_R.iloc[6, i] = rL_Precision_R.iloc[6,i][0]
print(rL_Precision_R)
# rL_Precision['DS 1.1']=rL_Precision['DS 1.1']+" ± "+rL_Precision['DS 1.1_']
rL_Precision_R=rL_Precision_R.astype(float)
rL_Precision_R=rL_Precision_R.apply(lambda x:round(x,2))
print(rL_Precision_R)

print(rL_Precision_R)
rL_Precision=rL_Precision.apply(lambda x:round(x,4))
#print(rL_Precision)
rL_Precision=rL_Precision.astype(str)
#print(type(rL_Precision.iloc[0,0]))
print(rL_Precision)
rL_Precision_C = rL_Precision_R.iloc[:,:].copy(deep=True)
print(rL_Precision_C)
rL_Precision_C["DS 1.1"]=rL_Precision.apply(lambda x: x["DS 1.1"] + "±" + x["DS 1.1_"],axis=1)
rL_Precision_C["DS 1.2"]=rL_Precision.apply(lambda x: x["DS 1.2"] + "±" + x["DS 1.2_"],axis=1)
rL_Precision_C["DS 2.1"]=rL_Precision.apply(lambda x: x["DS 2.1"] + "±" + x["DS 2.1_"],axis=1)
rL_Precision_C["DS 2.2"]=rL_Precision.apply(lambda x: x["DS 2.2"] + "±" + x["DS 2.2_"],axis=1)
rL_Precision_C["DS 3.1"]=rL_Precision.apply(lambda x: x["DS 3.1"] + "±" + x["DS 3.1_"],axis=1)
rL_Precision_C["DS 3.2"]=rL_Precision.apply(lambda x: x["DS 3.2"] + "±" + x["DS 3.2_"],axis=1)
print(rL_Precision_C)
rL_Precision_R.columns = ["R 1.1","R 1.2","R 2.1","R 2.2","R 3.1","R 3.2"]
rL_Precision_CC = pd.concat([rL_Precision_C.iloc[:,0],rL_Precision_R.iloc[:,0],rL_Precision_C.iloc[:,1],rL_Precision_R.iloc[:,1],rL_Precision_C.iloc[:,2],rL_Precision_R.iloc[:,2],rL_Precision_C.iloc[:,3],rL_Precision_R.iloc[:,3],rL_Precision_C.iloc[:,4],rL_Precision_R.iloc[:,4],rL_Precision_C.iloc[:,5],rL_Precision_R.iloc[:,5]],axis=1)
print(rL_Precision_CC)

#rL_Precision_C.iloc[:,0]=rL_Precision.iloc[:,0] + "±" + rL_Precision.iloc[:,1]
rL_Precision_w_t_l = pd.DataFrame(index=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"),columns=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"))
print(rL_Precision_w_t_l)
rL_Precision_RR = rL_Precision_R.drop(["Average"], axis=0).copy(deep=True)
print(rL_Precision_RR)
for m in range(len(rL_Precision_RR.columns)):
    for n in range(len(rL_Precision_RR.columns)):
        win = 0
        tie = 0
        los = 0
        for t in range(len(rL_Precision_RR)):
            if rL_Precision_RR.iloc[t,m] < rL_Precision_RR.iloc[t,n]:
                win = win + 1
            elif rL_Precision_RR.iloc[t,m] == rL_Precision_RR.iloc[t,n]:
                tie = tie + 1
            else:
                los = los + 1
        rL_Precision_w_t_l.iloc[m,n] = "{0}/{1}/{2}".format(win,tie,los)
#
print(rL_Precision_w_t_l)

Precision_data = np.array([list(rL_Precision_RR.iloc[0,:]), list(rL_Precision_RR.iloc[1,:]), list(rL_Precision_RR.iloc[2,:]), list(rL_Precision_RR.iloc[3,:]), list(rL_Precision_RR.iloc[4,:]),list(rL_Precision_RR.iloc[5,:])])
Precision_Tf = Friedman(6,6,Precision_data)
print(Precision_Tf)

#Recall

# rL_Recall = rL_Recall.drop(["DS 4.1"], axis=1)
# rL_Recall = rL_Recall.drop(["DS 4.2"], axis=1)
#print(rL_Recall)

rL_Recall_1_1 = rL_Recall["DS 1.1"].str.split('-', expand=True)
rL_Recall_1_1.columns = ["DS 1.1","DS 1.1_"]
rL_Recall_1_2 = rL_Recall["DS 1.2"].str.split('-', expand=True)
rL_Recall_1_2.columns = ["DS 1.2","DS 1.2_"]
rL_Recall_2_1 = rL_Recall["DS 2.1"].str.split('-', expand=True)
rL_Recall_2_1.columns = ["DS 2.1","DS 2.1_"]
rL_Recall_2_2 = rL_Recall["DS 2.2"].str.split('-', expand=True)
rL_Recall_2_2.columns = ["DS 2.2","DS 2.2_"]
rL_Recall_3_1 = rL_Recall["DS 3.1"].str.split('-', expand=True)
rL_Recall_3_1.columns = ["DS 3.1","DS 3.1_"]
rL_Recall_3_2 = rL_Recall["DS 3.2"].str.split('-', expand=True)
rL_Recall_3_2.columns = ["DS 3.2","DS 3.2_"]
#print(rL_Recall_1_1)
rL_Recall = rL_Recall.drop(["DS 1.1"], axis=1).join(rL_Recall_1_1)
rL_Recall = rL_Recall.drop(["DS 1.2"], axis=1).join(rL_Recall_1_2)
rL_Recall = rL_Recall.drop(["DS 2.1"], axis=1).join(rL_Recall_2_1)
rL_Recall = rL_Recall.drop(["DS 2.2"], axis=1).join(rL_Recall_2_2)
rL_Recall = rL_Recall.drop(["DS 3.1"], axis=1).join(rL_Recall_3_1)
rL_Recall = rL_Recall.drop(["DS 3.2"], axis=1).join(rL_Recall_3_2)
#print(rL_Recall)

rL_Recall_R = rL_Recall.iloc[:,[0,2,4,6,8,10]].copy(deep=True)
#print(rL_Recall_R)
rL_Recall_R.iloc[0,:] = rL_Recall.iloc[0,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Recall_R.iloc[1,:] = rL_Recall.iloc[1,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Recall_R.iloc[2,:] = rL_Recall.iloc[2,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Recall_R.iloc[3,:] = rL_Recall.iloc[3,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Recall_R.iloc[4,:] = rL_Recall.iloc[4,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)
rL_Recall_R.iloc[5,:] = rL_Recall.iloc[5,[0,2,4,6,8,10]].rank(method='average',ascending=False).copy(deep=True)

#print(rL_Recall_R)
for col in rL_Recall:
    rL_Recall[col] = pd.to_numeric(rL_Recall[col],errors='coerce')

rL_Recall.loc['Average'] = 0

#print(rL_Recall)
for i in range(len(rL_Recall.columns)):
    rL_Recall.iloc[6,i] = np.mean([rL_Recall.iloc[:-1,i]],1)
#print(rL_Recall)

rL_Recall_R.loc['Average'] = 0

print(rL_Recall_R)
for i in range(len(rL_Recall_R.columns)):
    rL_Recall_R.iloc[6,i] = np.mean([rL_Recall_R.iloc[:-1,i]],1)
    rL_Recall_R.iloc[6, i] = rL_Recall_R.iloc[6,i][0]
print(rL_Recall_R)
# rL_Recall['DS 1.1']=rL_Recall['DS 1.1']+" ± "+rL_Recall['DS 1.1_']
rL_Recall_R=rL_Recall_R.astype(float)
rL_Recall_R=rL_Recall_R.apply(lambda x:round(x,2))
print(rL_Recall_R)

print(rL_Recall_R)
rL_Recall=rL_Recall.apply(lambda x:round(x,4))
#print(rL_Recall)
rL_Recall=rL_Recall.astype(str)
#print(type(rL_Recall.iloc[0,0]))
print(rL_Recall)
rL_Recall_C = rL_Recall_R.iloc[:,:].copy(deep=True)
print(rL_Recall_C)
rL_Recall_C["DS 1.1"]=rL_Recall.apply(lambda x: x["DS 1.1"] + "±" + x["DS 1.1_"],axis=1)
rL_Recall_C["DS 1.2"]=rL_Recall.apply(lambda x: x["DS 1.2"] + "±" + x["DS 1.2_"],axis=1)
rL_Recall_C["DS 2.1"]=rL_Recall.apply(lambda x: x["DS 2.1"] + "±" + x["DS 2.1_"],axis=1)
rL_Recall_C["DS 2.2"]=rL_Recall.apply(lambda x: x["DS 2.2"] + "±" + x["DS 2.2_"],axis=1)
rL_Recall_C["DS 3.1"]=rL_Recall.apply(lambda x: x["DS 3.1"] + "±" + x["DS 3.1_"],axis=1)
rL_Recall_C["DS 3.2"]=rL_Recall.apply(lambda x: x["DS 3.2"] + "±" + x["DS 3.2_"],axis=1)
print(rL_Recall_C)
rL_Recall_R.columns = ["R 1.1","R 1.2","R 2.1","R 2.2","R 3.1","R 3.2"]
rL_Recall_CC = pd.concat([rL_Recall_C.iloc[:,0],rL_Recall_R.iloc[:,0],rL_Recall_C.iloc[:,1],rL_Recall_R.iloc[:,1],rL_Recall_C.iloc[:,2],rL_Recall_R.iloc[:,2],rL_Recall_C.iloc[:,3],rL_Recall_R.iloc[:,3],rL_Recall_C.iloc[:,4],rL_Recall_R.iloc[:,4],rL_Recall_C.iloc[:,5],rL_Recall_R.iloc[:,5]],axis=1)
print(rL_Recall_CC)

#rL_Recall_C.iloc[:,0]=rL_Recall.iloc[:,0] + "±" + rL_Recall.iloc[:,1]
rL_Recall_w_t_l = pd.DataFrame(index=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"),columns=("DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"))
print(rL_Recall_w_t_l)
rL_Recall_RR = rL_Recall_R.drop(["Average"], axis=0).copy(deep=True)
print(rL_Recall_RR)
for m in range(len(rL_Recall_RR.columns)):
    for n in range(len(rL_Recall_RR.columns)):
        win = 0
        tie = 0
        los = 0
        for t in range(len(rL_Recall_RR)):
            if rL_Recall_RR.iloc[t,m] < rL_Recall_RR.iloc[t,n]:
                win = win + 1
            elif rL_Recall_RR.iloc[t,m] == rL_Recall_RR.iloc[t,n]:
                tie = tie + 1
            else:
                los = los + 1
        rL_Recall_w_t_l.iloc[m,n] = "{0}/{1}/{2}".format(win,tie,los)
#
print(rL_Recall_w_t_l)

Recall_data = np.array([list(rL_Recall_RR.iloc[0,:]), list(rL_Recall_RR.iloc[1,:]), list(rL_Recall_RR.iloc[2,:]), list(rL_Recall_RR.iloc[3,:]), list(rL_Recall_RR.iloc[4,:]),list(rL_Recall_RR.iloc[5,:])])
Recall_Tf = Friedman(6,6,Recall_data)
print(Recall_Tf)

TF = pd.DataFrame(index=("Accuracy","F-score","Precision","Recall"),columns=("$F_{F}$","$Critical$ $value$"))
TF.iloc[0,0] = Accuracy_Tf
TF.iloc[1,0] = Fscore_Tf
TF.iloc[2,0] = Precision_Tf
TF.iloc[3,0] = Recall_Tf
TF.iloc[0,1] = 2.4900


writer_train = pd.ExcelWriter("..//data/new/pH_result_train_arrange.xlsx")
#writer_train = pd.ExcelWriter("..//data/new/pH_result_test_arrange.xlsx")
#writer_train = pd.ExcelWriter("..//data/new/pH_result_E_v_arrange.xlsx")

#writer_train = pd.ExcelWriter("..//data/new/pH_result_train_F_arrange.xlsx")
#writer_train = pd.ExcelWriter("..//data/new/pH_result_test_F_arrange.xlsx")
#writer_train = pd.ExcelWriter("..//data/new/pH_result_E_v_F_arrange.xlsx")
rL_Accuracy_CC.to_excel(writer_train, sheet_name="Accuracy-1")
rL_Accuracy_w_t_l.to_excel(writer_train, sheet_name="Accuracy-2")
rL_Fscore_CC.to_excel(writer_train, sheet_name="Fscore-1")
rL_Fscore_w_t_l.to_excel(writer_train, sheet_name="Fscore-2")
rL_Precision_CC.to_excel(writer_train, sheet_name="Precision-1")
rL_Precision_w_t_l.to_excel(writer_train, sheet_name="Precision-2")
rL_Recall_CC.to_excel(writer_train, sheet_name="Recall-1")
rL_Recall_w_t_l.to_excel(writer_train, sheet_name="Recall-2")
TF.to_excel(writer_train, sheet_name="TF")
writer_train.save()


