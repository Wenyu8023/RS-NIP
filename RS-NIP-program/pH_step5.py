#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
#rL_1_1 = pd.read_excel("..//data/new/pH_result_test_1_1.xlsx",sheet_name='test_pH_1_1',header=0,names=None,index_col=0)
#rL_1_2 = pd.read_excel("..//data/new/pH_result_test_1_1.xlsx",sheet_name='test_pH_1_2',header=0,names=None,index_col=0)
#rL_2_1 = pd.read_excel("..//data/new/pH_result_test_1_2.xlsx",sheet_name='test_pH_2_1',header=0,names=None,index_col=0)
#rL_2_2 = pd.read_excel("..//data/new/pH_result_test_1_2.xlsx",sheet_name='test_pH_2_2',header=0,names=None,index_col=0)
#rL_3_1 = pd.read_excel("..//data/new/pH_result_test_1_3.xlsx",sheet_name='test_pH_3_1',header=0,names=None,index_col=0)
#rL_3_2 = pd.read_excel("..//data/new/pH_result_test_1_3.xlsx",sheet_name='test_pH_3_2',header=0,names=None,index_col=0)
# # rL_4_1 = pd.read_excel("..//data/1/pH_result_test_1_4.xlsx",sheet_name='test_pH_4_1',header=0,names=None,index_col=0)
# # rL_4_2 = pd.read_excel("..//data/1/pH_result_test_1_4.xlsx",sheet_name='test_pH_4_2',header=0,names=None,index_col=0)
#rL_1_1 = pd.read_excel("..//data/new/pH_result_test_F_1_1.xlsx",sheet_name='test_pH_1_1',header=0,names=None,index_col=0)
#rL_1_2 = pd.read_excel("..//data/new/pH_result_test_F_1_1.xlsx",sheet_name='test_pH_1_2',header=0,names=None,index_col=0)
#rL_2_1 = pd.read_excel("..//data/new/pH_result_test_F_1_2.xlsx",sheet_name='test_pH_2_1',header=0,names=None,index_col=0)
#rL_2_2 = pd.read_excel("..//data/new/pH_result_test_F_1_2.xlsx",sheet_name='test_pH_2_2',header=0,names=None,index_col=0)
#rL_3_1 = pd.read_excel("..//data/new/pH_result_test_F_1_3.xlsx",sheet_name='test_pH_3_1',header=0,names=None,index_col=0)
#rL_3_2 = pd.read_excel("..//data/new/pH_result_test_F_1_3.xlsx",sheet_name='test_pH_3_2',header=0,names=None,index_col=0)

#rL_1_1 = pd.read_excel("..//data/new/pH_result_train_1_1.xlsx",sheet_name='train_pH_1_1',header=0,names=None,index_col=0)
#rL_1_2 = pd.read_excel("..//data/new/pH_result_train_1_1.xlsx",sheet_name='train_pH_1_2',header=0,names=None,index_col=0)
#rL_2_1 = pd.read_excel("..//data/new/pH_result_train_1_2.xlsx",sheet_name='train_pH_2_1',header=0,names=None,index_col=0)
#rL_2_2 = pd.read_excel("..//data/new/pH_result_train_1_2.xlsx",sheet_name='train_pH_2_2',header=0,names=None,index_col=0)
#rL_3_1 = pd.read_excel("..//data/new/pH_result_train_1_3.xlsx",sheet_name='train_pH_3_1',header=0,names=None,index_col=0)
#rL_3_2 = pd.read_excel("..//data/new/pH_result_train_1_3.xlsx",sheet_name='train_pH_3_2',header=0,names=None,index_col=0)
# # rL_4_1 = pd.read_excel("..//data/1/pH_result_train_1_4.xlsx",sheet_name='train_pH_4_1',header=0,names=None,index_col=0)
# # rL_4_2 = pd.read_excel("..//data/1/pH_result_train_1_4.xlsx",sheet_name='train_pH_4_2',header=0,names=None,index_col=0)
#rL_1_1 = pd.read_excel("..//data/new/pH_result_train_F_1_1.xlsx",sheet_name='train_F_pH_1_1',header=0,names=None,index_col=0)
#rL_1_2 = pd.read_excel("..//data/new/pH_result_train_F_1_1.xlsx",sheet_name='train_F_pH_1_2',header=0,names=None,index_col=0)
#rL_2_1 = pd.read_excel("..//data/new/pH_result_train_F_1_2.xlsx",sheet_name='train_F_pH_2_1',header=0,names=None,index_col=0)
#rL_2_2 = pd.read_excel("..//data/new/pH_result_train_F_1_2.xlsx",sheet_name='train_F_pH_2_2',header=0,names=None,index_col=0)
#rL_3_1 = pd.read_excel("..//data/new/pH_result_train_F_1_3.xlsx",sheet_name='train_F_pH_3_1',header=0,names=None,index_col=0)
#rL_3_2 = pd.read_excel("..//data/new/pH_result_train_F_1_3.xlsx",sheet_name='train_F_pH_3_2',header=0,names=None,index_col=0)

#rL_1_1 = pd.read_excel("..//data/new/pH_result_E_v_1.xlsx",sheet_name='follow_up_1_pH_1_1',header=0,names=None,index_col=0)
#rL_1_2 = pd.read_excel("..//data/new/pH_result_E_v_1.xlsx",sheet_name='follow_up_1_pH_1_2',header=0,names=None,index_col=0)
#rL_2_1 = pd.read_excel("..//data/new/pH_result_E_v_2.xlsx",sheet_name='follow_up_1_pH_2_1',header=0,names=None,index_col=0)
#rL_2_2 = pd.read_excel("..//data/new/pH_result_E_v_2.xlsx",sheet_name='follow_up_1_pH_2_2',header=0,names=None,index_col=0)
#rL_3_1 = pd.read_excel("..//data/new/pH_result_E_v_3.xlsx",sheet_name='follow_up_1_pH_3_1',header=0,names=None,index_col=0)
#rL_3_2 = pd.read_excel("..//data/new/pH_result_E_v_3.xlsx",sheet_name='follow_up_1_pH_3_2',header=0,names=None,index_col=0)
# # # rL_4_1 = pd.read_excel("..//data/new/pH_result_E_v.xlsx",sheet_name='follow_up_1_pH_4_1',header=0,names=None,index_col=0)
# # # rL_4_2 = pd.read_excel("..//data/new/pH_result_E_v.xlsx",sheet_name='follow_up_1_pH_4_2',header=0,names=None,index_col=0)

rL_1_1 = pd.read_excel("..//data/new/pH_result_E_v_F_1.xlsx",sheet_name='follow_up_1_pH_1_1',header=0,names=None,index_col=0)
rL_1_2 = pd.read_excel("..//data/new/pH_result_E_v_F_1.xlsx",sheet_name='follow_up_1_pH_1_2',header=0,names=None,index_col=0)
rL_2_1 = pd.read_excel("..//data/new/pH_result_E_v_F_2.xlsx",sheet_name='follow_up_1_pH_2_1',header=0,names=None,index_col=0)
rL_2_2 = pd.read_excel("..//data/new/pH_result_E_v_F_2.xlsx",sheet_name='follow_up_1_pH_2_2',header=0,names=None,index_col=0)
rL_3_1 = pd.read_excel("..//data/new/pH_result_E_v_F_3.xlsx",sheet_name='follow_up_1_pH_3_1',header=0,names=None,index_col=0)
rL_3_2 = pd.read_excel("..//data/new/pH_result_E_v_F_3.xlsx",sheet_name='follow_up_1_pH_3_2',header=0,names=None,index_col=0)



#rL_Accuracy = rL_1_1.iloc[0,:] + " " + rL_1_2.iloc[0,:]  + " " + rL_2_1.iloc[0,:] + " " + rL_2_2.iloc[0,:] + " " + rL_3_1.iloc[0,:] + " " + rL_3_2.iloc[0,:] + " " + rL_4_1.iloc[0,:] + " " + rL_4_2.iloc[0,:]
rL_Accuracy = pd.concat([rL_1_1.iloc[0,:],rL_1_2.iloc[0,:],rL_2_1.iloc[0,:],rL_2_2.iloc[0,:],rL_3_1.iloc[0,:],rL_3_2.iloc[0,:]], axis=1, ignore_index=True)
#rL_Accuracy = pd.DataFrame(rL_Accuracy)
#rL_Accuracy.columns = ["DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2","DS 4.1","DS 4.2"]
rL_Accuracy.columns = ["DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"]
# print(rL_1_1.iloc[0,:])
# print(rL_1_2.iloc[0,:])
# print(rL_Accuracy)
rL_Fscore = pd.concat([rL_1_1.iloc[1,:],rL_1_2.iloc[1,:],rL_2_1.iloc[1,:],rL_2_2.iloc[1,:],rL_3_1.iloc[1,:],rL_3_2.iloc[1,:]], axis=1, ignore_index=True)
rL_Fscore.columns = ["DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"]
rL_Precision = pd.concat([rL_1_1.iloc[2,:],rL_1_2.iloc[2,:],rL_2_1.iloc[2,:],rL_2_2.iloc[2,:],rL_3_1.iloc[2,:],rL_3_2.iloc[2,:]], axis=1, ignore_index=True)
rL_Precision.columns = ["DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"]
rL_Recall = pd.concat([rL_1_1.iloc[3,:],rL_1_2.iloc[3,:],rL_2_1.iloc[3,:],rL_2_2.iloc[3,:],rL_3_1.iloc[3,:],rL_3_2.iloc[3,:]], axis=1, ignore_index=True)
rL_Recall.columns = ["DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"]
rL_AUC = pd.concat([rL_1_1.iloc[4,:],rL_1_2.iloc[4,:],rL_2_1.iloc[4,:],rL_2_2.iloc[4,:],rL_3_1.iloc[4,:],rL_3_2.iloc[4,:]], axis=1, ignore_index=True)
rL_AUC.columns = ["DS 1.1","DS 1.2","DS 2.1","DS 2.2","DS 3.1","DS 3.2"]

#writer = pd.ExcelWriter('..//data/new/data_result_test_F.xlsx')
#writer = pd.ExcelWriter('..//data/new/data_result_train_F.xlsx')
writer = pd.ExcelWriter('..//data/new/data_result_E_v_F.xlsx')

#writer = pd.ExcelWriter('..//data/new/data_result_test.xlsx')
#writer = pd.ExcelWriter('..//data/new/data_result_train.xlsx')
#writer = pd.ExcelWriter('..//data/new/data_result_E_v.xlsx')

rL_Accuracy.to_excel(writer,sheet_name='Accuracy')
rL_Fscore.to_excel(writer,sheet_name='F-score')
rL_Precision.to_excel(writer,sheet_name='Precision')
rL_Recall.to_excel(writer,sheet_name='Recall')
rL_AUC.to_excel(writer,sheet_name='AUC')
writer.save()
