#!/usr/bin/python
# -*- coding:utf-8 -*-
#!/usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

# pd.set_option("display.unicode.east_asian_width",True)
# rL_181 = pd.read_excel("..//data/181-2022-03-31.xlsx",sheet_name="181",header=0,names=None,index_col=0)
# print(len(rL_181))
# rLWN = rL_181.dropna(axis=1,how='all')
# rLWN = rLWN.iloc[0:181,:]
# print(len(rLWN))
# # pH_1 = rLWN.loc[:,['WHO_functional_class','6MWD','NT-proBNP','mRAP_invas','SvO2','CI','danger_level']]
# statistics_145 = rLWN.iloc[:,[11,12,3,4,5,6,7,8,9,32,34,36,37,38,39,40,41]]
# statistics_145.columns = ['$Age$','$Height$','$WHO$ $FC$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$RS$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVC$-$CI$','$IVCD$','$RAA$','$RVA$','$DIAGNOSIS$']
# print(statistics_145)
#
# statistics_1 = statistics_145.loc[statistics_145['$DIAGNOSIS$'] == 1]
# print(statistics_1)
#
# statistics_4 = statistics_145.loc[statistics_145['$DIAGNOSIS$'] == 4]
# print(statistics_4)
#
# statistics_5 = statistics_145.loc[statistics_145['$DIAGNOSIS$'] == 5]
# print(statistics_5)
#
# statistics_45 = statistics_145.loc[(statistics_145['$DIAGNOSIS$'] == 4) | (statistics_145['$DIAGNOSIS$'] == 5)]
# print(statistics_45)
#
# writer = pd.ExcelWriter('..//data/statistics/statistics_145_1_4_5_45.xlsx')
# statistics_145.to_excel(writer,sheet_name='pH_145')
# statistics_1.to_excel(writer,sheet_name='pH_1')
# statistics_4.to_excel(writer,sheet_name='pH_4')
# statistics_5.to_excel(writer,sheet_name='pH_5')
# statistics_45.to_excel(writer,sheet_name='pH_45')
# writer.save()


# n_145 = pd.read_excel("..//data/statistics/statistics_145_1_4_5_45.xlsx",sheet_name="pH_145",header=0,names=None,index_col=0)
# c_145 = pd.read_excel("..//data/pH_risk_level_145_2.xlsx",sheet_name="pH_1_2",header=0,names=None,index_col=0)
# print(c_145)
# c_145 = c_145.drop(["number"], axis=1)
# c_145.columns = ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']
# print(c_145)
# n_df_145 = pd.DataFrame(index=['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$'],columns=['$mean_145$','$median_145$','$std_145$'])
#
# c_df_145 = pd.DataFrame(index=['$Age$','$WHO$ $FC$','$IVC$-$CI$','$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D'],columns=['$F$'])
#
# for i in ['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']:
#     n_df_145.loc[i,'$mean_145$'] = n_145[i].mean()
#     n_df_145.loc[i,'$median_145$'] = n_145[i].median()
#     n_df_145.loc[i,'$std_145$'] = n_145[i].std()
#
# print(n_df_145)
#
# for i in ['$Age$','$WHO$ $FC$','$IVC$-$CI$']:
#     a = dict(n_145[i].value_counts()/len(n_145))
#     print(str(a))
#     c_df_145.loc[i, '$F$'] = str(a)
# print(c_df_145)
#
#     # cat_df = cat_df.astype('category')
#     # print(cat_df.dtypes)
#     # cat_df.describe().transpose()
# for i in ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']:
#     b = dict(c_145[i].value_counts()/len(c_145))
#     print(str(b))
#     c_df_145.loc[i, '$F$'] = str(b)
# print(c_df_145)
# #
# # print(n_145['$Age$'].describe().transpose())
#
# writer = pd.ExcelWriter('..//data/statistics/statistics_145.xlsx')
# n_df_145.to_excel(writer,sheet_name='pH_145_n')
# c_df_145.to_excel(writer,sheet_name='pH_145_c')
# writer.save()


# n_1 = pd.read_excel("..//data/statistics/statistics_145_1_4_5_45.xlsx",sheet_name="pH_1",header=0,names=None,index_col=0)
# c_1 = pd.read_excel("..//data/pH_risk_level_1_2.xlsx",sheet_name="pH_1_2",header=0,names=None,index_col=0)
# print(c_1)
# c_1 = c_1.drop(["number"], axis=1)
# c_1.columns = ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']
# print(c_1)
# n_df_1 = pd.DataFrame(index=['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$'],columns=['$mean_1$','$median_1$','$std_1$'])
#
# c_df_1 = pd.DataFrame(index=['$Age$','$WHO$ $FC$','$IVC$-$CI$','$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D'],columns=['$F$'])
#
# for i in ['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']:
#     n_df_1.loc[i,'$mean_1$'] = n_1[i].mean()
#     n_df_1.loc[i,'$median_1$'] = n_1[i].median()
#     n_df_1.loc[i,'$std_1$'] = n_1[i].std()
#
# print(n_df_1)
#
# for i in ['$Age$','$WHO$ $FC$','$IVC$-$CI$']:
#     a = dict(n_1[i].value_counts()/len(n_1))
#     print(str(a))
#     c_df_1.loc[i, '$F$'] = str(a)
# print(c_df_1)
#
#     # cat_df = cat_df.astype('category')
#     # print(cat_df.dtypes)
#     # cat_df.describe().transpose()
# for i in ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']:
#     b = dict(c_1[i].value_counts()/len(c_1))
#     print(str(b))
#     c_df_1.loc[i, '$F$'] = str(b)
# print(c_df_1)
# #
# # print(n_1['$Age$'].describe().transpose())
#
# writer = pd.ExcelWriter('..//data/statistics/statistics_1.xlsx')
# n_df_1.to_excel(writer,sheet_name='pH_1_n')
# c_df_1.to_excel(writer,sheet_name='pH_1_c')
# writer.save()

# n_4 = pd.read_excel("..//data/statistics/statistics_145_1_4_5_45.xlsx",sheet_name="pH_4",header=0,names=None,index_col=0)
# c_4 = pd.read_excel("..//data/pH_risk_level_4_2.xlsx",sheet_name="pH_1_2",header=0,names=None,index_col=0)
# print(c_4)
# c_4 = c_4.drop(["number"], axis=1)
# c_4.columns = ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']
# print(c_4)
# n_df_4 = pd.DataFrame(index=['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$'],columns=['$mean_4$','$median_4$','$std_4$'])
#
# c_df_4 = pd.DataFrame(index=['$Age$','$WHO$ $FC$','$IVC$-$CI$','$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D'],columns=['$F$'])
#
# for i in ['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']:
#     n_df_4.loc[i,'$mean_4$'] = n_4[i].mean()
#     n_df_4.loc[i,'$median_4$'] = n_4[i].median()
#     n_df_4.loc[i,'$std_4$'] = n_4[i].std()
#
# print(n_df_4)
#
# for i in ['$Age$','$WHO$ $FC$','$IVC$-$CI$']:
#     a = dict(n_4[i].value_counts()/len(n_4))
#     print(str(a))
#     c_df_4.loc[i, '$F$'] = str(a)
# print(c_df_4)
#
#     # cat_df = cat_df.astype('category')
#     # print(cat_df.dtypes)
#     # cat_df.describe().transpose()
# for i in ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']:
#     b = dict(c_4[i].value_counts()/len(c_4))
#     print(str(b))
#     c_df_4.loc[i, '$F$'] = str(b)
# print(c_df_4)
# #
# # print(n_4['$Age$'].describe().transpose())
#
# writer = pd.ExcelWriter('..//data/statistics/statistics_4.xlsx')
# n_df_4.to_excel(writer,sheet_name='pH_4_n')
# c_df_4.to_excel(writer,sheet_name='pH_4_c')
# writer.save()


# n_5 = pd.read_excel("..//data/statistics/statistics_145_1_4_5_45.xlsx",sheet_name="pH_5",header=0,names=None,index_col=0)
# c_5 = pd.read_excel("..//data/pH_risk_level_5_2.xlsx",sheet_name="pH_1_2",header=0,names=None,index_col=0)
# print(c_5)
# c_5 = c_5.drop(["number"], axis=1)
# c_5.columns = ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']
# print(c_5)
# n_df_5 = pd.DataFrame(index=['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$'],columns=['$mean_5$','$median_5$','$std_5$'])
#
# c_df_5 = pd.DataFrame(index=['$Age$','$WHO$ $FC$','$IVC$-$CI$','$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D'],columns=['$F$'])
#
# for i in ['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']:
#     n_df_5.loc[i,'$mean_5$'] = n_5[i].mean()
#     n_df_5.loc[i,'$median_5$'] = n_5[i].median()
#     n_df_5.loc[i,'$std_5$'] = n_5[i].std()
#
# print(n_df_5)
#
# for i in ['$Age$','$WHO$ $FC$','$IVC$-$CI$']:
#     a = dict(n_5[i].value_counts()/len(n_5))
#     print(str(a))
#     c_df_5.loc[i, '$F$'] = str(a)
# print(c_df_5)
#
#     # cat_df = cat_df.astype('category')
#     # print(cat_df.dtypes)
#     # cat_df.describe().transpose()
# for i in ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']:
#     b = dict(c_5[i].value_counts()/len(c_5))
#     print(str(b))
#     c_df_5.loc[i, '$F$'] = str(b)
# print(c_df_5)
# #
# # print(n_5['$Age$'].describe().transpose())
#
# writer = pd.ExcelWriter('..//data/statistics/statistics_5.xlsx')
# n_df_5.to_excel(writer,sheet_name='pH_5_n')
# c_df_5.to_excel(writer,sheet_name='pH_5_c')
# writer.save()


n_45 = pd.read_excel("..//data/20230127/ph_data_20230123.xlsx",sheet_name="pH_3",header=0,names=None,index_col=0)
c_45 = pd.read_excel("..//data/pH_risk_level_45_2.xlsx",sheet_name="pH_1_2",header=0,names=None,index_col=0)
print(c_45)
c_45 = c_45.drop(["number"], axis=1)
c_45.columns = ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']
print(c_45)
n_df_45 = pd.DataFrame(index=['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$'],columns=['$mean_45$','$median_45$','$std_45$'])

c_df_45 = pd.DataFrame(index=['$Age$','$WHO$ $FC$','$IVC$-$CI$','$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D'],columns=['$F$'])

for i in ['$Height$','$6MWD$','$NT$-$proBNP$','$RAP_{invas}$','$SvO_2$','$CI$','$PVR_{echo}$','$TRV_{max}$','$TAPSE$','$IVCD$','$RAA$','$RVA$']:
    n_df_45.loc[i,'$mean_45$'] = n_45[i].mean()
    n_df_45.loc[i,'$median_45$'] = n_45[i].median()
    n_df_45.loc[i,'$std_45$'] = n_45[i].std()

print(n_df_45)

for i in ['$Age$','$WHO$ $FC$','$IVC$-$CI$']:
    a = dict(n_45[i].value_counts()/len(n_45))
    print(str(a))
    c_df_45.loc[i, '$F$'] = str(a)
print(c_df_45)

    # cat_df = cat_df.astype('category')
    # print(cat_df.dtypes)
    # cat_df.describe().transpose()
for i in ['$WHO$ $FC$_D','$6MWD$_D','$NT$-$proBNP$_D','$RAP_{invas}$_D','$SvO_2$_D','$CI$_D','$RS$_D']:
    b = dict(c_45[i].value_counts()/len(c_45))
    print(str(b))
    c_df_45.loc[i, '$F$'] = str(b)
print(c_df_45)
#
# print(n_45['$Age$'].describe().transpose())

writer = pd.ExcelWriter('..//data/statistics/statistics_45.xlsx')
n_df_45.to_excel(writer,sheet_name='pH_45_n')
c_df_45.to_excel(writer,sheet_name='pH_45_c')
writer.save()

