# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:57:10 2020

@author: Dibyajyoti
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from analysing_functions import *
from sklearn.decomposition import PCA
from sklearn import svm
from plot_derived_ellipse_PCA import *
from sklearn.preprocessing import Normalizer,StandardScaler

################ Create Dataset from the EEG files #############################
print("---------------------------------------Creating Dataset from EEG signals of all subjects --------------------------")
path = "I:/Video_Project_CB/Only_EEG/"

Id = []

v_subs = []
c_subs = []

subs_num_c = []
subs_num_v = []

CB_status = []
col_names = ['F3','F4','T3','T4','P3','P4','O1','O2','Sub-ID','Song-status','Features']
#feature_names = np.array(['dta','dtr','bta','btr','dba','dbr','atr','ata','ttr','tta','gtr','gta'])
#feature_names = np.array(['dtr','btr','dbr','atr','ttr','gtr'])
feature_names = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

l_f = len(feature_names)
cols_names = [0, 2, 4, 6, 8, 10, 12, 14]
fs=125
win = 100 * fs

c = 1
v = 1

for i in os.listdir(path):
    path_eachsub = os.path.join(path,i)
    sub_num = []
    for s in range(l_f):
        sub_num.append(i)

    sub_num = np.array(sub_num,dtype = str)
    num_channels = 8
    v_subs_dta = []
    v_subs_dtr = []
    v_subs_bta = []
    v_subs_btr = []
    v_subs_dba = []
    v_subs_dbr = []
    v_subs_ata = []
    v_subs_atr = []
    v_subs_tta = []
    v_subs_ttr = []
    v_subs_gta = []
    v_subs_gtr = []

    c_subs_dta = []
    c_subs_dtr = []
    c_subs_bta = []
    c_subs_btr = []
    c_subs_dba = []
    c_subs_dbr = []
    c_subs_ata = []
    c_subs_atr = []
    c_subs_tta = []
    c_subs_ttr = []
    c_subs_gta = []
    c_subs_gtr = []

    for i in os.listdir(path_eachsub):
        subs_dta_c = []
        subs_dtr_c = []
        subs_bta_c = []
        subs_btr_c = []
        subs_dba_c = []
        subs_dbr_c = []
        subs_ata_c = []
        subs_atr_c = []
        subs_tta_c = []
        subs_ttr_c = []
        subs_gta_c = []
        subs_gtr_c = []

        subs_dta_v = []
        subs_dtr_v = []
        subs_bta_v = []
        subs_btr_v = []
        subs_dba_v = []
        subs_dbr_v = []
        subs_ata_v = []
        subs_atr_v = []
        subs_tta_v = []
        subs_ttr_v = []
        subs_gta_v = []
        subs_gtr_v = []

        if "Song" in i:#Looks for Song trial files
            if len(os.listdir(os.path.join(path_eachsub,i))) > 0:
                temp = per_sub_band_power(subs_dta_c,subs_dtr_c,subs_bta_c,subs_btr_c,subs_dba_c,subs_dbr_c,subs_atr_c,subs_ata_c,subs_ttr_c,subs_tta_c,subs_gtr_c,subs_gta_c,path_eachsub, cols_names, num_channels, i, win, fs)
#                temp = [c_subs_dta,c_subs_dtr,c_subs_bta,c_subs_btr,c_subs_dba,c_subs_dbr,c_subs_atr,c_subs_ata,c_subs_ttr,c_subs_tta,c_subs_gtr,c_subs_gta]
                temp = np.array(temp)
                sum_of_rows = temp.sum(axis=0)
#                temp = temp / sum_of_rows[:, np.newaxis]
                CB = []
                for s in range(l_f):
                    CB.append(1)
                CB = np.array(CB,dtype = int)
                c_subs.append(temp.resize(l_f,8))
                print("Control of breathing analysing...")
                if (c==1):
                    df_c = pd.DataFrame(temp, columns = col_names[:8])
                    df_c['Sub_ID'] = sub_num
                    df_c['Song-status'] = CB
                    df_c['Features'] = feature_names
                    print("I am in first condition...")
                    c += 1
                else:
                    df1 = pd.DataFrame(temp, columns = col_names[:8])
                    df1['Sub_ID'] = sub_num
                    df1['Song-status'] = CB
                    df1['Features'] = feature_names
                    df_c = df_c.append(df1)
                    c += 1
                    print("I am in second condition...")


#            else:
#                temp = np.zeros((12,8))/0
#                c_subs.append(temp)
#                if (c==1):
#                    df_c = pd.DataFrame(temp, index = feature_names, columns = col_names)
#                    c += 1
#                else:
#                    df1 = pd.DataFrame(temp, index = feature_names, columns = col_names)
#                    df_c = df_c.append(df1)
#                    c+=1
        else:
            if len(os.listdir(os.path.join(path_eachsub,i))) > 0:
                temp = per_sub_band_power(subs_dta_v,subs_dtr_v,subs_bta_v,subs_btr_v,subs_dba_v,subs_dbr_v,subs_atr_v,subs_ata_v,subs_ttr_v,subs_tta_v,subs_gtr_v,subs_gta_v,path_eachsub, cols_names, num_channels, i, win, fs)
#                temp = [v_subs_dta,v_subs_dtr,v_subs_bta,v_subs_btr,v_subs_dba,v_subs_dbr,v_subs_atr,v_subs_ata,v_subs_ttr,v_subs_tta,v_subs_gtr,v_subs_gta]
                temp = np.array(temp)
                CB = []
                for s in range(l_f):
                    CB.append(0)
                CB = np.array(CB,dtype = int)
                temp.resize(l_f,8)
                sum_of_rows = temp.sum(axis=0)
#                temp = temp / sum_of_rows[:, np.newaxis]

                v_subs.append(temp)
                if (v==1):
                    df_v = pd.DataFrame(temp, columns = col_names[:8])
                    df_v['Sub_ID'] = sub_num
                    df_v['Song-status'] = CB
                    df_v['Features'] = feature_names
                    print("I am in first condition...")
                    v += 1
                else:
                    df1 = pd.DataFrame(temp, columns = col_names[:8])
                    df1['Sub_ID'] = sub_num
                    df1['Song-status'] = CB
                    df1['Features'] = feature_names
                    df_v = df_v.append(df1)
                    print("I am in second condition...")
                    v += 1
            else:
                temp = np.zeros((l_f,8))/0
                v_subs.append(temp)
                if (v==1):
                    df_v = pd.DataFrame(temp, columns = col_names[:8])
                    df_v['Sub_ID'] = sub_num
                    df_v['Song-status'] = CB
                    df_v['Features'] = feature_names
                    v += 1
                else:
                    df1 = pd.DataFrame(temp, columns = col_names[:8])
                    df1['Sub_ID'] = sub_num
                    df1['Song-status'] = CB
                    df1['Features'] = feature_names
                    df_v = df_v.append(df1)
                    v+=1
########End of Dataset creation ##########

plt.close('all')

#Get Covariance matrix to find high relevant features
width = []
height = []
area = []
deg_angle = []
#pathname = "E:/Video_Project_CB/Plots/Frequency_domain/EEG_ratios/"
N = len(feature_names)
for k in range(len(feature_names)):

    posy = 0
    for k1 in range(len(feature_names)):
        if k1 > k:
            plt.figure
            p = 0
            w_c = []
            h_c = []
            da_c = []
            ar_c = []

            posx=0
            fig, axs = plt.subplots(1,8)
            for i in col_names[:8]:
                w,h,d_a = plot_derived_ellipse(df_stack,i,feature_names[k],feature_names[k1],axs[posx])
                posx +=1
                p += 1
                ar = np.pi*np.array(w,dtype = float)*np.array(h,dtype = float)
                width.append(np.array(w,dtype = float))
                height.append(np.array(h,dtype = float))
                area.append(np.array(ar,dtype = float))
                deg_angle.append(np.array(d_a,dtype = float))
                w_c.append(w)
                h_c.append(h)
                da_c.append(d_a)
                ar_c.append(ar)
            w_c = np.array(w_c,dtype = float)
            h_c = np.array(h_c,dtype = float)
            da_c = np.array(da_c,dtype = float)
            ar_c = np.array(ar_c,dtype = float)

            print(ttest_ind(w_c[0],w_c[1],equal_var = False))
            print(ttest_ind(h_c[0],h_c[1],equal_var = False))
            print(ttest_ind(da_c[0],da_c[1],equal_var = False))
            print(ttest_ind(ar_c[0],ar_c[1],equal_var = False))

            #fname = pathname + str(k)+str(k1)+'Song_Nosong.png'
            #plt.savefig(fname)

width = np.array(width)
height = np.array(height)
area = np.array(area)
deg_angle = np.array(deg_angle)


df_stack = pd.concat([df_c, df_v], axis=0)
df_stack = df_stack.reset_index(drop=True)
df_stack.pop('Sub_ID')
#df_stack_values = df_stack[['F3','F4','T3','T4','P3','P4','O1','O2']]
#df_stack_values = df_stack[['F3','F4','T3','T4','P3','P4','O1','O2']]
#df_stack[['F3','F4','T3','T4','P3','P4','O1','O2']] = Normalizer(norm='l2').fit_transform(df_stack_values)

scaler = StandardScaler()
#scaler.fit(df_stack)
scaled_df = scaler.fit_transform(df_stack)

pca = PCA(n_components=4)
principalComponents = pca.fit_transform(scaled_df)

print("Shape of the scaled and PCA features:",np.shape(principalComponents))

########### 4 component PCA ########
PCA_df = pd.DataFrame(data = principalComponents,columns = ['PC1','PC2','PC3','PC4'])

########### Following plots would be to see which components are most relevant in classification ###############
plt.figure()
plt.xlabel('PC - 1',fontsize=14)
plt.ylabel('PC - 2',fontsize=14)
plt.title("PCA",fontsize=20)
targets = [1, 0]
colors = ['g', 'b']
for target, color in zip(targets,colors):
    indices = df_stack['Song-status'] == target
    plt.scatter(PCA_df.loc[indices, 'PC1']
               , PCA_df.loc[indices, 'PC2'], c = color, s = 25)

plt.legend(['Song','No-song'],prop={'size': 15})

plt.figure()
plt.xlabel('PC - 3',fontsize=14)
plt.ylabel('PC - 4',fontsize=14)
for target, color in zip(targets,colors):
    indices = df_stack['Song-status'] == target
    plt.scatter(PCA_df.loc[indices, 'PC3']
               , PCA_df.loc[indices, 'PC4'], c = color, s = 25)

plt.legend(['Song','No-song'],prop={'size': 15})

########## Variance ratio to check which component is responsible for how much percentage of all the features #########
####### PCA(0.9) and PCA.n_components can be used to check how many components would be needed to cover for 90% of features #########
feature_var = np.var(principalComponents, axis = 0)
feature_var_ratio = feature_var/(np.sum(feature_var))

print("Principal Components variance ratios:", feature_var_ratio)

plt.figure()
plt.xlabel('PC - 1',fontsize=14)
plt.ylabel('PC - 3',fontsize=14)
for target, color in zip(targets,colors):
    indices = df_stack['Song-status'] == target
    plt.scatter(PCA_df.loc[indices, 'PC1']
               , PCA_df.loc[indices, 'PC3'], c = color, s = 25)

plt.legend(['Song','No-song'],prop={'size': 15})


plt.figure()
plt.xlabel('PC - 1',fontsize=14)
plt.ylabel('PC - 4',fontsize=14)
for target, color in zip(targets,colors):
    indices = df_stack['Song-status'] == target
    plt.scatter(PCA_df.loc[indices, 'PC1']
               , PCA_df.loc[indices, 'PC4'], c = color, s = 25)

plt.legend(['Song','No-song'],prop={'size': 15})

"""
After this step the SVM step would have started which would include,
1. Splitting Dataset using StratifiedShuffleSplit and defined pipeline using sklearn.pipeline.
2. Use SVM for different basic function inputs.
3. Plot SVM decision boundrary to show the contour plot showing the classification contours.
"""