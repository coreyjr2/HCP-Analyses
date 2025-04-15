#Imports
try:
  import datetime as dt
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  import seaborn as sb
  from sklearn.decomposition import KernelPCA
  import pandas as pd
  import numpy as np
  import sklearn.model_selection
  from sklearn.svm import SVC 
  from sklearn.model_selection import GroupShuffleSplit
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler
  from sklearn.metrics import confusion_matrix
  from sklearn.decomposition import PCA
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import make_classification
  import json 
  import hashlib
  import os
  from joblib import Parallel, delayed
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)

uid = '69354adf'
# Assign output dirs
base = '/data/hx-hx1/kbaacke/datasets/UCLA_Decoding/'
meta_path = f'{base}metadata/'
confusion_path = f'{base}confusion/'
probs_path = f'{base}probs/'
comppath = f'{base}comparison/'

#Make output paths
base = '/data/hx-hx1/kbaacke/datasets/UCLA_Decoding/'
for pth in [base, meta_path, confusion_path, probs_path, comppath]:
    try:
        os.makedirs(pth)
    except:
        pass

# Read in full data
full_data = pd.read_csv(f'/data/hx-hx1/kbaacke/datasets/UCLA/{uid}_FunctionalConnectomes.csv')

parcel_info = pd.read_csv('/data/hx-hx1/kbaacke/Code/Parcellation/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
node_names = list(parcel_info['ROI Name'])

def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list

dummy_array = np.zeros((200,200))
colnames = connection_names(dummy_array, node_names)

##standardizing data 
def scale_subset(df, cols_to_exclude):
    df_excluded = df[cols_to_exclude]
    df_temp = df.drop(cols_to_exclude, axis=1, inplace=False)
    df_temp = mean_norm(df_temp)
    df_ret = pd.concat([df_excluded, df_temp], axis=1, join='inner')
    return df_ret

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

scaled_data = scale_subset(full_data, list(full_data.columns)[:16] + ['Subject','Task'])

full_data = scaled_data

#generating random unique identifier 
def generate_uid(metadata, length = 8):
  dhash = hashlib.md5()
  encoded = json.dumps(metadata, sort_keys=True).encode()
  dhash.update(encoded)
  # You can change the 8 value to change the number of characters in the unique id via truncation.
  run_uid = dhash.hexdigest()[:length]
  return run_uid

#################################
####### SVC of task data ########
#################################

#creating our function, what we want in the output, metadict  
def svc_predgen(X_train, y_train, X_test, y_test, split, C=1, random_state=812, kernel= 'rbf', class_weight= None, probability= True, Notes = ''):
    ucla_svc= SVC(random_state = random_state, C =C, kernel = kernel, class_weight= class_weight, probability= probability)
    ucla_svc.fit(X_train, y_train)
    preds = ucla_svc.predict(X_test)
    trainacc = ucla_svc.score(X_train, y_train)
    testacc = ucla_svc.score(X_test, y_test)
    confusionmatrix = confusion_matrix(y_test, preds) 
    probs = ucla_svc.predict_proba(X_test)
    metadict = {
        'classifier': 'svc', 
        'random_state': random_state,
        'C': C,
        'kernel': kernel, 
        'class_weight': class_weight, 
        'probability': probability, 
        'training_size': len(y_train),
        'testing_size': len(y_test),
        'split': split, 
        'version': 2
    }
    if len(Notes)>0:
        metadict['Notes'] = Notes
    run_uid= generate_uid(metadict)
    #opens new json and writes data there
    with open(f'{meta_path}{run_uid}_metadict.json', 'w') as outfile:
        json.dump(metadict, outfile)
    resultsdict = {
        'classifier': ['svc'], 
        'trainacc': [trainacc],
        'testacc': [testacc], 
        'runuid': [run_uid],
    }
    np.savetxt(f'{confusion_path}{run_uid}_confusion_matrix.csv', confusionmatrix, delimiter=",")
    np.savetxt(f'{probs_path}{run_uid}_probs.csv', probs, delimiter=",")
    # making 2 cols w/ true and pred val side by side 
    result_comparison = pd.DataFrame({'preds': preds, 'y_test': y_test})
    result_comparison.to_csv(f'{comppath}{run_uid}_results.csv')
    return(resultsdict)

#####training test split by diagnosis w/ controls still  
gss_holdout = GroupShuffleSplit(n_splits=10, train_size = .8, random_state = 812)
idx_1 = gss_holdout.split(
    X = full_data[colnames],
    y = full_data['diagnosis'],
    groups = full_data['participant_id']
)

cv_splitdict = {}
ind = 0 
for train, test in idx_1:
    cv_splitdict[ind] = (train, test) # save as a tuple of train, test
    ind +=1


##filter out controls##
no_controls_df = full_data.loc[full_data["diagnosis"] != "CONTROL"]

#####training test split by diagnosis w/ controls still  
gss_holdout = GroupShuffleSplit(n_splits=10, train_size = .8, random_state = 812)
idx_1_nc = gss_holdout.split(
    X = no_controls_df[colnames],
    y = no_controls_df['diagnosis'],
    groups = no_controls_df['participant_id']
)

cv_splitdict_nc = {}
ind = 0 
for train, test in idx_1_nc:
    cv_splitdict_nc[ind] = (train, test) # save as a tuple of train, test
    ind +=1


##confusion matrices 
def conf_mat_to_df(uid, labels):
  # From a run UID, converts the confusion matrix into columns in a dataframe
  # Column labels will be {i}_{j} {i}_{j}_n in the confusion matrix where i is the try value and j was the predicted value. first values represetned as proportion, second represents the number of subjects in that cell (for ease of plotting later)
  try:
    conf_mat = pd.read_csv(f'{confusion_path}{uid}_confusion_matrix.csv', header=None)
    conf_array = np.array(conf_mat)
    out_dict = {
      'metadata_ref':[uid]
    }
    for i in range(conf_array.shape[0]):
      i_sum = np.sum(conf_array[i,])
      for j in range(conf_array.shape[1]):
        out_dict[f'{labels[i]}_{labels[j]}'] = [conf_array[i,j]/i_sum]
        out_dict[f'{labels[i]}_{labels[j]}_n'] = [conf_array[i,j]]
    out_df = pd.DataFrame.from_dict(out_dict,orient='columns')
    return out_df
  except Exception as e:
    print(f'Failure to read {uid}, {type(uid)}: {e}')
    return pd.DataFrame()

#### Save Task and Diagnosis information per split ####
for ind in cv_splitdict.keys(): 
    info_df_train = full_data.iloc[cv_splitdict[ind][0]][['participant_id','Task','diagnosis']]
    info_df_test = full_data.iloc[cv_splitdict[ind][1]][['participant_id','Task','diagnosis']]
    info_df_train.to_csv(f'{base}SplitInfo_Train_{ind}.csv',index=False)
    info_df_test.to_csv(f'{base}SplitInfo_Test_{ind}.csv',index=False)

#   No controls
for ind in cv_splitdict_nc.keys(): 
    info_df_train = no_controls_df.iloc[cv_splitdict_nc[ind][0]][['participant_id','Task','diagnosis']]
    info_df_test = no_controls_df.iloc[cv_splitdict_nc[ind][1]][['participant_id','Task','diagnosis']]
    info_df_train.to_csv(f'{base}NC_SplitInfo_Train_{ind}.csv',index=False)
    info_df_test.to_csv(f'{base}NC_SplitInfo_Test_{ind}.csv',index=False)
    
###DIAGNOSIS STUFF w/SVC###
diag_datadict = {}
for ind in cv_splitdict.keys(): 
    diag_datadict[ind]= {
        'X_train': full_data.iloc[cv_splitdict[ind][0]][colnames],
        'y_train': full_data.iloc[cv_splitdict[ind][0]]['diagnosis'].values.ravel(),
        'X_test': full_data.iloc[cv_splitdict[ind][1]][colnames],
        'y_test': full_data.iloc[cv_splitdict[ind][1]]['diagnosis'].values.ravel()
    }

# ***** rerun this
#   No controls
diag_datadict_nc = {}
for ind in cv_splitdict_nc.keys(): 
    diag_datadict_nc[ind]= {
        'X_train': no_controls_df.iloc[cv_splitdict_nc[ind][0]][colnames],
        'y_train': no_controls_df.iloc[cv_splitdict_nc[ind][0]]['diagnosis'].values.ravel(),
        'X_test': no_controls_df.iloc[cv_splitdict_nc[ind][1]][colnames],
        'y_test': no_controls_df.iloc[cv_splitdict_nc[ind][1]]['diagnosis'].values.ravel()
    }
# *****

#hyperparam tune svc 
diagpredlist = Parallel(n_jobs = 14)(
    delayed(svc_predgen)(
        X_train = diag_datadict[ind]['X_train'],
        y_train = diag_datadict[ind]['y_train'],
        X_test = diag_datadict[ind]['X_test'],
        y_test = diag_datadict[ind]['y_test'],
        split = ind,
        C=c,
        kernel=kernel,
        class_weight=class_weights 
    ) for ind in diag_datadict.keys() for c in[.01, .1, 1, 10, 100] for kernel in['linear', 'rbf'] for class_weights in [None, 'balanced']
)

try:
    diag_acc_df=pd.read_csv(f'{base}Diag_Accuracy_V2.csv')
    diagconcatlist= [diag_acc_df]
except :
    diagconcatlist=[]

for resdict in diagpredlist: 
    output_df = pd.DataFrame(resdict)
    output_df['sessions_used'] = 'all'
    output_df['subjects_used'] = 'all'
    diagconcatlist.append(output_df)
    print(resdict['testacc'])

diag_acc_df = pd.concat(diagconcatlist) 
diag_acc_df.to_csv(f'{base}Diag_Accuracy_V2.csv', index=False) 

# Ran till here already

# no control version 
#hyperparam tune svc 
diagpredlist = Parallel(n_jobs = 12)(
    delayed(svc_predgen)(
        X_train = diag_datadict_nc[ind]['X_train'],
        y_train = diag_datadict_nc[ind]['y_train'],
        X_test = diag_datadict_nc[ind]['X_test'],
        y_test = diag_datadict_nc[ind]['y_test'],
        split = ind,
        C=c,
        kernel=kernel,
        class_weight=class_weights 
    ) for ind in diag_datadict_nc.keys() for c in[.01, .1, 1, 10, 100] for kernel in['linear', 'rbf'] for class_weights in [None, 'balanced']
)

diagconcatlist= [diag_acc_df]
for resdict in diagpredlist: 
    output_df = pd.DataFrame(resdict)
    output_df['sessions_used'] = 'all'
    output_df['subjects_used'] = 'No CONTROL'
    diagconcatlist.append(output_df)
    print(resdict['testacc'])

diag_acc_df = pd.concat(diagconcatlist) 
diag_acc_df.to_csv(f'{base}Diag_Accuracy_V2.csv', index=False)

diag_acc_df = pd.read_csv(f'{base}Diag_Accuracy_V2.csv')

task_diag_dfs = []
for task in full_data['Task'].unique():
  task_data = full_data[full_data['Task']==task]
  gss_holdout = GroupShuffleSplit(n_splits=10, train_size = .8, random_state = 812)
  idx_1 = gss_holdout.split(
      X = task_data[colnames],
      y = task_data['diagnosis'],
      groups = task_data['participant_id']
  )
  cv_splitdict_task = {}
  ind = 0
  for train, test in idx_1:
      cv_splitdict_task[ind] = (train, test) # save as a tuple of train, test
      ind +=1
  task_diag_datadict = {}
  for ind in cv_splitdict_task.keys():
    task_diag_datadict[ind]= {
        'X_train': task_data.iloc[cv_splitdict_task[ind][0]][colnames],
        'y_train': task_data.iloc[cv_splitdict_task[ind][0]]['diagnosis'].values.ravel(),
        'X_test': task_data.iloc[cv_splitdict_task[ind][1]][colnames],
        'y_test': task_data.iloc[cv_splitdict_task[ind][1]]['diagnosis'].values.ravel()
    }
    task_data.iloc[cv_splitdict_task[ind][0]][['participant_id','Task','diagnosis']].to_csv(f'{base}{task}_SplitInfo_Train_{ind}.csv',index=False)
    task_data.iloc[cv_splitdict_task[ind][1]][['participant_id','Task','diagnosis']].to_csv(f'{base}{task}_SplitInfo_Test_{ind}.csv',index=False)
  task_diag_predlist = Parallel(n_jobs = 14)(
    delayed(svc_predgen)(
        X_train = task_diag_datadict[ind]['X_train'],
        y_train = task_diag_datadict[ind]['y_train'],
        X_test = task_diag_datadict[ind]['X_test'],
        y_test = task_diag_datadict[ind]['y_test'],
        split = ind,
        Notes = f'{task}',
        C=c,
        kernel=kernel,
        class_weight=class_weights 
    ) for ind in task_diag_datadict.keys() for c in[.01, .1, 1, 10, 100] for kernel in['linear', 'rbf'] for class_weights in [None, 'balanced']
  )
  task_diag_concatlist = []
  for resdict in task_diag_predlist:
    output_df = pd.DataFrame(resdict)
    output_df['sessions_used'] = task
    output_df['subjects_used'] = 'all'
    task_diag_predlist.append(output_df)
    # print(resdict['testacc'])
  task_diag_acc_df = pd.concat(task_diag_concatlist)
  task_diag_acc_df['Task'] = task
  task_diag_acc_df.to_csv(f'{base}{task}_Diag_Accuracy_V2.csv', index=False)
  task_diag_dfs.append(task_diag_acc_df)

task_acc_df_all = pd.concat(task_diag_dfs)
# print(task_acc_df_all)


## Run from here 8/10
# No Controls
task_diag_dfs = []
for task in no_controls_df['Task'].unique():
  task_data = no_controls_df[no_controls_df['Task']==task]
  gss_holdout = GroupShuffleSplit(n_splits=10, train_size = .8, random_state = 812)
  idx_1 = gss_holdout.split(
      X = task_data[colnames],
      y = task_data['diagnosis'],
      groups = task_data['participant_id']
  )
  cv_splitdict_task = {}
  ind = 0
  for train, test in idx_1:
      cv_splitdict_task[ind] = (train, test) # save as a tuple of train, test
      ind +=1
  task_diag_datadict = {}
  for ind in cv_splitdict_task.keys():
    task_diag_datadict[ind]= {
        'X_train': task_data.iloc[cv_splitdict_task[ind][0]][colnames],
        'y_train': task_data.iloc[cv_splitdict_task[ind][0]]['diagnosis'].values.ravel(),
        'X_test': task_data.iloc[cv_splitdict_task[ind][1]][colnames],
        'y_test': task_data.iloc[cv_splitdict_task[ind][1]]['diagnosis'].values.ravel()
    }
    task_data.iloc[cv_splitdict_task[ind][0]][['participant_id','Task','diagnosis']].to_csv(f'{base}{task}_NC_SplitInfo_Train_{ind}.csv',index=False)
    task_data.iloc[cv_splitdict_task[ind][1]][['participant_id','Task','diagnosis']].to_csv(f'{base}{task}_NC_SplitInfo_Test_{ind}.csv',index=False)
  task_diag_predlist = Parallel(n_jobs = 14)(
    delayed(svc_predgen)(
        X_train = task_diag_datadict[ind]['X_train'],
        y_train = task_diag_datadict[ind]['y_train'],
        X_test = task_diag_datadict[ind]['X_test'],
        y_test = task_diag_datadict[ind]['y_test'],
        split = ind,
        Notes = f'{task}',
        C=c,
        kernel=kernel,
        class_weight=class_weights 
    ) for ind in task_diag_datadict.keys() for c in[.01, .1, 1, 10, 100] for kernel in['linear', 'rbf'] for class_weights in [None, 'balanced']
  )
  task_diag_concatlist = []
  for resdict in task_diag_predlist:
    output_df = pd.DataFrame(resdict)
    output_df['sessions_used'] = task
    output_df['subjects_used'] = 'No CONTROL'
    task_diag_concatlist.append(output_df)
    # print(resdict['testacc'])
  task_diag_acc_df = pd.concat(task_diag_concatlist)
  task_diag_acc_df['Task'] = task
  task_diag_acc_df.to_csv(f'{base}NC_{task}_Diag_Accuracy_V2.csv', index=False)
  task_diag_dfs.append(task_diag_acc_df)

task_acc_df_all_nc = pd.concat(task_diag_dfs)
# print(task_acc_df_all_nc)

acc_df_full = pd.concat([diag_acc_df, task_acc_df_all, task_acc_df_all_nc])

acc_df_full.to_csv(f'{base}Full_Diag_Accuracy_V2.csv', index=False)