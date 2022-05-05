# Script to read in connectomes from \\...\\UCLA/69354adf_FunctionalConnectomes

from mimetypes import init
import pandas as pd
import numpy as np

# Read in Parcellation Info

# You may want to know the names of the parcels. They are in the Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv file (should be included in the files sent to you)
# Change the path in the next line to the path where that file is located.
parcel_info = pd.read_csv('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
# The node labels are contained in the 'ROI Name' column of this dataframe
node_names = list(parcel_info['ROI Name'])

# Global variables
# Change this depending on where you store the files
outpath = 'C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\UCLA\\69354adf_FunctionalConnectomes\\'
subject_list = [
  '10159','10171','10189','10193','10206','10217','10225','10227','10228',
  '10235','10249','10269','10271','10273','10274','10280','10290','10292',
  '10304','10316','10321','10325','10329','10339','10340','10345','10347',
  '10356','10361','10365','10376','10377','10388','10429','10438','10440',
  '10448','10455','10460','10471','10478','10487','10492','10506','10517',
  '10523','10524','10525','10527','10530','10557','10565','10570','10575',
  '10624','10629','10631','10638','10668','10672','10674','10678','10680',
  '10686','10692','10696','10697','10704','10707','10708','10719','10724',
  '10746','10762','10779','10785','10788','10844','10855','10871','10877',
  '10882','10891','10893','10912','10934','10940','10948','10949','10958',
  '10963','10968','10975','10977','10987','10998','11019','11030','11044',
  '11050','11052','11059','11061','11062','11066','11067','11068','11077',
  '11082','11088','11090','11097','11098','11104','11105','11106','11108',
  '11112','11122','11128','11131','11142','11143','11149','11156','50004',
  '50005','50006','50007','50008','50010','50013','50014','50015','50016',
  '50020','50021','50022','50023','50025','50027','50029','50032','50033',
  '50034','50035','50036','50038','50043','50047','50048','50049','50050',
  '50051','50052','50053','50056','50058','50059','50060','50061','50064',
  '50066','50067','50069','50073','50075','50076','50077','50080','50081',
  '50083','50085','60001','60005','60006','60008','60010','60011','60012',
  '60014','60015','60017','60020','60021','60022','60028','60030','60033',
  '60036','60037','60038','60042','60043','60045','60046','60048','60049',
  '60051','60052','60053','60055','60056','60057','60060','60062','60065',
  '60066','60068','60070','60072','60073','60074','60076','60077','60078',
  '60079','60080','60084','60087','60089','70001','70002','70004','70007',
  '70010','70015','70017','70020','70021','70022','70026','70029','70034',
  '70037','70040','70046','70048','70049','70051','70052','70055','70057',
  '70058','70060','70074','70075','70076','70077','70079','70080','70081',
  '70083','70086'
  ]
session_list = [
  'scap','bart','pamret','pamenc',
  'taskswitch','stopsignal','rest','bht'
  ]
output_template = '{OUTPATH}sub-{SUBJECT}_task-{SESSION}.npy'

# Read in data

# Create empty dictionary
data_dict = {}
# Iterate through session labels
for session in session_list:
  # Create a new dictionary within the data dictionary to contain data for each session
  data_dict[session] = {}
  for subject in subject_list:
    # Read in data in try block to prevent errors from stopping the script due to missing data
    try:
      data_dict[session][subject] = np.load(output_template.format(OUTPATH=outpath, SUBJECT=subject, SESSION=session))
    except Exception as e:
      # This may print a lot. Don't worry, that is expected
      print(f'Error loading data from sub-{subject}_task-{session}: {e}')


# You can see the keys in your dictionary using data_dict.keys()
print(data_dict.keys())
# If you want to use data for a given task, you can refer to it by it's key to see what keys (subject IDs) are available for that task
print(data_dict['rest'].keys())
# Or, you can see how many are available for each task in a for loop:
for task in data_dict.keys():
  subs_available = len(data_dict[task].keys())
  print(f'{subs_available} subjects available with {task} data.')

c1 =data_dict['rest']['10159']

l1 = list(c1[np.triu_indices_from(c1, k=1)])

def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list

dummy_array = np.zeros((200,200))
colnames = connection_names(dummy_array, node_names)
colnames.append('Subject')
colnames.append('Task')

vector_dict = {}
for task in data_dict.keys():
  vector_dict[task] = {}
  print(task)
  for subject in data_dict[task].keys():
    print(subject)
    connectome = data_dict[task][subject]
    vector_dict[task][subject] = list(connectome[np.triu_indices_from(connectome, k=1)])
    vector_dict[task][subject].append(subject)
    vector_dict[task][subject].append(task)

dfs_to_concat = []
for task in vector_dict.keys():
  task_df = pd.DataFrame.from_dict(vector_dict[task], orient = 'index', columns=colnames)
  dfs_to_concat.append(task_df)

full_data = pd.concat(dfs_to_concat)

#generate and store rest data 
rest_data = full_data[full_data['Task']=='rest']
rest_data.to_csv ('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\rest_data.csv', index = False)

del vector_dict    

rest_data = pd.read_csv('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\rest_data.csv')     
#reading in phenotype/participant data 
participant_data = pd.read_csv('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\participants.tsv', sep = "\t")

#merging participant and rest data
rest_data['temp'] = 'sub-'
rest_data['participant_id'] = rest_data['temp'] + rest_data['Subject'].astype(str)

full_data = pd.merge(participant_data, rest_data, how='right', on='participant_id')

#tSNE
#Imports
try:
  # import platform
  # import logging
  # import pandas as pd
  # import os
  import datetime as dt
  # import json 
  # import paramiko
  # from scp import SCPClient
  # import getpass
  # import numpy as np
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  # import pickle as pk
  import seaborn as sb
  # import time
  from sklearn.decomposition import KernelPCA
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)

tsne_out = 'C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\'

def run_plt_tsne(data, label, perplexity = 77, learning_rate=50, init = 'random', verbose=1, n_iter=2000):
  start_time = dt.datetime.now()
  # Run tSNE
  try:
    tSNE_output = TSNE(
      n_components=2,
      perplexity=perplexity,
      learning_rate=learning_rate,
      init=init,
      verbose=verbose,
      n_iter=n_iter,
      ).fit_transform(data)
    sb.scatterplot(tSNE_output[:,0], tSNE_output[:,1], hue = full_data['diagnosis'], s=3).set(title=label)
    plt.savefig(f'{tsne_out}{label} tSNE.png', transparent=True)
    end_time = dt.datetime.now()
    runtime = end_time - start_time
    print(f'tSNE on {label} complete. Runtime: {runtime}')
  except Exception as e:
    print(f'Error running tSNE for {label}: {e}')
    
parcel_info = pd.read_csv('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
# The node labels are contained in the 'ROI Name' column of this dataframe
node_names = list(parcel_info['ROI Name'])

def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list

dummy_array = np.zeros((200,200))
colnames = connection_names(dummy_array, node_names)

run_plt_tsne(full_data[colnames], label = 'perpl10lr10', perplexity= 10, learning_rate= 10) 

run_plt_tsne(full_data[colnames], label = 'perpl15lr10it5000', perplexity= 15, learning_rate= 10, n_iter= 5000)     

run_plt_tsne(full_data[colnames], label = 'perpl30lr1000it5000', perplexity= 30, learning_rate= 1000, n_iter= 5000)       

#PCA
kernel_pca = KernelPCA(n_components=2, kernel = 'sigmoid', gamma=1)
kernel_pca_result = kernel_pca.fit_transform(full_data[colnames])

plt.figure(figsize=(24,20))
sb.scatterplot(
    x= kernel_pca_result[,0], y=kernel_pca_result[,1],
    hue=full_data['diagnosis'],
    palette = sb.color_palette("Spectral", as_cmap=True),
    # data= kernel_pca_result,
    legend="full",
    alpha=1,
    s=100
)


    



    

