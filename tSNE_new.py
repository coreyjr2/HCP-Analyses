## tSNE
# Imports
try:
  # import platform
  # import logging
  import pandas as pd
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
  import pickle as pk
  import plotly.express as px
  # import time
  from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA
  import numpy as np
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)

tsne_out = 'C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\tSNE\\'

##tSNE
def run_plt_tsne(data, category_column, label, perplexity = 77, learning_rate=50, init = 'random', verbose=1, n_iter=2000):
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
    sb.scatterplot(tSNE_output[:,0], tSNE_output[:,1], hue =category_column, s=3).set(title=label)
    plt.savefig(f'{tsne_out}{label} tSNE.png', transparent=True)
    end_time = dt.datetime.now()
    runtime = end_time - start_time
    print(f'tSNE on {label} complete. Runtime: {runtime}')
  except Exception as e:
    print(f'Error running tSNE for {label}: {e}')

def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list

full_data = pd.read_csv ('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\full_data.csv')

dummy_array = np.zeros((200,200))
parcel_info = pd.read_csv('C:\\Users\\sarah\\Local_Documents\\Data\\UCLA_Data\\Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
node_names = list(parcel_info['ROI Name'])
colnames = connection_names(dummy_array, node_names)
participants_df = pd.read_csv('C:\\Users\\sarah\Local_Documents\\Data\\UCLA_Data\\participants.tsv', sep='\t')
full_data['participant_id'] = 'sub-' + full_data['Subject'].astype(str)

full_data = pd.merge(full_data, participants_df[['participant_id','diagnosis']], left_on = 'participant_id', right_on = 'participant_id', how='left')
no_controls_df = full_data.loc[full_data["diagnosis"] != "CONTROL"]

run_perplexity = 250
run_learning_rate = 50
run_n_iter = 5000
# With all sessions
pca = PCA().fit(no_controls_df[colnames])

fig = px.line(pca.explained_variance_)
fig.show()

run_plt_tsne(
  pca.transform(no_controls_df[colnames])[:,0:20],
  category_column = no_controls_df['diagnosis'],
  label = f'PCA_AllTasks_Perplexity-{run_perplexity}_LearningRate-{run_learning_rate}_NIter-{run_n_iter}',
  perplexity = run_perplexity,
  learning_rate = run_learning_rate,
  n_iter = run_n_iter
  )

# for each session label
for session in no_controls_df['Task'].unique():
  task_df = no_controls_df[no_controls_df['Task']==session]
  pca = PCA().fit(task_df[colnames])
  run_plt_tsne(
    pca.transform(task_df[colnames]),
    category_column = task_df['diagnosis'],
    label = f'PCA_{session}_Perplexity-{run_perplexity}_LearningRate-{run_learning_rate}_NIter-{run_n_iter}',
    perplexity = run_perplexity,
    learning_rate = run_learning_rate,
    n_iter = run_n_iter
    )



# run_plt_tsne(full_data[colnames], label = 'perpl10lr10', perplexity= 10, learning_rate= 10) 

# run_plt_tsne(full_data[colnames], label = 'perpl15lr10it5000', perplexity= 15, learning_rate= 10, n_iter= 5000)     

# run_plt_tsne(full_data[colnames], label = 'perpl30lr1000it5000', perplexity= 30, learning_rate= 1000, n_iter= 5000)  