#!/usr/bin/env python3
try:
  import platform
  import logging
  import pandas as pd
  import os
  import datetime as dt
  import json 
  import paramiko
  from scp import SCPClient
  import getpass
  import numpy as np
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  import pickle as pk
  import seaborn as sb
  import time
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)


# Global Variables
try:
  sep = os.path.sep
  # source_path = '/home/kbaacke/HCP_Analyses/'
  source_path = os.path.dirname(os.path.abspath(__file__)) + sep
  sys_name = platform.system() 
  hostname = platform.node()
  remote_sep = '/'
  # output_path = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  # local_path = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  # output_path = 'C:\\Users\\Sarah Melissa\\Documents\\output\\'
  # local_path = 'C:\\Users\\Sarah Melissa\\Documents\\temp\\'
  run_uid = '8d2513'
  remote_outpath = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  username = 'kbaacke'#'solshan2'
  datahost = 'r2.psych.uiuc.edu'
  source_path = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  base = 'S:\\' #f'C:\\Users\\Sarah Melissa\\Documents\\output\\'
  remote_base = '/data/hx-hx1/kbaacke/datasets/'
  output_path = '{base}hcp_analysis_output{sep}8d2513{sep}'
except:
  pass

# Template Functions
try:
  def createSSHClient(server, user, password, port=22):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client
except Exception as e:
  print(f'Error defining template functions: ')
  raise Exception(e)

outpath = output_path
tsne_out = outpath + 't-SNE{sep}'
# Make folder specific to this run's output
try:
    os.makedirs(tsne_out.format(base=base, sep=sep))
except:
    pass

start_time = dt.datetime.now()
ssh = createSSHClient(f'{datahost}', f'{username}', getpass.getpass(f'Password for {username}@{datahost}:'), 22)
scp = SCPClient(ssh.get_transport())
scp.get(tsne_out.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)
end_time = dt.datetime.now()
print('Transfer Done. Runtime: ', end_time - start_time)

local_tsne = tsne_out.format(base = base, sep=sep)
plotting_df = pd.read_csv(f'{local_tsne}Plotting_DF.csv')

# Set up a list of labels to read in
length_list = [
  19900,
  19884,
  18540,
  13981,
  10027,
  7482,
  5847,
  4638,
  3771,
  3149,
  2679,
  2297,
  1980,
  1725,
  1533,
  1354,
  1205,
  1078,
  960,
  876,
  807,
  734,
  676,
  622,
  569,
  522,
  480,
  443,
  414
]
labels = []
perp_list = [
  50, 
  # 55, 
  # 60, 
  # 70, 
  75, 
  # 80, 
  # 85, 
  90
]

for perp in perp_list:
  labels.append(f'PCA_perp{perp}')
  for n in length_list[6:]:
    labels.append(f'PCA_{n}_perp{perp}')

for label in labels:
  try:
    tSNE_output = np.load(f'{local_tsne}{label}_tSNE.npy')
    sb.scatterplot(tSNE_output[:,0], tSNE_output[:,1], hue = plotting_df['label'], s=3).set(title=label)
    plt.savefig(f'{local_tsne}{label}_tSNE.png', transparent=True)
  except Exception as e:
    print(e)
