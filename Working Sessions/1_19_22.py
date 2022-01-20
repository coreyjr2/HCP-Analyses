import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
username = 'kbaacke' # Change this
hostname = 'r2.psych.uiuc.edu'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

sep = os.path.sep
dest_sep = '/'
source_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + sep
localpath = f'C:\\Users\\kyle\\repos\\HCP-Analyses\\Output\\' # Change this
base = '/mnt/usb1/hcp_analysis_output/'

# shutil.make_archive(f'{base}2c891e', 'zip', f'{base}2c891e')

archive_pattern = '{base}2c891e'
psswd = getpass.getpass(f'Password for {username}@{hostname}:')
start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', f'{psswd}')
scp = SCPClient(ssh.get_transport())

scp.get(archive_pattern.format(base = base), archive_pattern.format(base = localpath), recursive=True)

end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)



import matplotlib
import matplotlib.pyplot as plt
import numpy as np
pca1 = np.load('C:\\Users\\kyle\\repos\\HCP-Analyses\\Output\\2c891e\\FeatureSelection\\network_conneciton\\2c891e_train_pca-1.npy') # Change these
pca_df = pd.DataFrame(pca1)
yvals = np.load('C:\\Users\\kyle\\repos\\HCP-Analyses\\Output\\2c891e\\FeatureSelection\\network_sum\\2c891e_train_y.npy')
pca_df['Class'] = yvals



fig = plt.figure(figsize=(8,8))
plt.scatter(pca_df[0], pca_df[1], pca_df[2], c=pca_df['Class'])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pca_df[0], pca_df[1], pca_df[2], c=pca_df['Class'], alpha=.2)

