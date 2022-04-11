import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
username = 'kbaacke'
hostname = 'r2.psych.uiuc.edu'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

sep = os.path.sep
remote_sep = '/'
source_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + sep
base = f'S:\\'
remote_base = '/data/hx-hx1/kbaacke/datasets/'
# remote_base = '/mnt/usb1/'
# output_path = '{base}hcp_analysis_output{sep}8d2513{sep}'
output_path = '{base}MSC-HCP_analysis{sep}'

try:
  os.makedir(output_path.format(base = base, sep=sep))
except:
  pass

metapath = output_path + 'metadata{sep}'
confpath =  output_path +  'confusion{sep}'
classpath = output_path + 'classification{sep}'
inputpath = output_path + 'inputs{sep}'
weightpath = output_path + 'weights{sep}'
tsnepath = output_path+ 't-SNE{sep}'
accuracy_path = output_path + 'Prediction_Accuracies.csv'
length_path = output_path + 'feature_length_index.csv'

# psswd = getpass.getpass(f'Password for {username}@{hostname}:')
start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
# ssh = createSSHClient(f'{hostname}', 22, f'{username}', f'{psswd}')
scp = SCPClient(ssh.get_transport())

# scp.get(output_path.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)

# scp.get(metapath.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)
# scp.get(confpath.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)
# scp.get(classpath.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)
# scp.get(tsnepath.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)
scp.put(accuracy_path.format(base = base, sep=sep), accuracy_path.format(base = remote_base, sep=remote_sep))
# scp.put(length_path.format(base = base, sep=sep), length_path.format(base = remote_base, sep=remote_sep))

end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)