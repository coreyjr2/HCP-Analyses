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
net_uid = 'b0c37fe7' # b0c37fe7 d8b1c692
remote_base = '/data/hx-hx1/kbaacke/datasets/'
# remote_base = '/mnt/usb1/'
output_path = '{base}hcp_analysis_output{sep}89952a{sep}'
# output_path = '{base}MSC-HCP_analysis{sep}'

try:
  os.makedir(output_path.format(base = base, sep=sep))
except:
  pass

metapath = output_path + 'metadata'
confpath =  output_path +  'confusion'
classpath = output_path + 'classification'
inputpath = output_path + 'inputs'
weightpath = output_path + 'weights'
tsnepath = output_path+ 't-SNE{sep}'
length_path = output_path + 'feature_length_index.csv'
net_out = output_path + 'Network_Analysis{sep}' + net_uid

accuracy_path = output_path + 'Prediction_Accuracies.csv'
accuracy_path_dt1 = output_path + 'Prediction_Accuracies_dt1.csv'
accuracy_path_nmg = output_path + 'Prediction_Accuracies_MNGMachine.csv'
accuracy_path_r2 = output_path + 'Prediction_Accuracies_r2.csv'
accuracy_path_dx = output_path + 'Prediction_Accuracies_dx.csv'
accuracy_path_hx = output_path + 'Prediction_Accuracies_hx.csv'
accuracy_path_sx = output_path + 'Prediction_Accuracies_sx.csv'

accuracy_path_network_dt1 = output_path + 'Prediction_Accuracies_network_dt1.csv'
accuracy_path_network_nmg = output_path + 'Prediction_Accuracies_network_MNGMachine.csv'
accuracy_path_network_r2 = output_path + 'Prediction_Accuracies_network_r2.csv'
accuracy_path_network_dx = output_path + 'Prediction_Accuracies_network_dx.csv'
accuracy_path_network_hx = output_path + 'Prediction_Accuracies_network_hx.csv'
accuracy_path_network_sx = output_path + 'Prediction_Accuracies_network_sx.csv'



# psswd = getpass.getpass(f'Password for {username}@{hostname}:')
start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
# ssh = createSSHClient(f'{hostname}', 22, f'{username}', f'{psswd}')
scp = SCPClient(ssh.get_transport())

# scp.get(output_path.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)

# scp.put(accuracy_path_dt1.format(base = base, sep=sep), accuracy_path_dt1.format(base = remote_base, sep=remote_sep))
# scp.put(accuracy_path_nmg.format(base = base, sep=sep), accuracy_path_nmg.format(base = remote_base, sep=remote_sep))
# scp.put(accuracy_path_network_dt1.format(base = base, sep=sep), accuracy_path_network_dt1.format(base = remote_base, sep=remote_sep))
# scp.put(accuracy_path_network_nmg.format(base = base, sep=sep), accuracy_path_network_nmg.format(base = remote_base, sep=remote_sep))
# scp.put(confpath.format(base = base, sep=sep), output_path.format(base = remote_base, sep=remote_sep), recursive=True)
# scp.put(weightpath.format(base = base, sep=sep), output_path.format(base = remote_base, sep=remote_sep), recursive=True)
# scp.put(classpath.format(base = base, sep=sep), output_path.format(base = remote_base, sep=remote_sep), recursive=True)
# scp.put(metapath.format(base = base, sep=sep), output_path.format(base = remote_base, sep=remote_sep), recursive=True)
# scp.get(tsnepath.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)
# scp.put(accuracy_path.format(base = base, sep=sep), accuracy_path.format(base = remote_base, sep=remote_sep))
# scp.put(length_path.format(base = base, sep=sep), length_path.format(base = remote_base, sep=remote_sep))
scp.put(net_out.format(base = base, sep=sep), output_path.format(base = remote_base, sep=remote_sep) + net_uid + remote_sep, recursive=True)

end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)