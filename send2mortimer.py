import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
username = 'kbaacke'
hostname = 'raid-18.mortimer.hpc.uwm.edu'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

sep = os.path.sep
dest_sep = '/'


base = f'E:\\dr-fs\\'
dest_base = '/raid-18/LS/medinak/kbaacke/dr-fs/'


start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
scp = SCPClient(ssh.get_transport())

subject_folders = os.listdir(base)
for folder_name in subject_folders:
    if '.' not in folder_name:
        scp.put(base+folder_name, dest_base, recursive=True)
    else:
        scp.put(base+folder_name, dest_base, recursive=False)