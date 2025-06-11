import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
username = 'corey' # Change this to be your username
hostname = 'hx.psych.uiuc.edu'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

sep = os.path.sep
remote_sep = '/'
source_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + sep
base = f's:\\dr-fs\\'

ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
ftp = ssh.open_sftp()
scp = SCPClient(ssh.get_transport())

scp.get('/data/hx-hx1/Corey/Projects/DR_FS/data/HCP/FeatureSelection.zip', f'{base}hcp{sep}FeatureSelection.zip')