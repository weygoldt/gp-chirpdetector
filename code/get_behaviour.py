import os 
from paramiko import SSHClient
from scp import SCPClient
from IPython import embed
from pandas import read_csv

ssh = SSHClient()
ssh.load_system_host_keys()

ssh.connect(hostname='kraken',
            username='efish',
            password='fwNix4U',
            )


# SCPCLient takes a paramiko transport as its only argument
scp = SCPClient(ssh.get_transport())

data = read_csv('../recs.csv')
foldernames = data['recording'].values

directory = f'/Users/acfw/Documents/uni_tuebingen/chirpdetection/GP2023_chirp_detection/data/mount_data/'
for foldername in foldernames:

    if not os.path.exists(directory+foldername):
        os.makedirs(directory+foldername)

    files = [('-').join(foldername.split('-')[:3])+'.csv','chirp_ids.npy', 'chirps.npy', 'fund_v.npy', 'ident_v.npy', 'idx_v.npy', 'times.npy', 'spec.npy', 'LED_on_time.npy', 'sign_v.npy']


    for f in files:
        scp.get(f'/home/efish/behavior/2019_tube_competition/{foldername}/{f}',
                directory+foldername)

scp.close()
