#!/usr/bin/env python3
import urllib.request
import tarfile
import os

# Define download and extraction paths
download_path = r'D:\data dump'

# URLs for the .tar.gz files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

# Create download path if it doesn't exist
os.makedirs(download_path, exist_ok=True)

for idx, link in enumerate(links):
    fn = f'images_{idx+1:02d}.tar.gz'
    full_path = os.path.join(download_path, fn)
    
    # Download the .tar.gz file
    print(f'Downloading {fn}...')
    urllib.request.urlretrieve(link, full_path)
    
    # Decompress the .tar.gz file
    print(f'Decompressing {fn}...')
    try:
        extract_folder = os.path.join(download_path, f'images_{idx+1}')
        os.makedirs(extract_folder, exist_ok=True)
        with tarfile.open(full_path, "r:gz") as tar:
            tar.extractall(path=extract_folder)
        print(f'{fn} decompressed successfully into {extract_folder}.')
    except Exception as e:
        print(f'Error decompressing {fn}: {e}')
    
    # Delete the .tar.gz file after decompression
    os.remove(full_path)
    print(f'{fn} removed after decompression.')

print("Download, decompression, and cleanup complete.")
