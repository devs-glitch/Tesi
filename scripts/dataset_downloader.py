''' Communicate with the external: download raw files and save them'''

import os
import requests

urls = ["https://zenodo.org/records/5649212/files/H1_O1.csv?download=1",
        "https://zenodo.org/records/5649212/files/H1_O2.csv?download=1",
        "https://zenodo.org/records/5649212/files/H1_O3a.csv?download=1",
        "https://zenodo.org/records/5649212/files/H1_O3b.csv?download=1",
        "https://zenodo.org/records/5649212/files/L1_O1.csv?download=1",
        "https://zenodo.org/records/5649212/files/L1_O2.csv?download=1",
        "https://zenodo.org/records/5649212/files/L1_O3a.csv?download=1",
        "https://zenodo.org/records/5649212/files/L1_O3b.csv?download=1"]

destination_path = "data/raw/"

os.makedirs(destination_path, exist_ok = True)

for urls in urls:
    raw_file = urls.split('/')[-1]
    file = raw_file.split('?')[0]

    print(f"Downloading {file}...")

    final_path = os.path.join(destination_path, file)

    response = requests.get(urls, stream = True)

    with open(final_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
print(f"Download completed: {file} ready")
