import requests, os

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/"
DEST_DIR = "data"


os.makedirs(DEST_DIR, exist_ok=True)


resp = requests.get(BASE_URL)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")


files = []
for a in soup.find_all("a"):

    href = a.get("href")

    if href and not href.startswith("../"):
        files.append(href)

print(f"Found {len(files)} files to download.")


# Download each file with progress bar
for fname in files:

    file_url = urljoin(BASE_URL, fname)
    local_path = os.path.join(DEST_DIR, fname)

    # skip already-downloaded files
    if os.path.exists(local_path):
        print(f"Skipping {fname}, already exists.")
        continue

    with requests.get(file_url, stream=True) as r:

        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))

        with open(local_path, 'wb') as f, tqdm(
            desc=fname, total=total, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            
            for chunk in r.iter_content(chunk_size=8192):

                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


print("All downloads complete.")
