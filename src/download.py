import os
from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url,save_path):
  urlretrieve(url,save_path)
  try:
    with ZipFile(save_path) as z:
      z.extractall(os.path.split(save_path)[0])
    print("Done")
  except Exception as e:
    print("Invalid file")


