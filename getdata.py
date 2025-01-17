from zipfile import ZipFile
import os
import urllib
import urllib.request

FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'


if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}...')
    urllib.request.urlretrieve(URL, FILE)
print('Unzipping images...')

with ZipFile(FILE) as zip_images:
    zip_images.extractall(FOLDER)
print('Done!')