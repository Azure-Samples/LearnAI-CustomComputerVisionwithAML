import os
import pdb
import random
import pandas as pd
import urllib.request
import zipfile
from cv2 import imread

def download_and_place(kind, link, data_dir, index):
    if not os.path.exists(os.path.join(data_dir, kind)):
        os.makedirs(os.path.join(data_dir, kind))

    dstPath = os.path.join(data_dir, kind, str(index) + ".jpg")
    if os.path.isfile(dstPath):
        print("Already downloaded image: " + link)
    else:
        try:
            urllib.request.urlretrieve(link, dstPath)
            img = imread(dstPath)
            assert(img is not None) # test if image was loaded correctly
            print("Downloaded image {:4}: {}".format(index, link))
        except:
            if os.path.exists(dstPath):
                os.remove(dstPath)
            return 0
    return 1

# def download_all(dataset_location):
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     urls = pd.read_table(os.path.join(dir_path,"fashion_texture_urls.tsv"), names=["kind", "link"])
#     count = 0
#     for index, row in urls.iterrows():
#         count = count + download_and_place(row.kind, row.link, dataset_location, count)
#     print("Downloaded {} images.".format(count))

def download_all(dataset_location):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    urls = pd.read_table(os.path.join(dir_path,"fashion_texture_urls.tsv"), names=["kind", "link"])
    count = 0
    for index, row in urls.iterrows():
        count = count + download_and_place(row.kind, row.link, dataset_location, index)
    print("Downloaded {} images.".format(count))


if __name__ == "__main__":
    download_all("test")
