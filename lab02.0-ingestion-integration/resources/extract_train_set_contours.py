from cvtk.augmentation import augment_dataset
from cvtk.evaluation import ClassificationEvaluation
from cvtk.core import Context, ClassificationDataset, Image, Label
from cvtk.core.model import CNTKTLModel
from cvtk import Splitter
from types import FunctionType
import cv2
import numpy as np
import inspect

from imgaug import augmenters

import os

def extract_contour_dataset(dataset_location='classification/sample_data/imgs_recycling/',
                                  dataset_name='recycling', 
                                  enable_logging=True):
    # if we're not running inside AML WB, set up the share directory
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' not in os.environ:
        os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] = './share'

    # create a dataset from a directory with folders representing different classes
    dataset = ClassificationDataset.create_from_dir(dataset_name,
                                                    dataset_location,
                                                    enable_logging=enable_logging)

    extract_contour(dataset)

def extract_contour(dataset_orig):
    """
      Applies Canny edge detection and extracts contours

      Args:
        train_set_orig : The original set of images from 
        the training set
    """
    images = dataset_orig.images
    for item in images:
        # Convert image from 3x480x640 to 480x640x3
        color_img = np.array(np.transpose(item.as_np_array(), (1,2,0)))
        # Transform to grayscale and apply mean filter
        gray_img = cv2.GaussianBlur(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY), (0,0), 1)
        # Apply the Canny edge detector on the image
        contour_img = cv2.Canny(np.uint8(gray_img), 30, 120)
        contour_filename = item.storage_path.split('.')[0] + '_contour.jpg'
        print("Generated " + contour_filename)
        cv2.imwrite(contour_filename, contour_img)

def add_image_dataset(dataset, path_image, label_image):
    labels_dict = {}
    for item in dataset.labels:
        labels_dict[item.name] = item
    dataset.add_image(Image(storage_path=path_image, name=path_image.split('\\')[-1]), labels_dict[label_image])

def create_dataset_from_json():

    file_labels = "C:\\Users\\miprasad\\Downloads\\cvp-1.0.0b2-release5\\cvp-1.0.0b2-release\\cvp_project\\classification\\scripts\\file_labels.json"

    dataset = ClassificationDataset.create_from_json("recycling", file_labels, context=None)
    dataset.print_info()

if __name__ == '__main__':
   extract_contour_dataset()
#   create_dataset_from_json()