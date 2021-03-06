from cvtk.augmentation import augment_dataset
from cvtk.evaluation import ClassificationEvaluation
from cvtk.core import Context, ClassificationDataset
from cvtk.core.model import CNTKTLModel
from cvtk import Splitter
from imgaug import augmenters
from bokeh.plotting import show, output_notebook
from ui_utils.ui_confusion_matrix import ConfusionMatrixUI

import os

def classify(dataset_location='classification/sample_data/imgs_recycling/',
             dataset_name='recycling', do_augmentations=True,
             enable_logging=True):
    """
      a sample pipeline for classification.

      loads a dataset, optionally does some augmentations, creates and trains a
      model using transfer learning based on ResNet18, and returns the accuracy
      on a test set.

      Args:
        dataset_location: path to a dataset.  there should be a top level folder
          containing folders for each class.  see the sample recycling dataset for
          an example of the format
        dataset_name: the of the dataset.  will be used in the dataset
          management functionality
        do_augmentations: boolean.  specifies whether augmentations should be
          applied to the test set

      Returns:
        the accuracy on the test set
    """

    # if we're not running inside AML WB, set up the share directory
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' not in os.environ:
        os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] = './share'

    # create a dataset from a directory with folders representing different classes
    dataset = ClassificationDataset.create_from_dir(dataset_name,
                                                    dataset_location)

    # print out some info about the dataset
    print("DATASET INFO:")
    dataset.print_info()

    # split the full dataset into a train and test set
    # the stratify option will ensure that the different labels are balanced in the
    # train and test sets
    train_set_orig, test_set = dataset.split(train_size = 0.66, stratify = "label")

    # optionally augment images by cropping and rotating
    if do_augmentations:
        aug_sequence = augmenters.Sequential([
            augmenters.Fliplr(0.5),             # horizontally flip 50% of all images
            augmenters.Crop(percent=(0, 0.1))  # crop images by 0-10% of their height/width
        ])
        train_set = augment_dataset(train_set_orig, [aug_sequence])
        print("Number of original training images = {}, with augmented images included = {}.".format(train_set_orig.size(), train_set.size()))
    else:
        train_set = train_set_orig

    # model creation
    lr_per_mb = [0.05]*7 + [0.005]*7 +  [0.0005]
    mb_size = 32
    input_resoluton = 224
    base_model_name = 'ResNet18_ImageNet_CNTK'
    model = CNTKTLModel(train_set.labels,
                       base_model_name=base_model_name,
                       image_dims = (3, input_resoluton, input_resoluton))

    # train the model using cntk
    ce = ClassificationEvaluation(model, test_set, minibatch_size = mb_size)

    acc = ce.compute_accuracy()
    print("Accuracy = {:2.2f}%".format(100*acc))
    cm  = ce.compute_confusion_matrix()
    print("Confusion matrix = \n{}".format(cm))

    cm_ui = ConfusionMatrixUI(cm, [l.name for l in test_set.labels])
    show(cm_ui.ui)

    return acc

if __name__ == '__main__':
    print('test accuracy is', classify())