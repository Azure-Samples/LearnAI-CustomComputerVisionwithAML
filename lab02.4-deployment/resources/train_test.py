from cvtk.augmentation import augment_dataset
from cvtk.evaluation import ClassificationEvaluation
from cvtk.core import Context, ClassificationDataset
from cvtk.core.model import CNTKTLModel
from cvtk import Splitter

from imgaug import augmenters

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
                                                    dataset_location,
                                                    enable_logging=enable_logging)

    # print out some info about the dataset
    print("DATASET INFO:")
    dataset.print_info()

    # split the full dataset into a train and test set
    # the stratify option will ensure that the different labels are balanced in the
    # train and test sets
    splitter = Splitter(dataset, enable_logging=enable_logging)
    train_set_orig, test_set = splitter.split(train_size=.8, stratify='label')

    # optionally augment images by cropping and rotating
    if do_augmentations:
        # here we create two pipelines for doing augmentations.  the first
        # will rotate each image by between -45 and 45 degrees (the angle is
        # chosen at random).  then the rotated images will be flipped from left
        # to right with probability .5.  the second pipeline will randomly crop
        # images by between 0 and 10 percent.  each pipeline will be applied to
        # the original dataset.  the resulting dataset will three times as many
        # images as the original - the original dataset, the dataset after
        # augmentation by the rotate_and_flip pipeline, and the dataset
        # after augmentation by the crop pipeline
        rotate_and_flip = augmenters.Sequential([
            augmenters.Affine(rotate=(-45, 45)),
            augmenters.Fliplr(.5)])

        crop = augmenters.Sequential([augmenters.Crop(percent=(0, .1))])

        train_set = augment_dataset(train_set_orig, [rotate_and_flip, crop],
                                    enable_logging=enable_logging)
    else:
        train_set = train_set_orig

    # now create the model
    base_model_name = 'ResNet18_ImageNet_CNTK'
    model = CNTKTLModel(train_set.labels,
                        base_model_name = base_model_name,
                        output_path='.',
                        enable_logging=enable_logging)

    # train the model using cntk
    num_epochs = 45
    mb_size = 32
    model.train(train_set,
                lr_per_mb=[.01] * 20 + [.001] * 20 + [.0001],
                num_epochs=num_epochs,
                mb_size=mb_size)

    # return the accuracy
    ce = ClassificationEvaluation(model, test_set, minibatch_size=16,
                                  enable_logging=enable_logging)
    acc = ce.compute_accuracy()

    return acc

if __name__ == '__main__':
    print('test accuracy is', classify())