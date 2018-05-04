import os

from cvtk import Splitter
from cvtk.augmentation import augment_dataset
from cvtk.core import (Context, ClassificationDataset, StorageContext, CNTKTLModel)
from cvtk.evaluation import ClassificationEvaluation
from cvtk.operationalization import (AMLDeployment, delete_deployment,
                                     delete_if_service_exist)
from imgaug import augmenters

### NOTE: Please do ensure that you have set up the deployment targets 
### as per the readme instructions (prerequistes) before running this script.

def train_deploy(dataset_location='classification/sample_data/imgs_recycling',
                 dataset_name='recycling', do_augmentations=True,
                 deployment_name="testdeployment", azureml_rscgroup=None, azureml_cluster_name=None):
    """
      a sample pipeline for deploying themodel that is trained on a dataset.

      loads a dataset, optionally does some augmentations, creates and trains a
      model using transfer learning based on ResNet18, deploys the trained model
      on the specified Azure ML cluster or picks up the one set using the CLI.
      and returns the Scoring URL.

      Args:
        dataset_location: path to a dataset.  there should be a top level folder
          containing folders for each class.  see the sample recycling dataset for
          an example of the format
        dataset_name: the of the dataset.  will be used in the dataset
          management functionality
        do_augmentations: boolean.  specifies whether augmentations should be
          applied to the test set
        deployment_name: the deployment of the deployment. Will be used in deployment 
          management facility
        azureml_rscgroup: Azure ML resource group name of the model management account.
           If not set, default value will be picked up if set from CLI
        azureml_cluster_name: Azure ML cluster name where the model is deployed. If not set, 
           default value will be picked up if set from CLI.

      Returns:
        the scoring API URL of the deployment
    """
    # if we're not running inside AML WB, set up the share directory
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' not in os.environ:
        os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] = './share'
    context = Context.get_global_context()

    # create a dataset from a directory with folders representing different classes
    dataset = ClassificationDataset.create_from_dir(dataset_name, dataset_location)

    # print out some info about the dataset
    dataset.print_info()

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

        train_set = augment_dataset(dataset, [rotate_and_flip, crop])
    else:
        train_set = dataset

    # now create the model
    base_model_name = 'ResNet18_ImageNet_CNTK'
    model = CNTKTLModel(train_set.labels,
                        base_model_name = base_model_name,
                        output_path='.')

    # train the model using cntk
    num_epochs = 5
    mb_size = 32
    model.train(train_set,
                lr_per_mb=[.01] * 20 + [.001] * 20 + [.0001],
                num_epochs=num_epochs,
                mb_size=mb_size)

    print("Model state:", model.model_state)

    # check if the deployment exists, if yes remove it first
    AMLDeployment.delete_if_service_exist(deployment_name)

    #deploy the trained model
    deploy_obj = AMLDeployment(
            deployment_name=deployment_name, associated_DNNModel=model, aml_env = "cluster", replicas=1)
    deploy_obj.deploy()

    return deploy_obj.service_url

if __name__ == '__main__':
    print("The scoring API URL is:", train_deploy())
