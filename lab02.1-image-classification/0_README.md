# Image Classification

This hands-on lab demonstrates the application of the Azure ML Computer Vision Package for image classification.

In this lab, we will:
- Create a sample pipeline for image classification
- Ingest an image dataset
- Perform image augmentations
- Train a model using transfer learning based on ResNet18

### Learning Objectives ###

The objectives of this lab are to:

- Understand the image classification workflow
- Learn how to format a dataset in order to ingest and perform augmentations
- Train a model using transfer learning based on ResNet18 and evaluate the classifier

### Data

In this lab, we will use a sample classification dataset (resources/sample_data.zip) related to recyling dishes. The dataset consists of four classes: bowls, plates, cups and cutlery as shown below: 



| Bowl |Plate|Cup|Cutlery| 
|------|------|------|-----|
|![bowl](images\bowl.jpg)|![plate](images\plate.jpg)|![cup](images\cup.jpg)|![bowl](images\cutlery.jpg)

The sample dataset indicates the format required to ingest in the pipeline. You will find that there should be a top level folder containing folders for each class:
````
imgs_recycling\bowl\...
imgs_recycling\cup\...
imgs_recycling\cutlery\...
imgs_recycling\plate\...
````

### Execution

Launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt.

The script to perform training and evaluation is `resources/train.py`. Execute the train script by runing the below command and walk through the code:

```az ml experiment submit -c local train.py```

The main function that does training and evaluation is _classify_ in `train.py`. Ensure the argument _dataset_location_ points to the dataset folder provided in _resources/sample_data_.

#### Dataset creation from directory

The below python code will create a dataset from a directory with folders representing different classes. The _print_info()_ function provides detailed breakdown and distribution of the dataset as shown below.

````python
dataset = ClassificationDataset.create_from_dir(dataset_name, dataset_location, enable_logging=enable_logging)

dataset.print_info()
`````

```
Dataset name: recycling
Total classes: 4 ,total images: 63
Label-wise image count:
         bowl : 5 images
         cup : 19 images
         cutlery : 7 images
         plate : 32 images
Sample images for each class:
         bowl :
                 C:\AppData\Temp\2\azureml_runs\cvp_project_1523844450768\classification\sample_data\imgs_recycling\bowl\msft-plastic-bowl20170725152138800.jpg
                 C:AppData\Temp\2\azureml_runs\cvp_project_1523844450768\classification\sample_data\imgs_recycling\bowl\msft-plastic-bowl20170725152141939.jpg
                 C:AppData\Temp\2\azureml_runs\cvp_project_1523844450768\classification\sample_data\imgs_recycling\bowl\msft-plastic-bowl20170725152154282.jpg
```

#### Dataset splitting

The _cvtk_ package offers several utility functions to split the dataset into a train and test set. Additionally, the stratify option will ensure that the different labels are balanced in the train and test sets. In the below code snippet, the train and test sets are created using a 0.8/0.2 proportion and using the stratified option.

````python
splitter = Splitter(dataset, enable_logging=enable_logging)

train_set_orig, test_set = splitter.split(train_size=.8, stratify='label')
````

### Augmentation

To achieve good performance, deep networks require large amount of training data. To build a robust image classifier using very little training data, image augmentation is usually required to boost the performance of deep networks.  Image augmentation artificially creates training images through different ways of processing or combinations of multiple processing, such as random rotation, shifts, shear and flips, etc.

In this lab, we will demonstrate how you can perform a few augmentations using the _augmenters_ module:

1. We will first rotate each image randomly between -45 and 45 degrees. The rotated images are then flipped from left to right with 0.5 probability.  

````python
rotate_and_flip = augmenters.Sequential([augmenters.Affine(rotate=(-45, 45)), augmenters.Fliplr(.5)])
````

2. We will randomly crop images between 0 and 10 percent. 


````python
crop = augmenters.Sequential([augmenters.Crop(percent=(0, .1)])
````

The resulting dataset will be three times as many as the original - the original dataset, the dataset after augmentation by the rotate/flip and crop.

````python
train_set = augment_dataset(train_set_orig, [rotate_and_flip, crop], enable_logging=enable_logging)
````

### Model Training

In this lab, we will use Deep Residual Learning for classification. Deep residual networks took the deep learning world by storm when Microsoft Research released [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf). ResNets led to 1st-place winning entries in all five main tracks of the ImageNet and COCO 2015 competitions, covering image classification, object detection, and semantic segmentation.

The robustness of ResNets has since been proven by various visual recognition tasks and by non-visual tasks involving speech and language. Briefly, with ResNets, we explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. The below snippet trains a ResNet model using transfer learning and computes accuracy on _test_set_.


````python
    base_model_name = 'ResNet18_ImageNet_CNTK'
    model = CNTKTLModel(train_set.labels,
                        base_model_name = base_model_name,
                        output_path='.',
                        enable_logging=enable_logging)

    num_epochs = 45
    mb_size = 32
    model.train(train_set,
                lr_per_mb=[.01] * 20 + [.001] * 20 + [.0001],
                num_epochs=num_epochs,
                mb_size=mb_size)

    ce = ClassificationEvaluation(model, test_set, minibatch_size=16,
                                  enable_logging=enable_logging)
    acc = ce.compute_accuracy()

````

On completion of the script execution, you will see the location of the trained model and accuracy as shown below:

```
Finished Epoch[43 of 45]: [Training] loss = 0.003055 * 150, metric = 0.00% * 150 1.080s (138.9 samples/s);
Finished Epoch[44 of 45]: [Training] loss = 0.002405 * 150, metric = 0.00% * 150 1.114s (134.7 samples/s);
Finished Epoch[45 of 45]: [Training] loss = 0.005883 * 150, metric = 0.00% * 150 1.040s (144.2 samples/s);
Stored trained model at .\outputs\.\ImageClassification.model
test accuracy is 1.0
```

### Exercise

There are plenty of [online image datasets](http://clickdamage.com/sourcecode/cv_datasets.php) available for ingesting and training using Azure ML Computer Vision Package. Can you create (by setting the folder structure and formating) a small dataset using any of [online image datasets](http://clickdamage.com/sourcecode/cv_datasets.php) and integrate within the workflow?
