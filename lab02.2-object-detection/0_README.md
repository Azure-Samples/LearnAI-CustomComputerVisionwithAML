# Object Detection

Object detection is a process for identifying a specific object in a digital image. Specifically, detection is about not only finding the class of object but also localizing the extent of an object in the image. In recent times, Deep learning based methods have become the state of the art in object detection in image. They construct a representation in a hierarchial manner with increasing order of abstraction from lower to higher levels of neural network.

The problem we are solving in this lab is to identify grocery items present in refrigerators. Specifically, in this lab, we will:
- Train an object detection model using AML Package for Computer Vision using a GPU equipped DSVM
- Create and evaluate the results on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), one of the main object detection challenges in the Computer Vision field
- Visualize and save the detected objects on an unseen image

### Learning Objectives ###

The objectives of this lab are to:
- Learn how to create a dataset and it's format for object detection
- Learn how to train and evaluate an object detection model based on [Faster R-CNNs](https://arxiv.org/abs/1506.01497)
- Learn how to save the model and detect objects on new images

### DSVM equipped with GPU ###

Our first step is to make sure we have access to a VM with a GPU.

1. Open your web browser and go to the Azure portal
2. Select + New on the left of the portal. Search for Data Science Virtual Machine for Linux Ubuntu CSP in the marketplace. Choosing Ubuntu is critical.
3. Click Create to create an Ubuntu DSVM.
4. Fill in the Basics blade with the required information. When selecting the location for your VM, note that GPU VMs (e.g. NC-series) are only available in certain Azure regions, for example, South Central US. See compute products available by region. Click OK to save the Basics information.
5. Choose the size of the virtual machine. Select one of the sizes with NC-prefixed VMs, which are equipped with NVidia GPU chips. Click View All to see the full list as needed. Learn more about GPU-equipped Azure VMs.
6. Finish the remaining settings and review the purchase information. Click Purchase to create the VM. Take note of the IP address allocated to the virtual machine - you will need this (or a domain name) in the next section when you are configuring AML.

### Data

The dataset (resources\sample_data.zip) used in this lab consists of grocery items inside refrigerators and include the following classes: `Egg box, joghurt, ketchup, mushroom, mustard, orange, squash,water, etc`. An example of annotated image shown below:

![Annotated Image](images\sample_image.jpg)

In this lab, we will first create an "Object Detection Dataset" using the helper function _create_from_dir_pascal_voc_ that takes the image folder as an argument. The ObjectDetectionDataset object consists of a set of images, with their respective bounding box annotations.

````python
    image_folder = "C:\\path2sample_data\\liebherr\\liebherr_train"
    data_train = ObjectDetectionDataset.create_from_dir_pascal_voc(dataset_name='training_dataset', data_dir=image_folder)
````

The image training folder contains two folders:

- Annotations

The annotation files capture each object for the corresponding image (in JPEGImages folder) along with a bounding box that includes the top left corner and bottom right corner points.

At the time of developing this lab, support is present only for bounding boxes and not other shapes.

```
    <object>
            <name>tomato</name>
            <pose>Unspecified</pose>
            <bndbox>
                <xmin>828</xmin>
                <ymin>1068</ymin>
                <xmax>982</xmax>
                <ymax>1196</ymax>
            </bndbox>
    </object>
```

- JPEGImages

The JPEGImages folder contains the raw jpg images. For each jpg image in the folder, an annotation xml-file with similar name exists in Annotations folder.

### Execution

To execute the detection.py script located in resources, launch Azure Machine Learning Workbench and open CLI by selecting File -> Open Command Prompt. Run the below command and walk through the code. The script would need to be edited to change the path references to the datasets.


```az ml experiment submit -c local detection.py```

### Model Definition and Training

In this lab, we will use [Faster R-CNN](https://arxiv.org/abs/1506.01497), a significant improvement of R-CNNs for object detection. To define the [Faster R-CNN](https://arxiv.org/abs/1506.01497) model, we will set _score_threshold_ and _max_total_detections_. _score_threshold_ is used for thresholding the detection score and _max_total_detections_ is used for the number of detections allowed. The larger the value, slower the training (but may increase accuracy).


````python
    score_threshold = 0.0
    max_total_detections = 300
    my_detector = TFFasterRCNN(labels=data_train.labels, 
                           score_threshold=score_threshold, 
                           max_total_detections=max_total_detections)
````


After the model is defined, we can train the object detector. Training on a GPU machine using the refrigerator dataset can take up to 5 minutes. The number or training steps in the code is set to 350, so that training runs quickly in about 5 minutes. In practice, this should be set to at least 10 times the number of images in the training set.

#### Training Parameters

Two key training parameters are number of steps and learning rates. The argument _num_steps_ is for specifying the number of minibatches used to train the model. Since the minibatch size is set to 1 in this release, it equals the number of images considered during training.

Learning rates can be set as follows with the below arguments:
- _initial_learning_rate_: the initial learning rate.
- _learning_rate1_: the learning rate for steps since step1. For example, learning_rate1=0.0003 (default) and step1=0  (default) means that the learning rate is 0.0003 from step 0. This values overrides initial_learning_rate if step1=0.
- _learning_rate2_: the learning rate for steps since step2. For instance, learning_rate1=0.00003 (default) and step2=900000 (default) means that the learning rate is 0.00003 from step 900,000.
- _learning_rate3_: the learning rate for steps since step3. For example, learning_rate1=0.000003  (default) and step2=1200000  (default) means that the learning rate is 0.000003 from step 1,200,000.

Below is an example snippet where the parameters are passed for training:

````python
    num_steps = 350
    learning_rate = 0.001 # learning rate
    step1 = 200 

    start_train = time.time()
    my_detector.train(dataset=data_train, num_steps=num_steps, 
                    initial_learning_rate=learning_rate,
                    step1=step1,
                    learning_rate1=learning_rate)
    end_train = time.time()
````

### Evaluation

We will first create a validation dataset first by calling the helper function _create_from_dir_pascal_voc_ using the validation data as the argument:

````python
    image_folder = "C:\\path2sample_data\\liebherr\\liebherr_val"
    data_val = ObjectDetectionDataset.create_from_dir_pascal_voc(dataset_name='val_dataset', data_dir=image_folder)
````

Accuracy can be obtained for each category and overall using the below code:

````python
    eval_result = my_detector.evaluate(dataset=data_val)

    for label_obj in data_train.labels:
        label = label_obj.name
        key = 'PASCAL/PerformanceByCategory/AP@0.5IOU/' + label
        print('{0: <15}: {1: <3}'.format(label, round(eval_result[key], 2)))
        
    print('{0: <15}: {1: <3}'.format("overall:", round(eval_result['PASCAL/Precision/mAP@0.5IOU'], 2))) 
````

and accuracy metrics for each category looks as shown below. Notice the last category 'overall' that combines all the categories.

```
onion          : 1.0
avocado        : 1.0
eggBox         : 1.0
ketchup        : 0.92
orange         : 1.0
pepper         : 1.0
gerkin         : 1.0
joghurt        : 1.0
tomato         : 1.0
orangeJuice    : 1.0
champagne      : 1.0
tabasco        : 0.18
milk           : 1.0
butter         : 1.0
water          : 0.7
mustard        : 1.0
overall:       : 0.92
```
### Saving the model

To save the model to a particular path, run the below code and change the model_dir:

````python
    model_dir = "C:\\path2sample_data\\cvtk_output\\frozen_model"
    frozen_model_path, label_map_path = my_detector.save(model_dir)
    print("Frozen model written to path: " + frozen_model_path)
    print("Labels written to path: " + label_map_path)
````

### Scoring

We can obtain labels, scores and coordinates of all the detected objects in the image using the below code. The dictionary _detections_dict_ contains object attributes including class, scores, and bounding box coordinates for each object. We also threshold to view objects with scores more than 0.5:

````python
    image_path = 'C:\\path2sample_data\\liebherr\\liebherr_val\\JPEGImages\\WIN_20160803_11_46_03_Pro.jpg'
    dectections_dict = detection_utils.score(frozen_model_path, image_path)


    look_up = dict((v,k) for k,v in my_detector.class_map.items())
    n_obj = 0
    for i in range(dectections_dict['num_detections']):
        if dectections_dict['detection_scores'][i] > 0.5:
            n_obj += 1
            print("Object {}: label={:11}, score={:.2f}, location=(top: {:.2f}, left: {:.2f}, bottom: {:.2f}, right: {:.2f})".format(
                i, look_up[dectections_dict['detection_classes'][i]], 
                dectections_dict['detection_scores'][i], 
                dectections_dict['detection_boxes'][i][0],
                dectections_dict['detection_boxes'][i][1], 
                dectections_dict['detection_boxes'][i][2],
                dectections_dict['detection_boxes'][i][3]))    
            
    print("\nFound {} objects in image {}.".format(n_obj, image_path)) 
````

The attributes of each object are captured as follows:

```
Object 0: label=butter, score=0.95, location=(top: 0.35, left: 0.56, bottom: 0.41, right: 0.81)
Object 1: label=pepper, score=0.93, location=(top: 0.67, left: 0.30, bottom: 0.76, right: 0.51)
Object 2: label=tomato, score=0.91, location=(top: 0.51, left: 0.54, bottom: 0.62, right: 0.86)
Object 3: label=tomato, score=0.89, location=(top: 0.34, left: 0.38, bottom: 0.41, right: 0.54)
Object 4: label=eggBox, score=0.88, location=(top: 0.67, left: 0.01, bottom: 0.83, right: 0.30)
Object 5: label=avocado, score=0.79, location=(top: 0.52, left: 0.35, bottom: 0.61, right: 0.53)
Object 6: label=gerkin, score=0.73, location=(top: 0.45, left: 0.05, bottom: 0.64, right: 0.31)
```

### Visualization

For a given query image, you can visualize the objects detected using _detection_utils.visualize_:

````python
    image_size=(10.8, 19.2)
    path_save = "C:\\path2sample_data\\scored_image.jpg"
    _ = detection_utils.visualize(image_path, dectections_dict, label_map_path, path_save=path_save, image_size=image_size)
````

![scored image](images\scored_image.png)

### Exercise

1. Change the score threshold and visualize the objects on the query image.

2. Can you think of ways to segment the object from the bounding box?

