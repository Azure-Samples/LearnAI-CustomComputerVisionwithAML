import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import cntk
import os, shutil
import cvtk
import download_images
from cvtk.utils import Constants
from cvtk.core import Context, ClassificationDataset, Image, Label, Splitter, CNTKTLModel
from cvtk.core.ranker import ImagePairs, ImageSimilarityMetricRanker, ImageSimilarityLearnerRanker, ImageSimilarityRandomRanker, RankerEvaluation
from cvtk.utils.ranker_utils import visualize_ranked_images
from cvtk.augmentation import augment_dataset
print(cntk.__version__)

if __name__ == '__main__':


    # Dataset Creation
    if 'AZUREML_NATIVE_SHARE_DIRECTORY' not in os.environ:
        os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] = './share'
    context = Context.get_global_context()


    dataset_name = "fashion"
    dataset_location = os.path.join(Context.get_global_context().storage.outputs_path, "data", dataset_name)
    print("Dataset Location:", dataset_location)

    print("Downloading images to: " + dataset_location)
    download_images.download_all(dataset_location)

    dataset = ClassificationDataset.create_from_dir(dataset_name, dataset_location)
    print("Dataset consists of {} images with {} labels.".format(len(dataset.images), len(dataset.labels)))
    # Split the data into train and test
    splitter = Splitter(dataset)
    train_set, test_set = splitter.split(train_size = .5, random_state=1, stratify="label")
    print("Number of original training images = {}.".format(train_set.size()))

    num_train_sets = 20
    num_test_sets = 20
    num_different_label = 50
    trainPairs = ImagePairs(train_set, num_train_sets, num_different_label)
    print('There are {} sets of image pairs generated for all labels from training data.'.format(len(trainPairs.image_sets)))
    testPairs = ImagePairs(test_set, num_test_sets, num_different_label)
    print('There are {} sets of image pairs generated for all labels from training data.'.format(len(testPairs.image_sets)))

    # Model Training

    refineDNN = False # Use the pretrained model as-is or refine
    model = CNTKTLModel(train_set.labels, class_map = {i: l.name for i, l in enumerate(dataset.labels)}, base_model_name='ResNet18_ImageNet_CNTK')
    if refineDNN:
        model.train(train_set)

    similarityMethod = "L2" # Options: "random", "L2", "svm"

    if similarityMethod == "random":
        ranker = ImageSimilarityRandomRanker()
    elif similarityMethod == "L2":
        ranker = ImageSimilarityMetricRanker(model, metric="L2")
    elif similarityMethod == "svm":
        from sklearn.svm import LinearSVC
        # SVM-defined weighted L2-distance. Need to train, but this is fast.
        svmLearner = LinearSVC(C = 0.01)
        ranker = ImageSimilarityLearnerRanker(model, learner=svmLearner)

    # Train the ranker, random and L2 do not need training and .train() will do nothing
    ranker.train(trainPairs)

    re = RankerEvaluation(ranker, testPairs)
    mean_rank = re.compute_mean_rank()
    median_rank = re.compute_median_rank()
    print("mean rank:", mean_rank)
    print("median rank:", median_rank)
    acc_plot = re.top_n_acc_plot(n=32, visualize=True)
    re.visualize_results(n = 5, visualize=True)