import os, time
from cvtk.core import Context, ObjectDetectionDataset, TFFasterRCNN
from cvtk.utils import detection_utils

if __name__ == '__main__':

    Context.load_context_from_json("C:\\path2context\\context.json")
    
    image_folder = "C:\\path2sample_data\\foods\\train"
    data_train = ObjectDetectionDataset.create_from_dir(dataset_name='training_dataset', data_dir=image_folder,
                                                    annotations_dir="Annotations", image_subdirectory='JPEGImages')
    data_train.print_info()
    
    score_threshold = 0.0       # Threshold on the detection score, use to discard lower-confidence detections.
    max_total_detections = 300  # Maximum number of detections. A high value will slow down training but might increase accuracy.
    my_detector = TFFasterRCNN(labels=data_train.labels, 
                            score_threshold=score_threshold, 
                            max_total_detections=max_total_detections)
    
    print("tensorboard logdir:", my_detector.train_dir)

    num_steps = 350
    learning_rate = 0.001 # learning rate

    start_train = time.time()
    my_detector.train(dataset=data_train, num_steps=num_steps, 
                    initial_learning_rate=learning_rate)
    end_train = time.time()
    print(end_train-start_train)

    image_folder = "C:\\path2sample_data\\foods\\test"
    data_val = ObjectDetectionDataset.create_from_dir_pascal_voc(dataset_name='val_dataset', data_dir=image_folder)
    eval_result = my_detector.evaluate(dataset=data_val)
    print (eval_result)

    # print out the performance metric values
    for label_obj in data_train.labels:
        label = label_obj.name
        key = 'PASCAL/PerformanceByCategory/AP@0.5IOU/' + label
        print('{0: <15}: {1: <3}'.format(label, round(eval_result[key], 2)))
        
    print('{0: <15}: {1: <3}'.format("overall:", round(eval_result['PASCAL/Precision/mAP@0.5IOU'], 2)))

    image_path = data_val.images[0].storage_path
    scores = my_detector.score(image_path)
    path_save = "C:\\path2save\\scored_image_preloaded.jpg"
    ax = detection_utils.visualize(image_path, scores, image_size=(8, 12))
    path_save_dir = os.path.dirname(os.path.abspath(path_save))
    os.makedirs(path_save_dir, exist_ok=True)
    ax.get_figure().savefig(path_save)

    save_model_path = "C:\\path2model\\cvtk_output\\frozen_model" # Please save your model to outside of your AML workbench project folder because of the size limit of AML project
    my_detector.save(save_model_path)

    my_detector_loaded = TFFasterRCNN.load(save_model_path)
    detections_dict = my_detector_loaded.score(image_path)

    look_up = dict((v,k) for k,v in my_detector.class_map.items())
    n_obj = 0
    for i in range(detections_dict['num_detections']):
        if detections_dict['detection_scores'][i] > 0.5:
            n_obj += 1
            print("Object {}: label={:11}, score={:.2f}, location=(top: {:.2f}, left: {:.2f}, bottom: {:.2f}, right: {:.2f})".format(
                i, look_up[detections_dict['detection_classes'][i]], 
                detections_dict['detection_scores'][i], 
                detections_dict['detection_boxes'][i][0],
                detections_dict['detection_boxes'][i][1], 
                detections_dict['detection_boxes'][i][2],
                detections_dict['detection_boxes'][i][3]))    
            
    print("\nFound {} objects in image {}.".format(n_obj, image_path))

    path_save = "C:\\path2save\\scored_image_preloaded.jpg"
    ax = detection_utils.visualize(image_path, detections_dict, path_save=path_save, image_size=(8, 12))