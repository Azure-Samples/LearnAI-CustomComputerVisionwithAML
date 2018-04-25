import os, time
from cvtk.core import Context, ObjectDetectionDataset, TFFasterRCNN
from cvtk.utils import detection_utils

if __name__ == '__main__':

    # Initialize the context object
    
    Context.load_context_from_json("C:\\path2context\\context.json")
    

    image_folder = "C:\\path2data\\sample_data\\liebherr\\liebherr_train"
    data_train = ObjectDetectionDataset.create_from_dir_pascal_voc(dataset_name='training_dataset', data_dir=image_folder)
    score_threshold = 0.0       # Threshold on the detection score, use to discard lower-confidence detections.
    max_total_detections = 300  # Maximum number of detections. A high value will slow down training but might increase accuracy.
    my_detector = TFFasterRCNN(labels=data_train.labels, 
                           score_threshold=score_threshold, 
                           max_total_detections=max_total_detections)
    
    print("tensorboard logdir:", my_detector.train_dir)

    num_steps = 350
    learning_rate = 0.001 # learning rate
    step1 = 200 

    start_train = time.time()
    my_detector.train(dataset=data_train, num_steps=num_steps, 
                    initial_learning_rate=learning_rate,
                    step1=step1,
                    learning_rate1=learning_rate)
    end_train = time.time()
    print(end_train-start_train)

    image_folder = "C:\\path2data\\sample_data\\liebherr\\liebherr_val"
    data_val = ObjectDetectionDataset.create_from_dir_pascal_voc(dataset_name='val_dataset', data_dir=image_folder)
    eval_result = my_detector.evaluate(dataset=data_val)
    print (eval_result)

    # print out the performance metric values
    for label_obj in data_train.labels:
        label = label_obj.name
        key = 'PASCAL/PerformanceByCategory/AP@0.5IOU/' + label
        print('{0: <15}: {1: <3}'.format(label, round(eval_result[key], 2)))
        
    print('{0: <15}: {1: <3}'.format("overall:", round(eval_result['PASCAL/Precision/mAP@0.5IOU'], 2)))

    model_dir = "../../../cvtk_output/frozen_model"
    frozen_model_path, label_map_path = my_detector.save(model_dir)
    print("Frozen model written to path: " + frozen_model_path)
    print("Labels written to path: " + label_map_path) 


    # Score Image

    image_path = 'C:\\Users\\miprasad\\Downloads\\cvp-1.0.0b2-release5\\cvp-1.0.0b2-release\\cvp_project\\detection\\sample_data\\liebherr\\liebherr_val\\JPEGImages\\WIN_20160803_11_46_03_Pro.jpg'
    # scores = my_detector.score_image(frozen_model_path, image_path)
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

    image_size=(10.8, 19.2)
    path_save = "C:\\Users\\miprasad\\AppData\\Local\\Temp\\cvtk_output\\scored_image.jpg"
    _ = detection_utils.visualize(image_path, dectections_dict, label_map_path, path_save=path_save, image_size=image_size)