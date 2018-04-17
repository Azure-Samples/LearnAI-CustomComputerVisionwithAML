import base64
import json
import numpy as np
import requests
import sys, traceback
import timeit
from io import BytesIO
from PIL import Image
"""
Sample script to score a published AML webservice directly on test images.
"""

def get_class(class_index):
    class_map = dict()
    class_map[0] = 'bowl'
    class_map[1] = 'cup'
    class_map[2] = 'cutlery'
    class_map[3] = 'plate'
    if class_index in class_map.keys():
        return class_map[class_index]

def score_service_endpoint_with_images(images, service_endpoint_url,  parameters ={},  service_key=None):
    """Score image list against a service endpoint

    Args:
        images(list): list of (input image file path)
        service_endpoint_url(str): endpoint url
        service_key(str): service key, None for local deployment.
        parameters(dict): service additional paramters in dictionary
        image_resize_dims(list or tuple): resize image if provided. Format: [width, height].
        
    Returns:
        result (list): list of result for each image
    """
    routing_id = ""
    if service_key is None:
        headers = {'Content-Type': 'application/json',
                   'X-Marathon-App-Id': routing_id}
    else:
        headers = {'Content-Type': 'application/json',
                   "Authorization": ('Bearer ' + service_key), 'X-Marathon-App-Id': routing_id}
    payload = []
    for image in images:
        encoded = None
        img = Image.open(image).convert('RGB')
        image_buffer = BytesIO()
        img.save(image_buffer, format="png")
        encoded = base64.b64encode(image_buffer.getvalue())
        image_request = {"image_in_base64": "{0}".format(encoded), "parameters": parameters}
        payload.append(image_request)
    body = json.dumps(payload)
    r = requests.post(service_endpoint_url, data=body, headers=headers)
    try:
        result = json.loads(r.text)
    except:
        raise ValueError("Incorrect output format. Result cant not be parsed: " +r.text)
    return result

# Score images on disk using deployed endpoint. 
def main():
    service_endpoint_url = "http://40.84.40.11/api/v1/service/testdeployment/score" # Please replace this with your service endpoint url
    service_key = "73246bc47340467e97915fb2aed7c6d7" # Please replace this with your service key
    parameters = {}
    test_images = [
                    '../sample_data/imgs_recycling/cup/msft-plastic-cup20170725135025957.jpg',
                    '../sample_data/imgs_recycling/cup/msft-plastic-cup20170725135335923.jpg',
                    '../sample_data/imgs_recycling/cup/msft-plastic-cup20170725135216711.jpg',
                    ]

    for img in test_images:
        tic = timeit.default_timer()
        return_json = score_service_endpoint_with_images([img], service_endpoint_url,  parameters =parameters, service_key=service_key)[0]
        print('Scoring image {}'.format(img))
        print("   Time for scoring call: {:.2f} seconds".format(timeit.default_timer() - tic))
        # parse returned json string
        result = json.loads(return_json)
        class_index = np.argmax(np.array(result))
        print(return_json)
        print("classified label: {}".format(get_class(class_index)))

if __name__ == "__main__":
    main()
