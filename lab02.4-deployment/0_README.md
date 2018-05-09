# Deployment of pipeline

This hands-on lab demonstrates the application of the Azure ML Computer Vision Package for operationalization of image classification.

In this lab, we will:
- Deploy the trained model from lab 2.1
- Use the service endpoint for classifying unseen images 

### Learning Objectives ###

The objectives of this lab are to:
- Learn how to setup your environment for deploying
- Learn how to create a service endpoint by deploying a trained model
- Use the service endpoint for scoring 

### Deployment Setup

To deploy the trained model to production, there are several setup steps needed using CLI before running `train_deploy.py`.

1. Set up your environment using the following command:
    
    `az provider register -n Microsoft.MachineLearningCompute`

    `az provider register -n Microsoft.ContainerRegistry`

    `az provider register -n Microsoft.ContainerService`
    
    `az ml env setup --cluster -n <ENVIRONMENT_NAME> -l <AZURE_REGION e.g. eastus2> -g <RESOURCE_GROUP>`

    For example, `az ml env setup --cluster -n cvtkevn -l eastus2`

2. The _provisioning state_ of the environment setup should be  _Succeeded_ to proceed further. You can check the _provisioning state_ by running the below command:

    `az ml env show -n <ENVIRONMENT_NAME> -g <RESOURCE_GROUP>`

    For example, `az ml env show -g cvtkevnrg -n cvtkevn` would produce:

    ```
    {
        "Cluster Name": "cvtkevn",
        "Cluster Size": 2,
        "Created On": "2018-04-16T04:46:49.856Z",
        "Location": "eastus2",
        "Provisioning State": "Succeeded",
        "Resource Group": "cvtkevnrg",
        "Subscription": "5be49961-ea44-42ec-8021-b728be90d58c"
    }
    ```




3. Once _provisioning state_ changes from _Creating_ to _Succeeded_, we can set the above environment as our compute environment:

    `az ml env set -n <ENVIRONMENT_NAME> -g <RESOURCE_GROUP>`

    For exampe, `az ml env set -n cvtkevn -g cvtkevnrg`

4. A model management account is required for deploying models. We usually do this once per subscription, and can reuse the same account in multiple deployments. To create a new model management account and use the model management account, run the below commands:

    `az ml account modelmanagement create -l <AZURE_REGION e.g. eastus2> -n <ACCOUNT_NAME> -g <RESOURCE_GROUP> --sku-instances <NUMBER_OF_INSTANCES, e.g. 1> --sku-name <PRICING_TIER for example S1>`

    `az ml account modelmanagement set -n <ACCOUNT_NAME> -g <RESOURCE_GROUP>`

    ```
    az ml account modelmanagement create -l eastus2 -n cvtkmodel -g cvtkevnrg --sku-instances 1 --sku-name S1
    
    az ml account modelmanagement set -n cvtkmodel -g cvtkevnrg
    ```

5. You are now ready to deploy!!! Run the below script:

    `az ml experiment submit -c local scripts\train_deploy.py`

    and on success, you will see the scoring API URL as shown below along with the service key:

    ```
    Deployment finished
    Please keep the following informatioin for future reference:
    Service id: testdeployment.cvtkevn-55239986.eastus2
    Service endpoint: http://40.84.40.11/api/v1/service/testdeployment/score
    Serivce key: c5c5558a3f134e26b0644ffef511008a
    The scoring API URL is: http://40.84.40.11/api/v1/service/testdeployment/score
    ```

    *** Make a note of the scoring API URL and the service key.


    #### Class map

    When the script `train_deploy.py` is executed, you will also notice the model state is _trained_ and the class map (i.e. class labels with its corresponding index) is produced which can be used during scoring for identifying the classified class.

    ```
    Model state: trained
    Map: {0: 'bowl', 1: 'cup', 2: 'cutlery', 3: 'plate'}
    ```

### Scoring

The script _score_aml.py_ is used for scoring test images using a published AML webservice. The script consists of two main functions _score_service_endpoint_with_images_ and _get_class_.

_score_service_endpoint_with_images_ takes a list of images as arguments along with endpoint url and service key for scoring. In the script, the _io_ module allows us to manage the image-related i/o operations.

_get_class_ is used to obtain the predicted class from a vector of scores using _argMax_.

1. In the _main_ function, ensure that the _service_endpoint_url_ and _service_key_ are updated. You would have found this information when you ran the deployment script.

````python
    service_endpoint_url = "http://40.84.40.11/api/v1/service/testdeployment/score"
    service_key = "c5c5558a3f134e26b0644ffef511008a"
````

2. Execute the script by running `python score_aml.py`

At the end of the script execution, you will see the resulting json obtained from _score_service_endpoint_with_images_ for each of the three test images. The key information displayed are:
- sample image
- time taken for scoring
- the scores for each of the class and the classified label using _argMax_.

````
C:\cvp_project\classification\scripts>python score_aml.py

Scoring image ../sample_data/imgs_recycling/cup/msft-plastic-cup20170725135025957.jpg
   Time for scoring call: 0.39 seconds
[-1.5030657052993774, 7.572079181671143, -3.2485733032226562, -0.32960304617881775]
classified label: cup

Scoring image ../sample_data/imgs_recycling/cup/msft-plastic-cup20170725135335923.jpg
   Time for scoring call: 0.37 seconds
[-3.788055658340454, 7.7130560874938965, -2.6157686710357666, 0.17622099816799164]
classified label: cup

Scoring image ../sample_data/imgs_recycling/cup/msft-plastic-cup20170725135216711.jpg
   Time for scoring call: 0.38 seconds
[-0.9185317754745483, 7.899391174316406, -3.142271041870117, 1.5241585969924927]
classified label: cup
````

4. Open one of the images used for testing (for example: msft-plastic-cup20170725135025957.jpg) and verify that the classified label is _cup_.