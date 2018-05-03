# Setting up for the bootcamp

The following steps will get us up and running for the bootcamp. These activities take about 3 hours in total and should be completed **prior to attending** the bootcamp. Failure to do so will put you behind the rest of the class and divert your time and attention from the material covered throughout the bootcamp. As proof that you have completed the below steps, you must bring the "Web Service ID" code (described below) with you to sign in at the bootcamp.

##  What you will need for this workshop: 

### DSVM equipped with GPU ###

To use a DSVM equipped with GPU:

1. Open your web browser and go to the Azure portal
2. Select + New on the left of the portal. Search for Data Science Virtual Machine for Linux Ubuntu CSP in the marketplace. Choosing Ubuntu is critical.
3. Click Create to create an Ubuntu DSVM.
4. Fill in the Basics blade with the required information. When selecting the location for your VM, note that GPU VMs (e.g. NC-series) are only available in certain Azure regions, for example, South Central US. See compute products available by region. Click OK to save the Basics information.
5. Choose the size of the virtual machine. Select one of the sizes with NC-prefixed VMs, which are equipped with NVidia GPU chips. Click View All to see the full list as needed. Learn more about GPU-equipped Azure VMs.
6. Finish the remaining settings and review the purchase information. Click Purchase to create the VM. Take note of the IP address allocated to the virtual machine - you will need this (or a domain name) in the next section when you are configuring AML.

### Azure Machine Learning Workbench

 -  A Microsoft Azure account where you can create resources, including Application Insights. This could be an organization account, an MSDN subscription account, a Trial Account, or an account provided by your company.
 -  A Microsoft Azure Machine Learning Experimentation and Model Management Account.
 -  A Windows machine on which you can install software, and which can run Docker. In order to accomplish the steps below, this machine must have Docker and Azure Machine Learning installed on it.
 - #### Setting up your environment ####
    Once you have the above requirements in place, you should be able to execute most of the online Iris tutorial by performing [Part 1](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-1), [Part 2](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-2) and [Part 3](https://docs.microsoft.com/en-us/azure/machine-learning/preview/tutorial-classifying-iris-part-3).

###  AML Package for Computer Vision

- Go to AML Package for Computer Vision internal Private Preview site located [here](https://aka.ms/aml-cv)
- Follow the [getting started and installation instructions](https://microsoft.sharepoint.com/teams/ComputerVisionPackage/_layouts/WopiFrame.aspx?sourcedoc=%7B080506BD-B5F3-493C-B867-F29CF68421D8%7D&file=Quickstart%20AML%20Package%20For%20Computer%20Vision.docx&action=default)