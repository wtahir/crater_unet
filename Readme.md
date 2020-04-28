

Make sure you have requirements.txt satisfied before running any notebook or python file.

Genration of binary masks and later on, binarization filters applied on pobability maps generated by the unet are provided in jupyter notebook of 'maskfrom_json'. This notebook has all the steps from pre-processing of generation of binary masks from annotations (json) and application of binary image filters to extract features from lunar optical images. 

Crop_image.ipnyb is used to crop images of the sizes 256x256.

DataPrepare.ipnyb is for data preparation for u-net training

Plotting sigmoig_relu.ipnyb are the plotting functions for relu and sigmoid functions

TrainUnet.ipnyb is a jupyter notebook for training and prediction of dataset.

- Main.py is the model configuration

- Model.py is model architecture

 All other functions are named after the task.  


