
## Usage
Make sure you have ```requirements.txt``` satisfied before running any notebook or python file.

Genration of binary masks and later on, binarization filters applied on pobability maps generated by the unet are provided in jupyter notebook of:
```
maskfrom_json.ipynb
```
This notebook has all the steps from pre-processing of generation of binary masks from annotations (json) and application of binary image filters to extract features from lunar optical images. 

To crop images (size 256x256):
```
crop_image.ipnyb
```
Prepare data for u-net training using:
```
dataPrepare.ipnyb
```
For training and prediction, use the following notebook.
```
trainUnet.ipnyb
```
Model configuration is written in:
```
main.py
```
Model architecture is given in presented in following file. The model is based on u-net written by [Ronneberger et al., 2015] and keras implementation by [https://github.com/zhixuhao]
```
model.py is 
```


