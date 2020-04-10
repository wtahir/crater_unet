#%%
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
from PIL import Image
import numpy as np
 
def numpytoimage(numpy):
    numpy = numpy * 255
    image= Image.fromarray(numpy.astype(np.uint8))
    return image


reference = cv2.imread("img_5475.jpg",0)
_, thresh_ref = cv2.threshold(reference, 75, 1, 0)

extract = cv2.imread("marking.jpg",0)
_, thresh_extract = cv2.threshold(extract, 75, 2, 0)


C = np.zeros(shape=(len(thresh_ref), len(thresh_ref[0]), 3))

for i in range (0, thresh_ref.shape[0],1):
    for j in range(0, thresh_ref.shape[1], 1):
        if thresh_ref[i][j] == thresh_extract[i][j] and thresh_ref[i][j] == 0:
            C[i][j] = 1
        elif thresh_ref[i][j] == 0:
            C[i][j][0] = 0
            C[i][j][1] = 1
            C[i][j][2] = 0
        elif thresh_extract[i][j] == 0:
            C[i][j][0] = 1
            C[i][j][1] = 0
            C[i][j][2] = 0
        else:
            C[i][j][0] = 0.5
            C[i][j][1] = 0.5
            C[i][j][2] = 0.5

C_image = numpytoimage(C)
C_image.save("quality.png")

#%%
import cv2

dino = cv2.imread('marking.jpg', 0)
