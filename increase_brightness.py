#%%
import cv2
import numpy as np
import glob
import os

def contrast(img):
    img = cv2.imread(img, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1
#%%
# os.chdir(r"/home/waqas/projects/unet/clahe")
imgs = glob.glob('/home/waqas/projects/unet/')

c = 0
for i in imgs:
    j = contrast(i)
    cv2.imwrite("%s.png" % c, j)
    c +=1
