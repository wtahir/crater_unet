
#%%

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/waqas/projects/unet/0.png',0)
plt.hist(img.ravel(),256,[0,256])
axes = plt.gca()
axes.set_xlim([230,260])
axes.set_ylim([0,600])
plt.ylabel('Number of pixels')
plt.xlabel('Pixel intensity')
# plt.savefig('hist.eps', format='eps')
plt.show()


#%%

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
