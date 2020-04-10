#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

#%%

c = 0
for i in glob.glob('/home/waqas/projects/unet/*t.png'):
    img = cv2.imread(i, 0)
    ret, thresh = cv2.threshold(img, 99, 255, cv2.THRESH_BINARY)
    cv.imwrite('IMG%s.png' % c, thresh)
    c = c+1

#%%
img = cv2.imread('0_predict.png', 0)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,25,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imwrite('detected_circles.png',cimg)

