#%%
import cv2
import numpy as np

img_gray = cv2.imread('hcon_thesis-1.png',0)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)
cimg = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, minDist=15,
                     param1=50, param2=18, minRadius=12, maxRadius=22)

if circles is not None:
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imwrite('with_circles.png', cimg)
# %%
