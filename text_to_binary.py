#%%
from skimage.draw import circle
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import Point

#%%
df = pd.read_csv('test/13.txt', sep='\t', header=None)
# df = df[df[0] < 230]
# df.to_csv('waqas_dia.txt', sep='\t', header=False, index=False)
#%%
img = np.zeros((256, 256), dtype=np.uint8)
# cv2.imwrite('empty.png', img)

# img = cv2.imread('empty.png', 0)
for index, row in df.iterrows():
    try:
        rr, cc = circle(int(row[1]), int(row[2]), int(row[0]/2))
        img[rr, cc] = 255
    except:
        print(index)
    
    # img = Image.fromarray(img)
    cv2.imwrite('gt_test.png', img)