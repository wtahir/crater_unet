#%%
# Projection of craters

import cv2
import pandas as pd
import os

# os.chdir(r'/home/waqas/projects/unet')

# img = cv2.imread('/home/waqas/projects/unet/test/predictions')

#%%
df = pd.read_csv('/home/waqas/projects/unet/results/otsu/otsu.txt', sep="\t", header=None)
df = df[df[0] > 1]
img = cv2.imread('/home/waqas/projects/unet/moon_image.jpg')
for index, row in df.iterrows():
    i = cv2.circle(img, (int(row[2]), int(row[1])), int(row[0]/2), (0,0,255), thickness=2, lineType=8, shift=0)
    # cv2.circle(frame1,tuple(point),1,(0,0,255))
cv2.imwrite('/home/waqas/projects/unet/final_results/otsu/moon_otsu.png', i)



#%%
df = pd.read_csv('crater_brighted_all.txt', sep='\t')
df = df.drop('number', axis=1)
df = df.where(df['dia'] < 210)
df = df.dropna()
df.to_csv('crater_brighted_all_filtered.txt', sep='\t')
