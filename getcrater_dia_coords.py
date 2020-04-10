#%%
import scipy.ndimage
from skimage.measure import regionprops
import pandas as pd
import cv2
# import fire

def dia_coords(path):

    """ Takes image with prediction and returns a txt file with number, dia and coordinates

    Arguments:
        path {'str'} -- path to prediction image file
    """
    img = cv2.imread(path, 0)
    blobs = img > 0.1
    col = []
    labels, nlabels = scipy.ndimage.label(blobs)
    properties = regionprops(labels)
    # print ('Label \tLargest side')
    for p in properties:
        min_row, min_col, max_row, max_col = p.bbox
        s = ('%5d %14.2f %14.2d %14.2d' % (p.label, max(max_row - min_row, max_col - min_col), ((max_row - min_row)/2 + min_row), ((max_col - min_col)/2 + min_col)))
        col.append(s)
    df = pd.DataFrame(col,  columns=['number_craters'])
    df = pd.DataFrame(df.number_craters.str.split().tolist(), columns=['number', 'dia', 'y', 'x'])
    # df.to_csv('crater_dia.txt', sep='\t',index=False)
    return  df

# if __name__ == "__main__":
#     fire.Fire(dia_coords)

#%%
import glob

for path in glob.glob('/home/waqas/projects/unet/2nd_exp/brighted/*t.png'):
    img = cv2.imread(path, 0)
    blobs = img > 0.1
    col = []
    labels, nlabels = scipy.ndimage.label(blobs)
    properties = regionprops(labels)
    # print ('Label \tLargest side')
    for p in properties:
        min_row, min_col, max_row, max_col = p.bbox
        s = ('%5d %14.2f %14.2d %14.2d' % (p.label, max(max_row - min_row, max_col - min_col), ((max_row - min_row)/2 + min_row), ((max_col - min_col)/2 + min_col)))
        col.append(s)
    df = pd.DataFrame(col,  columns=['number_craters'])
    df = pd.DataFrame(df.number_craters.str.split().tolist(), columns=['number', 'dia', 'y', 'x'])
    



#%%
