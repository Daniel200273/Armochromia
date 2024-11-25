import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from glob import glob
import os
import math
from face_features_v3 import get_avg_color2




def get_colors_dict(image_folder):
    d = dict()
    model = KMeans(n_clusters = 1, n_init=10)

    for image_path in image_folder:
        # Load the image
        image = cv2.imread(image_path)
        # Get the average color of the image (bgr value)
        my_col, _ =get_avg_color2(image, [] , model)
        # Convert the avg color to rgb
        my_col=my_col[0][::-1]
        # Extract the label from the filename
        file_name = os.path.basename(image_path)
        label = os.path.splitext(file_name)[0]
        # Add an entry to the dictionary
        d[label] = my_col
    return d 


# PLOTTING THE DICTIONARY
def plot_dict(d):
    
    l= len(d)
    
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    label, pixel = zip(*lists) # unpack a list of pairs into two tuples
    
    grid_size = math.ceil(math.sqrt(l))
    
    fig , axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*1.5,grid_size*1.5))
    i=0
    for x in range(0,grid_size):
        for y in range(0,grid_size):
            if i < l:
                axs[x][y].imshow([[pixel[i]]])
                axs[x][y].set_title(label=label[i], fontsize=grid_size*0.8)
                axs[x][y].axis(False)
                
            else:
                fig.delaxes(axs[x][y])
                
            i+=1
            
    plt.subplots_adjust(wspace=1, hspace=0)
    
    pass


