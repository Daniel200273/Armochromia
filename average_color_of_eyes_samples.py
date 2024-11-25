import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from glob import glob
import os
import math
from face_features_v3 import get_avg_color2

def distance_from_black(rgb):
    """Calculate the Euclidean distance of an RGB value from black (0, 0, 0)."""
    return np.sqrt(rgb[0]**2 + rgb[1]**2 + rgb[2]**2)

def drop_closer_to_black(colors):
    """Return the RGB value that is farther from black."""

    dist1 = distance_from_black(colors[0])
    dist2 = distance_from_black(colors[1])
    if dist1 > dist2:
        return colors[:1]
    else:
        return colors[1:]


def get_avg_color(image, points, model):
    
    # If you want to apply kmeans on a region delimited by some points:
    if len(points)>0:
        # Create a mask for the ROI
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255, 255, 255))
                              
        # Extract the pixel values from the ROI (region of interest) using the mask
        roi = cv2.bitwise_and(image, image, mask=mask)

        (h,w,c) = roi.shape
        
    # If you have no points, apply kmeans on the whole image
    else:
        roi = image
        (h,w,c) = image.shape
    
    # Reshape the image from 3d to 2d
    roi_reshaped = roi.reshape(h*w, c) 
    
    # map the colors to the clusters
    cluster_labels = model.fit_predict(roi_reshaped) 
    
    # convert centroids values to good pixel values
    cols = model.cluster_centers_.round(0).astype(int)
    

    return cols

def get_colors_dict(image_folder):
    model = KMeans(n_clusters = 2, n_init=10)
    d = dict()
    for image_path in image_folder:
        # Read the image
        image = cv2.imread(image_path)
        # Get the masked eye image (it is already an rgb image)
        image = get_eye_mask(image)
        # Get 2 average colors of the eye (one is black, since there are many black details in the eyes)
        my_col, _ =get_avg_color2(image, [] , model)
        # Drop the color closer to black
        my_col= drop_closer_to_black(my_col)[0]
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

def get_eye_mask(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold for the pupil
    _, pupil_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Detect edges for the iris using Canny edge detector
    edges = cv2.Canny(blurred, 70, 120)
    
    # Use HoughCircles to detect circles in the edges
    # Some tuning of the parameters was needed to achive a satisfiable level of precision
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=400,
                               param1=100, param2=30, minRadius=40, maxRadius=200)
    
    iris_mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(iris_mask, (x, y), r, 255, -1)
    
    # Combine all masks
    final_mask = cv2.bitwise_and(image_rgb, cv2.cvtColor(iris_mask, cv2.COLOR_GRAY2RGB))
    
    # Plot the results
    '''
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Pre-processing')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Masked image (ready for K-Means)')
    plt.imshow(final_mask)
    plt.axis('off')
    '''
    return final_mask