import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from sklearn.cluster import KMeans
from math import sqrt
import skimage

def lab_dist(rgb1, rgb2):
    """Compute the similarity coefficient between two RGB values using CIELAB color format."""
    rgb1=np.array(rgb1)
    rgb2= np.array(rgb2)
    lab1=skimage.color.rgb2lab([[[rgb1/255]]])[0][0][0]
    lab2=skimage.color.rgb2lab([[[rgb2/255]]])[0][0][0]
    delta_e = skimage.color.deltaE_ciede2000(lab1, lab2)
    return delta_e


def drop_closer_to_x(colors, x):
    dists=[lab_dist(color,x) for color in colors]
    # drop closer to x
    return np.delete(colors, dists.index(min(dists)), axis=0)

    
        
    
def landmark_extractor(image):
    
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    face = detector(gray_image)
    
    # FACE LANDMARKS EXTRACTION
    # Load facial landmark predictor from Dlib
    predictor = dlib.shape_predictor('/Users/daniel/Desktop/Armochromia/shape_predictor_68_face_landmarks.dat')       
    # Get the facial landmarks for the first face
    landmarks = predictor(gray_image, face[0])
    landmarked_image = rgb_image.copy()
    # Draw landmarks
    dot_size = image.shape[0]//1000 + 3

    for i in range(0, len(landmarks.parts())):
        cv2.circle(landmarked_image, (landmarks.part(i).x,landmarks.part(i).y), dot_size, (255, 0,0), -1)
        
    return landmarked_image, landmarks



def display_features(masked_image, color, label):
    color_array = [color]

    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title(label=f'{label} Region')
    axs[1].imshow(color_array)
    axs[1].axis('off')
    axs[1].set_title(label=f'{label} Avg Color')
    
    plt.show()
    pass


def extract_colors(image,landmarked_image, landmarks, image_path):
    # Initialize model for kmeans
    model2 = KMeans(n_clusters = 2, n_init=10)
    model3 = KMeans(n_clusters = 3, n_init=10)

    plt.imshow(landmarked_image)
    plt.show()
    
    # GET AVG COLOR OF LEFT EYE -----------------------------------------------------------------
    # First get the coordinates of the landmarks of the left eye (Region of Interest)
    lms_list = np.array([(landmarks.part(i).x,landmarks.part(i).y) for i in [36,37,38,39,40,41]])

    # Apply kmeans to get avg color (and also the region of interest)
    # 3 is the number of clusters (2 colors to catch black and white details of the eye, 1 for the charecteristic color)
    left_eye_col, roi_eye = get_avg_color2(image,lms_list, model3)

    # Cleaning the output of K-means
    left_eye_col = [col[::-1] for col in left_eye_col] 
    left_eye_col = drop_closer_to_x(left_eye_col,[0,0,0])
    left_eye_col = drop_closer_to_x(left_eye_col,[255,255,255])
    
    display_features(roi_eye, left_eye_col, 'Left Eye')

    # GET AVG COLOR OF SKIN TONE -----------------------------------------------------------------
    # Extract coordinates of region of face we are interested in
    lms_list_skin = np.array([(landmarks.part(i).x,landmarks.part(i).y) for i in [1,41,31,3]])

    # Apply kmeans to get avg color (and also the region of interest)
    skin_tone_col, roi_skin = get_avg_color2(image,lms_list_skin,model2)
    
    skin_tone_col = [col[::-1] for col in skin_tone_col]
    skin_tone_col = drop_closer_to_x(skin_tone_col,[0,0,0])
        
    display_features(roi_skin ,skin_tone_col, 'Skin Tone')
    

    # GET AVERAGE COLOR OF HAIR -----------------------------------------------------------------
    # SInce this task cannot be solved easly with landmarks or simple masks, 
    # we opted to use a pretrained model: Google Mediapipe Hair Segmenter

    # FIrst get the hair mask
    hair_mask = get_hair_mask(image_path)

    # APply kmeans to get avg color
    hair_col , _ = get_avg_color2(hair_mask,[], model2 )
    hair_col = [col[::-1] for col in hair_col]

    hair_col = drop_closer_to_x(hair_col[:], [0,0,0])

    # Create an array containg the color we want to display
    color_array_hair = np.array([[hair_col[i]] for i in range(0,len(hair_col))])
    
    # PLOT!
    display_features(hair_mask, hair_col, 'Hair')
    
    return [left_eye_col[0], skin_tone_col[0],hair_col[0]]

 





# GET AVG COLOR OF A REGION DELIMITED BY N POINTS USING KMEANS ALGORITHM
def get_avg_color2(image, points, model):
    
    # If you want to apply kmeans on a region delimited by some points:
    if len(points)>0:
        # Create a mask for the ROI (region of interest)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255, 255, 255))
                              
        # Extract the pixel values from the ROI  using the mask
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
    bgr_cols = model.cluster_centers_.round(0).astype(int)
    

    return bgr_cols, roi

import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_hair_mask(image_path):
    BG_COLOR = (0, 0, 0) # black
    MASK_COLOR = (255, 255, 255) # white
    
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Create a image segmenter instance with the image mode:
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path='/Users/daniel/Desktop/Armochromia/hair_segmenter.tflite'),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)
    
    with ImageSegmenter.create_from_options(options) as segmenter:
        
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(image_path)
        image_data = cv2.imread(image_path)
    
        # Perform image segmentation on the provided single image.
        segmented_masks = segmenter.segment(mp_image)

        # Get the mask from the Segmented Image
        category_mask = segmented_masks.category_mask
        
        # Generate solid color images for showing the output segmentation mask.    
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
    
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        output_image = cv2.bitwise_and(output_image, image_data)
        
        return output_image
        
