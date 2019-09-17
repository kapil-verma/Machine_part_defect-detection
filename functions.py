# coding: utf-8

import pandas as pd
from PIL import Image #pillow
import matplotlib.pyplot as plt
import seaborn as sns

def pre_processing(image_path):
    """
     function performs minor processing of rotation, blurring, resizing and grayscale conversion and returns tuple containing 
     resized gray, blurred and original images
    """
    import cv2 as cv #openCV
    import numpy as np
    #Reading the image with opencv
    image=cv.imread(image_path)
    image = np.array(image, dtype=np.uint8)
    #changing to grayscale
    
    #Rotating if image is vertical
    if image.shape[1]<image.shape[0]:
        #compairing width and height
        image = np.rot90(image)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    #Applying Gaussian blur to the image, to reduce the noise
    blur=cv.GaussianBlur(image,(13,13),0)
    
    #resizing images to 10% of their original size, as they all are HD images
    blur=cv.resize(blur,(0,0),fx=.1,fy=.1)
    gray=cv.resize(gray,(0,0),fx=.1,fy=.1)

    return gray, blur, image

def make_classes(y_pred):
    """
    function takes in the prediction array from the model and gives classes of "Defective" and "Healthy" to the results
    along with the probability associated with our prediction in form of a tuple.
    """
    for i in y_pred:
        if i[0] > 0.5:
            return "Healthy", i[0]
        elif i[0] <= 0.5:
            return "Defective", i[0]

