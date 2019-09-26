# coding: utf-8

import pandas as pd
from PIL import Image #pillow

def pre_processing(image_path):
    """
     Function performs minor processing of rotation, blurring, resizing and grayscale conversion and returns tuple containing 
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
    Function takes in the prediction array from the model and gives classes of "Defective" and "Healthy" to the results
    along with the probability associated with our prediction in form of a tuple.
    """
    for i in y_pred:
        if i[0] > 0.5:
            return "Healthy", i[0]
        elif i[0] <= 0.5:
            return "Defective", i[0]

def pred(test_image_path):
    """
    Main function for image prediction which uses saved MobileNet model to return resulted class using make_classes function
    """
    import keras
    from keras.applications import MobileNet
    from keras import optimizers
    from keras.models import load_model, model_from_json
    import cv2 as cv
    import numpy as np
    from PIL import Image #pillow
    from functions import pre_processing, make_classes
    from keras.utils.generic_utils import CustomObjectScope
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        with open("MobileNet_model_keras.json") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            #print(loaded_model.summary())
            #load weights into new model
            loaded_model.load_weights("MobileNet_model_wieghts.h5")
            sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
            loaded_model.compile(loss='binary_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])
            X_test=[]
            X_test.append(cv.resize(pre_processing(test_image_path)[2],(224,224), interpolation=cv.INTER_CUBIC))
            img = np.array(X_test)
            pred= loaded_model.predict(img)
    return make_classes(pred)
        
