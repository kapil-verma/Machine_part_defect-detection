def pre_processing(image_path):
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
    for i in y_pred:
        if i[0] > 0.5:
            i[0]=1
            i[1]=0
        elif i[0] <= 0.5:
            i[1]=1
            i[0]=0
    return y_pred
