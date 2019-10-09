# Machine part defect detection application
*Live Demo available at http://kapilve.pythonanywhere.com/*<br>
Classification of a specific automotive part called **Fender Apron** (shown below) as defective and non-defective using Transfer Learning <br>
* `functions.py` contains functions for preprocessing of images and making classes<br>
* `Machine defect detection .ipynb` is for code walk-through <br>
* `routes.py` is the Flask api file
* `Results` folder contain processed images with different kernels of the machine part for manual defect detection, also a document about the approach followed.
* `templates` contain `index.html`, the frontend of the application 
* `MobileNet_model_keras.json` & `MobileNet_model_wieghts.h5` are saved model and its weights respectively, which are deployed in our application
#### Sample processed image of Fender Apron
<img src="Results/Sharpen_Gray.jpg" alt="Drawing" style="width: 250px;"/>

### Dataset
The data is already labelled having total 250 images with 139 images as healthy machine parts and rest 111 as defective parts. Images given in the dataset were captured from different angles and scales. Training and Test datasets were prepared by randomly selecting total 25 images (i.e 10%) in which 10 were defective and 15 were healthy parts. Training/validation split used is 90/10.<br>
[Dataset Link](https://drive.google.com/file/d/1k57jP_oy4c9VDZmlgqCvfErzVTzPeA_M/view?usp=sharing)