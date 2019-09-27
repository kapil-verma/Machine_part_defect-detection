# Machine part defect detection application
*Live Demo available at http://kapilve.pythonanywhere.com/*<br>
Classification of a specific automotive part called **Fender Apron** (shown below) as defective and non-defective using Transfer Learning <br>
* `functions.py` contains functions for preprocessing of images and making classes<br>
* `Machine defect detection .ipynb` is for code walk-through <br>
* `routes.py` is the Flask api file
* `Results` folder contain processed images with different kernels of the machine part for manual defect detection 
* `templates` contain `index.html`, the frontend of the application 
* `MobileNet_model_keras.json` & `MobileNet_model_wieghts.h5` are saved model and its weights respectively, which are deployed in our application
#### Sample processed image of Fender Apron
<img src="Results/Sharpen_Gray.jpg" alt="Drawing" style="width: 250px;"/>
