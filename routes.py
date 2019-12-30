import os
from functions import pred
from flask import Flask,request,render_template,jsonify
from keras import backend as K

app = Flask(__name__)
@app.route('/')
def home():
    """
    Renders a HTML page which allows us to input an image.
    """
    return render_template('index.html')

@app.route('/predict' ,methods=['POST'])
def predict():
    """
    Main API function which takes image from local storage with request and uses function pred for classification 
    and covert the result to JSON format
    """
    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file found'
    user_file = request.files['file']
    if user_file.filename == '':
        return 'file name not found …'
    else:
        path=os.path.join(os.getcwd()+user_file.filename)
        user_file.save(path)
        K.clear_session() 
        classes = pred(path)
        K.clear_session() 
        
        return jsonify({
        "status":"success",
        "prediction":classes[0],
        "confidence":str(classes[1])
        })

if __name__ == '__main__':
    app.run(ssl_context='adhoc')
            

   




            
           
          


