import numpy as np
import cv2
import os
from keras.models import load_model
from flask import Flask,render_template,Response
import tensorflow as tf
from gtts import gTTs
global graph
global Writer
from skimage.transform import resize

graph=tf.get_defauly_graph()
writer=None

model=load_model('aslpng1.h5');

vals=['A','B','c','D','E','F','G','H','I']

app=Flask(__name__)

print("[INFO] Accessing video Stream...")

vs=cv2.VideoCapture(0)

pred=""

def detect(frame):
    img=resize(frame,(64,64,1))
    img=np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img=img/255.0
    with graph.as_default():
        prediction=model.predict_classes(img)
    print(prediction)
    pred=vals[prediction[0]]
    print(pred)
    return pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace;boundary-frame')

if __name__=='__main__':
    app.run(debug=True)