import streamlit as st
import cv2
import sys
import os

ROOT_DIR = os.path.abspath('./yolov5/')
sys.path.append(ROOT_DIR)

from main_detection import *

#Headings for Web Application
st.title("Yolov5 Object Detection Application - Improved for Abandonned Luggage Detection ")
st.subheader("What type of weights do you need to use ?")

#Picking what NLP task you want to do
option = st.selectbox('Yolov5 Weights',('Small Original Version - Yolov5s', 'Luggage Trained Version - Yolov5s custom')) #option is stored in this variable

#Uploader for image files
image_selector = st.selectbox('Image', ('Image1', 'Image2', 'Image3', 'Image4'))

if image_selector == 'Image1':
    image_path = './yolov5/inference/images/image1.jpg'
elif image_selector == 'Image2':
    image_path = './yolov5/inference/images/image2.jpg'
elif image_selector == 'Image3':
    image_path = './yolov5/inference/images/image3.jpg'
elif image_selector == 'Image4':
    image_path = './yolov5/inference/images/image4.jpg'

if option == 'Small Original Version - Yolov5s':
    weights = './yolov5/weights/yolov5s.pt'
else :
    weights = './yolov5/weights/yolov5s_trained.pt'

#Display results 
st.header("Original Picture")

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
st.image(image, caption='Original Picture')

st.header("Yolov5 Processed Picture")

image_yolo = detect_function(weights,image_path, 640)
image_yolo = cv2.cvtColor(image_yolo, cv2.COLOR_BGR2RGB)
st.image(image_yolo, caption='Object Detection Processed Picture')



