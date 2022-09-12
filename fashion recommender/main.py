pip install streamlit
import streamlit as st
from PIL import Image
import os
import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2
features_list=np.array(pickle.load(open('embedding.pkl','rb')))
file_names=pickle.load(open('filenames.pkl','rb'))


model=ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tensorflow.keras.Sequential([model,GlobalMaxPooling2D()])

st.title('fashion recommender system')
#file upload
#uploaded_file=st.file_uploader("choose an image")
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb')as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    expand_img_arr = np.expand_dims(img_arr, axis=0)
    preprocessed_img = preprocess_input(expand_img_arr)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result
def recommendation(features,features_list):
    neigh = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neigh.fit(features_list)
    distances,indices=neigh.kneighbors([features])

    return indices
uploaded_file=st.file_uploader("choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
       display_image=Image.open(uploaded_file)
       st.image(display_image)
       features=extract_features(os.path.join("uploads",uploaded_file.name),model)

       st.text(features)
       indices=recommendation(features,features_list)
       for i in range (0,5):
         st.text(indices[0][i])


       for file in indices[0][1:6]:
           temp_img = cv2.imread(file_names[file])
           cv2.imshow('output', temp_img)
           cv2.waitKey(0)

       col1,col2,col3,col4,col5=st.columns(5)
       st.image(indices[0][0])
    else:
      st.header("some error occured")



