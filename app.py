import streamlit as st
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

# Ensure scikit-learn is imported correctly
from sklearn.neighbors import NearestNeighbors

import os
from numpy.linalg import norm

# Streamlit header
st.header('Fashion')
Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D, Input
from tensorflow.keras.models import Model

# Define the input layer
input_layer = Input(shape=(224, 224, 3))

# Load the ResNet50 model with pre-trained ImageNet weights, excluding the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)
base_model.trainable = False

# Add the GlobalMaxPooling2D layer
output_layer = GlobalMaxPooling2D()(base_model.output)

# Create the final model
model = Model(inputs=input_layer, outputs=output_layer)

# Display the model's architecture
model.summary()
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

upload_file=st.file_uploader("Upload")

if upload_file is not None:
    with open(os.path.join('upload',upload_file.name),'wb') as f:
        f.write(upload_file.getbuffer())
    st.subheader('Upload Image')
    st.image(upload_file)
    input_img_features=extract_features_from_images(upload_file,model)
    data = np.random.rand(5, 128) 
    n_neighbors = min(6, len(data))

# Find the neighbors
    distance, indices = neighbors.kneighbors([input_img_features], n_neighbors=n_neighbors)
    print(distance, indices)
    st.subheader('Recommended')
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    