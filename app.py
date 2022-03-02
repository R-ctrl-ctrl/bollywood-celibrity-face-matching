from distutils.command.upload import upload
from operator import ipow
from unittest import result
from chardet import detect
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
import numpy as np
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity



detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list = pickle.load(open('embedding.pkl','rb'))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_img):
    try:
         with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
             f.write(uploaded_img.getbuffer())
         return True
    except:
        return False



def extract_feature(img_path,model,detector):
    img = cv2.imread(img_path)
    res = detector.detect_faces(img)

    x,y,width,height = res[0]['box']
    face = img[y:y+height,x:x+width]

    image = Image.fromarray(face)
    image = image.resize((224,224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result



def recommed(feature,feature_list):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1,-1),feature_list[i].reshape(1,-1)))

    index_pos = sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]
    return index_pos




st.title("bollywood celibrity face matcing")

uploaded_img =  st.file_uploader('choose an image')

if uploaded_img is not None:
    if save_uploaded_image(uploaded_img):
        # load the image
        display_image = Image.open(uploaded_img)

        # extrat the feature
        features_extracted = extract_feature(os.path.join('uploads',uploaded_img.name),model,detector)
        # st.text(features_extracted)

        #recommended
        index_pos = recommed(features_extracted,feature_list)
        
        col1,col2 = st.columns(2)

        with col1:
            st.header('your uploaded image')
            st.image(display_image)

        with col2:
            st.header("seems like".join(filenames[index_pos].split('\\')[1].split('_')))
            st.image(filenames[index_pos],width=260)








    

