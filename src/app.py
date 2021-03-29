import streamlit as st
import cv2 
from predict import predict
import os
from PIL import Image

# User Interface
st.title("Cancer Detector AI")
st.markdown(">AI powered web app that can detect Diabetic Retinopathy")
st.write("")
uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg", "tif"))
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save('./test/uploaded/temp.png')
    image = cv2.imread('./test/uploaded/temp.png')
    st.image(uploaded_file, caption='Uploaded Image.', width=360)
    st.markdown("Hurray, AI is making prediction!")
    st.write("")
    st.markdown("**AI**: Chances of being Cancer is...")
    chance = predict(image)
    st.success(f"{chance}%")
    # Delete Image as soon as we make prediction
    os.remove('./test/uploaded/temp.png')

st.markdown("")
# st.markdown("[GitHub Repo](https://github.com/abhinand5/planets-recognizer-app)")