import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2
import os
import joblib

st.title("Text2Image Generator")
model = load_model('Text2Image1.h5')
tokenizer = joblib.load('tokenizer.joblib')

# Text input field
input = st.text_input("Enter some text", "")

# Submit button
if st.button("Submit"):
    st.write("You entered:", input)
    input = [input]
    input = tokenizer.texts_to_sequences(input)
    input = pad_sequences(input,maxlen=1,padding='post',truncating='post')
    input = input.reshape([1,1,1])
    res = model.predict(input)
    res = res.reshape([160,160,3])
    st.image(res, use_column_width=True, caption="Generated Image")



st.subheader("Created by Jibitesh Chakraborty")