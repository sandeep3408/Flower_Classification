import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image

model = load_model('p_flower.h5')

CLASS = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

st.title('FLOWER PREDICTION')

st.markdown('Upload an image of flower')

f_image = st.file_uploader("Choose an image...", type='png')
submit = st.button('Predict')


if submit:

    if f_image is not None:


        file_bytes = np.asarray(bytearray(f_image.read()),dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")

        opencv_image = cv2.resize(opencv_image, (64,64))

        opencv_image = np.expand_dims(opencv_image,axis=0)

        pred = model.predict(opencv_image)

        for i in range(0,5):
            if pred[0][i]==1:

                p = (list(CLASS.keys())
                [list(CLASS.values()).index(i)])

        st.title(str("the flower is : "+p))
