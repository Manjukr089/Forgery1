
#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('skin.h5')

#Name of Classes
CLASS_NAMES = ['Original', 'Forged']

#Setting Title of App
st.title("Image Forgery detection")
st.markdown("Upload an image")

#Uploading the image
skin_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:

    if skin_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(skin_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (180,180))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,180,180,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result))