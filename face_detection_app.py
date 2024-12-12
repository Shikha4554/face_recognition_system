import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Title of the app
st.title("Face Detection App")

# Upload image
uploaded_image = st.file_uploader("Upload an image for face detection", type=["jpg", "jpeg", "png"])

# Function for face detection using Haar Cascade
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

if uploaded_image is not None:
    # Convert uploaded file to image
    image = Image.open(uploaded_image)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Detect faces
    result_image = detect_faces(image)
    
    # Display the image with detected faces
    st.image(result_image, caption="Image with Detected Faces", use_column_width=True)
