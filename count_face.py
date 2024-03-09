from PIL import Image, ImageDraw, ExifTags
import numpy as np
import cv2
import streamlit as st

st.header("RCEE::AI&DS")
st.title("FACE RECOGNITION AND COUNT ")

uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    
    image_np = np.array(image)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in faces:
        draw.rectangle(((x, y), (x+w, y+h)), outline="blue", width=3)
    
    st.image(image, caption='Uploaded Image with Face Detection', use_column_width=True)
    
    num_faces = len(faces)
    st.write(f'Number of Faces Detected: {num_faces}')
