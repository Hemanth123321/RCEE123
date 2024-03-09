import face_recognition
from PIL import Image, ImageDraw, ExifTags
import numpy as np
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
    face_locations = face_recognition.face_locations(image_np)
    draw = ImageDraw.Draw(image)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        draw.rectangle(((left, top), (right, bottom)), outline="blue", width=3)
    st.image(image, caption='Uploaded Image with Face Recognition', use_column_width=True)
    num_faces = len(face_locations)
    st.write(f'Number of Faces Detected: {num_faces}')
