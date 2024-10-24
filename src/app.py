import streamlit as st

from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import os
import json

import tensorflow as tf
from io import StringIO
from PIL import Image


st.title("Egyption Hieroglyphs detection app")


def load_yolo_model():
    model = YOLO(
        "/home/hany_jr/Ai/Egyption-Hieroglyphs-detection/src/Object Detection/runs/detect/train2/weights/best.pt"
    )
    return model


def resnet_model():
    model = tf.keras.models.load_model(
        "/home/hany_jr/Ai/Egyption-Hieroglyphs-detection/src/model/ResNetModel.keras"
    )
    return model


yolo = load_yolo_model()
resnet = resnet_model()
classes = json.load(
    open("/home/hany_jr/Ai/Egyption-Hieroglyphs-detection/dataset/class.json")
)
# load image with streamlit in the sidebar

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:

    img = Image.open(uploaded_file)

    # get the image as a numpy array to be able to use it with opencv

    img = np.array(img)
    st.image(img, caption="Uploaded Image.", use_column_width=True)


def predict_with_yolo(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        "/home/hany_jr/Ai/Egyption-Hieroglyphs-detection/dataset/save_image/img.jpg",
        img,
    )

    img = "/home/hany_jr/Ai/Egyption-Hieroglyphs-detection/dataset/save_image/img.jpg"

    results = yolo(img)

    extracted_data = results[0].boxes

    clss = int(extracted_data.cls[0])

    box = extracted_data.xyxy

    x1 = int(box[0][0])
    y1 = int(box[0][1])
    x2 = int(box[0][2])
    y2 = int(box[0][3])

    return classes[str(clss)], x1, y1, x2, y2


if uploaded_file is not None:

    clss, x1, y1, x2, y2 = predict_with_yolo(img)

    print(clss, x1, y1, x2, y2)

    st.write(f"Detected class is {clss}")

    st.write("the box bounding detected class is:")

    img = cv2.imread(
        "/home/hany_jr/Ai/Egyption-Hieroglyphs-detection/dataset/save_image/img.jpg",
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    st.write("the image with the bounding box")

    st.image(img, caption="Uploaded Image.", use_column_width=True)
