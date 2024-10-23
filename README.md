<br />
<p align="center">
  <h3 align="center"> Egyption Hieroglyphs detection </h3>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Description](#description)
- [Dataset](#dataset)
- [Features](#features)
- [What you will learn?!](#what-you-will-learn-?!)
- [Packages and frameworks i used](#packages-and-frameworks-i-used)
- [Installing packages](#installing-packages)
- [Deplyment and run the application](#deplyment-the-application)

## Description

This project focuses on detecting and classifying Egyptian hieroglyphs using advanced DeepLearning techniques. It features two core components:

* **Image Classification:** Classifies images of Egyptian hieroglyphs into distinct categories.
* **Object Detection:** Utilizes `YOLO` (You Only Look Once) for detecting and localizing hieroglyphs in images.

With this repository, you can train custom models to identify Egyptian hieroglyphs, experiment with `pre-trained models`, and improve accuracy for image recognition tasks. This could be useful for archaeological research, digital archiving, or educational purposes.

## Dataset

The dataset can be sourced from:

* **Kaggle:** [Egyptian Hieroglyph Dataset](https://www.kaggle.com/datasets/alexandrepetit881234/egyptian-hieroglyphs/data)

**Some Notes:**

* Make sure the dataset is in the correct format:

1. **For image classification:** Organized into folders where each folder corresponds to a class.

2. **For object detection (YOLO):** Annotated in `YOLO` format (`.yaml` files with class and bounding box coordinates).


## Features

* **Hieroglyph Image Classification:** Predicts the type of hieroglyph in an image from a predefined set of classes.
* **Hieroglyph Object Detection (YOLO):** Detects and localizes multiple hieroglyphs in a single image.


## What you will learn?!

1. **How to read the data from directories**
2. **How to augment the data and how to crop them**
3. **How to build a CNN Architecture with the functional subclassing technique.**
4. **How to use the transfer learning.**
5. **To to make the bounding box coordinates in a YOLO format.**
6. **How to use YOLO model in training and testing data.**

## Packages and frameworks i used

- [Python 3.x]() 3.9 or more (Recommended)
- [numpy](https://keras.io/) for Algebra
- [pandas](https://pandas.pydata.org/docs/) for datasets
- [cv2](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) For images
- [ultralytics](https://docs.ultralytics.com/guides/) For `YOLO` model
- [TensorFlow](https://www.tensorflow.org/) For DeepLearning Models and Architectures
- [scikit-learn](https://scikit-learn.org/stable/) For evaluating the model results
- [os](https://docs.python.org/3/library/os.html) For control the dirictories
- [streamlit](https://docs.streamlit.io/) for deployment

## Installing packages

* **Clone the repository:**

        git clone https://github.com/ahmedhany14/Egyption-Hieroglyphs-detection.git
        cd Egyption-Hieroglyphs-detection

* **Install the required dependencies:**

        pip install -r requirements.txt

## Deplyment and run the application
        cd "src"
        streamlit run app.py