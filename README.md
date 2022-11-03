
# Facial Keypoints Detection

### A computer vision project to build a facial keypoints detection system.
<p align="center"><img src=https://raw.githubusercontent.com/ShashankKumbhare/facial-keypoints-detecter/main/auxil/images/_project_intro.png  width="500"></p>

## Table of Contents

- [**Project Overview**](#Project-Overview)
- [**Data Description**](#Data-Description)
- [**Methodology**](#Methodology)
- [**Python package `traffic_light_classifier`**](#python-package-traffic_light_classifier)
- [**Package Usage**](#Package-Usage)
- [**Results**](#Results)

---

## Project Overview

- Facial keypoints detection system has variety of applications, including: 
  - Facial tracking.
  - Facial pose recognition.
  - Facial filters.
  - Emotion recognition.
  - Medical diagnosis: Identifying dysmorphic facial symptoms.
- Detecting facial keypoints is a challenging problem given the variations in both facial features as well as image conditions. Facial features may differ according to size, position, pose and expression, while image qualtiy may vary with illumination and viewing angle.  
- In this project, Convolutional Neural Network (CNN) based facial keypoints detector system has been implemented to detect 68 facial keypoints (also called facial landmarks) around important areas of the face: the eyes, corners of the mouth, the nose, etc. using computer vision techniques and deep learning architectures.  
- The project is broken up into a few main parts in 4 Python notebooks:
  - [Notebook 1](https://github.com/ShashankKumbhare/facial-keypoints-detecter/blob/main/1.%20Load%20and%20Visualize%20Data.ipynb): Loading and Visualizing the Facial Keypoint Data.  
  - [Notebook 2](https://github.com/ShashankKumbhare/facial-keypoints-detecter/blob/main/2.%20Define%20the%20Network%20Architecture.ipynb): Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints.  
  - [Notebook 3](https://github.com/ShashankKumbhare/facial-keypoints-detecter/blob/main/3.%20Facial%20Keypoint%20Detection%2C%20Complete%20Pipeline.ipynb): Facial Keypoint Detection Using Haar Cascades and a Trained CNN.  
  - [Notebook 4](https://github.com/ShashankKumbhare/facial-keypoints-detecter/blob/main/4.%20Fun%20with%20Keypoints.ipynb): Applications of this project.  
- The implemented Python package code is [facial_keypoints_detecter](https://github.com/ShashankKumbhare/facial-keypoints-detecter/tree/main/facial_keypoints_detecter).

---

## Data Description

<p align="center">
  <img src="https://raw.githubusercontent.com/ShashankKumbhare/facial-keypoints-detecter/main/auxil/images/key_pts_example.png" height="200" />
  <img src="https://raw.githubusercontent.com/ShashankKumbhare/facial-keypoints-detecter/main/auxil/images/landmarks_numbered.jpg" height="200" />
</p>

- Facial keypoints are the small magenta dots shown on each of the faces in the image above.  
- In each training and test image, there is a single face and 68 keypoints, with coordinates (x, y), for that face.  
- These keypoints mark important areas of the face: the eyes, corners of the mouth, the nose, etc. Here they are, numbered, and you can see that specific ranges of points match different portions of the face.  

**Training and Testing Data**  
- This facial keypoints dataset consists of 5770 [color images](https://github.com/ShashankKumbhare/facial-keypoints-detecter/tree/main/data).
- 3462 are [training images](https://github.com/ShashankKumbhare/facial-keypoints-detecter/tree/main/data/training).
- 2308 are [test images](https://github.com/ShashankKumbhare/facial-keypoints-detecter/tree/main/data/test).  
- The information about the images and keypoints in this dataset are summarized in [CSV files](https://github.com/ShashankKumbhare/facial-keypoints-detecter/tree/main/data).  

Note: Datasets are explored in [Notebook 1](https://github.com/ShashankKumbhare/facial-keypoints-detecter/blob/main/1.%20Load%20and%20Visualize%20Data.ipynb).  
Note: All images come from this MIT self-driving car course and are licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
