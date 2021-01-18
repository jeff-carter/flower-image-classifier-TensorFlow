# flower-image-classifier-TensorFlow

This repo contains files submitted as a project for the Introduction to Machine Learning with TensorFlow Udacity Nanodegree.

## Description
The goal of this project was to train an image classifier to recognize 102 different species of flowers and then use it to perform inference on test flower images.

## Software library
This project involved the use of the following software and libraries
 - Python
 - NumPy
 - TensorFlow
 - Matplotlib

## Contents
**Project_Image_Classifier_Project.html**
 - The HTML export of the Jupyter Notebook used in the project
 - The following steps were performed in it:
    * Load the image dataset and create a pipeline
    * Build and train an image classifier on the dataset
    * Use the trained model to perform inference on flower images

**predict.py**
  - Python script that can be used to perform inference on flower images using a pre-trained model
  - Can be used in combination with the model trained in the Juypter Notebook
  - Outputs the names and probabilities of the top k predictions
