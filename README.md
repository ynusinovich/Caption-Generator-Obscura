# Capstone Project:
# README

## Problem Statement
Have you ever tried to post your travel photographs, and had a brain freeze about how to caption them? With my program, you can upload your photograph, and it will generate a suggested caption for you. The program is trained on 19,500 travel photos and captions from Atlas Obscura.

## Executive Summary
The goal of this project was to create an interactive model that can be used to generate a caption for an uploaded travel photograph. For this project, I differentiated my model from other caption generators by training it on about 19,500 captions and images from Atlas Obscura, a travel website that catalogs unusual and obscure travel destinations. The intention was not for the model to be able to perfectly identify each part of each image, but, rather, for the model to understand the general idea of an image and create an appropriate and entertaining caption. The inherent limitation was that the model, given the creative captions on the website, may fail to generate anything sensical at all. The model resulted in a single-word BLEU Score of about 8.5%, which was lower than my expected benchmark of 15%. Allowing the program to choose from among the model's top 7 predicted next caption words, rather than picking the top one, raised the BLEU score slightly to about 9%.

## Recommendations
With more time and computational power, some possible improvements to the model include:
- Training on captions from more travel websites, e.g., Lonely Planet
- Trying more types of models for processing images before passing them to the neural network I created, e.g., ResNet by Microsoft
- Adding more layers or other adjustments to the neural network I created

## Future Uses
Train on a larger variety of websites, e.g., cooking, real estate, social media<br>
Predict image captions for blind users for images that don’t have alt text or a caption

## Contents
- Data Source and References
- Data Download
    - Part 1: Image Captions and Image URLs Downloaded Using Scrapy
    - Part 2: Image Downloads in Jupyter Notebook
- Data Exploration
    - Remove Duplicate Captions and Images
    - Explore Caption Lengths
    - Delete Short Captions
- Process Data for Modeling
    - Generate Caption Sequence Dictionary and Vocabulary List
    - Run Images Through Inception V3 Model
    - Create Word Index and Find Maximum Caption Length
    - Create Data Generator
    - Embed Captions with Pre-Trained GloVe Vector
    - Run Image Data Through Custom Model
- Train Model (Using .py File)
- Reload Model and Make Predictions (Using .py File)
- Model Evaluation Using BLEU Score (Using .py File)

## Data Dictionary
|Feature or Term|Type|Description|
|---|---|---|
|**Images**|image files, imported as arrays|All images from the Atlas Obscura list of places.|
|**Captions**|strings|All captions from the Atlas Obscura list of places.|
|**GloVe**|text file, imported as an array|Pre-trained vector representations for words.|
|**Inception V3**|model|Pre-trained neural network model for image processing.|
|**Model**|model|Custom neural network model for combined image and text processing and caption prediction.|
|**BLEU Score**|float|Metric for evaluating generated captions compared to actual captions.|

## Required Packages and Downloads
- Keras (install using conda or pip3)
- TensorFlow (install using conda or pip3)
- InceptionV3 model (downloads in the Jupyter notebook)
- GloVe model (download at https://www.kaggle.com/incorpes/glove6b200d)
