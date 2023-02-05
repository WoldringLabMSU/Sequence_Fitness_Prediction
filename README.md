# Sequence_Fitness_Prediction
<p align="center"> <img src=https://user-images.githubusercontent.com/77028470/216752280-01adaf5c-c677-4872-9af2-d03f022b1021.png height="210" width="300" style="border: 0">



## Introduction

The goal of this project is to predict the function of protein sequences and enhance the fitness score using machine learning algorithms and language model techniques. The prediction process will be optimized by utilizing ensemble learning, as well as resolving imbalanced data issues through the use of sampling methods including SMOTE and R-oversampling. For protein sequence representation, four different methods are analyzed : One-Hot Encoding, Physiochemical Encoding, UniRep ( Next-Token Prediction Embedding) and ESM (Masked-Token Prediction Embedding). These analysis are implemented over two distint datasets for affinity and stability prediciton. 

The main questions addressed are:

* How do different representation methods perform in predicting distinct fitness attributes such as stability or affinity? 
* How do sampling methods perform in the imbalanced protein dataset?
* Is ensemble learning over different protein representations helpful in boosting the performance of discriminative models?


## Requirements

- Python 3.x
- Numpy
- Pandas
- Sklearn
- modlamp 
- Optuna
- seaborn
- Matplotlib
- Scipy

  
The 2 datasets used in this study are Affinity Binding and NESP for stability prediction. Please refer to the manuscipt for more detailed information about these datasets. 
