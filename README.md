# Sequence_Fitness_Prediction

## Introduction

The goal of this project is to predict the function of protein sequences and enhance the fitness score using machine learning algorithms and language model techniques. The prediction process will be optimized by utilizing ensemble learning techniques, as well as resolving imbalanced data issues through the use of sampling methods including SMOTE and R-oversampling. For protein sequence representation, four different methods are analyzed : One-Hot Encoding, Physiochemical Encoding, UniRep ( Next-Token Prediction Embedding) and ESM (Masked-Token Prediction Embedding). These analysis are implemented over two distint datasets for affinity and stability prediciton. 

The main questions addressed are:

* How do different representation methods perform in predicting distinct fitness attributes such as stability or affinity? 
* How do sampling methods perform in the imbalanced protein dataset?
* Is ensemble learning over different protein representations helpful in boosting the performance of discriminative models?
