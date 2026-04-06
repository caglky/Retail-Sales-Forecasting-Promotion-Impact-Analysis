## Retail Sales Forecasting & Promotion Impact Analysis

This project focuses on forecasting retail sales and analyzing the impact of promotion using time series data. The goal is to build a structered and reproducible machine learning pipeline, starting from raw data to model evaluation and simulation.

## Project Structure 

promocast/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   └── holidays_events.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_simulator.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── simulator.py
│
├── outputs/
│   ├── figures/
│   ├── predictions.csv
│   └── metrics.json
│
└── README.md

---

## Project Overview

- Perform exploratory data analysis (EDA)
- Build reusable feature engineering functions
- Train baseline and deep learning models (LSTM)
- Evaluate model performance using appropriate metrics
- Simulate promotion effects on sales

---

## Technologies Used

- Python
- Pandas / NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## Key Concepts

- Time Series Forecasting  
- Feature Engineering  
- Model Evaluation (MAE, etc.)  
- Deep Learning (LSTM)  
- Modular Code Design  

---


## Dataset

The dataset is not included in this repository due to size limitations.
You can download it from Kaggle: 
https://www.kaggle.com/competitions/store-sales-time-series-forecasting 
(after downloading, place the files inside a data/folder)

## Motivation

This project was built to better understand how machine learning models can be applied to real-world retail data, especially for forecasting and decision-making.

It also focuses on writing clean, reusable, and modular code suitable for real-world projects.

## Author 

Computer engineering student interested in Data Science, Machine Learning and real-world data problems