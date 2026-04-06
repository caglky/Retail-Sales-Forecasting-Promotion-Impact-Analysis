## Retail Sales Forecasting & Promotion Impact Analysis

Retail companies often struggle to accurate forecase future sales, especially when promotion, holidays, and external factors affect customer behavior. This project focuses to suggest best estimation by building machine learning pipeline that predicts future sales and analyzes the impact of promotional activities using time series data. 

## Project Structure 
```bash
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
```
---

## Project Overview

- Data analysis to understand patterns in sales (EDA)
- Build reusable feature engineering functions which improve model performance
- Baseline model for simple predictions
- LSTM model for more advanced forecasting
- Evaluation of model performance using appropriate metrics
- Simulator to test how promotions impact future sales

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

## Setup 

1. Clone the repository :
   ```bash
   git clone <repo_link>
   cd <repo_name>
2. Create and activate a virtual environment:
   pythın -m venv venv
   source venv/bin/activate

3. Install repuired packages :
   pip install -r requirements.txt

---

## Dataset

The dataset is not included in this repository due to size limitations.
You can download it from Kaggle: 
https://www.kaggle.com/competitions/store-sales-time-series-forecasting 
(after downloading, place the files inside a data/folder)

---

## Motivation

This project was built to better understand how machine learning models can be applied to real-world retail data, especially for forecasting and decision-making.

It also focuses on writing clean, reusable, and modular code suitable for real-world projects.

## Author 

Computer engineering student interested in Data Science, Machine Learning and real-world data problems
