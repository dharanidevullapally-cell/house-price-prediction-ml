# House Price Prediction using Machine Learning

## Project Overview

This project predicts house prices using Machine Learning techniques. Historical housing data is used to train a Random Forest Regression model and forecast house prices based on property features.

## Features

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Random Forest Regression Model
* Feature Importance Analysis
* Actual vs Predicted Visualization
* Interactive Streamlit Web Application

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

## Dataset Features

* Area
* Bedrooms
* Bathrooms
* Stories
* Parking
* Air Conditioning
* Main Road Access
* Guest Room
* Basement
* Furnishing Status

## Model Performance

* R² Score: 0.6179
* MAE: 1,013,365.42
* RMSE: 1,389,775.89

## Project Structure

House_Price_Prediction/
│
├── data/
├── models/
├── notebooks/
├── screenshots/
├── train.py
├── app.py
├── requirements.txt
└── README.md

## Run Locally

Install dependencies:

pip install -r requirements.txt

Train model:

python train.py

Run application:

streamlit run app.py

## Author

Dharani Devullapally
