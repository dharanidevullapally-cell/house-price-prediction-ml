import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("models/house_model.pkl")
feature_names = joblib.load("models/features.pkl")

st.title("🏠 House Price Prediction")

# Inputs
area = st.number_input("Area", min_value=500)

bedrooms = st.selectbox(
    "Bedrooms",
    [1, 2, 3, 4, 5, 6]
)

bathrooms = st.selectbox(
    "Bathrooms",
    [1, 2, 3, 4]
)

stories = st.selectbox(
    "Stories",
    [1, 2, 3, 4]
)

mainroad = st.selectbox(
    "Main Road",
    ["yes", "no"]
)

guestroom = st.selectbox(
    "Guest Room",
    ["yes", "no"]
)

basement = st.selectbox(
    "Basement",
    ["yes", "no"]
)

hotwaterheating = st.selectbox(
    "Hot Water Heating",
    ["yes", "no"]
)

airconditioning = st.selectbox(
    "Air Conditioning",
    ["yes", "no"]
)

parking = st.selectbox(
    "Parking",
    [0, 1, 2, 3]
)

prefarea = st.selectbox(
    "Preferred Area",
    ["yes", "no"]
)

furnishingstatus = st.selectbox(
    "Furnishing Status",
    [
        "furnished",
        "semi-furnished",
        "unfurnished"
    ]
)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    prediction = model.predict(input_df)

    st.success(
        f"Estimated House Price: ₹ {prediction[0]:,.0f}"
    )