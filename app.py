import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("scaler.pkl")

st.title("Restaurants Rating Prediction App")



st.caption("This app helps you predict a restuarants review class")

st.divider()

average_cost = st.number_input("Please enter the estimated average cost for two people", min_value=50, max_value=999999999, value=1000,step = 200)

book_table = st.selectbox("Does the restaurant allow table booking?", ["Yes", "No"])

online_order = st.checkbox("Does the restaurant accept online orders?", ["Yes", "No"])

price_range = st.selectbox("Please select the price range of the restaurant(1 Cheapest, 4 Most Expensive)", [1, 2, 3, 4])

predictionbutton = st.button("Predict the reviwe!")


st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if book_table == "Yes" else 0

online_order_status = 1 if online_order == "Yes" else 0

values =[[average_cost, bookingstatus, online_order_status, price_range]]

my_X_values = np.array(values)

X = scaler.transform(my_X_values)   

if predictionbutton:
    st.snow()

    prediction = model.predict(X)

    if prediction < 2.5:
        st.error("The restaurant is rated poorly")
    elif prediction < 3.5:
        st.warning("The restaurant is rated averagely")
    elif prediction < 4.0:
        st.success("The restaurant is rated well")
    elif prediction <= 4.5:
        st.success("The restaurant is rated Very well")
    else:
        st.success("The restaurant is rated Excellent")