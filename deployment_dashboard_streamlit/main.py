import eda_cluster, predict
import streamlit as st
from PIL import Image

st.sidebar.title("Credit Card Default Prediction App")
app_mode = st.sidebar.selectbox("Choose the app mode", ["EDA & Clustering", "Prediction"])
st.sidebar.image(Image.open('credit_card.png'), use_column_width=True)  


