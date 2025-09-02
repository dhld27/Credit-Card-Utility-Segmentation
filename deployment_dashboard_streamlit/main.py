import eda_cluster, predict
import streamlit as st
from PIL import Image

st.sidebar.title("Credit Card Default Prediction App")
st.sidebar.write(
    "Predict whether a credit card holder will default next month based on their financial behavior and segmentation."
)

# Add a sidebar image/logo for branding
st.sidebar.image("https://thumbs.dreamstime.com/b/credit-card-29413194.jpg", use_column_width=True)

# Add a sidebar info box
with st.sidebar.expander("About this app"):
    st.write(
        """
        - **EDA & Clustering:** Explore the dataset and see customer segments.
        - **Prediction:** Predict default risk for a customer.
        """
    )

# Add a theme selector
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #222;
            color: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #fff;
            color: #000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add a feedback widget
feedback = st.sidebar.text_area("Feedback", "Let us know your thoughts...")

app_mode = st.sidebar.selectbox("Choose the app mode", ["EDA & Clustering", "Prediction"])

if app_mode == "EDA & Clustering":
    eda_cluster.run_eda_cluster()
elif app_mode == "Prediction":
    predict.run()

# Show feedback submission confirmation
if feedback and st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")