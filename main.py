import eda_cluster, predict
import streamlit as st
from PIL import Image

# Add a sidebar image/logo for branding
st.sidebar.image("https://www.svgrepo.com/show/528100/card.svg", use_container_width=True)

# Set up the main app title and description
st.sidebar.title("Credit Card Default Prediction & Customer Segmentation")
st.sidebar.write(
    "Predict whether a credit card holder will default next month based on their financial behavior and segmentation."
)

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
feedback = st.sidebar.text_area(
    "Feedback",
    placeholder="Let us know your thoughts..."
)

# Show feedback submission confirmation
if feedback and st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")

app_mode = st.sidebar.selectbox("Choose the app mode", ["EDA & Clustering", "Prediction"])

if app_mode == "EDA & Clustering":
    eda_cluster.run_eda_cluster()
elif app_mode == "Prediction":
    predict.run()

