import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load the model
with open("modelScaler.pkl", "rb") as file_1:
    modelScaler = pickle.load(file_1)
with open("modelPCA.pkl", "rb") as file_2:
    modelPCA = pickle.load(file_2)
with open("modelKM.pkl", "rb") as file_3:
    modelKM = pickle.load(file_3)
with open("numCol.txt", "r") as file_4:
    numCol = json.load(file_4)

# Define the Streamlit app
def run():
    st.title("Credit Segmentation Classifier")

    st.write("Fill the form below for your Credit Segmentation.")

    with st.form("Credit Cluster"):
        st.title("Credit Usage")
        # Example: Add sliders, selectboxes, and explanations for engagement

        customer_id = st.text_input("Customer ID", placeholder="Enter your Customer ID")
        balance = st.slider("Credit Balance", min_value=0, max_value=100000, value=5000, step=100)
        balance_frequency = st.slider("Balance Frequency (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        purchases = st.slider("Total Purchases", min_value=0, max_value=100000, value=1000, step=100)
        oneoff_purchase = st.slider("One-off Purchases", min_value=0, max_value=100000, value=500, step=100)
        installment_purchases = st.slider("Installment Purchases", min_value=0, max_value=100000, value=500, step=100)
        cash_advance = st.slider("Cash Advance Amount", min_value=0, max_value=100000, value=200, step=100)
        purchases_frequency = st.slider("Purchases Frequency (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        oneoff_purchase_freq = st.slider("One-off Purchase Frequency (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        purchase_installment_freq = st.slider("Installment Purchase Frequency (0-1)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        cash_advance_freq = st.slider("Cash Advance Frequency (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        cash_advance_trx = st.slider("Cash Advance Transactions", min_value=0, max_value=100, value=2, step=1)
        purchase_trx = st.slider("Purchase Transactions", min_value=0, max_value=100, value=10, step=1)
        credit_limit = st.slider("Credit Limit", min_value=0, max_value=100000, value=5000, step=100)
        payments = st.slider("Payments", min_value=0, max_value=100000, value=1000, step=100)
        minimum_payments = st.slider("Minimum Payments", min_value=0, max_value=100000, value=500, step=100)
        prc_full_payment = st.slider("Percent Full Payment (0-1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        tenure = st.slider("Tenure (months)", min_value=0, max_value=100, value=12, step=1)

        st.markdown("#### 💡 Tip: Use the sliders to quickly estimate your credit usage profile!")
        st.markdown("---")
        submit_button = st.form_submit_button("Submit")

        dfInf = {
            'CUSTOMER_ID': customer_id,
            'BALANCE': balance,
            'BALANCE_FREQUENCY': balance_frequency,
            'PURCHASES': purchases,
            'ONEOFF_PURCHASES': oneoff_purchase,
            'INSTALLMENTS_PURCHASES': installment_purchases,
            'CASH_ADVANCE': cash_advance,
            'PURCHASES_FREQUENCY': purchases_frequency,
            'ONEOFF_PURCHASES_FREQUENCY': oneoff_purchase_freq,
            'PURCHASES_INSTALLMENTS_FREQUENCY': purchase_installment_freq,
            'CASH_ADVANCE_FREQUENCY': cash_advance_freq,
            'CASH_ADVANCE_TRX': cash_advance_trx,
            'PURCHASES_TRX': purchase_trx,
            'CREDIT_LIMIT': credit_limit,
            'PAYMENTS': payments,
            'MINIMUM_PAYMENTS': minimum_payments,
            'PRC_FULL_PAYMENT': prc_full_payment,
            'TENURE': tenure
        }

        dfInf = pd.DataFrame([dfInf])
        input_data = dfInf[numCol]

        if submit_button:
            # Scale the data
            scaled_data = modelScaler.transform(input_data[numCol])

            # Apply PCA
            pca_data = modelPCA.transform(scaled_data)

            # Predict the cluster
            cluster = modelKM.predict(pca_data)

            st.success(f"🎉 Your Credit Segmentation Cluster is: {cluster[0]}")
            st.write("Thank you for using our Credit Segmentation Classifier!")
            st.balloons()

if __name__ == "__main__":
    run()
