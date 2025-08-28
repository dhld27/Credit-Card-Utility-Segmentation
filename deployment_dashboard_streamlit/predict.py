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
        customer_id = st.number_input("Insert your customer ID", value=None, placeholder="Write down your Customer ID here")
        balance = st.number_input("Insert your credit's balance", value=None, placeholder="Write down your credit balance here")
        balance_frequency = st.number_input("Insert your balance frequency", value=None, placeholder="Write down your balance frequency here")
        purchases = st.number_input("Insert your amount of purchases", value=None, placeholder="Write down your purchases amount here")
        oneoff_purchase = st.number_input("Insert your amount of oneoff purchase", value=None, placeholder="Write down your purchase amount here")
        installment_purchases = st.number_input("Insert your amount of installment purchase", value=None, placeholder="Write down your purchase amount here")
        cash_advance = st.number_input("Insert your amount of cash in advance", value=None, placeholder="Write down your purchase amount here")
        purchases_frequency = st.number_input("Insert your frequency purchase", value=None, placeholder="Write down your purchase frequency here")
        oneoff_purchase_freq = st.number_input("Insert your oneoff purchase frequency", value=None, placeholder="Write down your purchase frequency here")
        purchase_installment_freq = st.number_input("Insert your installment purchase frequency", value=None, placeholder="Write down your installment purchase frequency here")
        cash_advance_freq = st.number_input("Insert your cash frequency in advance", value=None, placeholder="Write down your cash advance frequency here")
        cash_advance_trx = st.number_input("Insert your cash transaction in advance", value=None, placeholder="Write down your transaction in cash here")
        purchase_trx = st.number_input("Insert your purchase transaction", value=None, placeholder="Write down your purchase transaction here")
        credit_limit = st.number_input("Insert your credit limit", value=None, placeholder="Write down your credit limit here")
        payments = st.number_input("Insert your payments", value=None, placeholder="Write down your payments here")
        minimum_payments = st.number_input("Insert your minimum payment", value=None, placeholder="Write down your minimum payment here")
        prc_full_payment = st.number_input("Insert your percentage of full payment", value=None, placeholder="Write down your percentage of full payment here")
        tenure = st.number_input("Insert your tenure", value=None, placeholder="Write down your tenure here")
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

            st.success(f"Your Credit Segmentation Cluster is: {cluster[0]}")
            st.write("Thank you for using our Credit Segmentation Classifier!")
            st.balloons()

if __name__ == "__main__":
    run()
