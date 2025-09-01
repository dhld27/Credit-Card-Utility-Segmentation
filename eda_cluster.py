import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st
from feature_engine.outliers import Winsorizer
import numpy as np
import io

import matplotlib.pyplot as plt

# Load your data
@st.cache_data
def load_data():
    # Update the path to your dataset as needed
    df = pd.read_csv('credit_card_default.csv')
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("EDA & Clustering App")
tab = st.sidebar.radio("Choose Analysis", ["EDA Before Clustering", "EDA After Clustering"])

# --- PCA and n_cluster decision section ---

st.header("PCA & Cluster Selection")

# Select only numeric columns for PCA and clustering
pdfNum = df.select_dtypes(include=['float64', 'int64'])

# Fill NaN/missing values with 0
pdfNum = pdfNum.fillna(0)

# Outlier treatment using Winsorizer
columns = ['BALANCE', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

normal = []
skew = []
extremeSkew = []

for col in columns:
    if col in pdfNum.columns:
        skewness = pdfNum[col].skew()
        if -0.5 < skewness < 0.5:
            normal.append(col)
        elif -1 < skewness <= -0.5 or 0.5 <= skewness < 1:
            skew.append(col)
        else:
            extremeSkew.append(col)

# Apply Winsorization
winsorizer = Winsorizer(capping_method='iqr', tail='both', fold=3)
for col in extremeSkew:
    if col != 'TENURE' and col in pdfNum.columns:
        pdfNum[col] = winsorizer.fit_transform(pdfNum[[col]])


# Scaling
scaler = MinMaxScaler()
pdfNumScaled = scaler.fit_transform(pdfNum)

# PCA fit
pca = PCA()
pca.fit(pdfNumScaled)

# Plot Cumulative Explained Variance Ratio
fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_ * 100), marker='o')
ax1.set_xlabel('Number of components')
ax1.set_ylabel('Explained Variance Ratio - Cumulative (%)')
ax1.set_title('Cumulative Explained Variance by PCA Components')
ax1.grid()
st.pyplot(fig1)
plt.clf()

# Plot Eigenvalues
fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
ax2.set_xlabel('Number of components')
ax2.set_ylabel('Eigenvalues')
ax2.set_title('PCA Eigenvalues')
ax2.grid()
st.pyplot(fig2)
plt.clf()

# Number of features to retain 95% variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
num_features = np.argmax(cumsum >= 0.95) + 1
st.write(f"Number of PCA components to retain 95% variance: **{num_features}**")

# Transform data with optimal PCA components
pca_opt = PCA(n_components=num_features)
pdfNumScaled_pca = pca_opt.fit_transform(pdfNumScaled)

# Elbow Method for KMeans
wcss = []
random_state = 10
max_cluster = 9
for i in range(2, max_cluster+1):
    km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
    km.fit(pdfNumScaled_pca)
    wcss.append(km.inertia_)

fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(range(2, max_cluster+1), wcss, marker='o')
ax3.set_xlabel('Number of Clusters')
ax3.set_ylabel('WCSS')
ax3.set_title('Elbow Method For Optimal Clusters')
ax3.grid()
st.pyplot(fig3)
plt.clf()

# --- EDA Before and After Clustering ---

if tab == "EDA Before Clustering":
    st.title("Exploratory Data Analysis - Before Clustering")
    st.checkbox("Show the dataframe", value=True)
    st.dataframe(df, height=300)

    st.write("Data Info:")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": [df[col].notnull().sum() for col in df.columns],
        "Dtype": [df[col].dtype for col in df.columns]
    })
    st.dataframe(info_df)

    st.write("Summary Statistics:")
    st.dataframe(df.describe())

    st.write("Missing Values:")
    st.dataframe(df.isnull().sum())

    # Custom EDA: Credit Limit vs Purchases, Balance, OneOff Purchases, Installments Purchases
    st.write("Credit Limit vs Various Features")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.scatterplot(x='PURCHASES', y='CREDIT_LIMIT', data=df, ax=axes[0,0])
    axes[0,0].set_title('Credit Limit And Purchases')
    sns.scatterplot(x='BALANCE', y='CREDIT_LIMIT', data=df, ax=axes[0,1])
    axes[0,1].set_title('Credit Limit And Balance')
    sns.scatterplot(x='ONEOFF_PURCHASES', y='CREDIT_LIMIT', data=df, ax=axes[1,0])
    axes[1,0].set_title('Credit Limit And One Off Purchases')
    sns.scatterplot(x='INSTALLMENTS_PURCHASES', y='CREDIT_LIMIT', data=df, ax=axes[1,1])
    axes[1,1].set_title('Credit Limit And Installments Purchases')
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

elif tab == "EDA After Clustering":
    st.title("Exploratory Data Analysis - After Clustering")

    # Cluster selection
    k = st.sidebar.slider("Number of clusters", 2, 9, 4)
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
    pred = km.fit_predict(pdfNumScaled_pca)

    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = pred

    # PCA for 2D visualization
    pca_2d = PCA(n_components=2)
    pdfNumScaled_pca_2d = pca_2d.fit_transform(pdfNumScaled)

    st.write("PCA 2D Scatter Plot by Cluster")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pdfNumScaled_pca_2d[:,0], y=pdfNumScaled_pca_2d[:,1], hue=pred, palette='coolwarm', ax=ax4)
    ax4.set_title('Clusters visualized in PCA 2D space')
    st.pyplot(fig4)
    plt.clf()

    # Credit Limit vs Purchases
    st.write("Credit Limit vs Purchases by Cluster")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=df_clustered, x='PURCHASES', y='CREDIT_LIMIT', hue='cluster', palette='coolwarm', ax=ax5)
    ax5.set_title('CREDIT LIMIT vs PURCHASES')
    st.pyplot(fig5)
    plt.clf()

    # Credit Limit vs Balance
    st.write("Credit Limit vs Balance by Cluster")
    fig6, ax6 = plt.subplots()
    sns.scatterplot(data=df_clustered, x='BALANCE', y='CREDIT_LIMIT', hue='cluster', palette='coolwarm', ax=ax6)
    ax6.set_title('CREDIT LIMIT vs BALANCE')
    st.pyplot(fig6)
    plt.clf()

    # Credit Limit vs One-Off Purchases
    st.write("Credit Limit vs One-Off Purchases by Cluster")
    fig7, ax7 = plt.subplots()
    sns.scatterplot(data=df_clustered, x='ONEOFF_PURCHASES', y='CREDIT_LIMIT', hue='cluster', palette='coolwarm', ax=ax7)
    ax7.set_title('CREDIT LIMIT vs ONE-OFF PURCHASES')
    st.pyplot(fig7)
    plt.clf()

    # Credit Limit vs Installments Purchases
    st.write("Credit Limit vs Installments Purchases by Cluster")
    fig8, ax8 = plt.subplots()
    sns.scatterplot(data=df_clustered, x='INSTALLMENTS_PURCHASES', y='CREDIT_LIMIT', hue='cluster', palette='coolwarm', ax=ax8)
    ax8.set_title('CREDIT LIMIT vs INSTALLMENTS PURCHASES')
    st.pyplot(fig8)
    plt.clf()