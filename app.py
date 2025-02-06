import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests  # For interacting with Ollama

# File Uploader to Choose CSV File
st.set_page_config(layout="wide")  # Enable wide layout
st.title("Data Visualization & AI Chat")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
else:
    file_path = "data.csv"
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        st.error("No CSV file uploaded, and default file 'data.csv' not found!")
        st.stop()

# Assign column names
df.columns = [f"Feature_{i}" for i in range(df.shape[1])]
data = df.copy()

# Visualization Functions
def plot_heatmap():
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, cmap="coolwarm", annot=False, cbar=True)
    st.pyplot(plt.gcf())

def plot_pca():
    data_transposed = data.T
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    st.pyplot(plt.gcf())

def plot_kmeans_clusters():
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3, key="kmeans_slider")
    data_transposed = data.T
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_transposed)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    st.pyplot(plt.gcf())

def plot_histogram():
    selected_feature = st.selectbox("Select Feature for Histogram", data.columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(data[selected_feature], kde=True, bins=30)
    st.pyplot(plt.gcf())

def plot_boxplot():
    selected_feature = st.selectbox("Select Feature for Boxplot", data.columns, key="boxplot")
    plt.figure(figsize=(10, 6))
    sns.boxplot(y=data[selected_feature])
    st.pyplot(plt.gcf())

# AI Integration with Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt, data_context):
    full_prompt = f"CSV Data Sample (first 3 rows):\n{data_context}\n\nUser Query:\n{prompt}"
    payload = {"model": "gemma2:9b-instruct-q4_K_M", "prompt": full_prompt, "stream": False}
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "No response returned.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"

# Streamlit Layout
col1, col2 = st.columns([2, 3])

with col1:
    st.write("### Dataset Overview")
    st.dataframe(data.head())
    st.write("### Descriptive Statistics")
    st.dataframe(data.describe().T)
    
    st.write("### Select Visualization")
    visualization = st.selectbox("Choose a visualization", [
        "Heatmap", "PCA Scatter Plot", "K-Means Clustering", "Histogram", "Boxplot"
    ])
    
    if visualization == "Heatmap":
        plot_heatmap()
    elif visualization == "PCA Scatter Plot":
        plot_pca()
    elif visualization == "K-Means Clustering":
        plot_kmeans_clusters()
    elif visualization == "Histogram":
        plot_histogram()
    elif visualization == "Boxplot":
        plot_boxplot()

with col2:
    st.header("Ask Mewo A.I. about the Data")
    user_prompt = st.text_area("Enter your question about the data:")
    
    include_context = st.checkbox("Include CSV context (first 3 rows)", value=True)
    data_context = data.head(3).to_csv(index=False) if include_context else ""
    
    if st.button("Submit Query"):
        with st.spinner("Thinking..."):
            ai_response = query_ollama(user_prompt, data_context)
        st.subheader("AI Response:")
        st.write(ai_response)
