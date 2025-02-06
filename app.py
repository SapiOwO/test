import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests  # For interacting with Ollama

# File Uploader to Choose CSV File
st.sidebar.title("Upload or Use Default CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
else:
    # Default CSV file
    file_path = "data.csv"
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        st.error("No CSV file uploaded, and default file 'data.csv' not found!")
        st.stop()

df.columns = [f"Feature_{i}" for i in range(df.shape[1])]

# Sidebar: Data Transformations & Filtering
st.sidebar.header("Data Transformations")
normalize = st.sidebar.checkbox("Normalize Data")
standardize = st.sidebar.checkbox("Standardize Data")

if normalize:
    df = (df - df.min()) / (df.max() - df.min())
elif standardize:
    df = (df - df.mean()) / df.std()

st.sidebar.header("Data Filtering")
selected_feature = st.sidebar.selectbox("Select Feature to Filter", df.columns)
min_val, max_val = df[selected_feature].min(), df[selected_feature].max()
filter_range = st.sidebar.slider(f"Select range for {selected_feature}", min_val, max_val, (min_val, max_val))

# Apply filtering
df_filtered = df[(df[selected_feature] >= filter_range[0]) & (df[selected_feature] <= filter_range[1])]
use_filtered = st.sidebar.checkbox("Use Filtered Data", value=False)
data = df_filtered.copy() if use_filtered else df.copy()

# Visualization Functions
def plot_heatmap():
    st.subheader("Heatmap of Dataset")
    plt.figure(figsize=(12, 5))
    sns.heatmap(data, cmap="coolwarm", annot=False, cbar=True)
    st.pyplot(plt.gcf())

def plot_pca():
    st.subheader("PCA Scatter Plot")
    data_transposed = data.T
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df)
    st.pyplot(plt.gcf())

def plot_kmeans_clusters():
    st.subheader("K-Means Clustering")
    num_clusters = st.slider("Select Number of Clusters", 2, 10, 3, key="kmeans_slider")
    data_transposed = data.T
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_transposed)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_transposed)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
    plt.title(f"K-Means Clustering with {num_clusters} Clusters")
    st.pyplot(plt.gcf())

# AI Integration with Ollama (Fixed)
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Make sure Ollama is running!

def query_ollama(prompt, data_context):
    """
    Query Ollama API with a prompt that includes CSV data context.
    """
    full_prompt = f"CSV Data Sample (first 3 rows):\n{data_context}\n\nUser Query:\n{prompt}"
    payload = {
        "model": "gemma2:9b-instruct-q4_K_M",  # Change to your installed Ollama model
        "prompt": full_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "No response returned.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"

# Streamlit UI
st.title("Data Visualization & AI Chat")

# Create two columns
col1, col2 = st.columns([2, 1])

# Left column for visualizations
with col1:
    st.write("### Dataset Overview")
    st.dataframe(data.head())
    st.write("### Descriptive Statistics")
    st.dataframe(data.describe().T)

    # Visualization Selection
    visualization = st.selectbox("Choose a visualization", [
        "Heatmap",
        "PCA Scatter Plot",
        "K-Means Clustering"
    ])

    if visualization == "Heatmap":
        plot_heatmap()
    elif visualization == "PCA Scatter Plot":
        plot_pca()
    elif visualization == "K-Means Clustering":
        plot_kmeans_clusters()

# Right column for AI chat
with col2:
    st.markdown("---")
    st.header("Ask Mewo A.I. about the Data")
    user_prompt = st.text_area("Enter your question about the data:")

    include_context = st.checkbox("Include CSV context (first 3 rows)", value=True)
    data_context = data.head(3).to_csv(index=False) if include_context else ""

    if st.button("Submit Query"):
        with st.spinner("Thinking..."):
            ai_response = query_ollama(user_prompt, data_context)
        st.subheader("AI Response:")
        st.write(ai_response)

st.write("AI is running on local machine!")
