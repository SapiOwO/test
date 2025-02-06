import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Page Title
st.title("CSV Data Visualization")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Load the CSV file
    df = pd.read_csv(uploaded_file)

    # Convert column names to strings (if necessary)
    df.columns = df.columns.astype(str)

    # Ensure only numeric columns are used
    df = df.select_dtypes(include=['number'])

    # Select a subset of the data for visualization (first 10 numeric columns)
    df_subset = df.iloc[:, :10]

    # Show dataset preview
    st.write("### Data Preview", df.head())

    # Line Chart
    st.write("### Line Chart (First 10 Features)")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in df_subset.columns:
        ax.plot(df_subset.index, df_subset[col], label=col)
    ax.legend()
    ax.set_title("Line Chart of First 10 Features")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    st.pyplot(fig)  # Render in Streamlit

    # Bar Chart (Mean values of first 10 columns)
    st.write("### Bar Chart (Mean Values of First 10 Features)")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_subset.mean().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Bar Chart of Mean Values (First 10 Features)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Mean Value")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Scatter Plot (First two columns, if available)
    if df_subset.shape[1] >= 2:
        st.write(f"### Scatter Plot: {df_subset.columns[0]} vs {df_subset.columns[1]}")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df_subset.iloc[:, 0], df_subset.iloc[:, 1], alpha=0.5, color='red')
        ax.set_title(f"Scatter Plot: {df_subset.columns[0]} vs {df_subset.columns[1]}")
        ax.set_xlabel(df_subset.columns[0])
        ax.set_ylabel(df_subset.columns[1])
        st.pyplot(fig)

    # Heatmap (First 10 Columns)
    st.write("### Heatmap (First 10 Features Correlation)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Heatmap of First 10 Features")
    st.pyplot(fig)

