import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from data.csv
data = pd.read_csv('data.csv')

# Display the dataframe
st.write("Here's the data:")
st.write(data)

# Create a plot
fig, ax = plt.subplots()
data.plot(kind='line', x='Column1', y='Column2', ax=ax)  # Replace 'Column1' and 'Column2' with the actual column names

# Display the plot
st.pyplot(fig)

# Streamlit command to run the app
# Run this command in your terminal: streamlit run your_script_name.py
