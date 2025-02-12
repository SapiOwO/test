import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# Load CSV file without header
df = pd.read_csv("selamat.csv", header=None)

# Convert dataframe to numpy array
data = df.to_numpy()

# Streamlit App Title
st.title("Signal Data Analysis")

# Sidebar Menu
st.sidebar.title("Menu")
option = st.sidebar.selectbox("Select a visualization", 
                              ["Dataset Overview", "Descriptive Statistics", "Line Plot", "Spectrogram", "Histogram", "Population vs Sample Analysis"])

if option == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape of the dataset:", df.shape)
    st.write("First few rows of data:")
    st.write(df.head())

elif option == "Descriptive Statistics":
    st.subheader("Descriptive Statistics")
    st.write(pd.DataFrame(data).describe().T)

elif option == "Line Plot":
    st.subheader("Line Plot of Signal Amplitudes")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data[0], label="Signal 1", alpha=0.7)
    ax.plot(data[1], label="Signal 2", alpha=0.7)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Line Plot of Signal Amplitudes")
    ax.legend()
    st.pyplot(fig)

elif option == "Spectrogram":
    st.subheader("Spectrogram of Signal 1")
    fs = 1  # Assuming a sampling frequency of 1 Hz
    f, t, Sxx = spectrogram(data[0], fs)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram of Signal 1")
    fig.colorbar(cax, label="Power (dB)")
    st.pyplot(fig)

elif option == "Histogram":
    st.subheader("Histogram of Signal Amplitudes")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data[0], bins=50, alpha=0.7, label="Signal 1", density=True)
    ax.hist(data[1], bins=50, alpha=0.7, label="Signal 2", density=True)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of Signal Amplitudes")
    ax.legend()
    st.pyplot(fig)

elif option == "Population vs Sample Analysis":
    st.subheader("Population vs Sample Analysis")

    def calculate_population_sample_stats(data):
        population_mean = np.mean(data)
        sample_size = int(len(data) * 0.7)
        sample = np.random.choice(data, size=sample_size, replace=False)
        sample_mean = np.mean(sample)
        sample_percentage = (sample_size / len(data)) * 100
        population_percentage = 100 - sample_percentage
        return population_mean, sample_mean, sample, population_percentage, sample_percentage

    pop_mean_1, samp_mean_1, sample_1, pop_perc_1, samp_perc_1 = calculate_population_sample_stats(data[0])
    pop_mean_2, samp_mean_2, sample_2, pop_perc_2, samp_perc_2 = calculate_population_sample_stats(data[1])

    st.write(f"Signal 1 - Population Mean: {pop_mean_1}, Sample Mean: {samp_mean_1}")
    st.write(f"Signal 1 - Population: {pop_perc_1:.2f}%, Sample: {samp_perc_1:.2f}%")
    st.write(f"Signal 2 - Population Mean: {pop_mean_2}, Sample Mean: {samp_mean_2}")
    st.write(f"Signal 2 - Population: {pop_perc_2:.2f}%, Sample: {samp_perc_2:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data[0], bins=50, alpha=0.5, label="Population Signal 1", density=True, color='blue')
    ax.hist(sample_1, bins=50, alpha=0.7, label="Sample Signal 1", density=True, color='red')
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Population vs Sample - Signal 1")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data[1], bins=50, alpha=0.5, label="Population Signal 2", density=True, color='green')
    ax.hist(sample_2, bins=50, alpha=0.7, label="Sample Signal 2", density=True, color='orange')
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Density")
    ax.set_title("Population vs Sample - Signal 2")
    ax.legend()
    st.pyplot(fig)

st.write("Analysis Completed! 🚀")

# Create requirements.txt file
requirements = """
streamlit
pandas
matplotlib
numpy
scipy
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

st.write("Generated requirements.txt for deployment.")