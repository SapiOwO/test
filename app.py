import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data.csv"  # Change this if needed
df = pd.read_csv(file_path)

# Convert column names to strings (some may be numbers)
df.columns = df.columns.astype(str)

# Ensure only numeric columns are used
df = df.select_dtypes(include=['number'])

# Select a subset of the data for visualization (first 10 numeric columns)
df_subset = df.iloc[:, :10]

# Line Chart
plt.figure(figsize=(10, 5))
for col in df_subset.columns:
    plt.plot(df_subset.index, df_subset[col], label=col)
plt.legend()
plt.title("Line Chart of First 10 Features")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

# Bar Chart (Mean values of first 10 columns)
plt.figure(figsize=(10, 5))
df_subset.mean().plot(kind='bar', color='skyblue')
plt.title("Bar Chart of Mean Values (First 10 Features)")
plt.xlabel("Features")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.show()

# Scatter Plot (First two columns, if available)
if df_subset.shape[1] >= 2:
    plt.figure(figsize=(6, 6))
    plt.scatter(df_subset.iloc[:, 0], df_subset.iloc[:, 1], alpha=0.5, color='red')
    plt.title(f"Scatter Plot: {df_subset.columns[0]} vs {df_subset.columns[1]}")
    plt.xlabel(df_subset.columns[0])
    plt.ylabel(df_subset.columns[1])
    plt.show()

# Heatmap (First 10 Columns)
plt.figure(figsize=(10, 6))
sns.heatmap(df_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap of First 10 Features")
plt.show()
