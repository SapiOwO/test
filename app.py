import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = "data.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Rename columns if necessary (assuming they are incorrectly formatted)
df.columns = [f"Feature_{i}" for i in range(df.shape[1])]

# Select a subset of the data for visualization (e.g., first 10 columns)
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

# Scatter Plot (First two columns)
plt.figure(figsize=(6, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.5, color='red')
plt.title("Scatter Plot of First Two Features")
plt.xlabel("Feature_0")
plt.ylabel("Feature_1")
plt.show()

# Heatmap (First 10 Columns)
plt.figure(figsize=(10, 6))
sns.heatmap(df_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of First 10 Features")
plt.show()
