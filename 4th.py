# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
df = pd.read_csv('/content/Iris.csv')

# Display the first few rows of the dataset
print(df.head())

print(df.info())

# Calculate descriptive statistics
print(df.describe())

# Plot a histogram of the Sepal Length feature
plt.figure(figsize=(8, 6))
sns.histplot(x='SepalLengthCm', data=df, bins=50)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Plot a scatter plot of Sepal Length vs. Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', data=df)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

df.replace({'Species':{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}},inplace=True)

# Calculate the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Group the data by Species and calculate the mean Sepal Length and Petal Length
grouped = df.groupby('Species').agg({'SepalLengthCm': 'mean', 'PetalLengthCm': 'mean'})
print(grouped)

# Plot a bar chart of the mean Sepal Length and Petal Length by Species
plt.figure(figsize=(8, 8))
sns.barplot(x='Species', y='SepalLengthCm', data=grouped)
plt.title('Mean Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Sepal Length (cm)')
plt.show()

plt.figure(figsize=(8, 8))
sns.barplot(x='Species', y='PetalLengthCm', data=grouped)
plt.title('Mean Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Mean Petal Length (cm)')
plt.show()
