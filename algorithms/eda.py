# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("customers.csv")  # Replace with your CSV path

# 1. Initial Data Overview
print("First 5 rows:")
print(df.head())  # View first 5 rows

print("\nData shape (rows, columns):", df.shape)

print("\nColumn names and data types:")
print(df.info())  # Check data types and missing values

print("\nSummary statistics:")
print(df.describe(include='all'))  # For numerical and categorical columns

# 2. Check Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Visualize missing values (if any)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# 3. Analyze Categorical Variables
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"\nUnique values in {col}: {df[col].nunique()}")
    print(df[col].value_counts().head())  # Top 5 categories

# 4. Analyze Numerical Variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    # Histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

    # Boxplot (check for outliers)
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# 5. Correlation Analysis (for numerical columns)
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 6. Relationships Between Variables (Example: Pairplot)
# Warning: Use subsample if dataset is large
sns.pairplot(df[numerical_cols].sample(1000))  # Adjust sample size
plt.show()

# 7. Target Variable Analysis (if applicable)
# Example for a regression/classification problem:
if 'target_column' in df.columns:
    sns.histplot(df['target_column'], kde=True)
    plt.title("Distribution of Target Variable")
    plt.show()