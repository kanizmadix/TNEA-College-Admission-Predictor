import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
file_path = "Vocational_2023_Mark_Cutoff.csv"
df = pd.read_csv(file_path)

# Cutoff Trends by Category
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean())
plt.title('Cutoff Trends by Category')
plt.xlabel('Category')
plt.ylabel('Average Cutoff')
plt.savefig('cutoff_trends_by_category.png')

# Branch-Wise Comparison
branch_df = df.groupby('RANCH NAME')[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean()
plt.figure(figsize=(10, 6))
sns.barplot(data=branch_df)
plt.title('Branch-Wise Comparison')
plt.xlabel('Branch')
plt.ylabel('Average Cutoff')
plt.savefig('branch_wise_comparison.png')

# Top Colleges by Cutoff
college_df = df.groupby('COLLEGE NAME')[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean()
top_colleges = college_df.sort_values(by='OC', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_colleges)
plt.title('Top Colleges by Cutoff')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.savefig('top_colleges_by_cutoff.png')

# Category-Wise Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']])
plt.title('Category-Wise Distribution')
plt.xlabel('Cutoff')
plt.ylabel('Frequency')
plt.savefig('category_wise_distribution.png')

# Missing Data Analysis
missing_data_df = df.isnull().sum()
plt.figure(figsize=(10, 6))
sns.heatmap(missing_data_df, annot=True, cmap='Blues')
plt.title('Missing Data Analysis')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.savefig('missing_data_analysis.png')

# Branch Popularity
branch_popularity_df = df.groupby('RANCH NAME')[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean().mean(axis=1)
top_branches = branch_popularity_df.sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_branches)
plt.title('Branch Popularity')
plt.xlabel('Branch')
plt.ylabel('Average Cutoff')
plt.savefig('branch_popularity.png')

# College Rankings
college_rankings_df = df.groupby('COLLEGE NAME')[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean().mean(axis=1)
top_colleges = college_rankings_df.sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_colleges)
plt.title('College Rankings')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.savefig('college_rankings.png')