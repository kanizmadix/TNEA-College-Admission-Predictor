import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import squarify

# Load data
file_path = "C:\\Users\\Kaniz\\Pictures\\TNEA_fct\\Vocational_2023_Mark_Cutoff.csv"
df = pd.read_csv(file_path)

# Group data by college and calculate mean cutoff for each category
college_df = df.groupby('COLLEGE NAME')[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean().reset_index()

# Top Colleges by Cutoff (OC) - Bar Chart
top_colleges = college_df.sort_values(by='OC', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_colleges, x='COLLEGE NAME', y='OC')
plt.title('Top Colleges by Cutoff (OC)')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('top_colleges_by_cutoff_oc.png')
plt.show()

# Top Colleges by Cutoff (OC) - Grouped Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(data=top_colleges, x='COLLEGE NAME', y='OC')
plt.title('Top Colleges by Cutoff (OC) - Grouped Bar Chart')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('top_colleges_by_cutoff_oc_grouped.png')
plt.show()

# Category-Wise Best Colleges - Stacked Bar Chart
plt.figure(figsize=(10, 6))
college_df.set_index('COLLEGE NAME').plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Category-Wise Best Colleges - Stacked Bar Chart')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('category_wise_best_colleges_stacked.png')
plt.show()

# Category-Wise Best Colleges - Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(college_df[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].corr(), annot=True, cmap='Blues')
plt.title('Category-Wise Best Colleges - Heatmap')
plt.savefig('category_wise_best_colleges_heatmap.png')
plt.show()

# Branch Performance Across Colleges - Boxplot
branch_df = df.groupby('BRANCH NAME')[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean()
plt.figure(figsize=(10, 6))
sns.boxplot(data=branch_df)
plt.title('Branch Performance Across Colleges - Boxplot')
plt.xlabel('Branch')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('branch_performance_across_colleges_boxplot.png')
plt.show()

# Branch Performance Across Colleges - Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=branch_df, x=branch_df.index, y='OC')
plt.title('Branch Performance Across Colleges - Scatter Plot')
plt.xlabel('Branch')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('branch_performance_across_colleges_scatterplot.png')
plt.show()

# Average Cutoff Comparison - Line Chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=college_df, x='COLLEGE NAME', y='OC')
plt.title('Average Cutoff Comparison - Line Chart')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('average_cutoff_comparison_linechart.png')
plt.show()

# Average Cutoff Comparison - Radar Chart
fig = px.line_polar(college_df, r='OC', theta='COLLEGE NAME', line_close=True)
fig.update_layout(title='Average Cutoff Comparison - Radar Chart')
fig.write_image('average_cutoff_comparison_radar.png')
fig.show()

# Overall College Rankings - Treemap
plt.figure(figsize=(10, 6))
squarify.plot(sizes=college_df['OC'], label=college_df['COLLEGE NAME'], alpha=0.8)
plt.title('Overall College Rankings - Treemap')
plt.axis('off')
plt.savefig('overall _college_rankings_treemap.png')
plt.show()

# Overall College Rankings - Bubble Chart
plt.figure(figsize=(10, 6))
sns.scatterplot(data=college_df, x='COLLEGE NAME', y='OC', size='OC', sizes=(20, 500))
plt.title('Overall College Rankings - Bubble Chart')
plt.xlabel('College')
plt.ylabel('Average Cutoff')
plt.xticks(rotation=45)
plt.savefig('overall_college_rankings_bubblechart.png')
plt.show()