import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read and clean the data
def clean_college_data(data):
    # Create a DataFrame with the columns
    df = pd.DataFrame([
        {
            'College Code': row['COLLEGE CODE'],
            'College Name': row['COLLEGE NAME'].strip().replace('\n', ' '),
            'Branch Code': row['BRANCH CODE'],
            'Branch Name': row['RANCH NAME'].strip().replace('\n', ' '),
            'OC': row['OC'],
            'BC': row['BC'],
            'BCM': row['BCM'],
            'MBC': row['MBC'],
            'SC': row['SC'],
            'SCA': row['SCA'],
            'ST': row['ST']
        }
        for _, row in data.iterrows()
    ])
    
    # Convert cutoff marks to numeric, replacing empty strings with NaN
    for col in ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def analyze_cutoffs(df):
    # Calculate average cutoffs by branch
    branch_stats = df.groupby('Branch Name').agg({
        'OC': ['mean', 'min', 'max', 'count'],
        'BC': ['mean', 'min', 'max', 'count'],
        'MBC': ['mean', 'min', 'max', 'count'],
        'SC': ['mean', 'min', 'max', 'count']
    }).round(2)
    
    # Find top colleges by cutoff
    top_colleges = df.nsmallest(5, 'OC')[['College Name', 'Branch Name', 'OC']]
    
    # Calculate community-wise statistics
    community_stats = pd.DataFrame({
        'Community': ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST'],
        'Average Cutoff': [df[col].mean() for col in ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']],
        'Min Cutoff': [df[col].min() for col in ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']],
        'Max Cutoff': [df[col].max() for col in ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']],
    }).round(2)
    
    return {
        'branch_stats': branch_stats,
        'top_colleges': top_colleges,
        'community_stats': community_stats
    }

def predict_colleges(df, cutoff, community, branch_name=None):
    # Filter based on cutoff and community
    if branch_name:
        eligible_colleges = df[
            (df[community] <= cutoff) & 
            (df['Branch Name'] == branch_name)
        ][['College Name', 'Branch Name', community]]
    else:
        eligible_colleges = df[
            df[community] <= cutoff
        ][['College Name', 'Branch Name', community]]
    
    return eligible_colleges.sort_values(community, ascending=False)

# Load and analyze the data
raw_data = pd.read_csv('VVocational_2023_Mark_Cutoff.csv')  # Replace with your data loading method
df = clean_college_data(raw_data)
analysis_results = analyze_cutoffs(df)

# Print analysis results
print("\nBranch-wise Statistics:")
print(analysis_results['branch_stats'])

print("\nTop Colleges by Cutoff:")
print(analysis_results['top_colleges'])

print("\nCommunity-wise Statistics:")
print(analysis_results['community_stats'])

# Example prediction
cutoff = 150
community = 'OC'
branch = 'COMPUTER SCIENCE AND ENGINEERING'
predictions = predict_colleges(df, cutoff, community, branch)
print(f"\nPredicted colleges for {branch} with cutoff {cutoff} in {community} category:")
print(predictions)