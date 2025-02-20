import pandas as pd
import glob

# Merge all yearly files
all_files = glob.glob("data/Vocational_*_Mark_Cutoff.csv")
merged_data = pd.concat([pd.read_csv(f) for f in all_files])
merged_data.to_csv("data/merged_tnea_data.csv", index=False)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from typing import Dict, List

class TNEACalculator:
    def __init__(self):
        self.data = None
        self.categories = ['OC', 'BC', 'BCM', 'MBC_DNC', 'SC', 'SCA', 'ST']
        self.focus_branches = ['MECH', 'CS']
        self.focus_categories = ['BC', 'MBC_DNC']
        self.load_data()
    
    def load_data(self):
        """Load and merge data from all year files"""
        data_frames = []
        data_dir = 'data'
        
        for year in range(2019, 2024):
            file_name = f"Vocational_{year}_Mark_Cutoff.csv"
            file_path = os.path.join(data_dir, file_name)
            
            try:
                df = pd.read_csv(file_path)
                df['Year'] = year  # Add year column
                data_frames.append(df)
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")
        
        if data_frames:
            self.data = pd.concat(data_frames, ignore_index=True)
            print("Data loaded successfully!")
        else:
            raise ValueError("No data files could be loaded!")
    
    def calculate_cutoff(self, math: float, physics: float, chemistry: float) -> float:
        """Calculate cutoff score from PCM marks"""
        math_converted = (math / 200) * 100
        physics_converted = (physics / 100) * 50
        chemistry_converted = (chemistry / 100) * 50
        return math_converted + physics_converted + chemistry_converted
    
    def analyze_branch_trends(self) -> Dict:
        """Analyze trends for specific branches"""
        trends = {}
        
        for branch in self.focus_branches:
            branch_data = self.data[self.data['Branch code'].str.contains(branch, case=False, na=False)]
            if branch_data.empty:
                continue
            
            trends[branch] = {
                'yearly_stats': {},
                'predictions': {}
            }
            
            # Analyze each category
            for category in self.focus_categories:
                yearly_stats = {}
                years = []
                means = []
                
                for year in range(2019, 2024):
                    year_data = branch_data[branch_data['Year'] == year][category]
                    if not year_data.empty:
                        mean_cutoff = year_data.mean()
                        yearly_stats[year] = round(mean_cutoff, 2)
                        years.append(year)
                        means.append(mean_cutoff)
                
                trends[branch]['yearly_stats'][category] = yearly_stats
                
                # Predict 2024 cutoff
                if len(years) >= 2:  # Need at least 2 points for prediction
                    X = np.array(years).reshape(-1, 1)
                    y = np.array(means)
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    prediction_2024 = model.predict([[2024]])[0]
                    trend_coef = model.coef_[0]
                    
                    trends[branch]['predictions'][category] = {
                        'predicted_2024': round(prediction_2024, 2),
                        'trend': round(trend_coef, 2)
                    }
        
        return trends
    
    def evaluate_chances(self, cutoff: float) -> Dict:
        """Evaluate admission chances based on cutoff score"""
        current_year = 2023  # Using most recent year
        current_data = self.data[self.data['Year'] == current_year]
        
        chances = {}
        for branch in self.focus_branches:
            branch_data = current_data[current_data['Branch code'].str.contains(branch, case=False, na=False)]
            if branch_data.empty:
                continue
            
            chances[branch] = {
                'branch_name': branch_data['Branch Name'].iloc[0],
                'categories': {}
            }
            
            for category in self.focus_categories:
                cutoffs = branch_data[category].dropna()
                if not cutoffs.empty:
                    min_cutoff = cutoffs.min()
                    eligible = cutoff >= min_cutoff
                    margin = cutoff - min_cutoff
                    
                    chances[branch]['categories'][category] = {
                        'eligible': eligible,
                        'min_required': round(min_cutoff, 2),
                        'margin': round(margin, 2),
                        'status': 'Eligible' if eligible else f'Need {abs(round(margin, 2))} more points'
                    }
        
        return chances

def print_analysis_report(calculator: TNEACalculator, math: float, physics: float, chemistry: float):
    """Print comprehensive analysis report"""
    cutoff = calculator.calculate_cutoff(math, physics, chemistry)
    
    print("\n" + "="*60)
    print("TNEA CUTOFF ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nYour Details:")
    print(f"Mathematics: {math}/200")
    print(f"Physics: {physics}/100")
    print(f"Chemistry: {chemistry}/100")
    print(f"Calculated Cutoff: {cutoff:.2f}")
    
    # Get trends and predictions
    trends = calculator.analyze_branch_trends()
    
    print("\nTRENDS AND PREDICTIONS")
    print("-"*60)
    
    for branch, data in trends.items():
        print(f"\n{branch} Branch Analysis:")
        for category in calculator.focus_categories:
            if category in data['yearly_stats']:
                print(f"\n{category} Category Historical Cutoffs:")
                for year, cutoff_val in data['yearly_stats'][category].items():
                    print(f"  {year}: {cutoff_val:.2f}")
                
                if category in data['predictions']:
                    pred = data['predictions'][category]
                    print(f"  Predicted 2024: {pred['predicted_2024']:.2f}")
                    trend_direction = "Increasing" if pred['trend'] > 0 else "Decreasing"
                    print(f"  Trend: {trend_direction} by {abs(pred['trend']):.2f} points per year")
    
    # Get chances
    chances = calculator.evaluate_chances(cutoff)
    
    print("\nYOUR ELIGIBILITY STATUS")
    print("-"*60)
    
    for branch, data in chances.items():
        print(f"\n{branch}:")
        for category, cat_data in data['categories'].items():
            print(f"\n{category}:")
            print(f"Status: {cat_data['status']}")
            print(f"Minimum Required: {cat_data['min_required']}")
            if cat_data['eligible']:
                print(f"Margin above cutoff: +{cat_data['margin']:.2f}")

def main():
    print("\nTNEA Cutoff Calculator and Analyzer")
    print("="*60)
    
    try:
        calculator = TNEACalculator()
        
        while True:
            try:
                math = float(input("\nEnter Mathematics marks (out of 200): "))
                physics = float(input("Enter Physics marks (out of 100): "))
                chemistry = float(input("Enter Chemistry marks (out of 100): "))
                
                if not (0 <= math <= 200 and 0 <= physics <= 100 and 0 <= chemistry <= 100):
                    print("\nError: Please enter valid marks within the specified range!")
                    continue
                
                print_analysis_report(calculator, math, physics, chemistry)
                
            except ValueError:
                print("\nError: Please enter valid numerical marks!")
                continue
            
            choice = input("\nWould you like to calculate for another student? (y/n): ")
            if choice.lower() != 'y':
                break
        
        print("\nThank you for using the TNEA Calculator!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please ensure all data files are present in the 'data' directory.")

if __name__ == "__main__":
    main()