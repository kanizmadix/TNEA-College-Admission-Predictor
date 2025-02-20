import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List
import os

class TNEAMultiYearAnalyzer:
    def __init__(self):
        self.yearly_data: Dict[int, pd.DataFrame] = {}
        self.categories = ['OC', 'BC', 'BCM', 'MBC_DNC', 'SC', 'SCA', 'ST']
        self.merged_data = None
        
    def load_multiple_years(self, base_path: str):
        """
        Load mark cutoff data for multiple years
        """
        years = range(2019, 2024)  # 2019 to 2023
        
        for year in years:
            file_name = f"Vocational_{year}_Mark_Cutoff.csv"
            file_path = os.path.join(base_path, file_name)
            
            try:
                df = pd.read_csv(file_path)
                # Add year column
                df['Year'] = year
                self.yearly_data[year] = df
                print(f"Successfully loaded data for {year}")
            except FileNotFoundError:
                print(f"Warning: File not found for year {year}: {file_path}")
            except Exception as e:
                print(f"Error loading data for {year}: {str(e)}")
        
        if self.yearly_data:
            self.merge_yearly_data()
        else:
            raise ValueError("No data was loaded successfully")

    def merge_yearly_data(self):
        """
        Merge data from all years into a single DataFrame
        """
        if not self.yearly_data:
            return
        
        self.merged_data = pd.concat(self.yearly_data.values(), ignore_index=True)

    def analyze_trends(self, branch_code: str = None):
        """
        Analyze cutoff trends across years for specific branch or all branches
        """
        if self.merged_data is None:
            return None
        
        trends = {}
        
        if branch_code:
            branch_data = self.merged_data[self.merged_data['Branch code'] == branch_code]
            branch_name = branch_data['Branch Name'].iloc[0] if not branch_data.empty else "Unknown"
            trends[branch_code] = self._analyze_branch_trends(branch_data, branch_name)
        else:
            for code in self.merged_data['Branch code'].unique():
                branch_data = self.merged_data[self.merged_data['Branch code'] == code]
                branch_name = branch_data['Branch Name'].iloc[0]
                trends[code] = self._analyze_branch_trends(branch_data, branch_name)
        
        return trends

    def _analyze_branch_trends(self, branch_data: pd.DataFrame, branch_name: str):
        """
        Analyze trends for a specific branch
        """
        trend_data = {
            'branch_name': branch_name,
            'categories': {}
        }
        
        for category in self.categories:
            if category in branch_data.columns:
                yearly_stats = branch_data.groupby('Year')[category].agg(['mean', 'min', 'max']).round(2)
                
                # Calculate year-over-year changes
                yoy_changes = yearly_stats['mean'].pct_change() * 100
                
                trend_data['categories'][category] = {
                    'yearly_stats': yearly_stats.to_dict('index'),
                    'overall_trend': {
                        'start': yearly_stats['mean'].iloc[0],
                        'end': yearly_stats['mean'].iloc[-1],
                        'change': ((yearly_stats['mean'].iloc[-1] - yearly_stats['mean'].iloc[0]) / 
                                 yearly_stats['mean'].iloc[0] * 100).round(2)
                    },
                    'year_over_year_changes': yoy_changes.dropna().to_dict()
                }
        
        return trend_data

    def predict_cutoffs(self, branch_code: str, category: str):
        """
        Predict cutoffs for next year using linear regression
        """
        if self.merged_data is None:
            return None
            
        branch_data = self.merged_data[self.merged_data['Branch code'] == branch_code]
        if branch_data.empty or category not in branch_data.columns:
            return None
            
        # Prepare data for prediction
        X = branch_data['Year'].values.reshape(-1, 1)
        y = branch_data[category].values
        
        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for next year (2024)
        next_year = np.array([[2024]])
        predicted_cutoff = model.predict(next_year)[0]
        
        # Calculate confidence metrics
        r2_score = model.score(X, y)
        
        return {
            'predicted_cutoff': round(predicted_cutoff, 2),
            'confidence_score': round(r2_score, 2),
            'trend_coefficient': round(model.coef_[0], 2)
        }

    def evaluate_admission_chances(self, cutoff: float, year: int = None):
        """
        Evaluate admission chances based on cutoff score
        Returns chances for the specified year or latest year if not specified
        """
        if self.merged_data is None:
            return None
            
        if year is None:
            year = max(self.yearly_data.keys())
            
        year_data = self.yearly_data[year]
        chances = {}
        
        for branch_code in year_data['Branch code'].unique():
            branch_data = year_data[year_data['Branch code'] == branch_code]
            branch_name = branch_data['Branch Name'].iloc[0]
            
            chances[branch_code] = {
                'branch_name': branch_name,
                'categories': {}
            }
            
            for category in self.categories:
                if category in branch_data.columns:
                    cat_data = branch_data[category].dropna()
                    if not cat_data.empty:
                        min_cutoff = cat_data.min()
                        chances[branch_code]['categories'][category] = {
                            'eligible': cutoff >= min_cutoff,
                            'min_required': min_cutoff,
                            'margin': cutoff - min_cutoff,
                            'colleges_available': len(cat_data[cat_data <= cutoff])
                        }
        
        return chances

def print_trend_analysis(trends):
    """Print comprehensive trend analysis results"""
    print("\nTNEA Cutoff Trends Analysis")
    print("==========================")
    
    for branch_code, branch_data in trends.items():
        print(f"\nBranch: {branch_code} - {branch_data['branch_name']}")
        
        for category, cat_data in branch_data['categories'].items():
            print(f"\n{category} Category:")
            print("Yearly Statistics:")
            for year, stats in cat_data['yearly_stats'].items():
                print(f"  {year}: Avg={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
            
            trend = cat_data['overall_trend']
            print(f"Overall Trend: {trend['change']:+.2f}% change from {trend['start']:.2f} to {trend['end']:.2f}")

def main():
    # Initialize analyzer
    analyzer = TNEAMultiYearAnalyzer()
    
    # Load data from multiple years
    base_path = base_path = r"C:\Users\Kaniz\Pictures\TNEA_fct"  # Corrected path using raw string
  # Change this to your data directory path
    analyzer.load_multiple_years(base_path)
    
    # Get user input
    print("\nEnter your marks:")
    math = float(input("Mathematics (out of 200): "))
    physics = float(input("Physics (out of 100): "))
    chemistry = float(input("Chemistry (out of 100): "))
    
    # Calculate cutoff
    math_converted = (math / 200) * 100
    physics_converted = (physics / 100) * 50
    chemistry_converted = (chemistry / 100) * 50
    cutoff = math_converted + physics_converted + chemistry_converted
    
    print(f"\nCalculated Cutoff: {cutoff:.2f}")
    
    # Get current year chances
    chances = analyzer.evaluate_admission_chances(cutoff)
    
    # Print chances and predictions for eligible branches
    print("\nAdmission Chances and Predictions:")
    print("=================================")
    
    for branch_code, branch_data in chances.items():
        eligible_categories = []
        
        # Check eligibility in any category
        for category, cat_data in branch_data['categories'].items():
            if cat_data['eligible']:
                eligible_categories.append(category)
        
        if eligible_categories:
            print(f"\nBranch: {branch_code} - {branch_data['branch_name']}")
            
            # Print current chances
            for category in eligible_categories:
                cat_data = branch_data['categories'][category]
                print(f"\n{category}:")
                print(f"  Currently eligible in {cat_data['colleges_available']} colleges")
                print(f"  Margin above minimum: {cat_data['margin']:.2f} points")
                
                # Get prediction for next year
                prediction = analyzer.predict_cutoffs(branch_code, category)
                if prediction:
                    print("  Prediction for next year:")
                    print(f"    Expected cutoff: {prediction['predicted_cutoff']:.2f}")
                    print(f"    Trend direction: {prediction['trend_coefficient']:+.2f} points/year")
                    print(f"    Confidence score: {prediction['confidence_score']:.2f}")

if __name__ == "__main__":
    main()