import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List
import os
import glob

class TNEADataLoader:
    def __init__(self):
        self.yearly_data: Dict[int, pd.DataFrame] = {}
        self.categories = ['OC', 'BC', 'BCM', 'MBC_DNC', 'SC', 'SCA', 'ST']
        self.merged_data = None

    def find_data_files(self, base_path: str) -> List[str]:
        """Find all relevant CSV files in the directory"""
        pattern = os.path.join(base_path, "Vocational_*_Mark_Cutoff.csv")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No matching files found in {base_path}")
            print("Please ensure your files follow the naming pattern: Vocational_YYYY_Mark_Cutoff.csv")
            return []
            
        return files

    def load_data(self, base_path: str) -> bool:
        """Load all available mark cutoff data files"""
        files = self.find_data_files(base_path)
        
        if not files:
            return False
            
        print("\nFound the following data files:")
        for file in files:
            print(f"- {os.path.basename(file)}")
        
        for file_path in files:
            try:
                # Extract year from filename using string operations
                filename = os.path.basename(file_path)
                year = int(filename.split('_')[1])  # Extracts year from Vocational_YYYY_Mark_Cutoff.csv
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                df['Year'] = year
                self.yearly_data[year] = df
                print(f"\nSuccessfully loaded data for year {year}")
                print(f"Number of records: {len(df)}")
                print(f"Columns found: {', '.join(df.columns)}")
                
            except Exception as e:
                print(f"\nError loading {filename}: {str(e)}")
                continue
        
        if self.yearly_data:
            print("\nData loading completed successfully!")
            self.merge_yearly_data()
            return True
        else:
            print("\nNo data was loaded successfully. Please check your files and try again.")
            return False

    def merge_yearly_data(self):
        """Merge data from all years into a single DataFrame"""
        if not self.yearly_data:
            return
        
        self.merged_data = pd.concat(self.yearly_data.values(), ignore_index=True)
        print(f"\nMerged data from {len(self.yearly_data)} years")
        print(f"Total records: {len(self.merged_data)}")

    def get_data_summary(self):
        """Get a summary of loaded data"""
        if not self.yearly_data:
            return "No data loaded"
            
        summary = "\nData Summary:\n"
        summary += "=" * 50 + "\n"
        
        for year, df in self.yearly_data.items():
            summary += f"\nYear {year}:\n"
            summary += f"- Number of records: {len(df)}\n"
            summary += f"- Number of unique colleges: {df['College Name'].nunique()}\n"
            summary += f"- Number of unique branches: {df['Branch Name'].nunique()}\n"
            
            # Show some basic statistics for each category
            for category in self.categories:
                if category in df.columns:
                    stats = df[category].describe()
                    summary += f"\n{category} Statistics:\n"
                    summary += f"  Mean: {stats['mean']:.2f}\n"
                    summary += f"  Min:  {stats['min']:.2f}\n"
                    summary += f"  Max:  {stats['max']:.2f}\n"
            
            summary += "-" * 50 + "\n"
            
        return summary

def setup_data_directory():
    """Create a data directory if it doesn't exist and return its path"""
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Create a 'data' directory if it doesn't exist
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"\nCreated data directory at: {data_dir}")
    
    return data_dir

def main():
    print("\nTNEA Data Loading Utility")
    print("=" * 50)
    
    # Setup data directory
    data_dir = setup_data_directory()
    print(f"\nLooking for data files in: {data_dir}")
    
    # Create an instance of the data loader
    loader = TNEADataLoader()
    
    # First try to load from data directory
    if not loader.load_data(data_dir):
        # If no files found in data directory, try current directory
        print("\nTrying current directory...")
        if not loader.load_data(os.getcwd()):
            print("\nPlease ensure your data files are present and follow these guidelines:")
            print("1. File naming format: Vocational_YYYY_Mark_Cutoff.csv")
            print("2. Place files either in the 'data' subdirectory or the current directory")
            print("3. Required columns: College Name, Branch Name, Branch code, and category columns (OC, BC, etc.)")
            return
    
    # Print data summary
    print(loader.get_data_summary())
    
    # Keep the window open (for Windows users)
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()