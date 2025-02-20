import pandas as pd
import numpy as np
import os
from datetime import datetime

class TNEASetup:
    def __init__(self):
        self.categories = ['OC', 'BC', 'BCM', 'MBC_DNC', 'SC', 'SCA', 'ST']
        self.years = range(2019, 2024)  # 2019 to 2023
        
    def create_sample_data(self, base_path: str):
        """Create sample CSV files with realistic data"""
        # Create data directory if it doesn't exist
        data_dir = os.path.join(base_path, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Sample college and branch data
        colleges = [
            "Anna University CEG Campus",
            "Anna University MIT Campus",
            "Anna University ACT Campus",
            "Government College of Engineering - Coimbatore",
            "Government College of Engineering - Salem"
        ]
        
        branches = [
            ("CS", "Computer Science and Engineering"),
            ("IT", "Information Technology"),
            ("ECE", "Electronics and Communication Engineering"),
            ("EEE", "Electrical and Electronics Engineering"),
            ("MECH", "Mechanical Engineering")
        ]
        
        for year in self.years:
            # Create DataFrame for each year
            data = []
            
            for college in colleges:
                for branch_code, branch_name in branches:
                    row = {
                        'College Name': college,
                        'Branch Name': branch_name,
                        'Branch code': branch_code
                    }
                    
                    # Generate realistic cutoff marks for each category
                    base_cutoff = np.random.normal(175, 10)  # Base cutoff around 175
                    for i, category in enumerate(self.categories):
                        # Gradually decrease cutoff for each category
                        category_cutoff = max(100, min(200, base_cutoff - (i * 5) + np.random.normal(0, 2)))
                        row[category] = round(category_cutoff, 2)
                    
                    data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Save to CSV
            file_name = f"Vocational_{year}_Mark_Cutoff.csv"
            file_path = os.path.join(data_dir, file_name)
            df.to_csv(file_path, index=False)
            print(f"Created sample file: {file_name}")

def main():
    print("\nTNEA Data Setup Utility")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.getcwd()
    
    setup = TNEASetup()
    
    print("\nChecking for existing data files...")
    data_dir = os.path.join(current_dir, 'data')
    
    # Check if files exist
    files_exist = False
    if os.path.exists(data_dir):
        existing_files = [f for f in os.listdir(data_dir) if f.endswith('Mark_Cutoff.csv')]
        if existing_files:
            files_exist = True
            print("\nFound existing files:")
            for file in existing_files:
                print(f"- {file}")
    
    if not files_exist:
        print("\nNo existing data files found.")
        create_sample = input("Would you like to create sample data files? (y/n): ").lower()
        
        if create_sample == 'y':
            print("\nCreating sample data files...")
            setup.create_sample_data(current_dir)
            print("\nSample data files have been created in the 'data' directory.")
            print("You can now use these files with the main TNEA analysis program.")
        else:
            print("\nPlease ensure you have the following files in a 'data' directory:")
            for year in range(2019, 2024):
                print(f"- Vocational_{year}_Mark_Cutoff.csv")
    
    print("\nSetup complete!")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()