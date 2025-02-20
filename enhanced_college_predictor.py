import pandas as pd
import numpy as np

def create_college_data():
    """Create initial CSV file from the college data"""
    data = {
        'COLLEGE CODE': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1013, 1013, 1013, 1013, 1014, 1015, 1015, 1015, 1015, 1026, 1026, 1110, 1110, 1112, 1113, 1114, 1114, 1116, 1118],
        'COLLEGE NAME': [
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University Departments of Anna University Chennai - CEG Campus",
            "University College of Engineering, Villupuram",
            "University College of Engineering, Villupuram",
            "University College of Engineering, Villupuram",
            "University College of Engineering, Villupuram",
            "University College of Engineering Tindivanam",
            "University College of Engineering, Arni",
            "University College of Engineering, Arni",
            "University College of Engineering, Arni",
            "University College of Engineering, Arni",
            "University College of Engineering Kancheepuram",
            "University College of Engineering Kancheepuram",
            "Prathyusha Engineering College",
            "Prathyusha Engineering College",
            "R M D Engineering College",
            "R M K Engineering College",
            "S A Engineering College",
            "S A Engineering College",
            "Sri Venkateswara College of Engineering and Technology",
            "Vel Tech Multi Tech Dr. Rangarajan Dr. Sakunthala Engineering College"
        ],
        'BRANCH CODE': ['BY', 'CE', 'CM', 'CS', 'EC', 'EM', 'GI', 'IE', 'IM', 'MA', 'ME', 'MI', 'MN', 'XC', 'XM', 'EC', 'IT', 'ME', 'XM', 'IT', 'CS', 'EC', 'ME', 'XM', 'CS', 'ME', 'BT', 'CS', 'CS', 'CS', 'AD', 'CS', 'CS', 'ME'],
        'BRANCH NAME': [
            "BIO MEDICAL ENGINEERING (SS)", 
            "CIVIL ENGINEERING",
            "COMPUTER SCIENCE AND ENGINEERING (SS)",
            "COMPUTER SCIENCE AND ENGINEERING",
            "ELECTRONICS AND COMMUNICATION ENGINEERING",
            "ELECTRONICS AND COMMUNICATION ENGINEERING (SS)",
            "GEO INFORMATICS",
            "INDUSTRIAL ENGINEERING",
            "INFORMATION TECHNOLOGY (SS)",
            "MATERIAL SCIENCE AND ENGINEERING (SS)",
            "MECHANICAL ENGINEERING",
            "MINING ENGINEERING",
            "MANUFACTURING ENGINEERING",
            "CIVIL ENGINEERING (TAMIL MEDIUM)",
            "MECHANICAL ENGINEERING (TAMIL MEDIUM)",
            "ELECTRONICS AND COMMUNICATION ENGINEERING",
            "INFORMATION TECHNOLOGY",
            "MECHANICAL ENGINEERING",
            "MECHANICAL ENGINEERING (TAMIL MEDIUM)",
            "INFORMATION TECHNOLOGY",
            "COMPUTER SCIENCE AND ENGINEERING",
            "ELECTRONICS AND COMMUNICATION ENGINEERING",
            "MECHANICAL ENGINEERING",
            "MECHANICAL ENGINEERING (TAMIL MEDIUM)",
            "COMPUTER SCIENCE AND ENGINEERING",
            "MECHANICAL ENGINEERING",
            "BIO TECHNOLOGY",
            "COMPUTER SCIENCE AND ENGINEERING",
            "COMPUTER SCIENCE AND ENGINEERING",
            "COMPUTER SCIENCE AND ENGINEERING",
            "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
            "COMPUTER SCIENCE AND ENGINEERING",
            "COMPUTER SCIENCE AND ENGINEERING",
            "MECHANICAL ENGINEERING"
        ]
    }
    
    # Add community cutoff marks
    data['OC'] = [None, None, 193.5, None, 191.5, None, 172, None, 185, None, 197.5, 157.5, 171.5, None, None, None, None, None, 118.5, 137.5, 132, None, None, None, 136, None, None, 139.5, 146.5, None, None, None, None, None]
    data['BC'] = [None, None, 185.5, 187.5, None, None, None, None, None, None, None, None, None, 190.5, 180, 140, None, None, None, None, None, 146.5, None, None, None, None, None, None, None, None, None, 154.5, 129.5, None]
    data['BCM'] = [None, None, None, None, None, None, None, None, None, 144, None, None, None, None, None, None, 124.5, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    data['MBC'] = [None, 151.5, 183.5, None, None, None, None, None, 164.5, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 133, None, 131, None, None, None, None, 132.5, None, None, None]
    data['SC'] = [None, None, None, None, None, 151.5, None, 137, None, None, None, None, None, None, None, None, None, 148.5, None, None, None, None, 131.5, None, None, None, 130, None, None, 155, None, None, None, 144.5]
    data['SCA'] = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    data['ST'] = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv('college_data.csv', index=False)
    return df

def calculate_cutoff(maths, physics, chemistry):
    """Calculate cutoff mark based on TN Engineering formula"""
    return maths + (physics/2) + (chemistry/2)

def get_user_marks():
    """Get marks input from user with validation"""
    while True:
        try:
            print("\nEnter your marks (0-100):")
            maths = float(input("Mathematics: "))
            physics = float(input("Physics: "))
            chemistry = float(input("Chemistry: "))
            
            if not all(0 <= mark <= 100 for mark in [maths, physics, chemistry]):
                print("Error: Marks should be between 0 and 100")
                continue
            
            return maths, physics, chemistry
        except ValueError:
            print("Error: Please enter valid numbers")

def get_filter_mode():
    """Get the filtering mode from user"""
    print("\nChoose filtering mode:")
    print("1. Show top 10 colleges across all branches")
    print("2. Show top 10 colleges for specific branch")
    print("3. Show all branches for specific college")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if 1 <= choice <= 3:
                return choice
            print("Please enter a number between 1 and 3")
        except ValueError:
            print("Please enter a valid number")

def get_branch_choice(df):
    """Get branch preference from user"""
    unique_branches = sorted(df['BRANCH NAME'].unique())
    print("\nAvailable Branches:")
    for idx, branch in enumerate(unique_branches, 1):
        print(f"{idx}. {branch}")
    
    while True:
        try:
            choice = int(input("\nEnter branch number: ")) - 1
            if 0 <= choice < len(unique_branches):
                return unique_branches[choice]
            print("Please enter a valid branch number")
        except ValueError:
            print("Please enter a valid number")

def get_college_choice(df):
    """Get college preference from user"""
    unique_colleges = sorted(df['COLLEGE NAME'].unique())
    print("\nAvailable Colleges:")
    for idx, college in enumerate(unique_colleges, 1):
        print(f"{idx}. {college}")
    
    while True:
        try:
            choice = int(input("\nEnter college number: ")) - 1
            if 0 <= choice < len(unique_colleges):
                return unique_colleges[choice]
            print("Please enter a valid college number")
        except ValueError:
            print("Please enter a valid number")

def predict_all_colleges(df, cutoff_mark):
    """Mode 1: Predict top 10 colleges across all branches"""
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    all_predictions = []
    
    for community in communities:
        eligible = df[pd.notna(df[community]) & (df[community] <= cutoff_mark)].copy()
        if not eligible.empty:
            eligible['Community'] = community
            eligible['Cutoff'] = eligible[community]
            all_predictions.append(
                eligible[['COLLEGE NAME', 'BRANCH NAME', 'BRANCH CODE', 'Community', 'Cutoff']]
            )
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        return combined_predictions.nlargest(10, 'Cutoff')
    return pd.DataFrame()

def predict_branch_specific(df, cutoff_mark, branch_name):
    """Mode 2: Predict top 10 colleges for specific branch"""
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    all_predictions = []
    
    branch_df = df[df['BRANCH NAME'] == branch_name]
    for community in communities:
        eligible = branch_df[pd.notna(branch_df[community]) & (branch_df[community] <= cutoff_mark)].copy()
        if not eligible.empty:
            eligible['Community'] = community
            eligible['Cutoff'] = eligible[community]
            all_predictions.append(
                eligible[['COLLEGE NAME', 'BRANCH NAME', 'BRANCH CODE', 'Community', 'Cutoff']]
            )
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        return combined_predictions.nlargest(10, 'Cutoff')
    return pd.DataFrame()

def show_college_branches(df, college_name):
    """Mode 3: Show all branches for specific college"""
    college_df = df[df['COLLEGE NAME'] == college_name]
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    
    print(f"\nBranches available at {college_name}:")
    print("=" * 100)
    
    for _, row in college_df.iterrows():
        print(f"\nBranch: {row['BRANCH NAME']} ({row['BRANCH CODE']})")
        print("Cutoff marks by community:")
        for community in communities:
            if pd.notna(row[community]):
                print(f"- {community}: {row[community]:.2f}")
        print("-" * 50)

def main():
    try:
        print("Preparing college data...")
        df = create_college_data()
        
        # Get filter mode
        mode = get_filter_mode()
        
        if mode in [1, 2]:
            # Get marks and calculate cutoff
            maths, physics, chemistry = get_user_marks()
            cutoff_mark = calculate_cutoff(maths, physics, chemistry)
            print(f"\nYour calculated cutoff mark is: {cutoff_mark:.2f}")
            print(f"(Formula: Maths({maths}) + Physics({physics})/2 + Chemistry({chemistry})/2)")
            
            if mode == 1:
                # Mode 1: All colleges
                print("\nFinding top 10 eligible colleges across all branches...")
                predictions = predict_all_colleges(df, cutoff_mark)
            else:
                # Mode 2: Branch specific
                branch_name = get_branch_choice(df)
                print(f"\nFinding top 10 eligible colleges for {branch_name}...")
                predictions = predict_branch_specific(df, cutoff_mark, branch_name)
            
            if not predictions.empty:
                print("\nTop 10 Predicted Colleges:")
                print("=" * 100)
                for idx, row in predictions.iterrows():
                    print(f"\nRank {idx + 1}")
                    print(f"College: {row['COLLEGE NAME']}")
                    print(f"Branch: {row['BRANCH NAME']} ({row['BRANCH CODE']})")
                    print(f"Community: {row['Community']}")
                    print(f"Cutoff Mark: {row['Cutoff']:.2f}")
                    print("-" * 50)
            else:
                print("\nNo colleges found matching your criteria.")
                
        else:
            # Mode 3: College specific
            college_name = get_college_choice(df)
            show_college_branches(df, college_name)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()