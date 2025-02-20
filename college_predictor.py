import pandas as pd
import numpy as np
import os

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
    return (maths) + (physics/2) + (chemistry/2)

def get_user_marks():
    """Get marks input from user with validation"""
    while True:
        try:
            print("\nEnter your marks (0-100 for Mathematics, 0-100 for Physics and Chemistry):")
            maths = float(input("Mathematics: "))
            physics = float(input("Physics: "))
            chemistry = float(input("Chemistry: "))
            
            if not (0 <= maths <= 100 and 0 <= physics <= 100 and 0 <= chemistry <= 100):
                print("Error: Invalid marks range. Mathematics should be 0-200, Physics and Chemistry should be 0-100")
                continue
            
            return maths, physics, chemistry
        except ValueError:
            print("Error: Please enter valid numbers")

def get_filter_mode():
    """Get the filtering mode from user"""
    while True:
        print("\nChoose filtering mode:")
        print("1. Show top 10 colleges across all branches")
        print("2. Show top 10 colleges for specific branch")
        print("3. Show all branches for specific college")
        print("4. Exit program")
        
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if 1 <= choice <= 4:
                return choice
            print("Please enter a number between 1 and  4")
        except ValueError:
            print("Please enter a valid number")

def get_branch_choice(df):
    """Get branch preference from user with pagination"""
    unique_branches = sorted(df['BRANCH NAME'].unique())
    branches_per_page = 20
    total_pages = (len(unique_branches) + branches_per_page - 1) // branches_per_page
    current_page = 1

    while True:
        start_idx = (current_page - 1) * branches_per_page
        end_idx = start_idx + branches_per_page
        current_branches = unique_branches[start_idx:end_idx]

        print(f"\nAvailable Branches (Page {current_page} of {total_pages}):")
        print("=" * 100)
        for idx, branch in enumerate(current_branches, start_idx + 1):
            print(f"{idx}. {branch}")
        
        print("\nNavigation options:")
        print("n - Next page")
        print("p - Previous page")
        print("q - Quit")
        print("Or enter a branch number to select")
        
        choice = input("\nEnter your choice: ").lower()
        
        if choice == 'n':
            if current_page < total_pages:
                current_page += 1
            else:
                print("Already on last page")
        elif choice == 'p':
            if current_page > 1:
                current_page -= 1
            else:
                print("Already on first page")
        elif choice == 'q':
            return None
        else:
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(unique_branches):
                    return unique_branches[choice_num - 1]
                print("Please enter a valid branch number")
            except ValueError:
                print("Please enter a valid choice")

def predict_all_colleges(df, cutoff_mark):
    """Mode 1: Predict top 10 colleges across all branches"""
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    all_predictions = []
    
    for community in communities:
        eligible = df[pd.notna(df[community]) & (df[community] <= cutoff_mark)].copy()
        if not eligible.empty:
            eligible['Community'] = community
            eligible['Cutoff'] = eligible[community]
            eligible['Margin'] = cutoff_mark - eligible[community]
            all_predictions.append(
                eligible[['COLLEGE NAME', 'BRANCH NAME', 'BRANCH CODE', 'Community', 'Cutoff', 'Margin']]
            )
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        sorted_predictions = combined_predictions.sort_values(['Cutoff', 'Margin'], 
                                                           ascending=[False, True])
        return sorted_predictions.head(10)
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
            eligible['Margin'] = cutoff_mark - eligible[community]
            all_predictions.append(
                eligible[['COLLEGE NAME', 'BRANCH NAME', 'BRANCH CODE', 'Community', 'Cutoff', 'Margin']]
            )
    
    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        sorted_predictions = combined_predictions.sort_values(['Cutoff', 'Margin'], 
                                                           ascending=[False, True])
        return sorted_predictions.head(10)
    return pd.DataFrame()

def show_college_branches(df, college_name):
    """Mode 3: Show all branches for specific college with enhanced visualization"""
    college_df = df[df['COLLEGE NAME'] == college_name]
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    
    print(f"\nBranches available at {college_name}")
    print("=" * 100)
    
    # Calculate maximum cutoff for each branch and sort
    college_df['max_cutoff'] = college_df[communities].max(axis=1)
    college_df = college_df.sort_values('max_cutoff', ascending=False)
    
    # Get the overall maximum cutoff for scaling
    overall_max_cutoff = college_df['max_cutoff'].max()
    
    for idx, (_, row) in enumerate(college_df.iterrows , 1):
        print(f"\nBranch {idx}:")
        print(f"Name: {row['BRANCH NAME']}")
        print(f"Code: {row['BRANCH CODE']}")
        print("\nCutoff marks by community:")
        
        # Display cutoffs with visual representation
        for community in communities:
            if pd.notna(row[community]):
                # Calculate stars based on relative cutoff value
                stars = int((row[community] / overall_max_cutoff) * 20)
                print(f"{community:4} : {row[community]:6.2f} {'★' * stars}")
        
        # Show highest cutoff if available
        if pd.notna(row['max_cutoff']):
            print(f"\nHighest cutoff: {row['max_cutoff']:.2f}")
        print("-" * 50)

def get_college_choice(df):
    """Get college preference from user with pagination"""
    unique_colleges = sorted(df['COLLEGE NAME'].unique())
    colleges_per_page = 10
    total_pages = (len(unique_colleges) + colleges_per_page - 1) // colleges_per_page
    current_page = 1

    while True:
        start_idx = (current_page - 1) * colleges_per_page
        end_idx = start_idx + colleges_per_page
        current_colleges = unique_colleges[start_idx:end_idx]

        print(f"\nAvailable Colleges (Page {current_page} of {total_pages}):")
        print("=" * 100)
        for idx, college in enumerate(current_colleges, start_idx + 1):
            print(f"{idx}. {college}")
        
        print("\nNavigation options:")
        print("n - Next page")
        print("p - Previous page")
        print("q - Quit")
        print("Or enter a college number to select")
        
        choice = input("\nEnter your choice: ").lower()
        
        if choice == 'n':
            if current_page < total_pages:
                current_page += 1
            else:
                print("Already on last page")
        elif choice == 'p':
            if current_page > 1:
                current_page -= 1
            else:
                print("Already on first page")
        elif choice == 'q':
            return None
        else:
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(unique_colleges):
                    return unique_colleges[choice_num - 1]
                print("Please enter a valid college number")
            except ValueError:
                print("Please enter a valid choice")

def display_predictions(predictions):
    """Display prediction results in a formatted table"""
    if predictions.empty:
        print("\nNo colleges found matching your criteria.")
        return
    
    print("\nTop Recommendations:")
    print("=" * 100)
    
    for idx, (_, row) in enumerate(predictions.iterrows(), 1):
        print(f"\nRank {idx}:")
        print(f"College: {row['COLLEGE NAME']}")
        print(f"Branch: {row['BRANCH NAME']} ({row['BRANCH CODE']})")
        print(f"Community: {row['Community']}")
        print(f"Cutoff Mark: {row['Cutoff']:.2f}")
        print(f"Your Margin: {row['Margin']:.2f}")
        print("-" * 50)

def main():
    """Main program loop"""
    print("Welcome to the College Predictor Program!")
    print("=" * 50)
    
    # Create or load college data
    if os.path.exists('college_data.csv'):
        df = pd.read_csv('college_data.csv')
    else:
        df = create_college_data()
    
    while True:
        # Get user marks
        maths, physics, chemistry = get_user_marks()
        cutoff_mark = calculate_cutoff(maths, physics, chemistry)
        print(f"\nYour calculated cutoff mark is: {cutoff_mark:.2f}")
        
        while True:
            mode = get_filter_mode()
            
            if mode == 1:
                # Show top 10 colleges across all branches
                predictions = predict_all_colleges(df, cutoff_mark)
                display_predictions(predictions)
            
            elif mode == 2:
                # Show top 10 colleges for specific branch
                branch_name = get_branch_choice(df)
                if branch_name:
                    predictions = predict_branch_specific(df, cutoff_mark, branch_name)
                    display_predictions(predictions)
            
            elif mode == 3:
                # Show all branches for specific college
                college_name = get_college_choice(df)
                if college_name:
                    show_college_branches(df, college_name)
            
            elif mode == 4:
                print("\nThank you for using the College Predictor Program!")
                return
            
            # ```python
            # Ask if user wants to try another filter mode
            choice = input("\nWould you like to try another filter mode? (y/n): ").lower()
            if choice != 'y':
                break
        
        # Ask if user wants to enter new marks
        choice = input("\nWould you like to enter new marks? (y/n): ").lower()
        if choice != 'y':
            print("\nThank you for using the College Predictor Program!")
            break

if __name__ == "__main__":
    main()