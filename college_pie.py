import pandas as pd
import numpy as np
import os

def load_college_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.fillna(value=np.nan, inplace=True)
        return df
    else:
        print(f"Error: File '{file_path}' not found.")
        return None

def calculate_cutoff(maths, physics, chemistry):
    return (maths) + (physics / 2) + (chemistry / 2)

def get_user_marks():
    while True:
        try:
            print("\nEnter your marks (0-100 for Mathematics, 0-100 for Physics and Chemistry):")
            maths = float(input("Mathematics: "))
            physics = float(input("Physics: "))
            chemistry = float(input("Chemistry: "))

            if not (0 <= maths <= 100 and 0 <= physics <= 100 and 0 <= chemistry <= 100):
                print("Error: Invalid marks range. Mathematics should be 0-100, Physics and Chemistry should be 0-100")
                continue

            return maths, physics, chemistry
        except ValueError:
            print("Error: Please enter valid numbers")

def get_filter_mode():
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
            print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")

def get_branch_choice(df):
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
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    all_predictions = []

    for community in communities:
        community_name = community.strip()  # Remove leading and trailing spaces
        if community_name in df.columns:
            eligible = df[pd.notna(df[community_name]) & (df[community_name] >= cutoff_mark)].copy()
            if not eligible.empty:
                eligible['Community'] = community_name
                eligible['Cutoff'] = eligible[community_name]
                eligible['Margin'] = eligible[community_name] - cutoff_mark
                all_predictions.append(
                    eligible[['COLLEGE NAME', 'BRANCH NAME', 'BRANCH CODE', 'Community', 'Cutoff', 'Margin']]
                )

    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        sorted_predictions = combined_predictions.sort_values(['Cutoff', 'Margin'], 
                                                           ascending=[True, True])
        return sorted_predictions.head(10)
    return pd.DataFrame()

def predict_branch_specific(df, cutoff_mark, branch_name):
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
    college_df = df[df['COLLEGE NAME'] == college_name]
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']

    print(f"\nBranches available at {college_name}")
    print("=" * 100)

    college_df['max_cutoff'] = college_df[communities].max(axis=1)
    college_df = college_df.sort_values('max_cutoff', ascending=False)

    overall_max_cutoff = college_df['max_cutoff'].max()

    for idx, (_, row) in enumerate(college_df.iterrows(), 1):
        print(f"\nBranch {idx}:")
        print(f"Name: {row['BRANCH NAME']}")
        print(f"Code: {row['BRANCH CODE']}")
        print("\nCutoff marks by community:")

        for community in communities:
            if pd.notna(row[community]):
                stars = int((row[community] / overall_max_cutoff) * 20)
                print(f"{community:4} : {row[community]:6.2f} {'★' * stars}")

        if pd.notna(row['max_cutoff']):
            print(f"\nHighest cutoff: {row['max_cutoff']:.2f}")
        print("-" * 50)

def get_college_choice(df):
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
    print("Welcome to the College Predictor Program!")
    print("=" * 50)

    file_path = "Unique_Colleges_Max_Cutoff.csv"
    df = load_college_data(file_path)

    if df is None:
        return

    while True:
        maths, physics, chemistry = get_user_marks()
        cutoff_mark = calculate_cutoff (maths, physics, chemistry)
        print(f"\nYour calculated cutoff mark is: {cutoff_mark:.2f}")

        while True:
            mode = get_filter_mode()

            if mode == 1:
                predictions = predict_all_colleges(df, cutoff_mark)
                display_predictions(predictions)

            elif mode == 2:
                branch_name = get_branch_choice(df)
                if branch_name:
                    predictions = predict_branch_specific(df, cutoff_mark, branch_name)
                    display_predictions(predictions)

            elif mode == 3:
                college_name = get_college_choice(df)
                if college_name:
                    show_college_branches(df, college_name)

            elif mode == 4:
                print("\nThank you for using the College Predictor Program!")
                return

            choice = input("\nWould you like to try another filter mode? (y/n): ").lower()
            if choice != 'y':
                break

        choice = input("\nWould you like to enter new marks? (y/n): ").lower()
        if choice != 'y':
            print("\nThank you for using the College Predictor Program!")
            break

if __name__ == "__main__":
    main()