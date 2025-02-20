import pandas as pd
import numpy as np
import os

def load_college_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
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
    unique_branches = sorted(df['RANCH NAME'].unique())
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
                choice_num =int(choice)
                if 1 <= choice_num <= len(unique_colleges):
                    return unique_colleges[choice_num - 1]
                print("Please enter a valid college number")
            except ValueError:
                print("Please enter a valid choice")

def predict_all_colleges(df, cutoff_mark):
    all_predictions = []

    eligible = df[pd.notna(df['OC']) & (df['OC'] >= cutoff_mark)].copy()
    if not eligible.empty:
        eligible['Margin'] = eligible['OC'] - cutoff_mark
        all_predictions.append(
            eligible[['COLLEGE NAME', 'RANCH NAME', 'OC', 'Margin']]
        )

    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        sorted_predictions = combined_predictions.sort_values(['OC', 'Margin'], 
                                                           ascending=[True, True])
        return sorted_predictions.head(10)
    return pd.DataFrame()

def predict_branch_specific(df, cutoff_mark, branch_name):
    all_predictions = []

    branch_df = df[df['RANCH NAME'] == branch_name]
    eligible = branch_df[pd.notna(branch_df['OC']) & (branch_df['OC'] >= cutoff_mark)].copy()
    if not eligible.empty:
        eligible['Margin'] = eligible['OC'] - cutoff_mark
        all_predictions.append(
            eligible[['COLLEGE NAME', 'RANCH NAME', 'OC', 'Margin']]
        )

    if all_predictions:
        combined_predictions = pd.concat(all_predictions)
        sorted_predictions = combined_predictions.sort_values(['OC', 'Margin'], 
                                                           ascending=[True, True])
        return sorted_predictions.head(10)
    return pd.DataFrame()

def show_college_branches(df, college_name):
    college_df = df[df['COLLEGE NAME'] == college_name]

    print(f"\nBranches available at {college_name}")
    print("=" * 100)

    for idx, (_, row) in enumerate(college_df.iterrows(), 1):
        print(f"\nBranch {idx}:")
        print(f"Name: {row['RANCH NAME']}")
        print(f"Code: {row['BRANCH CODE']}")
        print(f"Max Cutoff: {row['OC']:.2f}")
        print("-" * 50)

def display_predictions(predictions):
    if predictions.empty:
        print("\nNo colleges found matching your criteria.")
        return

    print("\nTop Recommendations:")
    print("=" * 100)

    for idx, (_, row) in enumerate(predictions.iterrows(), 1):
        print(f"\nRank {idx}:")
        print(f"College: {row['COLLEGE NAME']}")
        print(f"Branch: {row['RANCH NAME']}")
        print(f"Max Cutoff: {row['OC']:.2f}")
        print(f"Your Margin: {row['Margin']:.2f}")
        print("-" * 50)

def main():
    print("Welcome to the College Predictor Program!")
    print("=" * 50)

    file_path = "C:\\Users\\Kaniz\\Pictures\\TNEA_fct\\Vocational_2023_Mark_Cutoff.csv"
    df = load_college_data(file_path)

    if df is None:
        return

    while True:
        maths, physics, chemistry = get_user_marks()
        cutoff_mark = calculate_cutoff(maths, physics, chemistry)
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