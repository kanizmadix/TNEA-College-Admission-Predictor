import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class CollegePredictorML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = ['college_name_encoded', 'branch_name_encoded']  # Update feature names

    def prepare_historical_data(self, df):
        """Prepare historical data for training"""
        print("\nAvailable columns in dataset:", df.columns.tolist())
        
        categorical_columns = ['COLLEGE NAME', 'BRANCH NAME']
        
        encoded_features = pd.DataFrame()
        for column in categorical_columns:
            if column not in df.columns:
                raise KeyError(f"Required column '{column}' not found in dataset")
            self.label_encoders[column] = LabelEncoder()
            encoded_features[f'{column.lower().replace(" ", "_")}_encoded'] = \
                self.label_encoders[column].fit_transform(df[column])
        
        if 'MAX CUTOFF' not in df.columns:
            raise ValueError("MAX CUTOFF column not found in dataset")
            
        return encoded_features, df['MAX CUTOFF']

    def train_model(self, df):
        try:
            X, y = self.prepare_historical_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=self.feature_names
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=self.feature_names
            )
            
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            return mse, r2
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return None, None

    def predict_cutoff(self, college_name, branch_name):
        if not self.is_trained:
            return None
        
        try:
            college_encoded = self.label_encoders['COLLEGE NAME'].transform([college_name])[0]
            branch_encoded = self.label_encoders['BRANCH NAME'].transform([branch_name])[0]
            
            X = pd.DataFrame(
                [[college_encoded, branch_encoded]],
                columns=self.feature_names  # Use feature names from training
            )
            
            # Add assertion to check feature names
            assert set(X.columns.tolist()) == set(self.model.feature_names_in_), "Feature names mismatch"
            
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names
            )
            
            return self.model.predict(X_scaled)[0]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

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
            print("Please enter a number between 1-4")
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

        if choice == 'n' and current_page < total_pages:
            current_page += 1
        elif choice == 'p' and current_page > 1:
            current_page -= 1
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

        if choice == 'n' and current_page < total_pages:
            current_page += 1
        elif choice == 'p' and current_page > 1:
            current_page -= 1
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

def predict_all_colleges(predictor, df, cutoff_mark):
    predictions = []
    
    for _, row in df.iterrows():
        predicted_cutoff = predictor.predict_cutoff(
            row['COLLEGE NAME'],
            row['BRANCH NAME']
        )
        
        if predicted_cutoff is not None:
            margin = cutoff_mark - predicted_cutoff
            chance = calculate_admission_chance(margin)
            
            predictions.append({
                'COLLEGE NAME': row['COLLEGE NAME'],
                'BRANCH NAME': row['BRANCH NAME'],
                'Predicted Cutoff': predicted_cutoff,
                'Your Cutoff': cutoff_mark,
                'Margin': margin,
                'Admission Chance': chance
            })
    
    return pd.DataFrame(predictions)

def predict_branch_specific(predictor, df, cutoff_mark, branch_name):
    branch_df = df[df['BRANCH NAME'] == branch_name]
    predictions = []
    
    for _, row in branch_df.iterrows():
        predicted_cutoff = predictor.predict_cutoff(
            row['COLLEGE NAME'],
            row['BRANCH NAME']
        )
        
        if predicted_cutoff is not None:
            margin = cutoff_mark - predicted_cutoff
            chance = calculate_admission_chance(margin)
            
            predictions.append({
                'COLLEGE NAME': row['COLLEGE NAME'],
                'BRANCH NAME': row['BRANCH NAME'],
                'Predicted Cutoff': predicted_cutoff,
                'Your Cutoff': cutoff_mark,
                'Margin': margin,
                'Admission Chance': chance
            })
    
    return pd.DataFrame(predictions)

def show_college_branches(predictor, df, college_name, user_cutoff):
    college_df = df[df['COLLEGE NAME'] == college_name]
    
    print(f"\nBranches available at {college_name}")
    print("=" * 100)
    
    predictions = []
    for _, row in college_df.iterrows():
        predicted_cutoff = predictor.predict_cutoff(
            row['COLLEGE NAME'],
            row['BRANCH NAME']
        )
        
        if predicted_cutoff is not None:
            margin = user_cutoff - predicted_cutoff
            chance = calculate_admission_chance(margin)
            
            predictions.append({
                'BRANCH NAME': row['BRANCH NAME'],
                'Predicted Cutoff': predicted_cutoff,
                'Your Cutoff': user_cutoff,
                'Margin': margin,
                'Admission Chance': chance
            })
    
    predictions_df = pd.DataFrame(predictions)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values('Predicted Cutoff', ascending=False)
        
        for idx, (_, row) in enumerate(predictions_df.iterrows(), 1):
            print(f"\nBranch {idx}:")
            print(f"Name: {row['BRANCH NAME']}")
            print(f"Predicted Cutoff: {row['Predicted Cutoff']:.2f}")
            print(f"Your Cutoff: {row['Your Cutoff']:.2f}")
            print(f"Margin: {row['Margin']:.2f}")
            print_admission_chance(row['Admission Chance'])
            print("-" * 50)

def calculate_admission_chance(margin):
    if margin >= 0:
        return min((margin + 5) / 10, 1) * 100
    else:
        return max(0, (1 + margin / 20) * 100)

def print_admission_chance(chance):
    if chance >= 80:
        chance_str = f"\033[92mVery High ({chance:.1f}%)\033[0m"
    elif chance >= 60:
        chance_str = f"\033[93mGood ({chance:.1f}%)\033[0m"
    elif chance >= 40:
        chance_str = f"\033[93mModerate ({chance:.1f}%)\033[0m"
    else:
        chance_str = f"\033[91mLow ({chance:.1f}%)\033[0m"
    print(f"Admission Chance: {chance_str}")

def display_predictions(predictions):
    if predictions.empty:
        print("\nNo predictions available.")
        return

    predictions_sorted = predictions.sort_values('Admission Chance', ascending=False).head(10)
    
    print("\nTop 10 College Recommendations:")
    print("=" * 100)
    
    for idx, (_, row) in enumerate(predictions_sorted.iterrows(), 1):
        print(f"\nRank {idx}:")
        print(f"College: {row['COLLEGE NAME']}")
        print(f"Branch: {row['BRANCH NAME']}")
        print(f"Predicted Cutoff: {row['Predicted Cutoff']:.2f}")
        print(f"Your Cutoff: {row['Your Cutoff']:.2f}")
        print(f"Margin: {row['Margin']:.2f}")
        print_admission_chance(row['Admission Chance'])
        print("-" * 50)

def main():
    print("Welcome to the Enhanced College Predictor Program!")
    print("=" * 50)

    file_path = "Unique_Colleges_Max_Cutoff.csv"
    df = pd.read_csv(file_path)
    if df is None:
        return

    print("\nInitializing ML model...")
    predictor = CollegePredictorML()
    mse, r2 = predictor.train_model(df)
    
    if mse is not None and r2 is not None:
        print(f"Model trained successfully!")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
    else:
        print("Failed to train model. Exiting...")
        return

    while True:
        print("\nEnter your marks (0-100):")
        try:
            maths = float(input("Mathematics: "))
            physics = float(input("Physics: "))
            chemistry = float(input("Chemistry: "))

            if not all(0 <= x <= 100 for x in [maths, physics, chemistry]):
                print("Error: All marks should be between 0 and 100")
                continue

            cutoff_mark = maths + (physics / 2) + (chemistry / 2)
            print(f"\nYour calculated cutoff mark is: {cutoff_mark:.2f}")

            while True:
                mode = get_filter_mode()

                if mode == 1:
                    predictions = predict_all_colleges(predictor, df, cutoff_mark)
                    display_predictions(predictions)
                elif mode == 2:
                    branch_name = get_branch_choice(df)
                    if branch_name:
                        predictions = predict_branch_specific(predictor, df, cutoff_mark, branch_name)
                        display_predictions(predictions)
                elif mode == 3:
                    college_name = get_college_choice(df)
                    if college_name:
                        show_college_branches(predictor, df, college_name, cutoff_mark)
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

        except ValueError:
            print("Error: Please enter valid numbers")

if __name__ == "__main__":
    main()