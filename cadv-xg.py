import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CollegePredictorML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.is_trained = False
        self.feature_names = ['college_name_encoded', 'branch_name_encoded']

    def prepare_historical_data(self, df):
        """Prepare historical data for training with encoding and scaling"""
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

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using GridSearchCV"""
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4]
        }

        grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def evaluate_model(self, X, y, model):
        """Evaluate model using multiple metrics and cross-validation"""
        cv_rmse = np.sqrt(-cross_val_score(model, X, y, 
                                         scoring='neg_mean_squared_error', 
                                         cv=5))
        cv_mae = -cross_val_score(model, X, y, 
                                scoring='neg_mean_absolute_error', 
                                cv=5)
        cv_r2 = cross_val_score(model, X, y, 
                               scoring='r2', 
                               cv=5)

        return {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }

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

            self.model = self.hyperparameter_tuning(X_train_scaled, y_train)
            self.model.fit(X_train_scaled, y_train)

            print("\nEvaluating model performance...")
            evaluation_metrics = self.evaluate_model(X_train_scaled, y_train, self.model)

            y_pred = self.model.predict(X_test_scaled)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)

            self.is_trained = True

            return {
                'cross_validation_metrics': evaluation_metrics,
                'test_metrics': {
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
            }

        except Exception as e:
            print(f"Error during model training: {str(e)}")
            return None

    def predict_cutoff(self, college_name, branch_name):
        """Make predictions for a given college and branch"""
        if not self.is_trained:
            return None

        try:
            college_encoded = self.label_encoders['COLLEGE NAME'].transform([college_name])[0]
            branch_encoded = self.label_encoders['BRANCH NAME'].transform([branch_name])[0]

            X = pd.DataFrame(
                [[college_encoded, branch_encoded]],
                columns=self.feature_names
            )

            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names
            )

            return self.model.predict(X_scaled)[0]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

# Additional functions remain unchanged...

def main():
    print("Welcome to the Enhanced College Predictor Program (XGBoost Edition)!")

    # Load the dataset
    try:
        file_path = "Unique_Colleges_Max_Cutoff.csv"
        if not os.path.exists(file_path):
            print(f"Error: Dataset file '{file_path}' not found!")
            return
        
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded dataset with {len(df)} records.")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
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

def calculate_admission_chance(margin):
    """Calculate admission chance based on margin"""
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

def print_model_performance(metrics):
    """Print detailed model performance metrics"""
    if metrics is None:
        print("No metrics available.")
        return
    
    cv_metrics = metrics['cross_validation_metrics']
    test_metrics = metrics['test_metrics']
    
    print("\nCross-Validation Metrics:")
    print(f"RMSE: {cv_metrics['cv_rmse_mean']:.2f} (±{cv_metrics['cv_rmse_std']:.2f})")
    print(f"MAE: {cv_metrics['cv_mae_mean']:.2f} (±{cv_metrics['cv_mae_std']:.2f})")
    print(f"R²: {cv_metrics['cv_r2_mean']:.2f} (±{cv_metrics['cv_r2_std']:.2f})")
    
    print("\nTest Set Metrics:")
    print(f"RMSE: {test_metrics['test_rmse']:.2f}")
    print(f"MAE: {test_metrics['test_mae']:.2f}")
    print(f"R²: {test_metrics['test_r2']:.2f}")

def main():
    print("Welcome to the Enhanced College Predictor Program (XGBoost Edition)!")
    print("=" * 60)

    # Load the dataset
    try:
        file_path = "Unique_Colleges_Max_Cutoff.csv"
        if not os.path.exists(file_path):
            print(f"Error: Dataset file '{file_path}' not found!")
            return
        
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded dataset with {len(df)} records.")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Initialize and train model
    print("\nInitializing and training XGBoost model...")
    predictor = CollegePredictorML()
    metrics = predictor.train_model(df)
    
    if metrics:
        print("\nModel training completed successfully!")
        print_model_performance(metrics)
    else:
        print("Failed to train model. Exiting...")
        return

    # Main program loop
    while True:
        print("\nEnter your marks (0-100):")
        try:
            # Get user marks
            maths = float(input("Mathematics: "))
            physics = float(input("Physics: "))
            chemistry = float(input("Chemistry: "))

            # Validate marks
            if not all(0 <= x <= 100 for x in [maths, physics, chemistry]):
                print("Error: All marks should be between 0 and 100")
                continue

            # Calculate cutoff
            cutoff_mark = maths + (physics / 2) + (chemistry / 2)
            print(f"\nYour calculated cutoff mark is: {cutoff_mark:.2f}")

            # Inner loop for filtering options
            while True:
                mode = get_filter_mode()

                if mode == 1:
                    # Show top 10 colleges across all branches
                    predictions = predict_all_colleges(predictor, df, cutoff_mark)
                    display_predictions(predictions)
                
                elif mode == 2:
                    # Show top 10 colleges for specific branch
                    branch_name = get_branch_choice(df)
                    if branch_name:
                        predictions = predict_branch_specific(predictor, df, cutoff_mark, branch_name)
                        display_predictions(predictions)
                
                elif mode == 3:
                    # Show all branches for specific college
                    college_name = get_college_choice(df)
                    if college_name:
                        show_college_branches(predictor, df, college_name, cutoff_mark)
                
                elif mode == 4:
                    print("\nThank you for using the College Predictor Program!")
                    return

                # Ask if user wants to try another filter mode
                choice = input("\nWould you like to try another filter mode? (y/n): ").lower()
                if choice != 'y':
                    break

            # Ask if user wants to enter new marks
            choice = input("\nWould you like to enter new marks? (y/n): ").lower()
            if choice != 'y':
                print("\nThank you for using the College Predictor Program!")
                break

        except ValueError:
            print("Error: Please enter valid numbers")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()