import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import time

class EnhancedCollegePredictorML:
    def __init__(self):
        self.model = None
        self.college_encoder = LabelEncoder()
        self.branch_encoder = LabelEncoder()
        self.feature_importance = None
        self.metrics = {}
        
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        processed_df = df.copy()
        
        # Encode categorical variables
        processed_df['COLLEGE_CODE'] = self.college_encoder.fit_transform(processed_df['COLLEGE NAME'])
        processed_df['BRANCH_CODE'] = self.branch_encoder.fit_transform(processed_df['BRANCH NAME'])
        
        return processed_df
    
    def train_model(self, df):
        """Train the XGBoost model and calculate metrics"""
        start_time = time.time()
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Prepare features and target
        X = processed_df[['COLLEGE_CODE', 'BRANCH_CODE']]
        y = processed_df['MAX CUTOFF']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define model parameters
        model_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**model_params)
        self.model.fit(X_train, y_train)
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Make predictions
        pred_start_time = time.time()
        y_pred = self.model.predict(X_test)
        pred_time = (time.time() - pred_start_time) * 1000  # Convert to milliseconds
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': ['College', 'Branch'],
            'importance': self.model.feature_importances_
        })
        
        # Store metrics
        self.metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'train_time': train_time,
            'pred_time': pred_time,
            'model_params': model_params,
            'feature_importance': feature_importance
        }
        
        return self.metrics
    
    def predict_cutoff(self, college_name, branch_name):
        """Predict cutoff for a given college and branch"""
        if self.model is None:
            return None
            
        try:
            college_code = self.college_encoder.transform([college_name])[0]
            branch_code = self.branch_encoder.transform([branch_name])[0]
            
            prediction = self.model.predict([[college_code, branch_code]])[0]
            return float(prediction)
        except:
            return None