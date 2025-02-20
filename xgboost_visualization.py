import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import matplotlib.pyplot as plt
from xgboost import plot_tree
from dtreeviz.trees import DTreeVizAPI

# Import your predictor class
from cadv_new import EnhancedCollegePredictorML  # Make sure this file exists

class XGBoostVisualizer:
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def plot_feature_importance(self):
        """Interactive Feature Importance using Plotly"""
        importance = self.model.feature_importances_
        features = self.X_train.columns

        fig = px.bar(
            x=importance, 
            y=features, 
            orientation='h', 
            title="Feature Importance",
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='viridis'  # More vibrant color scheme
        )
        fig.show()

    def plot_decision_tree(self, save_path="xgboost_tree.svg"):
        """Beautiful decision tree visualization with dtreeviz"""
        viz = dtreeviz(
            self.model, 
            self.X_train, 
            self.y_train, 
            target_name="Max Cutoff",
            feature_names=self.X_train.columns.to_list()
        )
        viz.save(save_path)
        viz.view()  # Opens interactive visualization

# Step 1: Load Dataset
df = pd.read_csv("Unique_Colleges_Max_Cutoff.csv")  # Ensure the file exists

# Step 2: Train Model
predictor = EnhancedCollegePredictorML()
metrics = predictor.train_model(df)

# Extract training data for visualization
processed_df = predictor.preprocess_data(df)
X = processed_df[['COLLEGE_CODE', 'BRANCH_CODE']]
y = processed_df['MAX CUTOFF']

# Step 3: Interactive Visualizations
visualizer = XGBoostVisualizer(predictor.model, X, y)
visualizer.plot_feature_importance()
visualizer.plot_decision_tree()
