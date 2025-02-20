import streamlit as st
import pandas as pd
import numpy as np
from cadv_new import EnhancedCollegePredictorML
import plotly.express as px
import plotly.graph_objects as go
import time

# Set page configuration
st.set_page_config(
    page_title="College Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .excellent-chance {
        color: #0f5132;
        background-color: #d1e7dd;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .good-chance {
        color: #084298;
        background-color: #cfe2ff;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .fair-chance {
        color: #664d03;
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .low-chance {
        color: #842029;
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load and train the model"""
    try:
        df = pd.read_csv("Unique_Colleges_Max_Cutoff.csv")
        predictor = EnhancedCollegePredictorML()
        metrics = predictor.train_model(df)
        return predictor, df, metrics
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def display_model_metrics(metrics):
    """Display model metrics with visualizations"""
    st.subheader("Model Performance Metrics")
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "R² Score",
            f"{metrics['r2']:.4f}",
            help="Coefficient of determination (higher is better)"
        )
    with col2:
        st.metric(
            "RMSE",
            f"{metrics['rmse']:.4f}",
            help="Root Mean Square Error (lower is better)"
        )
    with col3:
        st.metric(
            "MAE",
            f"{metrics['mae']:.4f}",
            help="Mean Absolute Error (lower is better)"
        )
    
    # Visualizations
    with st.expander("View Model Analysis"):
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Feature Importance Plot
            fig_importance = px.bar(
                metrics['feature_importance'],
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with viz_col2:
            # Cross-validation Scores
            cv_df = pd.DataFrame({
                'Fold': range(1, len(metrics['cv_scores']) + 1),
                'Score': metrics['cv_scores']
            })
            fig_cv = px.bar(
                cv_df,
                x='Fold',
                y='Score',
                title="Cross-Validation Scores"
            )
            st.plotly_chart(fig_cv, use_container_width=True)
        
        # Model Parameters
        st.write("### Model Parameters")
        params_df = pd.DataFrame({
            'Parameter': metrics['model_params'].keys(),
            'Value': metrics['model_params'].values()
        })
        st.table(params_df)

def format_admission_chance(chance):
    """Format admission chance with appropriate styling"""
    if chance >= 80:
        return f'<div class="excellent-chance">Excellent Chance ({chance:.1f}%)<br><small>Very high probability of admission</small></div>'
    elif chance >= 60:
        return f'<div class="good-chance">Good Chance ({chance:.1f}%)<br><small>Strong possibility of admission</small></div>'
    elif chance >= 40:
        return f'<div class="fair-chance">Fair Chance ({chance:.1f}%)<br><small>Moderate possibility of admission</small></div>'
    else:
        return f'<div class="low-chance">Low Chance ({chance:.1f}%)<br><small>Consider alternative options</small></div>'

def calculate_admission_chance(margin):
    """Calculate admission chance based on margin"""
    if margin >= 0:
        return 95
    elif margin >= -10:
        return 80
    elif margin >= -20:
        return 60
    elif margin >= -30:
        return 40
    else:
        return 20

def predict_colleges(predictor, df, cutoff_mark, **filters):
    """Predict cutoffs with filters applied"""
    predictions = []
    progress_bar = st.progress(0)
    
    filtered_df = df.copy()
    for key, value in filters.items():
        if value:
            filtered_df = filtered_df[filtered_df[key] == value]
    
    total_rows = len(filtered_df)
    
    for idx, row in filtered_df.iterrows():
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
        
        progress = min(1.0, (idx + 1) / total_rows)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return pd.DataFrame(predictions)

def display_predictions(predictions):
    """Display predictions with visualizations"""
    if predictions.empty:
        st.warning("No predictions available.")
        return

    # Sort predictions by admission chance
    predictions_sorted = predictions.sort_values('Admission Chance', ascending=False)
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=predictions_sorted['Admission Chance'].head(10),
        y=[f"{col} - {br}" for col, br in zip(
            predictions_sorted['COLLEGE NAME'].head(10),
            predictions_sorted['BRANCH NAME'].head(10)
        )],
        orientation='h',
        marker=dict(
            color=predictions_sorted['Admission Chance'].head(10),
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title='Top 10 College Recommendations',
        xaxis_title='Admission Chance (%)',
        yaxis_title='College - Branch',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed predictions
    with st.expander("View Detailed Predictions"):
        for idx, row in predictions_sorted.iterrows():
            st.markdown(f"""
            **{idx + 1}. {row['COLLEGE NAME']} - {row['BRANCH NAME']}**
            * Predicted Cutoff: {row['Predicted Cutoff']:.2f}
            * Your Cutoff: {row['Your Cutoff']:.2f}
            * Margin: {row['Margin']:.2f}
            """)
            st.markdown(format_admission_chance(row['Admission Chance']), unsafe_allow_html=True)
            st.markdown("---")

def main():
    st.title("🎓 College Admission Predictor")
    st.write("Enter your marks and explore college recommendations based on XGBoost predictions")

    # Load model and display metrics
    predictor, df, metrics = load_model()
    
    if predictor is None or df is None:
        st.error("Failed to initialize the prediction model.")
        return
    
    display_model_metrics(metrics)
    
    # Input marks
    st.subheader("Enter Your Marks")
    with st.form("marks_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            maths = st.number_input("Mathematics", 0.0, 100.0, step=0.5)
        with col2:
            physics = st.number_input("Physics", 0.0, 100.0, step=0.5)
        with col3:
            chemistry = st.number_input("Chemistry", 0.0, 100.0, step=0.5)
            
        calculate = st.form_submit_button("Calculate Cutoff")
    
    if calculate:
        cutoff_mark = maths + physics/2 + chemistry/2
        st.info(f"Your calculated cutoff mark is: {cutoff_mark:.2f}")
        
        st.session_state.cutoff_mark = cutoff_mark
        st.session_state.marks_submitted = True
    
    # Prediction section
    if 'marks_submitted' in st.session_state:
        st.subheader("College Predictions")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_branch = st.selectbox(
                "Filter by Branch",
                ["All"] + sorted(df['BRANCH NAME'].unique().tolist())
            )
        with col2:
            min_chance = st.slider(
                "Minimum Admission Chance",
                0, 100, 30
            )
        
        if st.button("Get Predictions"):
            filters = {}
            if selected_branch != "All":
                filters['BRANCH NAME'] = selected_branch
            
            with st.spinner("Generating predictions..."):
                predictions = predict_colleges(
                    predictor,
                    df,
                    st.session_state.cutoff_mark,
                    **filters
                )
                
                # Filter by minimum chance
                predictions = predictions[
                    predictions['Admission Chance'] >= min_chance
                ]
                
                display_predictions(predictions)

if __name__ == "__main__":
    main()