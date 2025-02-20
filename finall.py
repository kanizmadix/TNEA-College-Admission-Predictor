import streamlit as st
import pandas as pd
import numpy as np
from cadv_new import EnhancedCollegePredictorML
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="College Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .high-chance {
        color: #0f5132;
        background-color: #d1e7dd;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .medium-chance {
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

def format_admission_chance(chance):
    """Format admission chance with appropriate styling"""
    if chance >= 80:
        return f'<div class="high-chance">Very High ({chance:.1f}%)</div>'
    elif chance >= 60:
        return f'<div class="medium-chance">Good ({chance:.1f}%)</div>'
    elif chance >= 40:
        return f'<div class="medium-chance">Moderate ({chance:.1f}%)</div>'
    else:
        return f'<div class="low-chance">Low ({chance:.1f}%)</div>'

def main():
    st.title("🎓 College Admission Predictor")
    st.write("Enter your marks and explore college recommendations based on XGBoost predictions")

    # Load model
    predictor, df, metrics = load_model()
    
    if predictor is None or df is None:
        st.error("Failed to initialize the prediction model. Please check your data file.")
        return

    # Display model metrics in expander
    with st.expander("Model Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with col3:
            st.metric("MAE", f"{metrics['mae']:.4f}")

    # Input marks
    st.subheader("Enter Your Marks")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        maths = st.number_input("Mathematics", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    with col2:
        physics = st.number_input("Physics", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    with col3:
        chemistry = st.number_input("Chemistry", min_value=0.0, max_value=100.0, value=60.0, step=0.5)

    cutoff_mark = maths + (physics / 2) + (chemistry / 2)
    st.info(f"Your calculated cutoff mark is: {cutoff_mark:.2f}")

    # Filtering options
    st.subheader("Explore Predictions")
    
    filter_mode = st.radio(
        "Choose filtering mode:",
        ["Show top colleges across all branches", 
         "Show top colleges for specific branch",
         "Show all branches for specific college"]
    )

    if filter_mode == "Show top colleges across all branches":
        predictions = predict_all_colleges(predictor, df, cutoff_mark)
        display_predictions(predictions)
        
    elif filter_mode == "Show top colleges for specific branch":
        branch_name = st.selectbox(
            "Select Branch",
            sorted(df['BRANCH NAME'].unique())
        )
        if branch_name:
            predictions = predict_branch_specific(predictor, df, cutoff_mark, branch_name)
            display_predictions(predictions)
            
    else:  # Show all branches for specific college
        college_name = st.selectbox(
            "Select College",
            sorted(df['COLLEGE NAME'].unique())
        )
        if college_name:
            show_college_branches(predictor, df, college_name, cutoff_mark)

def predict_all_colleges(predictor, df, cutoff_mark):
    """Predict cutoffs for all colleges"""
    predictions = []
    progress_bar = st.progress(0)
    
    for idx, row in df.iterrows():
        progress_bar.progress((idx + 1) / len(df))
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
    
    progress_bar.empty()
    return pd.DataFrame(predictions)

def predict_branch_specific(predictor, df, cutoff_mark, branch_name):
    """Predict cutoffs for specific branch"""
    branch_df = df[df['BRANCH NAME'] == branch_name]
    return predict_all_colleges(predictor, branch_df, cutoff_mark)

def calculate_admission_chance(margin):
    """Calculate admission chance based on margin"""
    if margin >= 0:
        return min((margin + 5) / 10, 1) * 100
    else:
        return max(0, (1 + margin / 20) * 100)

def show_college_branches(predictor, df, college_name, user_cutoff):
    """Show predictions for all branches in a college"""
    college_df = df[df['COLLEGE NAME'] == college_name]
    predictions = predict_all_colleges(predictor, college_df, user_cutoff)
    
    if not predictions.empty:
        predictions = predictions.sort_values('Predicted Cutoff', ascending=False)
        
        st.subheader(f"Branches available at {college_name}")
        
        # Create a comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Predicted Cutoff',
            x=predictions['BRANCH NAME'],
            y=predictions['Predicted Cutoff'],
            marker_color='royalblue'
        ))
        
        fig.add_trace(go.Scatter(
            name='Your Cutoff',
            x=predictions['BRANCH NAME'],
            y=[user_cutoff] * len(predictions),
            mode='lines',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Branch-wise Cutoff Comparison',
            xaxis_title='Branch',
            yaxis_title='Cutoff Mark',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed predictions
        for _, row in predictions.iterrows():
            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Branch:** {row['BRANCH NAME']}")
                with col2:
                    st.write(f"**Predicted Cutoff:** {row['Predicted Cutoff']:.2f}")
                with col3:
                    st.markdown(format_admission_chance(row['Admission Chance']), unsafe_allow_html=True)

def display_predictions(predictions):
    """Display top 10 predictions with visualizations"""
    if predictions.empty:
        st.warning("No predictions available.")
        return

    predictions_sorted = predictions.sort_values('Admission Chance', ascending=False).head(10)
    
    # Create a horizontal bar chart for admission chances
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=predictions_sorted['Admission Chance'],
        y=predictions_sorted['COLLEGE NAME'] + ' - ' + predictions_sorted['BRANCH NAME'],
        orientation='h',
        marker=dict(
            color=predictions_sorted['Admission Chance'],
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title='Top 10 College Recommendations',
        xaxis_title='Admission Chance (%)',
        yaxis_title='College - Branch',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed predictions in an expander
    with st.expander("View Detailed Predictions"):
        for idx, row in predictions_sorted.iterrows():
            st.markdown(f"""
            **Rank {idx + 1}:**
            * College: {row['COLLEGE NAME']}
            * Branch: {row['BRANCH NAME']}
            * Predicted Cutoff: {row['Predicted Cutoff']:.2f}
            * Your Cutoff: {row['Your Cutoff']:.2f}
            * Margin: {row['Margin']:.2f}
            """)
            st.markdown(format_admission_chance(row['Admission Chance']), unsafe_allow_html=True)
            st.markdown("---")

if __name__ == "__main__":
    main()