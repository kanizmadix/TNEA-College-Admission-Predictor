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
    small {
        opacity: 0.8;
        font-size: 0.85em;
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
    """
    Format admission chance with appropriate styling and meaningful messages
    
    Args:
        chance: Calculated admission chance percentage
    Returns:
        Formatted HTML string with appropriate styling
    """
    if chance >= 80:
        return f'<div class="excellent-chance">Excellent Chance - You are above or very close to the cutoff! ({chance:.1f}%)<br/><small>Strong position for admission</small></div>'
    elif chance >= 60:
        return f'<div class="good-chance">Good Chance - Within competitive range ({chance:.1f}%)<br/><small>Consider this a serious option</small></div>'
    elif chance >= 40:
        return f'<div class="fair-chance">Fair Chance - May need improvement ({chance:.1f}%)<br/><small>Keep as a backup option</small></div>'
    elif chance >= 20:
        return f'<div class="low-chance">Low Chance - Significant gap ({chance:.1f}%)<br/><small>Consider alternative choices</small></div>'
    else:
        return f'<div class="low-chance">Very Low Chance - Large gap from cutoff ({chance:.1f}%)<br/><small>Focus on other options</small></div>'


def predict_all_colleges(predictor, df, cutoff_mark):
    """Predict cutoffs for all colleges with batched processing"""
    predictions = []
    total_rows = len(df)
    
    batch_size = 10
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_rows)
        
        batch_df = df.iloc[start_idx:end_idx]
        
        for _, row in batch_df.iterrows():
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
        
        progress = min(1.0, (batch_idx + 1) / num_batches)
        progress_bar.progress(progress)
    
    progress_bar.empty()
    return pd.DataFrame(predictions)

def predict_branch_specific(predictor, df, cutoff_mark, branch_name):
    """Predict cutoffs for specific branch"""
    branch_df = df[df['BRANCH NAME'] == branch_name].copy()
    return predict_all_colleges(predictor, branch_df, cutoff_mark)

def calculate_admission_chance(margin):
    """
    Calculate admission chance based on margin between student's cutoff and predicted cutoff
    
    Args:
        margin: Student's cutoff - Predicted cutoff
    Returns:
        Admission chance percentage
    """
    # If margin is positive or within -10 marks
    if margin >= 0:
        return 95  # Excellent chance (above cutoff)
    elif margin >= -10:
        return 80  # Very good chance (within 10 marks below)
    elif margin >= -15:
        return 60  # Good chance (11-15 marks below)
    elif margin >= -20:
        return 40  # Average chance (16-20 marks below)
    elif margin >= -25:
        return 20  # Low chance (21-25 marks below)
    else:
        return 5   # Very poor chance (>25 marks below)

def show_college_branches(predictor, df, college_name, user_cutoff):
    """Show predictions for all branches in a college"""
    college_df = df[df['COLLEGE NAME'] == college_name].copy()
    predictions = predict_all_colleges(predictor, college_df, user_cutoff)
    
    if not predictions.empty:
        predictions = predictions.sort_values('Predicted Cutoff', ascending=False)
        
        st.subheader(f"Branches available at {college_name}")
        
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
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        for _, row in predictions.iterrows():
            with st.container():
                cols = st.columns([2, 1, 1])
                with cols[0]:
                    st.write(f"**Branch:** {row['BRANCH NAME']}")
                with cols[1]:
                    st.write(f"**Predicted Cutoff:** {row['Predicted Cutoff']:.2f}")
                with cols[2]:
                    st.markdown(format_admission_chance(row['Admission Chance']), unsafe_allow_html=True)

def display_predictions(predictions):
    """Display top 10 predictions with visualizations"""
    if predictions.empty:
        st.warning("No predictions available.")
        return

    predictions_sorted = predictions.sort_values('Admission Chance', ascending=False).head(10)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=predictions_sorted['Admission Chance'],
        y=[f"{col} - {br}" for col, br in zip(predictions_sorted['COLLEGE NAME'], predictions_sorted['BRANCH NAME'])],
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
        height=500,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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
        cols = st.columns(3)
        with cols[0]:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with cols[1]:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with cols[2]:
            st.metric("MAE", f"{metrics['mae']:.4f}")

    # Input marks section
    st.subheader("Enter Your Marks")
    
    with st.form("marks_form"):
        cols = st.columns(3)
        with cols[0]:
            maths = st.number_input("Mathematics", min_value=0.0, max_value=100.0, step=0.5)
        with cols[1]:
            physics = st.number_input("Physics", min_value=0.0, max_value=100.0, step=0.5)
        with cols[2]:
            chemistry = st.number_input("Chemistry", min_value=0.0, max_value=100.0, step=0.5)
        
        calculate_button = st.form_submit_button("Calculate Cutoff")
    
    if calculate_button:
        cutoff_mark = maths + (physics / 2) + (chemistry / 2)
        st.session_state.cutoff_mark = cutoff_mark
        st.info(f"Your calculated cutoff mark is: {cutoff_mark:.2f}")
        
        # Store the marks in session state
        st.session_state.marks_submitted = True
        st.session_state.maths = maths
        st.session_state.physics = physics
        st.session_state.chemistry = chemistry
        st.rerun()  # Changed from st.experimental_rerun() to st.rerun()

    # Only show prediction section if marks have been submitted
    if 'marks_submitted' in st.session_state and st.session_state.marks_submitted:
        st.subheader("Explore Predictions")
        
        # Create columns for displaying current marks
        with st.container():
            st.write("Current Marks:")
            cols = st.columns(4)
            with cols[0]:
                st.write(f"**Mathematics:** {st.session_state.maths}")
            with cols[1]:
                st.write(f"**Physics:** {st.session_state.physics}")
            with cols[2]:
                st.write(f"**Chemistry:** {st.session_state.chemistry}")
            with cols[3]:
                st.write(f"**Cutoff Mark:** {st.session_state.cutoff_mark:.2f}")
        
        # Create a form for predictions
        with st.form("prediction_form"):
            filter_mode = st.radio(
                "Choose filtering mode:",
                ["Show top colleges across all branches", 
                 "Show top colleges for specific branch",
                 "Show all branches for specific college"]
            )
            
            # Add necessary selection boxes based on filter mode
            if filter_mode == "Show top colleges for specific branch":
                branch_name = st.selectbox(
                    "Select Branch",
                    sorted(df['BRANCH NAME'].unique())
                )
            elif filter_mode == "Show all branches for specific college":
                college_name = st.selectbox(
                    "Select College",
                    sorted(df['COLLEGE NAME'].unique())
                )
            
            get_predictions = st.form_submit_button("Get Predictions")
        
        if get_predictions:
            progress_text = "Calculating predictions..."
            my_bar = st.progress(0, text=progress_text)
            
            try:
                if filter_mode == "Show top colleges across all branches":
                    with st.spinner("Processing all colleges..."):
                        predictions = predict_all_colleges(predictor, df, st.session_state.cutoff_mark)
                        my_bar.progress(1.0, text="Predictions complete!")
                        display_predictions(predictions)
                
                elif filter_mode == "Show top colleges for specific branch":
                    with st.spinner("Processing selected branch..."):
                        predictions = predict_branch_specific(predictor, df, st.session_state.cutoff_mark, branch_name)
                        my_bar.progress(1.0, text="Predictions complete!")
                        display_predictions(predictions)
                
                else:  # Show all branches for specific college
                    with st.spinner("Processing college branches..."):
                        show_college_branches(predictor, df, college_name, st.session_state.cutoff_mark)
                        my_bar.progress(1.0, text="Predictions complete!")
            
            except Exception as e:
                st.error(f"An error occurred while generating predictions: {str(e)}")
            finally:
                my_bar.empty()
        
        # Add a button to reset the form
        if st.button("Reset Calculator"):
            for key in ['marks_submitted', 'maths', 'physics', 'chemistry', 'cutoff_mark']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()  # Changed from st.experimental_rerun() to st.rerun()

if __name__ == "__main__":
    main()