import streamlit as st
import pandas as pd
import numpy as np
from cadv_new import EnhancedCollegePredictorML
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image  # Import Pillow for image handling

# Set page configuration
st.set_page_config(
    page_title="TNEA College Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .very-high-chance {
        color: #155b1a;
        background-color: #d5f4d7;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .high-chance {
        color: #198754;
        background-color: #c6ead9;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .good-chance {
        color: #206c59;
        background-color: #c8e6c9;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .moderate-chance {
        color: #856404;
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .low-chance {
        color: #9f5132;
        background-color: #f4dcd7;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .very-low-chance {
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
        predictor = EnhancedCollegePredictorML()

        # Load data
        df_vocational = pd.read_csv("Vocational_2023_Mark_Cutoff.csv")
        df_maxcutoff = pd.read_csv("Unique_Colleges_Max_Cutoff.csv")

        # Rename columns in df_maxcutoff for consistency
        df_maxcutoff = df_maxcutoff.rename(columns={'COLLEGE NAME': 'College Name', 'BRANCH NAME': 'Branch Name', 'MAX CUTOFF': 'OC'})

        # Combine DataFrames
        combined_df = pd.concat([df_vocational, df_maxcutoff], ignore_index=True)

        # Check for required columns before training
        required_columns = ['College Name', 'Branch Name', 'OC']
        for col in required_columns:
            if col not in combined_df.columns:
                st.error(f"Required column '{col}' is missing in the combined data.")
                return None, None, None

        # Convert all values in 'College Name' and 'Branch Name' to string
        combined_df['College Name'] = combined_df['College Name'].astype(str)
        combined_df['Branch Name'] = combined_df['Branch Name'].astype(str)

        # Convert column names to string type
        combined_df.columns = combined_df.columns.astype(str)

        metrics = predictor.train_model(combined_df)

        return predictor, combined_df, metrics
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def format_admission_chance(chance):
    """Format admission chance with finer-grained categories for higher accuracy"""
    if chance >= 90:
        return f'<div class="very-high-chance">Almost Guaranteed – Strong profile, very high chance! ({chance:.1f}%)</div>'
    elif chance >= 75:
        return f'<div class="high-chance">Very Likely – Competitive, but minor improvements can help. ({chance:.1f}%)</div>'
    elif chance >= 60:
        return f'<div class="good-chance">Likely – Good profile, but a strong applicant pool may impact outcomes. ({chance:.1f}%)</div>'
    elif chance >= 45:
        return f'<div class="moderate-chance">Moderate – Competitive but uncertain, consider strengthening weak areas. ({chance:.1f}%)</div>'
    elif chance >= 30:
        return f'<div class="low-chance">Low – Admission is possible but highly competitive. ({chance:.1f}%)</div>'
    else:
        return f'<div class="very-low-chance">Very Low – Unlikely, consider alternative options. ({chance:.1f}%)</div>'

def predict_colleges(predictor, df, cutoff_mark, category=None):
    """Predict cutoffs for all colleges"""
    predictions = []

    for _, row in df.iterrows():
        predicted_cutoff = predictor.predict_cutoff(
            row['College Name'],
            row['Branch Name']
        )

        if predicted_cutoff is not None:
            margin = cutoff_mark - predicted_cutoff
            chance = calculate_admission_chance(margin)
            chance = predictor.adjust_chance_for_category(chance, category)
            predictions.append({
                'COLLEGE NAME': row['College Name'],
                'BRANCH NAME': row['Branch Name'],
                'Predicted Cutoff': predicted_cutoff,
                'Your Cutoff': cutoff_mark,
                'Margin': margin,
                'Admission Chance': chance
            })

    return pd.DataFrame(predictions)

def calculate_admission_chance(margin):
    """Calculate admission chance based on margin"""
    if margin >= 0:
        return min(100, max(0, (margin + 5) * 10))
    else:
        return max(0, min(100, (1 + margin / 20) * 100))

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

    # Display detailed predictions in an expander
    with st.expander("View Detailed Predictions"):
        for idx, row in predictions_sorted.iterrows():
            st.markdown(f"""
            **Rank: {idx+1}**
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

    # Add logo
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=100)
    except FileNotFoundError:
        st.warning("Logo file 'logo.png' not found. Please add it to the same directory as the script.")

    # Load model
    predictor, df, metrics = load_model()

    if predictor is None or df is None:
        st.error("Failed to initialize the prediction model. Please check your data file.")
        return

    # Seat matrix display
    st.subheader("Seat Matrix Summary")
    st.write(f"Total Unique Courses (Branches) Registered: 🏫 117")
    st.write("Total Seats Allocated by Category:")
    seat_matrix_summary = pd.DataFrame(predictor.seat_matrix.items(), columns=['Category', 'Seats'])
    st.table(seat_matrix_summary)
    st.write(f"Total Seats Across All Categories: {predictor.total_seats}")

    # Display model metrics in expander
    with st.expander("Model Performance Metrics"):
        cols = st.columns(3)
        with cols[0]:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with cols[1]:
            st.metric("RMSE", f"{metrics['rmse']:.4f}")
        with cols[2]:
            st.metric("MAE", f"{metrics['mae']:.4f}")

    # Input marks
    st.subheader("Enter Your Marks")
    cols = st.columns(3)

    with cols[0]:
        maths = st.number_input("Mathematics", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    with cols[1]:
        physics = st.number_input("Physics", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    with cols[2]:
        chemistry = st.number_input("Chemistry", min_value=0.0, max_value=100.0, value=60.0, step=0.5)

    cutoff_mark = maths + (physics / 2) + (chemistry / 2)
    st.info(f"Your calculated cutoff mark is: {cutoff_mark:.2f}")

    # Category Selection
    category = st.selectbox(
        "Select Category (Optional):",
        options=[None, 'OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST'],
        index=0  # Default to None
    )

    # --- FILTERING OPTIONS ---
    st.subheader("Filtering Options")

    filter_mode = st.selectbox(
        "Select Filtering Mode:",
        ["Top 10 Colleges", "College-wise Courses", "Branch-wise Colleges"]
    )

    if filter_mode == "Top 10 Colleges":
        if st.button("Calculate Top 10", key="top10"):  # Added key for the button
            with st.spinner("Calculating predictions..."):
                predictions = predict_colleges(predictor, df, cutoff_mark, category)
                display_predictions(predictions)

    elif filter_mode == "College-wise Courses":
        selected_college = st.selectbox("Select College", sorted(df['College Name'].unique()))
        if st.button("List Courses", key="college_courses"):  # Added key for the button
            with st.spinner(f"Listing courses for {selected_college}..."):
                college_df = df[df['College Name'] == selected_college]
                predictions = predict_colleges(predictor, college_df, cutoff_mark, category)
                if not predictions.empty:
                    st.write(f"Available courses at {selected_college}:")
                    for _, row in predictions.iterrows():
                        st.write(f"- {row['BRANCH NAME']} (Predicted Cutoff: {row['Predicted Cutoff']:.2f}, Margin: {row['Margin']:.2f}, Admission Chance: {row['Admission Chance']:.2f}%)")
                else:
                    st.warning(f"No courses found for {selected_college}.")

    elif filter_mode == "Branch-wise Colleges":
        selected_branch = st.selectbox("Select Branch", sorted(df['Branch Name'].unique()))
        if st.button("List Colleges", key="branch_colleges"):
            with st.spinner(f"Listing colleges for {selected_branch}..."):
                branch_df = df[df['Branch Name'] == selected_branch]
                predictions = predict_colleges(predictor, branch_df, cutoff_mark, category)
                if not predictions.empty:
                    # Sort by admission chance
                    predictions = predictions.sort_values(by='Admission Chance', ascending=False).head(10)
                    st.write(f"Top 10 colleges for {selected_branch}:")
                    for _, row in predictions.iterrows():
                        st.write(f"- {row['COLLEGE NAME']} (Predicted Cutoff: {row['Predicted Cutoff']:.2f}, Margin: {row['Margin']:.2f}, Admission Chance: {row['Admission Chance']:.2f}%)")
                else:
                    st.warning(f"No colleges found for {selected_branch}.")

if __name__ == "__main__":
    main()