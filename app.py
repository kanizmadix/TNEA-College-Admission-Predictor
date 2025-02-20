import streamlit as st
import pandas as pd
import numpy as np
from cadv_new import EnhancedCollegePredictorML
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image  # Import Pillow for image handling
from sklearn.model_selection import train_test_split

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
    .cutoff-value {
        color: #007bff; /* Example: Blue color */
        font-weight: bold;
    }
    .predicted-cutoff-value {
        color: #28a745; /* Example: Green color */
        font-weight: bold;
    }
    .margin-value {
        color: #dc3545; /* Example: Red color */
        font-weight: bold;
    }
    .highlighted-cutoff {
        background-color: #90EE90; /* Light Green */
        color: #000000; /* Black */
        padding: 0.2rem 0.4rem;
        border-radius: 0.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def load_and_clean_data(vocational_path, maxcutoff_path):
    """Loads and cleans the data, handling NaN values and inconsistencies."""
    try:
        df_vocational = pd.read_csv(vocational_path)
        df_maxcutoff = pd.read_csv(maxcutoff_path)

        # Rename columns in df_maxcutoff for consistency
        df_maxcutoff = df_maxcutoff.rename(columns={'COLLEGE NAME': 'College Name', 'BRANCH NAME': 'Branch Name', 'MAX CUTOFF': 'OC'})

        # Convert all values in 'College Name' and 'Branch Name' to string
        df_vocational['COLLEGE NAME'] = df_vocational['COLLEGE NAME'].astype(str)
        df_vocational['BRANCH NAME'] = df_vocational['BRANCH NAME'].astype(str)
        df_maxcutoff['College Name'] = df_maxcutoff['College Name'].astype(str)
        df_maxcutoff['Branch Name'] = df_maxcutoff['Branch Name'].astype(str)

        # Drop rows with NaN values in 'College Name' or 'Branch Name'
        df_vocational = df_vocational.dropna(subset=['COLLEGE NAME', 'BRANCH NAME'])
        df_maxcutoff = df_maxcutoff.dropna(subset=['College Name', 'Branch Name'])

        # Fill remaining NaN values with 'Unknown'
        df_vocational = df_vocational.fillna('Unknown')
        df_maxcutoff = df_maxcutoff.fillna('Unknown')

        # Standardize column names before combining
        df_vocational.rename(columns={'COLLEGE NAME': 'College Name', 'BRANCH NAME': 'Branch Name'}, inplace=True)

        # Combine DataFrames
        combined_df = pd.concat([df_vocational, df_maxcutoff], ignore_index=True)

        # Convert 'OC' column to numeric, coercing errors to NaN
        combined_df['OC'] = pd.to_numeric(combined_df['OC'], errors='coerce')

        # Remove rows with any missing values after converting 'OC' to numeric
        combined_df = combined_df.dropna()

        return combined_df

    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None
    except Exception as e:
        st.error(f"Error cleaning data: {e}")
        return None

@st.cache_resource
def load_model(combined_df):
    """Load and train the model"""
    try:
        predictor = EnhancedCollegePredictorML()

        # Check for required columns before training
        required_columns = ['College Name', 'Branch Name', 'OC']
        for col in required_columns:
            if col not in combined_df.columns:
                st.error(f"Required column '{col}' is missing in the combined data.")
                return None, None

        # Convert column names to string type
        combined_df.columns = combined_df.columns.astype(str)

        # Check if the DataFrame is empty after cleaning
        if combined_df.empty:
            st.error("The DataFrame is empty after cleaning.  Check your data and cleaning steps.")
            return None, None

        # Check the size of the DataFrame
        if len(combined_df) < 2:  # Need at least 2 samples for train/test split
            st.error(f"The DataFrame has too few rows ({len(combined_df)}) to train the model.  Need at least 2.")
            return None, None

        metrics = predictor.train_model(combined_df)

        return predictor, metrics
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

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

        if predicted_cutoff is not None and not np.isnan(predicted_cutoff):  # Check for NaN here
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

def display_predictions(predictions, display_chart=True):
    """Displays predictions with consistent formatting and limits to top 10."""
    if predictions.empty:
        st.warning("No predictions available.")
        return

    predictions = predictions.copy()
    predictions.loc[:, 'COLLEGE NAME'] = predictions['COLLEGE NAME'].astype(str)
    predictions.loc[:, 'BRANCH NAME'] = predictions['BRANCH NAME'].astype(str)
    predictions = predictions.dropna(subset=['COLLEGE NAME', 'BRANCH NAME'])

    # Sort and limit to top 10 *before* anything else
    predictions_sorted = predictions.sort_values('Admission Chance', ascending=False).head(10)

    if display_chart:
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
        # Create a container for better spacing
        container = st.container()
        
        # Add custom CSS for better formatting
        st.markdown("""
            <style>
            .prediction-box {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                border-left: 4px solid #007bff;
            }
            .college-name {
                font-size: 1.1rem;
                font-weight: bold;
                color: #2c3e50;
            }
            .branch-name {
                font-size: 1rem;
                color: #34495e;
                font-style: italic;
            }
            .metrics {
                margin: 0.5rem 0;
                color: #576574;
            }
            .predicted-value {
                color: #007bff;
                font-weight: bold;
                background-color: rgba(0, 123, 255, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .cutoff-value {
                color: #28a745;
                font-weight: bold;
                background-color: rgba(40, 167, 69, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .margin-value {
                color: #dc3545;
                font-weight: bold;
                background-color: rgba(220, 53, 69, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .rank-number {
                font-weight: bold;
                color: #6c757d;
                background-color: rgba(108, 117, 125, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Display each prediction in the sorted order
        for idx, row in predictions_sorted.iterrows():
            container.markdown(f"""
            <div class="prediction-box">
                <div class="college-name">Rank: <span class="rank-number">{idx + 1}</span></div>
                <div class="college-name">College: {row['COLLEGE NAME']}</div>
                <div class="branch-name">Branch: {row['BRANCH NAME']}</div>
                <div class="metrics">
                    Predicted Cutoff: <span class="predicted-value">{row['Predicted Cutoff']:.2f}</span><br>
                    Your Cutoff: <span class="cutoff-value">{row['Your Cutoff']:.2f}</span><br>
                    Margin: <span class="margin-value">{row['Margin']:.2f}</span>
                </div>
                {format_admission_chance(row['Admission Chance'])}
            </div>
            """, unsafe_allow_html=True)

def main():
    st.title("🎓 TNEA College Admission Predictor")
    st.write("Enter your marks and explore college recommendations based on XGBoost predictions")

    # Add logo
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=100)
    except FileNotFoundError:
        st.warning("Logo file 'logo.png' not found. Please add it to the same directory as the script.")

    # Load data
    vocational_path = "cleaned_vocational_data.csv"
    maxcutoff_path = "cleaned_maxcutoff_data.csv"

    combined_df = load_and_clean_data(vocational_path, maxcutoff_path)

    if combined_df is None:
        st.error("Failed to load and clean the data. Please check the file paths.")
        return

    predictor, metrics = load_model(combined_df)

    if predictor is None:
        st.error("Failed to initialize the prediction model. Please check your data file.")
        return

    # Seat matrix display
    if hasattr(predictor, 'seat_matrix'):
        st.subheader("Seat Matrix Summary")
        st.write(f"Total Unique Courses (Branches) Registered: 🏫 117")
        st.write("Total Seats Allocated by Category:")
        seat_matrix_summary = pd.DataFrame(predictor.seat_matrix.items(), columns=['Category', 'Seats'])
        st.table(seat_matrix_summary)
        st.write(f"Total Seats Across All Categories: {predictor.total_seats}")

    # Display model metrics in expander
    if metrics:
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

    # Calculate Cutoff Button
    if st.button("Calculate Cutoff"):
        st.session_state['cutoff_mark'] = maths + (physics / 2) + (chemistry / 2)
        st.session_state['cutoff_message'] = f"Your calculated cutoff mark is: <span class=\"highlighted-cutoff\">{st.session_state['cutoff_mark']:.2f}</span>"

    # Display calculated cutoff mark
    if 'cutoff_mark' in st.session_state and st.session_state['cutoff_mark'] is not None:
        st.markdown(st.session_state['cutoff_message'], unsafe_allow_html=True)

    if st.session_state.get('cutoff_mark') is not None:
        # Category Selection
        category = st.selectbox(
            "Select Category (Optional):",
            options=[None, 'OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST'],
            index=0
        )
        st.session_state['category'] = category

        # Filtering Options
        st.subheader("Filtering Options")
        filter_mode = st.selectbox(
            "Select Filtering Mode:",
            ["Top 10 Colleges", "College-wise Courses", "Branch-wise Colleges"]
        )

        if filter_mode == "Top 10 Colleges":
            if st.button("Calculate Top 10", key="top10"):
                with st.spinner("Calculating predictions..."):
                    predictions = predict_colleges(predictor, combined_df, st.session_state['cutoff_mark'], st.session_state['category'])
                    display_predictions(predictions, display_chart=True)

        elif filter_mode == "College-wise Courses":
            selected_college = st.selectbox("Select College", sorted(combined_df['College Name'].unique()))
            if st.button("List Courses", key="college_courses"):
                with st.spinner(f"Listing courses for {selected_college}..."):
                    college_df = combined_df[combined_df['College Name'] == selected_college]
                    if not college_df.empty:
                        predictions = predict_colleges(predictor, college_df, st.session_state['cutoff_mark'], st.session_state['category'])
                        display_predictions(predictions, display_chart=False)
                    else:
                        st.warning(f"No courses found for {selected_college}.")

        elif filter_mode == "Branch-wise Colleges":
            selected_branch = st.selectbox("Select Branch", sorted(combined_df['Branch Name'].unique()))
            if st.button("List Colleges", key="branch_colleges"):
                with st.spinner(f"Listing colleges for {selected_branch}..."):
                    if not combined_df.empty:
                        branch_df = combined_df[combined_df['Branch Name'] == selected_branch]
                        if not branch_df.empty:
                            predictions = predict_colleges(predictor, branch_df, st.session_state['cutoff_mark'], st.session_state['category'])
                            display_predictions(predictions, display_chart=False)
                        else:
                            st.warning(f"No colleges found for {selected_branch}.")
                    else:
                        st.warning("Combined data is empty.")

if __name__ == "__main__":
    main()