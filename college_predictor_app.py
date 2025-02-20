import streamlit as st
import pandas as pd
import numpy as np
import os

# Set page config as the FIRST command
st.set_page_config(
    page_title="TNEA Cutoff Predictor For Students",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #ffffff;
        color: #262730;
    }

    /* Sidebar background */
    .css-1d391kg {
        background-color: #f0f2f6 !important;
    }

    /* Widgets (buttons, sliders, etc.) */
    .stButton>button, .stSlider>div>div>div>div, .stSelectbox>div>div>div {
        background-color: #f63366 !important;
        color: #ffffff !important;
        border-radius: 8px;
    }

    /* Header decoration */
    .css-1v3fvcr {
        background: linear-gradient(90deg, #f63366, #fffd80);
        padding: 10px;
        border-radius: 8px;
        color: #ffffff !important;
    }

    /* Text color */
    .stMarkdown, .stText, .stWrite {
        color: #262730 !important;
    }

    /* Dark mode toggle */
    .stCheckbox>label {
        color: #262730 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to load college data
def load_college_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        st.error(f"Error: File '{file_path}' not found.")
        return None

# Function to calculate cutoff
def calculate_cutoff(maths, physics, chemistry):
    return (maths) + (physics / 2) + (chemistry / 2)

# Function to predict all colleges
def predict_all_colleges(df, cutoff_mark):
    eligible = df[pd.notna(df['OC']) & (df['OC'] >= cutoff_mark)].copy()
    if not eligible.empty:
        eligible['Margin'] = eligible['OC'] - cutoff_mark
        sorted_predictions = eligible.sort_values(['OC', 'Margin'], ascending=[True, True])
        return sorted_predictions.head(10)
    return pd.DataFrame()

# Function to predict branch-specific colleges
def predict_branch_specific(df, cutoff_mark, branch_name):
    branch_df = df[df['RANCH NAME'] == branch_name]
    eligible = branch_df[pd.notna(branch_df['OC']) & (branch_df['OC'] >= cutoff_mark)].copy()
    if not eligible.empty:
        eligible['Margin'] = eligible['OC'] - cutoff_mark
        sorted_predictions = eligible.sort_values(['OC', 'Margin'], ascending=[True, True])
        return sorted_predictions.head(10)
    return pd.DataFrame()

# Function to show all branches for a specific college
def show_college_branches(df, college_name):
    college_df = df[df['COLLEGE NAME'] == college_name]
    st.subheader(f"Branches available at {college_name}")
    for idx, (_, row) in enumerate(college_df.iterrows(), 1):
        st.write(f"**Branch {idx}:**")
        st.write(f"Name: {row['RANCH NAME']}")
        st.write(f"Code: {row['BRANCH CODE']}")
        st.write(f"Max Cutoff: {row['OC']:.2f}")
        st.write("-" * 50)

# Main function to run the Streamlit app
def main():
    # Title and header
    st.title("🎓 TNEA Cutoff Predictor For Students")
    st.markdown(
        "<div class='css-1v3fvcr'> Welcome to the TNEA Cutoff Predictor! Enter your marks and explore college options.</div>",
        unsafe_allow_html=True,
    )

    # Load data
    file_path = "C:\\Users\\Kaniz\\Pictures\\TNEA_fct\\Vocational_2023_Mark_Cutoff.csv"
    df = load_college_data(file_path)

    if df is None:
        return

    # Sidebar for user input
    st.sidebar.header("Enter Your Marks")
    maths = st.sidebar.number_input("Mathematics", min_value=0, max_value=100, value=0 )
    physics = st.sidebar.number_input("Physics", min_value=0, max_value=100, value=0)
    chemistry = st.sidebar.number_input("Chemistry", min_value=0, max_value=100, value=0)

    if st.sidebar.button("Calculate Cutoff"):
        cutoff_mark = calculate_cutoff(maths, physics, chemistry)
        st.sidebar.success(f"Your calculated cutoff mark is: {cutoff_mark:.2f}")

        # Store cutoff mark in session state
        st.session_state.cutoff_mark = cutoff_mark

        # Filter mode selection
        mode = st.selectbox("Choose filtering mode:", 
                            ["Show top 10 colleges across all branches", 
                             "Show top 10 colleges for specific branch", 
                             "Show all branches for specific college"])

        if mode == "Show top 10 colleges across all branches":
            predictions = predict_all_colleges(df, cutoff_mark)
            if not predictions.empty:
                st.subheader("Top Recommendations:")
                for idx, row in predictions.iterrows():
                    st.write(f"**Rank {idx + 1}:** College: {row['COLLEGE NAME']}, Branch: {row['RANCH NAME']}, Max Cutoff: {row['OC']:.2f}, Your Margin: {row['Margin']:.2f}")
            else:
                st.warning("No colleges found matching your criteria.")

        elif mode == "Show top 10 colleges for specific branch":
            branch_name = st.selectbox("Select Branch", sorted(df['RANCH NAME'].unique()))
            predictions = predict_branch_specific(df, cutoff_mark, branch_name)
            if not predictions.empty:
                st.subheader("Top Recommendations for Branch:")
                for idx, row in predictions.iterrows():
                    st.write(f"**Rank {idx + 1}:** College: {row['COLLEGE NAME']}, Max Cutoff: {row['OC']:.2f}, Your Margin: {row['Margin']:.2f}")
            else:
                st.warning("No colleges found matching your criteria.")

        elif mode == "Show all branches for specific college":
            college_name = st.selectbox("Select College", sorted(df['COLLEGE NAME'].unique()))
            show_college_branches(df, college_name)

    # Home button to restart the app
    if st.sidebar.button("HOME"):
        # Clear session state to reset the app
        st.session_state.clear()
        st.experimental_rerun()

if __name__ == "__main__":
    main()