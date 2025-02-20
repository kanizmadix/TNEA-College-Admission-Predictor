import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional

@dataclass
class ThemeColor:
    primaryColor: str
    backgroundColor: str
    secondaryBackgroundColor: str
    textColor: str
    inputTextColor: str

# Set page configuration
st.set_page_config(
    page_title="TNEA Vocational Cutoff Predictor",
    page_icon="🎓",
    layout="wide"
)

# Theme colors
preset_colors = [
    ("Default light", ThemeColor(
        primaryColor="#ff4b4b",
        backgroundColor="#ffffff",
        secondaryBackgroundColor="#f0f2f6",
        textColor="#000000",
        inputTextColor="#31333F"
    )),
    ("Default dark", ThemeColor(
        primaryColor="#ff4b4b",
        backgroundColor="#0e1117",
        secondaryBackgroundColor="#262730",
        textColor="#ffffff",
        inputTextColor="#fafafa"
    ))
]

# Custom CSS to improve UI
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
    }
    .theme-toggle {
        position: fixed;
        top: 0.5rem;
        right: 8rem;
        z-index: 1000;
        padding: 0.5rem;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    .info-box h3 {
        margin-bottom: 10px;
    }
    .info-box ol {
        margin-left: 20px;
    }
    .info-message {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        font-weight: 500;
        margin: 10px 0;
    }
    .stCheckbox {
        filter: brightness(1.2);
    }
    .stCheckbox > label {
        color: inherit !important;
    }
    [data-testid="stToggleSwitch"] {
        border: 2px solid #4a4a4a;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Create container for theme toggle
theme_container = st.container()
with theme_container:
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        # Theme toggle in the header with improved visibility
        is_dark_mode = st.toggle("🌓", key="dark_mode", help="Toggle dark/light mode")
        
        # Update theme based on toggle
        if is_dark_mode:
            theme = preset_colors[1][1]
            st.session_state.theme = "dark"
        else:
            theme = preset_colors[0][1]
            st.session_state.theme = "light"

        # Apply theme using custom CSS
        st.markdown(f"""
    <style>
        .stApp {{
            background-color: {theme.backgroundColor};
            color: {theme.textColor};
        }}
        [data-testid="stToggleSwitch"] {{
            background-color: {theme.secondaryBackgroundColor} !important;
            border: 2px solid {theme.textColor} !important;
        }}
        .stMarkdown, .stDataFrame {{
            background-color: {theme.secondaryBackgroundColor};
        }}
        .stButton>button {{
            background-color: {theme.primaryColor} !important;
            color: white !important;
        }}
        .stSelectbox label, .stNumberInput label {{
            color: {theme.textColor} !important;
        }}
        .stSelectbox select, .stNumberInput input {{
            color: {theme.inputTextColor} !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {theme.textColor} !important;
        }}
        .dataframe {{
            color: {theme.textColor} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {theme.secondaryBackgroundColor};
        }}
        .info-box {{
            background-color: {theme.secondaryBackgroundColor};
            color: {theme.textColor};
        }}
        .info-box h3, .info-box li {{
            color: {theme.textColor} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_college_data():
    try:
        df = pd.read_csv('Vocational_2023_Mark_Cutoff.csv')
        df['COLLEGE NAME'] = df['COLLEGE NAME'].str.replace('\n', ' ').str.strip()
        df.fillna('', inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'Vocational_2023_Mark_Cutoff.csv' file not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_cutoff(maths, physics, chemistry):
    return (maths) + (physics / 2) + (chemistry / 2)

# Load college data
df = load_college_data()

if df is not None:
    st.title("TNEA Vocational Cutoff Predictor")
    
    # Sidebar with default theme preserved
    st.sidebar.header("Enter Your Marks")
    
    maths = st.sidebar.number_input("Mathematics Marks", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    physics = st.sidebar.number_input("Physics Marks", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    chemistry = st.sidebar.number_input("Chemistry Marks", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    
    if st.sidebar.button("CALCULATE CUTOFF", key="calculate"):
        cutoff_mark = calculate_cutoff(maths, physics, chemistry)
        st.sidebar.markdown(f"<p class='big-font'>Your Cutoff: {cutoff_mark:.2f}</p>", unsafe_allow_html=True)
        st.session_state['cutoff'] = cutoff_mark
    
    # Main content area
    st.header("College Predictor")
    
    if 'cutoff' not in st.session_state:
        st.markdown("""
            <div class='info-message'>
                Please calculate your cutoff mark first using the sidebar.
            </div>
        """, unsafe_allow_html=True)
    
    mode = st.selectbox(
        "Choose filtering mode:",
        ["Show top colleges across all branches",
         "Show colleges for specific branch",
         "Show all branches for specific college"]
    )
    
    # Get unique sorted lists for filters
    unique_colleges = sorted(df['COLLEGE NAME'].unique())
    unique_branches = sorted(df['BRANCH NAME'].unique())
    communities = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    
    if mode == "Show top colleges across all branches":
        selected_community = st.selectbox("Select Community", communities)
        
        if 'cutoff' in st.session_state:
            cutoff = st.session_state['cutoff']
            # Filter colleges based on cutoff and community
            eligible_colleges = df[
                (df[selected_community] != '') & 
                (pd.to_numeric(df[selected_community], errors='coerce') <= cutoff)
            ].copy()
            
            if not eligible_colleges.empty:
                eligible_colleges['Cutoff'] = pd.to_numeric(eligible_colleges[selected_community], errors='coerce')
                eligible_colleges['Margin'] = cutoff - eligible_colleges['Cutoff']
                eligible_colleges = eligible_colleges.sort_values('Cutoff', ascending=False)
                
                st.write(f"### Eligible Colleges for {selected_community} Category")
                display_cols = ['COLLEGE NAME', 'BRANCH NAME', 'Cutoff', 'Margin']
                st.dataframe(eligible_colleges[display_cols], hide_index=True)
            else:
                st.warning("No eligible colleges found for your cutoff mark.")
                
    elif mode == "Show colleges for specific branch":
        selected_branch = st.selectbox("Select Branch", unique_branches)
        selected_community = st.selectbox("Select Community", communities)
        
        if 'cutoff' in st.session_state:
            cutoff = st.session_state['cutoff']
            # Filter colleges based on branch, cutoff and community
            eligible_colleges = df[
                (df['BRANCH NAME'] == selected_branch) &
                (df[selected_community] != '') & 
                (pd.to_numeric(df[selected_community], errors='coerce') <= cutoff)
            ].copy()
            
            if not eligible_colleges.empty:
                eligible_colleges['Cutoff'] = pd.to_numeric(eligible_colleges[selected_community], errors='coerce')
                eligible_colleges['Margin'] = cutoff - eligible_colleges['Cutoff']
                eligible_colleges = eligible_colleges.sort_values('Cutoff', ascending=False)
                
                st.write(f"### Eligible Colleges for {selected_branch} - {selected_community} Category")
                display_cols = ['COLLEGE NAME', 'BRANCH NAME', 'Cutoff', 'Margin']
                st.dataframe(eligible_colleges[display_cols], hide_index=True)
            else:
                st.warning(f"No eligible colleges found for {selected_branch} with your cutoff mark.")
                
    elif mode == "Show all branches for specific college":
        selected_college = st.selectbox("Select College", unique_colleges)
        
        # Show all branches for selected college
        college_branches = df[df['COLLEGE NAME'] == selected_college].copy()
        
        if not college_branches.empty:
            st.write(f"### Branches Available at {selected_college}")
            
            # Create a more readable display of cutoffs by community
            display_data = college_branches.copy()
            for community in communities:
                display_data[f"{community} Cutoff"] = pd.to_numeric(display_data[community], errors='coerce')
            
            display_cols = ['BRANCH NAME', 'BRANCH CODE'] + [f"{comm} Cutoff" for comm in communities]
            st.dataframe(display_data[display_cols], hide_index=True)
        else:
            st.warning("No data available for selected college.")

    # Add How to Use box with white background
        st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div class='info-box'>
        <h3>How to use:</h3>
        <ol>
            <li>Enter your marks in the sidebar</li>
            <li>Click CALCULATE CUTOFF</li>
            <li>Choose a filtering mode</li>
            <li>Select required filters</li>
            <li>View eligible colleges and branches</li>
        </ol>
        </div>
    """, unsafe_allow_html=True)
    
else:
    st.error("Unable to load college data. Please check if the CSV file exists and is correctly formatted.")