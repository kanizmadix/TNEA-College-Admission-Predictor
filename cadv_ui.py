import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from cadv_new import EnhancedCollegePredictorML

# Set page configuration with dark theme
st.set_page_config(
    page_title="College Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Black-Red Theme
st.markdown("""
    <style>
    /* Theme Colors */
    :root {
        --primary-black: #000000;
        --secondary-red: #FF0000;
        --accent-blue: #0066cc;
        --pure-white: #FFFFFF;
        --dark-gray: #1E1E1E;
        --light-gray: #2D2D2D;
    }
    
    /* Main Container */
    .main {
        background-color: var(--primary-black);
        color: var(--pure-white);
    }
    
    .stApp {
        background-color: var(--primary-black);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-black) 0%, var(--secondary-red) 100%);
        color: var(--pure-white);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.2);
        border: 2px solid var(--secondary-red);
    }
    
    /* Calculator Card */
    .calculator-card {
        background: var(--dark-gray);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid var(--secondary-red);
    }
    
    .calculator-header {
        color: var(--secondary-red);
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Filter Section */
    .filter-section {
        background: var(--dark-gray);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid var(--secondary-red);
    }
    
    /* Action Buttons */
    .calculate-button {
        background: linear-gradient(135deg, var(--secondary-red) 0%, #990000 100%) !important;
        color: var(--pure-white) !important;
        padding: 1rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.2) !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    .predict-button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #004999 100%) !important;
        color: var(--pure-white) !important;
        padding: 1rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.2) !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        background-color: var(--light-gray) !important;
        color: var(--pure-white) !important;
        border: 1px solid var(--secondary-red) !important;
        border-radius: 10px !important;
    }
    
    .stSelectbox > div > div > select {
        background-color: var(--light-gray) !important;
        color: var(--pure-white) !important;
        border: 1px solid var(--secondary-red) !important;
        border-radius: 10px !important;
    }
    
    /* Results Card */
    .results-card {
        background: var(--dark-gray);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid var(--secondary-red);
    }
    
    /* Prediction Cards */
    .high-chance {
        background: linear-gradient(135deg, var(--accent-blue) 0%, #004999 100%);
        color: var(--pure-white);
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    
    .medium-chance {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        color: var(--pure-white);
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    
    .low-chance {
        background: linear-gradient(135deg, var(--secondary-red) 0%, #990000 100%);
        color: var(--pure-white);
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    
    /* Metrics */
    .metric-card {
        background: var(--light-gray);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--secondary-red);
    }
    
    .stMetric {
        background-color: var(--light-gray) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--dark-gray);
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid var(--secondary-red);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--light-gray);
        border-radius: 8px;
        color: var(--pure-white);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--secondary-red);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the model and data"""
    try:
        category_df = pd.read_csv("college_data.csv")
        max_cutoff_df = pd.read_csv("Unique_Colleges_Max_Cutoff.csv")
        predictor = EnhancedCollegePredictorML()
        metrics = predictor.train_model(max_cutoff_df)
        return predictor, category_df, max_cutoff_df, metrics
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def calculate_cutoff(maths, physics, chemistry):
    """Calculate cutoff mark based on subject marks"""
    return maths + (physics/2) + (chemistry/2)

def calculate_admission_chance(cutoff_diff):
    """Calculate admission chance based on cutoff difference"""
    if cutoff_diff >= 0:
        if cutoff_diff > 10:
            return 99.0, "Guaranteed"
        elif cutoff_diff > 5:
            return 95.0, "Almost Certain"
        else:
            return 90.0, "Excellent"
    else:
        if cutoff_diff >= -5:
            return 70.0, "Good"
        elif cutoff_diff >= -10:
            return 50.0, "Moderate"
        elif cutoff_diff >= -15:
            return 30.0, "Low"
        else:
            return 10.0, "Very Low"

def format_chance(chance, label):
    """Format the admission chance with styling"""
    if chance >= 80:
        return f'<div class="high-chance">{label} ({chance:.1f}%)</div>'
    elif chance >= 50:
        return f'<div class="medium-chance">{label} ({chance:.1f}%)</div>'
    else:
        return f'<div class="low-chance">{label} ({chance:.1f}%)</div>'

def display_predictions(predictions, cutoff_mark):
    """Display predictions with interactive visualizations"""
    if predictions.empty:
        st.warning("No matching colleges found.")
        return

    predictions_sorted = predictions.sort_values('Chance', ascending=False)
    
    # Overview metrics
    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Colleges Found", len(predictions))
    with col2:
        high_chances = len(predictions[predictions['Chance'] >= 80])
        st.metric("High Chance Colleges", high_chances)
    with col3:
        avg_chance = predictions['Chance'].mean()
        st.metric("Average Admission Chance", f"{avg_chance:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # Visualization tabs
    tab1, tab2 = st.tabs(["📊 Admission Chances", "📈 Cutoff Analysis"])

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=predictions_sorted['Chance'][:10],
            y=[f"{col} - {br}" for col, br in zip(
                predictions_sorted['COLLEGE NAME'][:10],
                predictions_sorted['BRANCH NAME'][:10])],
            orientation='h',
            marker=dict(
                color=predictions_sorted['Chance'][:10],
                colorscale='RdYlGn',
                showscale=True
            )
        ))
        
        fig1.update_layout(
            title='Top 10 Recommendations by Admission Chance',
            xaxis_title='Admission Chance (%)',
            yaxis_title='College - Branch',
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            name='Maximum Cutoff',
            x=predictions_sorted['BRANCH NAME'][:10],
            y=predictions_sorted['Max Cutoff'][:10],
            marker_color='royalblue'
        ))
        
        fig2.add_trace(go.Scatter(
            name='Your Cutoff',
            x=predictions_sorted['BRANCH NAME'][:10],
            y=[cutoff_mark] * 10,
            mode='lines',
            line=dict(color='red', dash='dash')
        ))
        
        fig2.update_layout(
            title='Cutoff Comparison - Top 10 Colleges',
            xaxis_title='Branch',
            yaxis_title='Cutoff Mark',
            height=600,
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # Detailed predictions
    st.markdown("### 📋 Detailed College Recommendations")
    for _, row in predictions_sorted.iterrows():
        with st.expander(f"{row['COLLEGE NAME']} - {row['BRANCH NAME']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Maximum Cutoff", f"{row['Max Cutoff']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Cutoff Difference", f"{row['Cutoff Diff']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown(
                    format_chance(row['Chance'], row['Label']),
                    unsafe_allow_html=True
                )

def main():
    st.markdown('<div class="main-header"><h1>🎓 College Admission Predictor</h1></div>', unsafe_allow_html=True)
    
    # Load data and model
    predictor, category_df, max_cutoff_df, metrics = load_model()
    
    if predictor is None:
        return

    # Cutoff Calculator Section
    st.markdown('<div class="calculator-card">', unsafe_allow_html=True)
    st.markdown('<p class="calculator-header">📝 Enter Your Marks</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        maths = st.number_input("Mathematics", 
                               min_value=0.0, 
                               max_value=100.0, 
                               step=0.5,
                               help="Enter your Mathematics mark (max 100)")
    
    with col2:
        physics = st.number_input("Physics", 
                                min_value=0.0, 
                                max_value=100.0, 
                                step=0.5,
                                help="Enter your Physics mark (max 100)")
    
    with col3:
        chemistry = st.number_input("Chemistry", 
                                  min_value=0.0, 
                                  max_value=100.0, 
                                  step=0.5,
                                  help="Enter your Chemistry mark (max 100)")

    cutoff_mark = 0
    if st.button("CALCULATE CUTOFF", key="calc_button", help="Calculate your cutoff mark"):
        cutoff_mark = maths + (physics/2) + (chemistry/2)
        st.markdown(
            f'<div style="background-color: #1E1E1E; padding: 1rem; border-radius: 10px; border: 1px solid #FF0000; margin: 1rem 0;">'
            f'<h3 style="color: #FFFFFF; text-align: center;">Your Calculated Cutoff: {cutoff_mark:.2f}</h3>'
            '</div>',
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if cutoff_mark > 0:  # Only show filter section if cutoff is calculated
        # Filter Section
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.markdown('<p class="calculator-header">🔍 Select Your Filter</p>', unsafe_allow_html=True)
        
        filter_type = st.radio(
            "Filter Type",
            ["Branch Specific", "College Specific", "Category Specific"],
            horizontal=True
        )

        if filter_type == "Branch Specific":
            branch = st.selectbox(
                "Select Branch",
                sorted(max_cutoff_df['BRANCH NAME'].unique())
            )
            filtered_df = max_cutoff_df[max_cutoff_df['BRANCH NAME'] == branch]

        elif filter_type == "College Specific":
            college = st.selectbox(
                "Select College",
                sorted(max_cutoff_df['COLLEGE NAME'].unique())
            )
            filtered_df = max_cutoff_df[max_cutoff_df['COLLEGE NAME'] == college]

        else:  # Category Specific
            category = st.selectbox(
                "Select Category",
                ["OC", "BC", "BCM", "MBC", "SC", "SCA", "ST"]
            )
            filtered_df = max_cutoff_df.copy()

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("GET PREDICTIONS 🎯", type="primary", key="predict_button"):
            with st.spinner("Analyzing your chances... Please wait"):
                predictions = []
                
                for _, row in filtered_df.iterrows():
                    max_cutoff = row['MAX CUTOFF']
                    cutoff_diff = st.session_state.cutoff_mark - max_cutoff
                    chance, label = calculate_admission_chance(cutoff_diff)
                    
                    predictions.append({
                        'COLLEGE NAME': row['COLLEGE NAME'],
                        'BRANCH NAME': row['BRANCH NAME'],
                        'Max Cutoff': max_cutoff,
                        'Cutoff Diff': cutoff_diff,
                        'Chance': chance,
                        'Label': label
                    })
                
                predictions_df = pd.DataFrame(predictions)
                display_predictions(predictions_df, st.session_state.cutoff_mark)

if __name__ == "__main__":
    main()