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

# CSS with fixed positioning for key elements
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
        padding: 20px;
    }
    
    .stApp {
        background-color: var(--primary-black);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-black) 0%, var(--secondary-red) 100%);
        color: var(--pure-white);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.2);
        border: 2px solid var(--secondary-red);
        position: relative;
        z-index: 1000;
    }
    
    /* Calculator Card */
    .calculator-card {
        background: var(--dark-gray);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid var(--secondary-red);
        position: relative;
        z-index: 900;
    }
    
    /* Filter Section */
    .filter-section {
        background: var(--dark-gray);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid var(--secondary-red);
        position: relative;
        z-index: 800;
    }
    
    /* Results Section */
    .results-section {
        background: var(--dark-gray);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid var(--secondary-red);
        position: relative;
        z-index: 700;
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
    
    /* Fixed position container for results */
    .fixed-container {
        position: relative;
        z-index: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'cutoff_mark' not in st.session_state:
    st.session_state.cutoff_mark = 0
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Colleges Found", len(predictions))
    with col2:
        high_chances = len(predictions[predictions['Chance'] >= 80])
        st.metric("High Chance Colleges", high_chances)
    with col3:
        avg_chance = predictions['Chance'].mean()
        st.metric("Average Admission Chance", f"{avg_chance:.1f}%")

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
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
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
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # Detailed predictions
    st.markdown("### 📋 Detailed College Recommendations")
    for _, row in predictions_sorted.iterrows():
        with st.expander(f"{row['COLLEGE NAME']} - {row['BRANCH NAME']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Maximum Cutoff", f"{row['Max Cutoff']:.2f}")
            
            with col2:
                st.metric("Cutoff Difference", f"{row['Cutoff Diff']:.2f}")
            
            with col3:
                st.markdown(
                    format_chance(row['Chance'], row['Label']),
                    unsafe_allow_html=True
                )

def calculate_predictions(filtered_df, cutoff_mark):
    """Calculate predictions for the filtered colleges"""
    predictions = []
    for _, row in filtered_df.iterrows():
        max_cutoff = row['MAX CUTOFF']
        cutoff_diff = cutoff_mark - max_cutoff
        chance, label = calculate_admission_chance(cutoff_diff)
        
        predictions.append({
            'COLLEGE NAME': row['COLLEGE NAME'],
            'BRANCH NAME': row['BRANCH NAME'],
            'Max Cutoff': max_cutoff,
            'Cutoff Diff': cutoff_diff,
            'Chance': chance,
            'Label': label
        })
    
    return pd.DataFrame(predictions)

def main():
    # Load data and model
    predictor, category_df, max_cutoff_df, metrics = load_model()
    
    if predictor is None:
        return

    # Header
    st.markdown('<div class="main-header"><h1>🎓 College Admission Predictor</h1></div>', unsafe_allow_html=True)
    
    # Create a container for the calculator
    calculator_container = st.container()
    
    with calculator_container:
        st.markdown('<div class="calculator-card">', unsafe_allow_html=True)
        st.markdown('### 📝 Enter Your Marks', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            maths = st.number_input("Mathematics", 
                                  min_value=0.0, 
                                  max_value=100.0, 
                                  step=0.5,
                                  value=0.0,
                                  key='maths')
        
        with col2:
            physics = st.number_input("Physics", 
                                    min_value=0.0, 
                                    max_value=100.0, 
                                    step=0.5,
                                    value=0.0,
                                    key='physics')
        
        with col3:
            chemistry = st.number_input("Chemistry", 
                                      min_value=0.0, 
                                      max_value=100.0, 
                                      step=0.5,
                                      value=0.0,
                                      key='chemistry')

        if st.button("CALCULATE CUTOFF", key="calc_button"):
            st.session_state.cutoff_mark = maths + (physics/2) + (chemistry/2)
            st.markdown(
                f'<div style="background-color: #1E1E1E; padding: 1rem; border-radius: 10px; '
                f'border: 1px solid #FF0000; margin: 1rem 0;">'
                f'<h3 style="color: #FFFFFF; text-align: center;">'
                f'Your Calculated Cutoff: {st.session_state.cutoff_mark:.2f}</h3>'
                '</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Create a container for filters and predictions
    if st.session_state.cutoff_mark > 0:
        filter_container = st.container()
        
        with filter_container:
            st.markdown('<div class="filter-section">', unsafe_allow_html=True)
            st.markdown('### 🔍 Select Your Filters', unsafe_allow_html=True)
            
            # Filter Type Selection
            filter_type = st.radio(
                "Filter Type",
                ["Branch Specific", "College Specific", "Category Specific"],
                horizontal=True,
                key='filter_type'
            )
            
            # Filter Options based on Type
            filtered_df = max_cutoff_df.copy()
            
            if filter_type == "Branch Specific":
                selected_filter = st.selectbox(
                    "Select Branch",
                    sorted(filtered_df['BRANCH NAME'].unique()),
                    key='branch_select'
                )
                filtered_df = filtered_df[filtered_df['BRANCH NAME'] == selected_filter]
                
            elif filter_type == "College Specific":
                selected_filter = st.selectbox(
                    "Select College",
                    sorted(filtered_df['COLLEGE NAME'].unique()),
                    key='college_select'
                )
                filtered_df = filtered_df[filtered_df['COLLEGE NAME'] == selected_filter]
                
            else:  # Category Specific
                selected_filter = st.selectbox(
                    "Select Category",
                    ["OC", "BC", "BCM", "MBC", "SC", "SCA", "ST"],
                    key='category_select'
                )
            
            if st.button("GET PREDICTIONS 🎯", type="primary", key="predict_button"):
                with st.spinner("Analyzing your chances... Please wait"):
                    st.session_state.predictions_df = calculate_predictions(
                        filtered_df, 
                        st.session_state.cutoff_mark
                    )
                    st.session_state.show_predictions = True
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Create a container for results
        # Create a container for results
        if st.session_state.show_predictions and st.session_state.predictions_df is not None:
            results_container = st.container()
            
            with results_container:
                st.markdown('<div class="results-section">', unsafe_allow_html=True)
                display_predictions(st.session_state.predictions_df, st.session_state.cutoff_mark)
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()