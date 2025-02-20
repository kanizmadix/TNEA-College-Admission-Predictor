import pandas as pd
import plotly.express as px

# Load data
file_path = "C:\\Users\\Kaniz\\Pictures\\TNEA_fct\\Vocational_2023_Mark_Cutoff.csv"
df = pd.read_csv(file_path)

# Debugging: Print column names to verify
print("Columns in the dataset:", df.columns)

# Check if 'RANCH NAME' exists
if 'RANCH NAME' not in df.columns:
    raise KeyError("Column 'RANCH NAME' not found in the dataset. Please check the column names.")

# Group data by college and branch, then calculate mean cutoff for each category
college_branch_df = df.groupby(['COLLEGE NAME', 'RANCH NAME'])[['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']].mean().reset_index()

# Sort by OC cutoff and select top 5 colleges with their respective branches
top_colleges_branches = college_branch_df.sort_values(by='OC', ascending=False).head(5)

# Create a 3D scatter plot with custom design
fig = px.scatter_3d(
    top_colleges_branches,
    x='COLLEGE NAME',  # College names on the x-axis
    y='RANCH NAME',    # Branch names on the y-axis
    z='OC',            # OC cutoff on the z-axis
    color='OC',        # Color by OC cutoff for better visualization
    size='OC',         # Size of markers based on OC cutoff
    text='COLLEGE NAME',  # Display college names as labels
    title='<b>Top 5 Colleges with Highest Cutoff (OC) by Branch</b><br><i>Interactive 3D Visualization</i>',
    labels={'OC': 'OC Cutoff', 'RANCH NAME': 'Branch Name', 'COLLEGE NAME': 'College Name'},
    height=900,  # Adjust height of the chart
    color_continuous_scale=px.colors.sequential.Plasma,  # Use a vibrant gradient
    opacity=0.8,  # Adjust marker opacity
)

# Update layout for better readability and design
fig.update_layout(
    scene=dict(
        xaxis=dict(title='<b>College Name</b>', tickangle=45, title_font=dict(size=14)),
        yaxis=dict(title='<b>Branch Name</b>', title_font=dict(size=14)),
        zaxis=dict(title='<b>OC Cutoff</b>', title_font=dict(size=14)),
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),  # Adjust camera view for better perspective
    ),
    margin=dict(l=0, r=0, b=0, t=80),  # Adjust margins
    title_font=dict(size=20, family='Arial', color='darkblue'),  # Customize title font
    font=dict(family='Arial', size=12, color='black'),  # Customize global font
    coloraxis_colorbar=dict(title='<b>OC Cutoff</b>', title_font=dict(size=14)),  # Customize color bar
)

# Add annotations for better context
# Add annotations for better context
for i, row in top_colleges_branches.iterrows():
    fig.add_annotation(
        x=row['COLLEGE NAME'],
        y=row['RANCH NAME'],
        text=f"{row['COLLEGE NAME']}<br>{row['RANCH NAME']}<br>OC: {row['OC']:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        ax=20,
        ay=-40,
        font=dict(size=10, color='black'),
    )

# Customize marker style
fig.update_traces(
    marker=dict(
        symbol='diamond',  # Use diamond-shaped markers
        size=10,  # Adjust marker size
        line=dict(width=2, color='black'),  # Add a border to markers
    ),
    selector=dict(mode='markers+text'),
)

# Show the 3D chart
fig.show()