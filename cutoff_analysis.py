import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data for course trends
trends_data = {
    'year': [2019, 2020, 2021, 2022, 2023],
    'Computer Science': [185.5, 188.0, 190.5, 192.0, 187.5],
    'Electronics': [191.5, 189.5, 192.5, 193.0, 191.5],
    'Mechanical': [197.5, 195.0, 193.5, 190.0, 197.5],
    'Civil': [151.5, 155.0, 158.5, 160.0, 151.5]
}

# Category data
category_data = {
    'Category': ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA'],
    'Cutoff': [197.5, 190.5, 144.0, 183.5, 151.5, 137.0],
    'Course': ['Mechanical', 'Civil (TM)', 'Material Science', 
              'Computer Science', 'Electronics', 'Industrial']
}

# Create DataFrames
trends_df = pd.DataFrame(trends_data)
category_df = pd.DataFrame(category_data)

# Create subplots with more vertical spacing
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        '<b>Engineering Course Cutoff Trends (2019-2023)</b>',
        '<b>Category-wise Cutoff Distribution (2023)</b>'
    ),
    vertical_spacing=0.3,
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)

# Enhanced color palette
colors = {
    'Computer Science': '#FF6B6B',  # Coral Red
    'Electronics': '#4ECDC4',       # Turquoise
    'Mechanical': '#45B7D1',        # Sky Blue
    'Civil': '#96CEB4'             # Sage Green
}

# First plot: Course trends with enhanced styling
for course in colors.keys():
    fig.add_trace(
        go.Scatter(
            x=trends_df['year'],
            y=trends_df[course],
            name=course,
            line=dict(
                color=colors[course],
                width=4,
                dash='solid'
            ),
            mode='lines+markers',
            marker=dict(
                size=12,
                symbol='circle',
                line=dict(
                    color='white',
                    width=2
                )
            ),
            hovertemplate=(
                '<b>Course:</b> ' + course +
                '<br><b>Year:</b> %{x}' +
                '<br><b>Cutoff:</b> %{y:.1f}' +
                '<extra></extra>'
            )
        ),
        row=1, col=1
    )

# Second plot: Category analysis with enhanced styling
fig.add_trace(
    go.Bar(
        x=category_df['Category'],
        y=category_df['Cutoff'],
        marker_color=[colors[course.split()[0]] if course.split()[0] in colors else '#117A65' 
                     for course in category_df['Course']],
        text=category_df['Cutoff'].round(1),
        textposition='auto',
        textfont=dict(size=14),
        width=0.6,
        customdata=category_df['Course'],
        hovertemplate=(
            '<b>Category:</b> %{x}' +
            '<br><b>Cutoff:</b> %{y:.1f}' +
            '<br><b>Course:</b> %{customdata}' +
            '<extra></extra>'
        ),
        name='Category Cutoffs'
    ),
    row=2, col=1
)

# Update layout with enhanced styling
fig.update_layout(
    height=1400,  # Increased height
    title=dict(
        text='<b>Engineering Cutoff Analysis (2019-2023)</b>',
        x=0.5,
        y=0.98,
        xanchor='center',
        yanchor='top',
        font=dict(
            size=28,
            color='#2C3E50'
        )
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='#CCCCCC',
        borderwidth=1
    ),
    template="plotly_white",
    hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Arial"
    ),
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Update axes with enhanced styling
for i in [1, 2]:
    # Y-axes
    fig.update_yaxes(
        title_text="Cutoff Marks",
        range=[130, 200],
        row=i,
        col=1,
        tickfont=dict(size=12),
        title_font=dict(size=14),
        gridcolor='#E5E5E5',
        showgrid=True,
        zeroline=False,
        tickformat='.1f'
    )
    
    # X-axes
    fig.update_xaxes(
        title_text="Year" if i == 1 else "Category",
        row=i,
        col=1,
        tickfont=dict(size=12),
        title_font=dict(size=14),
        gridcolor='#E5E5E5',
        showgrid=True,
        zeroline=False
    )

# Add timestamp to avoid file clash
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save interactive HTML with timestamp
html_file = f"engineering_cutoff_analysis_{timestamp}.html"
fig.write_html(html_file)

# Save static image with timestamp
png_file = f"engineering_cutoff_analysis_{timestamp}.png"
fig.write_image(png_file, width=1500, height=1700)

# Show the plot
fig.show()

print("\nAnalysis completed! Files saved with timestamp:")
print(f"1. {html_file} (Interactive)")
print(f"2. {png_file} (Static)")