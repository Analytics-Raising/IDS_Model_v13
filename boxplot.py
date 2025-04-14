import plotly.express as px

# Create boxplot
fig = px.box(
    IDS_raw,
    y='hot_pepper_tins_organic_pesticides',
    title='Distribution of Hot Pepper Tins Organic Pesticides',
    labels={'hot_pepper_tins_organic_pesticides': 'Hot Pepper Tins Organic Pesticides'},
)

# Update layout for better visualization
fig.update_layout(
    height=600,
    width=800,
    margin=dict(l=50, r=50, t=50, b=50),
    yaxis_title='Number of Hot Pepper Tins',
    showlegend=False
)

# Add hover information
fig.update_traces(
    boxpoints='all',  # Show all points
    jitter=0.3,       # Add some jitter to the points
    pointpos=-1.8,    # Position of the points
    hoverinfo='y'     # Show y value on hover
)

fig.show() 