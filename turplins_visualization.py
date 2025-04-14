import plotly.express as px

# Create a grouped bar chart
fig = px.bar(
    IDS_2,
    x='pre_village',
    y='turplins_tools_cat',
    color='turplins_tools_cat',
    title='Distribution of Turplins Tools Categories by Village',
    labels={
        'pre_village': 'Village',
        'turplins_tools_cat': 'Category'
    },
    color_discrete_sequence=px.colors.qualitative.Set3
)

# Update layout for better visualization
fig.update_layout(
    height=800,
    width=1200,
    margin=dict(l=50, r=50, t=50, b=50),
    xaxis_title='Village',
    yaxis_title='Turplins Tools Category',
    showlegend=True,
    legend_title='Category',
    xaxis=dict(
        tickangle=45,  # Rotate x-axis labels for better readability
        tickfont=dict(size=8)  # Smaller font size for village names
    )
)

# Add hover information
fig.update_traces(
    hovertemplate='<b>Village:</b> %{x}<br><b>Category:</b> %{y}<extra></extra>'
)

fig.show() 