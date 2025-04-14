import plotly.express as px
from plotly.graph_objects import FigureWidget

# Assuming IDS_raw is your DataFrame

# Method 1: Using rangeslider (already implemented)
fig1 = px.bar(
    IDS_raw,
    x='pre_village',
    y='hot_pepper_tins_organic_pesticides',
    title='Method 1: Using Rangeslider',
    labels={'pre_village': 'Pre-village', 'hot_pepper_tins_organic_pesticides': 'Hot Pepper Tins Organic Pesticides'},
)
fig1.update_layout(
    height=400,
    width=800,
    xaxis=dict(
        rangeslider=dict(visible=True),
        type='category'
    ),
    margin=dict(l=50, r=50, t=50, b=50)
)

# Method 2: Using FigureWidget for interactive scrolling
fig2 = FigureWidget(px.bar(
    IDS_raw,
    x='pre_village',
    y='hot_pepper_tins_organic_pesticides',
    title='Method 2: Using FigureWidget',
    labels={'pre_village': 'Pre-village', 'hot_pepper_tins_organic_pesticides': 'Hot Pepper Tins Organic Pesticides'},
))
fig2.update_layout(
    height=400,
    width=800,
    margin=dict(l=50, r=50, t=50, b=50)
)

# Method 3: Using a fixed container with scroll
fig3 = px.bar(
    IDS_raw,
    x='pre_village',
    y='hot_pepper_tins_organic_pesticides',
    title='Method 3: Fixed Container with Scroll',
    labels={'pre_village': 'Pre-village', 'hot_pepper_tins_organic_pesticides': 'Hot Pepper Tins Organic Pesticides'},
)
fig3.update_layout(
    height=400,
    width=800,
    xaxis=dict(
        fixedrange=False,  # Allow zooming
        type='category'
    ),
    margin=dict(l=50, r=50, t=50, b=50)
)

# Method 4: Using a horizontal scroll with fixed width
fig4 = px.bar(
    IDS_raw,
    x='pre_village',
    y='hot_pepper_tins_organic_pesticides',
    title='Method 4: Horizontal Scroll',
    labels={'pre_village': 'Pre-village', 'hot_pepper_tins_organic_pesticides': 'Hot Pepper Tins Organic Pesticides'},
)
fig4.update_layout(
    height=400,
    width=1200,  # Wider width
    xaxis=dict(
        fixedrange=False,
        type='category'
    ),
    margin=dict(l=50, r=50, t=50, b=50)
)

# Show all figures
fig1.show()
fig2.show()
fig3.show()
fig4.show() 