import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import plotly.io as pio

# Read the Parquet file
df = pd.read_parquet('230P_PS1_data.parquet')

# Remove outliers from macro_state_1
q1 = df['macro_state_1'].quantile(0.25)
q3 = df['macro_state_1'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['macro_state_1'] >= lower_bound) & (df['macro_state_1'] <= upper_bound)]

# Calculate average outcome for each combination of category and innovation_success
avg_outcome = df.groupby(['category', 'innovation_success'])['outcome'].mean().reset_index()

# Prepare data for linear model
X = avg_outcome[['category', 'innovation_success']]
y = avg_outcome['outcome']

# Fit the linear model
model = LinearRegression()
model.fit(X, y)

# Predict the outcome on the grid
category_range = np.linspace(X['category'].min(), X['category'].max(), 10)
innovation_range = np.linspace(X['innovation_success'].min(), X['innovation_success'].max(), 10)
category_grid, innovation_grid = np.meshgrid(category_range, innovation_range)
X_grid = np.c_[category_grid.ravel(), innovation_grid.ravel()]
y_pred = model.predict(X_grid).reshape(category_grid.shape)

# Create 3D scatter plot with Plotly
fig = go.Figure()

# Add scatter plot
fig.add_trace(go.Scatter3d(
    x=avg_outcome['category'],
    y=avg_outcome['innovation_success'],
    z=avg_outcome['outcome'],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Data Points'
))

# Add fitted plane
fig.add_trace(go.Surface(
    x=category_grid,
    y=innovation_grid,
    z=y_pred,
    colorscale='Viridis',
    opacity=0.6,
    name='Fitted Plane'
))

# Update layout
fig.update_layout(
    title='3D Scatter Plot with Fitted Plane',
    scene=dict(
        xaxis_title='Category',
        yaxis_title='Innovation Success',
        zaxis_title='Average Outcome'
    )
)

# Save the plot as an HTML file
pio.write_html(fig, file='3d_scatter_plot_with_plane.html', auto_open=True)

# Show the plot
fig.show()

# Display model coefficients and intercept
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)

# Calculate mean and std for outcome grouped by category and year
category_year_stats = df.groupby(['category', 'year']).agg(
    mean_outcome=('outcome', 'mean'),
    std_outcome=('outcome', 'std')
).reset_index()

# Calculate mean and std for outcome grouped by innovation_success and year
innovation_year_stats = df.groupby(['innovation_success', 'year']).agg(
    mean_outcome=('outcome', 'mean'),
    std_outcome=('outcome', 'std')
).reset_index()

# Create bar plots for category-year statistics
fig_category = go.Figure()

for year in df['year'].unique():
    subset = category_year_stats[category_year_stats['year'] == year]
    fig_category.add_trace(go.Bar(
        x=subset['category'],
        y=subset['mean_outcome'],
        error_y=dict(type='data', array=subset['std_outcome']),
        name=f'Year {year}'
    ))

# Update layout for category-year bar plot
fig_category.update_layout(
    title='Mean and Std of Outcome by Category and Year',
    xaxis_title='Category',
    yaxis_title='Outcome',
    barmode='group'
)

# Save the category-year bar plot as an HTML file
pio.write_html(fig_category, file='category_year_bar_plot.html', auto_open=True)

# Show the category-year bar plot
fig_category.show()

# Create bar plots for innovation-year statistics
fig_innovation = go.Figure()

for year in df['year'].unique():
    subset = innovation_year_stats[innovation_year_stats['year'] == year]
    fig_innovation.add_trace(go.Bar(
        x=subset['innovation_success'],
        y=subset['mean_outcome'],
        error_y=dict(type='data', array=subset['std_outcome']),
        name=f'Year {year}'
    ))

# Update layout for innovation-year bar plot
fig_innovation.update_layout(
    title='Mean and Std of Outcome by Innovation Success and Year',
    xaxis_title='Innovation Success',
    yaxis_title='Outcome',
    barmode='group'
)

# Save the innovation-year bar plot as an HTML file
pio.write_html(fig_innovation, file='innovation_year_bar_plot.html', auto_open=True)

# Show the innovation-year bar plot
fig_innovation.show()

# Calculate mean and std for macro_state_1 grouped by year
macro_state_1_year_stats = df.groupby('year').agg(
    mean_macro_state_1=('macro_state_1', 'mean'),
    std_macro_state_1=('macro_state_1', 'std')
).reset_index()

# Calculate mean and std for macro_state_2 grouped by year
macro_state_2_year_stats = df.groupby('year').agg(
    mean_macro_state_2=('macro_state_2', 'mean'),
    std_macro_state_2=('macro_state_2', 'std')
).reset_index()

# Create bar plot for macro_state_1 by year
fig_macro_state_1 = go.Figure()

fig_macro_state_1.add_trace(go.Bar(
    x=macro_state_1_year_stats['year'],
    y=macro_state_1_year_stats['mean_macro_state_1'],
    error_y=dict(type='data', array=macro_state_1_year_stats['std_macro_state_1']),
    name='Macro State 1'
))

# Update layout for macro_state_1 bar plot
fig_macro_state_1.update_layout(
    title='Mean and Std of Macro State 1 by Year',
    xaxis_title='Year',
    yaxis_title='Macro State 1'
)

# Save the macro_state_1 bar plot as an HTML file
pio.write_html(fig_macro_state_1, file='macro_state_1_year_bar_plot.html', auto_open=True)

# Show the macro_state_1 bar plot
fig_macro_state_1.show()

# Create bar plot for macro_state_2 by year
fig_macro_state_2 = go.Figure()

fig_macro_state_2.add_trace(go.Bar(
    x=macro_state_2_year_stats['year'],
    y=macro_state_2_year_stats['mean_macro_state_2'],
    error_y=dict(type='data', array=macro_state_2_year_stats['std_macro_state_2']),
    name='Macro State 2'
))

# Update layout for macro_state_2 bar plot
fig_macro_state_2.update_layout(
    title='Mean and Std of Macro State 2 by Year',
    xaxis_title='Year',
    yaxis_title='Macro State 2'
)

# Save the macro_state_2 bar plot as an HTML file
pio.write_html(fig_macro_state_2, file='macro_state_2_year_bar_plot.html', auto_open=True)

# Show the macro_state_2 bar plot
fig_macro_state_2.show()
