import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Read the Parquet file
df = pd.read_parquet('230P_PS1_data.parquet')

# Replace outliers in macro_state_1 with the median
q1 = df['macro_state_1'].quantile(0.25)
q3 = df['macro_state_1'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
median_macro_state_1 = df['macro_state_1'].median()
df.loc[(df['macro_state_1'] < lower_bound) | (df['macro_state_1'] > upper_bound), 'macro_state_1'] = median_macro_state_1

# Replace outliers in macro_state_2 with the median
q1 = df['macro_state_2'].quantile(0.25)
q3 = df['macro_state_2'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
median_macro_state_2 = df['macro_state_2'].median()
df.loc[(df['macro_state_2'] < lower_bound) | (df['macro_state_2'] > upper_bound), 'macro_state_2'] = median_macro_state_2

# Display the DataFrame information after outlier replacement
print(df.info())

# Initialize a DataFrame to store the model summaries
model_summaries = pd.DataFrame(columns=['Variable', 'Degree', 'R²', 'Coefficients', 'Intercept'])

# Create directory to save plots
output_dir = 'polynomial_regression_plots'
os.makedirs(output_dir, exist_ok=True)

# Function to plot polynomial regression and summarize the model
def plot_and_summarize_polynomial_regression(data, x_var, y_var, degree, category_col, model_summaries, output_dir):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(data[[x_var]])
    model = LinearRegression()
    model.fit(X_poly, data[y_var])
    x_range = np.linspace(data[x_var].min(), data[x_var].max(), 300)
    x_range_poly = poly.transform(x_range.reshape(-1, 1))
    y_range = model.predict(x_range_poly)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_var, y=y_var, hue=category_col, palette='viridis')
    plt.plot(x_range, y_range, color='red', label=f'Polynomial Degree {degree}')
    plt.title(f'Polynomial Regression (Degree {degree}) of {x_var} vs {y_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{x_var}_degree_{degree}.png'))
    plt.close()

    # Model summary
    r2 = model.score(X_poly, data[y_var])
    coefficients = model.coef_
    intercept = model.intercept_

    # Append the summary to the DataFrame
    summary = pd.DataFrame({
        'Variable': [x_var],
        'Degree': [degree],
        'R²': [r2],
        'Coefficients': [coefficients],
        'Intercept': [intercept]
    })
    model_summaries = pd.concat([model_summaries, summary], ignore_index=True)

    return model_summaries, model, poly


# Visualize polynomial regression for different degrees and categorize by 'category'
model_summaries, model_1, poly_1 = plot_and_summarize_polynomial_regression(df, 'macro_state_1', 'outcome', degree=1,
                                                                            category_col='category',
                                                                            model_summaries=model_summaries,
                                                                            output_dir=output_dir)
model_summaries, model_2, poly_2 = plot_and_summarize_polynomial_regression(df, 'macro_state_1', 'outcome', degree=2,
                                                                            category_col='category',
                                                                            model_summaries=model_summaries,
                                                                            output_dir=output_dir)
model_summaries, model_3, poly_3 = plot_and_summarize_polynomial_regression(df, 'macro_state_1', 'outcome', degree=3,
                                                                            category_col='category',
                                                                            model_summaries=model_summaries,
                                                                            output_dir=output_dir)

model_summaries, model_4, poly_4 = plot_and_summarize_polynomial_regression(df, 'macro_state_2', 'outcome', degree=1,
                                                                            category_col='category',
                                                                            model_summaries=model_summaries,
                                                                            output_dir=output_dir)
model_summaries, model_5, poly_5 = plot_and_summarize_polynomial_regression(df, 'macro_state_2', 'outcome', degree=2,
                                                                            category_col='category',
                                                                            model_summaries=model_summaries,
                                                                            output_dir=output_dir)
model_summaries, model_6, poly_6 = plot_and_summarize_polynomial_regression(df, 'macro_state_2', 'outcome', degree=3,
                                                                            category_col='category',
                                                                            model_summaries=model_summaries,
                                                                            output_dir=output_dir)

# Display the model summaries
print(model_summaries)

# Additional plots for engineered features
def plot_engineered_features(data, y_var, features, category_col, models, polys, output_dir):
    for feature, model, poly, degree in zip(features, models, polys, [1, 2, 3, 1, 2, 3]):
        transformed_feature = model.predict(poly.transform(data[[feature]]))
        data[f'{feature}_trans'] = transformed_feature
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x=f'{feature}_trans', y=y_var, hue=category_col, palette='viridis')
        sns.regplot(data=data, x=f'{feature}_trans', y=y_var, scatter=False, color='red', label='Linear Fit')
        plt.title(f'Linear Relation between Transformed {feature} (Degree {degree}) and {y_var}')
        plt.xlabel(f'{feature}_trans')
        plt.ylabel(y_var)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{feature}_transformed_degree_{degree}.png'))
        plt.close()


# Models and polynomial transformers
models = [model_1, model_2, model_3, model_4, model_5, model_6]
polys = [poly_1, poly_2, poly_3, poly_4, poly_5, poly_6]

# Plotting linear relations for transformed features for macro_state_1 and macro_state_2
engineered_features = ['macro_state_1', 'macro_state_1', 'macro_state_1',
                       'macro_state_2', 'macro_state_2', 'macro_state_2']
plot_engineered_features(df, 'outcome', engineered_features, 'category', models, polys, output_dir)
