import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
import pickle

# Read the Parquet file
df = pd.read_parquet('230P_PS1_data.parquet')

# Identify and save outliers for macro_state_1
q1 = df['macro_state_1'].quantile(0.25)
q3 = df['macro_state_1'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_macro_state_1 = df[(df['macro_state_1'] < lower_bound) | (df['macro_state_1'] > upper_bound)]

# Save the outliers
outliers_macro_state_1.to_csv('outliers_macro_state_1.csv', index=False)

# Replace outliers in macro_state_1 with the median
median_macro_state_1 = df['macro_state_1'].median()
df.loc[
    (df['macro_state_1'] < lower_bound) | (df['macro_state_1'] > upper_bound), 'macro_state_1'] = median_macro_state_1

# Identify and save outliers for macro_state_2
q1 = df['macro_state_2'].quantile(0.25)
q3 = df['macro_state_2'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_macro_state_2 = df[(df['macro_state_2'] < lower_bound) | (df['macro_state_2'] > upper_bound)]

# Save the outliers
outliers_macro_state_2.to_csv('outliers_macro_state_2.csv', index=False)

# Replace outliers in macro_state_2 with the median
median_macro_state_2 = df['macro_state_2'].median()
df.loc[
    (df['macro_state_2'] < lower_bound) | (df['macro_state_2'] > upper_bound), 'macro_state_2'] = median_macro_state_2

# Transform macro_state_1 and macro_state_2 with polynomial features including bias
poly = PolynomialFeatures(degree=2, include_bias=True)

macro_state_1_poly = poly.fit_transform(df[['macro_state_1']])
macro_state_2_poly = poly.fit_transform(df[['macro_state_2']])

# Extract the required polynomial terms
macro_state_1_poly_terms = macro_state_1_poly[:, :3]  # 1, x, x^2
macro_state_2_poly_terms = macro_state_2_poly[:, :3]  # 1, x, x^2

# Prepare the final DataFrame
df['macro_state_1_1'] = macro_state_1_poly_terms[:, 1]
df['macro_state_1_2'] = macro_state_1_poly_terms[:, 2]
df['macro_state_2_1'] = macro_state_2_poly_terms[:, 1]
df['macro_state_2_2'] = macro_state_2_poly_terms[:, 2]

# Possible combinations of input variables
combinations = {
    'Combination 1': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'innovation_success',
                      'category', 'year'],
    'Combination 2': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'innovation_success',
                      'year'],
    'Combination 3': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'innovation_success',
                      'category'],
    'Combination 4': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'year', 'category'],
    'Combination 5': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'innovation_success'],
    'Combination 6': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'category'],
    'Combination 7': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'year'],
    'Combination 8': ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2']
}

# All possible features
all_features = ['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'innovation_success',
                'category', 'year']

results = []

for name, features in combinations.items():
    # Prepare data for linear model
    X = df[features]
    y = df['outcome']

    # Build the linear model
    model = LinearRegression()

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mean_cv_score = np.mean(cv_scores)

    # Fit the model on the entire dataset
    model.fit(X, y)

    # Display the model coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    # Create a dictionary of coefficients
    coefficients_dict = {'Intercept': intercept}
    for feature, coefficient in zip(features, coefficients):
        coefficients_dict[feature] = coefficient

    # Save the results
    result = {
        'Combination': name,
        'Average CV R²': mean_cv_score
    }
    for feature in ['Intercept'] + all_features:
        result[feature] = coefficients_dict.get(feature, '')

    results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print the results
print(results_df)

# Save the results to a CSV file
results_df.to_csv('model_comparison_results.csv', index=False)

# Save the model parameters for the best combination
best_combination = results_df.loc[results_df['Average CV R²'].idxmax()]

# Save the best model parameters
model_params = {
    'intercept': best_combination['Intercept'],
    'coefficients': [best_combination[feature] for feature in all_features if best_combination[feature] != '']
}
with open('linear_model_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)

# Save the polynomial feature names
poly_feature_names = {
    'macro_state_1': ['macro_state_1', 'macro_state_1^2'],
    'macro_state_2': ['macro_state_2', 'macro_state_2^2']
}
with open('poly_feature_names.pkl', 'wb') as f:
    pickle.dump(poly_feature_names, f)
