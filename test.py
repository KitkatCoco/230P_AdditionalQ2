import pandas as pd
import numpy as np

# Hard-coded model parameters
intercept = 123456.78  # Replace with the actual intercept value
coefficients = [1.23, 4.56, -7.89, 0.12, -3.45, 6.78]  # Replace with the actual coefficients

# Function to apply polynomial transformations
def transform_macro_state(x):
    return np.column_stack((x, x**2))

# Function to apply transformations and make predictions
def predict_outcome(df):
    macro_state_1_poly = transform_macro_state(df['macro_state_1'])
    macro_state_2_poly = transform_macro_state(df['macro_state_2'])

    df['macro_state_1_1'] = macro_state_1_poly[:, 0]
    df['macro_state_1_2'] = macro_state_1_poly[:, 1]
    df['macro_state_2_1'] = macro_state_2_poly[:, 0]
    df['macro_state_2_2'] = macro_state_2_poly[:, 1]

    # Prepare the feature matrix
    X = df[['macro_state_1_1', 'macro_state_1_2', 'macro_state_2_1', 'macro_state_2_2', 'innovation_success', 'category']]

    # Make predictions
    predictions = intercept + np.dot(X, coefficients)
    return predictions

# Function to load data and predict
def main(input_csv):
    df = pd.read_csv(input_csv)
    predictions = predict_outcome(df)
    df['predicted_outcome'] = predictions
    return df

# Example usage
if __name__ == "__main__":
    input_csv = 'test_data.csv'  # Replace with your test data file
    df_predictions = main(input_csv)
    df_predictions.to_csv('predicted_outcomes.csv', index=False)
    print(df_predictions.head())
