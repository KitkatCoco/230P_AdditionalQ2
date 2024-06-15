import pandas as pd


def test(input_parquet):

    # read the parquet file
    df = pd.read_parquet(input_parquet)

    # keep the columns that are needed, and the outcome column for R2 calculation
    columns = ['macro_state_1', 'macro_state_2', 'category', 'outcome']
    num_columns = len(columns) - 1
    df = df[columns]

    # Define coefficients from trained model
    intercept = 351.302
    coeff_category = 41.569
    coeff_x1_1 = -168.362
    coeff_x1_2 = 25.925
    coeff_x2_1 = 149.402994
    coeff_x2_2 = -25.1421868
    median_macro_state_1 = 4.994214932023571
    median_macro_state_2 = 5.021826165828092
    lower_bound_macro_state_1 = -1.0606604863376519
    upper_bound_macro_state_1 = 11.06503133597478
    lower_bound_macro_state_2 = -4.896123344053041
    upper_bound_macro_state_2 = 14.986664428211359

    # Replace outliers with the median
    df.loc[(df['macro_state_1'] < lower_bound_macro_state_1) | (
            df['macro_state_1'] > upper_bound_macro_state_1), 'macro_state_1'] = median_macro_state_1
    df.loc[(df['macro_state_2'] < lower_bound_macro_state_2) | (
            df['macro_state_2'] > upper_bound_macro_state_2), 'macro_state_2'] = median_macro_state_2

    # generate Feature1 and Feature2
    df['Feature1'] = coeff_x1_1 * df['macro_state_1'] + coeff_x1_2 * df['macro_state_1'] ** 2
    df['Feature2'] = coeff_x2_1 * df['macro_state_2'] + coeff_x2_2 * df['macro_state_2'] ** 2

    # Function to predict the outcome using the linear model
    df['predicted_outcome'] = intercept + df['Feature1'] + df['Feature2'] + coeff_category * df['category']

    # compute the R2 - between "column" and "predicted_outcome"
    y = df['outcome']
    y_hat = df['predicted_outcome']
    y_bar = y.mean()
    ss_total = ((y - y_bar) ** 2).sum()
    ss_res = ((y - y_hat) ** 2).sum()
    r2 = 1 - (ss_res / ss_total)

    # # Save the DataFrame to csv
    # df.to_csv('predicted_outcome.csv', index=False)

    return num_columns, df['predicted_outcome'], r2


# Example usage
if __name__ == "__main__":

    # path to the parquet file
    input_parquet = '230P_PS1_data.parquet'

    # test the function
    num_columns, df_predicted_outcomes, r2 = test(input_parquet)

    # print the results
    print(f'Number of columns: {num_columns}')
    print(f'R2: {r2}')
