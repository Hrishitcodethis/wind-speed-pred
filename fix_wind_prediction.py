import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load and preprocess data
df = pd.read_csv('/Users/hrishityelchuri/Documents/windPred/raw/8.52 hrishit data.csv')
df['PeriodEnd'] = pd.to_datetime(df['PeriodEnd'])
df['PeriodStart'] = pd.to_datetime(df['PeriodStart'])
df = df.sort_values('PeriodEnd')

def create_multivariate_lagged_dataset(df, target_col, feature_cols, lag=3):
    """
    Create supervised learning data from multivariate time series.
    
    Parameters:
    - df: pandas DataFrame containing time series data for multiple variables
    - target_col: string, name of the target variable column (e.g., 'WindSpeed10m')
    - feature_cols: list of strings, names of feature columns to use
    - lag: number of past time steps to include as input features
    
    Returns:
    - X: 2D NumPy array of shape (samples, features * lag)
    - y: 1D NumPy array of target values (samples,)
    """
    # Get feature data (excluding target column)
    feature_cols_without_target = [col for col in feature_cols if col != target_col]
    feature_data = df[feature_cols_without_target].values
    # Get target data separately
    target_data = df[target_col].values
    
    X, y = [], []
    for i in range(lag, len(df)):
        # extract lagged observations for all features
        X.append(feature_data[i-lag:i].flatten())  # flatten to 1D array of length features*lag
        y.append(target_data[i])
    return np.array(X), np.array(y)

# Set columns you want to use as features
feature_columns = ['AirTemp','Azimuth','CloudOpacity','DewpointTemp','Dhi','Dni','Ebh','WindDirection10m','Ghi','RelativeHumidity','SurfacePressure','WindSpeed10m']
target_column = 'WindSpeed10m'
lag_steps = 3  # number of past hours to use

# Create the dataset
X, y = create_multivariate_lagged_dataset(df, target_column, feature_columns, lag=lag_steps)

print(f"Dataset created successfully!")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split the data
n_samples = X.shape[0]
train_end = int(0.7 * n_samples)
val_end = int(0.85 * n_samples)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Now you can use this data with your RELM model
print("\nData is ready for training! You can now:")
print("1. Train your RELM model with X_train and y_train")
print("2. Validate with X_val and y_val")
print("3. Test with X_test and y_test") 