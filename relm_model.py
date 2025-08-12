import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RELM:
    def __init__(self, n_hidden, C=1.0, activation='sigmoid'):
        self.n_hidden = n_hidden
        self.C = C
        self.activation = activation
        self.input_weights = None
        self.bias = None
        self.beta = None

    def _activation(self, x):
        if self.activation == 'sigmoid':
            # Add clipping to prevent overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'sin':
            return np.sin(x)
        elif self.activation == 'hardlim':
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError("Unsupported activation function")

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.input_weights = np.random.randn(n_features, self.n_hidden)
        self.bias = np.random.randn(self.n_hidden)

        H = self._activation(np.dot(X, self.input_weights) + self.bias)

        if n_samples >= self.n_hidden:
            I = np.identity(self.n_hidden)
            self.beta = np.dot(np.linalg.inv(H.T @ H + I / self.C), H.T @ y)
        else:
            I = np.identity(n_samples)
            self.beta = H.T @ np.linalg.inv(H @ H.T + I / self.C) @ y

    def predict(self, X):
        H = self._activation(np.dot(X, self.input_weights) + self.bias)
        return np.dot(H, self.beta)

def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true, y_pred, set_name="Test"):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    
    print(f"{set_name} Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape_val:.2f}%")
    print("-" * 30)
    
    return mae, rmse, mape_val

if __name__ == "__main__":
    # Import the fixed dataset
    from fix_wind_prediction import X_train, y_train, X_val, y_val, X_test, y_test
    
    print("Training RELM model...")
    
    # Create and train the model
    model = RELM(n_hidden=50, C=1.0, activation='sigmoid')
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Evaluate on all sets
    print("\nModel Performance:")
    evaluate_model(y_train, y_pred_train, "Training")
    evaluate_model(y_val, y_pred_val, "Validation")
    evaluate_model(y_test, y_pred_test, "Test") 