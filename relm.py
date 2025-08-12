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
    
    def _activation(self, x):
        if self.activation == 'sigmoid':
            # Clip input to a safe range to avoid overflow
            x = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x))
