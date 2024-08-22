import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy import stats

Realslope = np.random.rand()

class LAD:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_outputs = y.shape[1] if y.ndim > 1 else 1
        
        self.coef_ = np.zeros((n_features, n_outputs))
        self.intercept_ = np.zeros(n_outputs)
        
        for i in range(n_outputs):
            y_i = y[:, i] if y.ndim > 1 else y
            
            def objective(params):
                return np.sum(np.abs(y_i - np.dot(X, params[:-1]) - params[-1]))
            
            initial_guess = np.concatenate([np.linalg.lstsq(X, y_i, rcond=None)[0], [0]])
            result = minimize(objective, initial_guess, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.success:
                self.coef_[:, i] = result.x[:-1]
                self.intercept_[i] = result.x[-1]
            else:
                print(f"LAD optimization warning for output {i}. Message: {result.message}")
                print(f"Using the best solution found. Optimization status: {result.status}")
                self.coef_[:, i] = result.x[:-1]
                self.intercept_[i] = result.x[-1]

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

class RLS:


    def __init__(self, num_vars, num_outputs, lam=0.9999999999, delta=100000000000):
        self.num_vars = num_vars
        self.num_outputs = num_outputs
        self.lam = lam
        self.P = delta * np.eye(self.num_vars)
        self.w = np.zeros((self.num_vars, self.num_outputs))

    def fit(self, X, Y):
        for i in range(len(X)):
            self.update(X[i], Y[i])

    def update(self, x, y):
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)
        
        e = y - self.w.T @ x
        g = self.P @ x / (self.lam + x.T @ self.P @ x)
        self.w += g @ e.T
        self.P = (self.P - g @ x.T @ self.P) / self.lam

    def predict(self, X):
        return X @ self.w

def generate_random_points(num_points, input_dims=1, output_dims=1, x_range=(0, 2), noise=0.25):
    X = np.random.uniform(x_range[0], x_range[1], (num_points, input_dims))
    Y = np.dot(X, Realslope)
    Y += np.random.normal(0, noise, (num_points, output_dims))
    return X, Y

def generate_random_outliers(num_points, input_dims=1, output_dims=1, x_range=(0, 2), noise=0.25, outlier_ratio=0.8):

    X, Y = generate_random_points(num_points, input_dims, output_dims, x_range, noise)
    num_outliers = int(num_points * outlier_ratio)

    outlier_indices = np.random.choice(num_points, size=num_outliers, replace=False)
    Y[outlier_indices] += np.random.uniform(low=-5, high=5, size=(num_outliers, output_dims))
    return X,Y

def generate_random_zeros(num_points, input_dims=1, output_dims=1, x_range=(0, 2), noise=0.25):
    X = np.random.uniform(x_range[0], x_range[1], (num_points, input_dims))
    Y = np.zeros((num_points, output_dims))
    return X, Y

def save_data_to_csv(X, Y, filename):
    data_df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(X.shape[1])])
    data_df = pd.concat([data_df, pd.DataFrame(Y, columns=[f'y{i+1}' for i in range(Y.shape[1])])], axis=1)
    data_df.to_csv(filename, index=False)

def calculate_errors(Y_true, Y_pred):
    mse = mean_squared_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)
    return mse, rmse

def calculate_confidence_interval(Y_true, Y_pred, confidence=0.95):
    residuals = Y_true - Y_pred
    mean_residual = np.mean(residuals, axis=0)  # Mean residual for each output
    se_residual = stats.sem(residuals, axis=0)  # Standard error of the residuals
    ci = se_residual * stats.t.ppf((1 + confidence) / 2., residuals.shape[0] - 1)
    return mean_residual - ci, mean_residual + ci

def run_regression(X_train, Y_train, X_test, Y_test, n_components=1):
    # Train models on training data
    reg_linear = LinearRegression(n_jobs=-1).fit(X_train, Y_train)
    reg_ridge = Ridge(alpha=1.0, solver='auto').fit(X_train, Y_train)
    reg_rls = RLS(num_vars=X_train.shape[1], num_outputs=Y_train.shape[1])
    reg_rls.fit(X_train, Y_train)
    reg_pls = PLSRegression(n_components=n_components)
    reg_pls.fit(X_train, Y_train)
    reg_lad = LAD()
    reg_lad.fit(X_train, Y_train)

    # Predict on test data
    Y_pred_linear = reg_linear.predict(X_test)
    Y_pred_ridge = reg_ridge.predict(X_test)
    Y_pred_rls = reg_rls.predict(X_test)
    Y_pred_pls = reg_pls.predict(X_test)
    Y_pred_lad = reg_lad.predict(X_test)

    plt.figure(figsize=(13, 8.8))
    plt.scatter(X_test[:, 0], Y_test[:, 0], color='blue', alpha=0.5, label='Actual')
    
    plt.plot (Realslope, color='blue', linestyle='--', label='Actual Line')

    models = {
        "LLS": Y_pred_linear,
        "Ridge": Y_pred_ridge,
        "RLS": Y_pred_rls,
        "PLS": Y_pred_pls,
        "LAD": Y_pred_lad
    }

    mse_results = {}
    
    for name, Y_pred in models.items():
        mse, rmse = calculate_errors(Y_test, Y_pred)
        ci_low, ci_high = calculate_confidence_interval(Y_test, Y_pred)
        mse_results[name] = mse  # Store MSE for comparison
        print(f"{name} - MSE: {mse:.20f}, RMSE: {rmse:.20f}")
        print(f"{name} - 95% Confidence Interval:")
        for dim in range(Y_test.shape[1]):
            print(f"  Output {dim + 1}: [{ci_low[dim]:.20f}, {ci_high[dim]:.20f}]")

    # Determine the best model based on the lowest MSE
    best_model = min(mse_results, key=mse_results.get)
    print("\nBest Model: ", best_model)
    
    # Save training and test data
    save_data_to_csv(X_train, Y_train, 'training_data.csv')
    save_data_to_csv(X_test, Y_test, 'testing_data.csv')

    # Plot predictions
    for name, Y_pred in models.items():
        slope, intercept = np.polyfit(X_test[:,0], Y_pred[:, 0], 1)
        line_x = np.array([X_test.min(), X_test.max()])
        line_y = slope * line_x + intercept
        plt.scatter(X_test, Y_pred[:, 0], alpha=0.5, label=name)
        plt.plot(line_x, line_y, linestyle='-', label=f'{name} Line')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Regression Methods')
    plt.legend()
    plt.grid(True)

    equations = []
    for name, Y_pred in models.items():
        slope, intercept = np.polyfit(X_test[:,0], Y_pred[:, 0], 1)
        equation = f"{name}: y = {slope:.2f}*x + {intercept:.2f}"
        equations.append(equation)

    equation_text = '\n'.join(equations)
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    #plt.show()

# Generate a large dataset
X, Y = generate_random_points(num_points=2_000_000)
#X, Y = generate_random_outliers(num_points=2_000_000)
#X, Y = generate_random_zeros(num_points=2_000_000)

# Split into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
run_regression(X_train, Y_train, X_test, Y_test)
