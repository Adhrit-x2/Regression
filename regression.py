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


def generate_random_points(num_points, input_dims=2, output_dims=2, x_range=(0, 5), noise=0.5):
    X = np.random.uniform(x_range[0], x_range[1], (num_points, input_dims))
    Y = np.dot(X, np.random.rand(input_dims, output_dims))
    Y += np.random.normal(0, noise, (num_points, output_dims))
    return X, Y

#def generate_data_with_outliers(num_outliers=10, outlier_intensity=100):
    #np.random.seed()

    #X = np.random.uniform(0, 5, (100, 3))
    #Y = np.dot(X, np.array([[1.5], [2.5], [3.5]])) + np.random.normal(0, 0.5, (100, 1))
    
    #Y = np.hstack([Y, Y + np.random.normal(0, 0.5, (100, 1))])
    
    #num_outliers = min(num_outliers, X.shape[0])
    
    #outlier_indices = np.random.choice(Y.shape[0], num_outliers, replace=False)
    #Y[outlier_indices] += outlier_intensity * np.random.rand(num_outliers, Y.shape[1])
    
    #return X, Y

#def generate_nonlinear_points(num_points, degree=2):
    #X = np.linspace(0, 10, num_points).reshape(-1, 1)
    #y = 3 * X**2 + 2 * X + 1 + np.random.normal(0, 10, X.shape)
    #return X, y

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

def run_regression(X_train, Y_train, X_test, Y_test, n_components=2):
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
    plt.scatter(Y_test[:, 0], Y_test[:, 1], color='blue', alpha=0.5, label='Actual')

    actual_slope, actual_intercept = np.polyfit(Y_test[:, 0], Y_test[:, 1], 1)
    actual_line_x = np.array([Y_test[:, 0].min(), Y_test[:, 0].max()])
    actual_line_y = actual_slope * actual_line_x + actual_intercept
    plt.plot(actual_line_x, actual_line_y, color='blue', linestyle='--', label='Actual Line')

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
        slope, intercept = np.polyfit(Y_pred[:, 0], Y_pred[:, 1], 1)
        line_x = np.array([Y_pred[:, 0].min(), Y_pred[:, 0].max()])
        line_y = slope * line_x + intercept
        plt.scatter(Y_pred[:, 0], Y_pred[:, 1], alpha=0.5, label=name)
        plt.plot(line_x, line_y, linestyle='-', label=f'{name} Line')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Regression Methods')
    #plt.legend()
    plt.grid(True)

    equations = []
    for name, Y_pred in models.items():
        slope, intercept = np.polyfit(Y_pred[:, 0], Y_pred[:, 1], 1)
        equation = f"{name}: y = {slope:.2f}*x + {intercept:.2f}"
        equations.append(equation)

    equation_text = '\n'.join(equations)
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    #plt.show()

# Generate a large dataset
X, Y = generate_random_points(num_points=2_000_000)

# Split into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
run_regression(X_train, Y_train, X_test, Y_test)






def plot_nonlinear_regression(X, y):
    plt.figure(figsize=(13, 8.8))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')

    # Fit and plot linear regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_linear = linear_model.predict(X_plot)
    plt.plot(X_plot, y_linear, color='red', label='Linear Regression')

    # Fit and plot polynomial regression
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, y)
    y_poly = poly_model.predict(X_plot)
    plt.plot(X_plot, y_poly, color='green', label='Polynomial Regression')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear models in non linear data')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate and plot data
#X, y = generate_nonlinear_points(100)
#plot_nonlinear_regression(X, y)

