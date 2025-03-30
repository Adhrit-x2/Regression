import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
from sklearn.model_selection import train_test_split

# Random true beta for data generation
true_beta = np.random.normal(0, 1)

# Define LAD (Least Absolute Deviations) regression
class LAD:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if y.ndim > 1:
            y = y.ravel()
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        def objective(params):
            return np.sum(np.abs(y - (np.dot(X, params[:-1]) + params[-1])))

        initial_guess = np.concatenate([np.linalg.lstsq(X, y, rcond=None)[0].flatten(), [0]])
        result = minimize(objective, initial_guess, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.success:
            self.coef_ = result.x[:-1]
            self.intercept_ = result.x[-1]
        else:
            print("LAD optimization warning. Message:", result.message)
            self.coef_ = result.x[:-1]
            self.intercept_ = result.x[-1]

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

# Define RLS (Recursive Least Squares) regression
class RLS:
    def __init__(self, num_vars, lam=0.999, delta=1e5):
        self.num_vars = num_vars
        self.lam = lam
        self.P = delta * np.eye(self.num_vars)
        self.w = np.zeros(self.num_vars)

    def fit(self, X, Y):
        for i in range(len(X)):
            self.update(X[i], Y[i])

    def update(self, x, y):
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)
        e = y - self.w.reshape(-1, 1).T @ x
        g = self.P @ x / (self.lam + x.T @ self.P @ x)
        self.w += (g.flatten() * e.flatten())
        self.P = (self.P - g @ x.T @ self.P) / self.lam

    def predict(self, X):
        return np.dot(X, self.w)

# Function to generate random points
def generate_random_points(num_points, input_dims=1, output_dims=1, x_range=(-10, 10), noise=0.5):
    X = np.random.uniform(x_range[0], x_range[1], (num_points, input_dims))
    Y = np.dot(X, true_beta)
    Y = Y.reshape(-1, output_dims)
    Y += np.random.normal(0, noise, (num_points, output_dims))
    return X, Y

# Calculate error metrics
def calculate_errors(Y_true, Y_pred):
    mae = mean_absolute_error(Y_true, Y_pred)
    r2 = r2_score(Y_true, Y_pred)
    n = len(Y_true)
    p = Y_true.shape[1] if Y_true.ndim > 1 else 1
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return mae, r2, adjusted_r2

# Fit all regression models
def fit_all_models(X_train, Y_train):
    models = {}
    models["LLS"] = LinearRegression(n_jobs=-1).fit(X_train, Y_train)
    models["Ridge"] = Ridge(alpha=1.0).fit(X_train, Y_train)
    
    rls = RLS(num_vars=X_train.shape[1])
    rls.fit(X_train, Y_train)
    models["RLS"] = rls
    
    pls = PLSRegression(n_components=1)
    pls.fit(X_train, Y_train)
    models["PLS"] = pls
    
    lad = LAD()
    lad.fit(X_train, Y_train)
    models["LAD"] = lad
    
    return models

# Run regression on training and testing splits and calculate errors
def run_regression(X_train, Y_train, X_test, Y_test):
    models = fit_all_models(X_train, Y_train)
    predictions = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        predictions[name] = pred

    mae_results = {}
    for name, Y_pred in predictions.items():
        mae, r2, adjusted_r2 = calculate_errors(Y_test, Y_pred)
        mae_results[name] = {"mae": mae, "r2": r2, "adjusted_r2": adjusted_r2}
    best_model = min(mae_results, key=lambda k: mae_results[k]["mae"])
    return {"best_model": best_model, "mae_results": mae_results}

# Run regression multiple times to gather error distributions
def run_regression_multiple_times(iterations=100):
    best_models = []
    mae_results_all = []
    for i in range(iterations):
        X, Y = generate_random_points(num_points=1000)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=i)
        results = run_regression(X_train, Y_train, X_test, Y_test)
        best_models.append(results["best_model"])
        mae_results_all.append(results["mae_results"])
    return mae_results_all

# Plot the error distribution with zoomed y-axis
def plot_error_distribution(mae_results_all):
    methods = mae_results_all[0].keys()
    mae_data = {method: [] for method in methods}
    for res in mae_results_all:
        for method in methods:
            mae_data[method].append(res[method]["mae"])
    
    plt.figure(figsize=(10, 6))
    box = plt.boxplot([mae_data[method] for method in methods], labels=methods, patch_artist=True)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title("Comparative Performance Distributions (MAE)")
    plt.ylabel("Mean Absolute Error")
    plt.xlabel("Regression Method")
    plt.show()

if __name__ == "__main__":
    mae_results_all = run_regression_multiple_times(iterations=100)
    plot_error_distribution(mae_results_all)
