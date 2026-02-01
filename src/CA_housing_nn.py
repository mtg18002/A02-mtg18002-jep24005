from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]

# Quick check
print(housing.frame.head())
print(housing.frame.shape)

# Train, val, and test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# MLPRegressor with Early Stopping
mlp = MLPRegressor(
    hidden_layer_sizes=(12, 4),
    activation="relu",
    max_iter=500,
    early_stopping=True,        # <-- early stopping ON
    validation_fraction=0.2,    # uses 10% of TRAIN as internal validation
    random_state=42
)

mlp.fit(X_train_scaled, y_train)

# Train predictions
y_pred_train = mlp.predict(X_train_scaled)

# Train predictions metrics
def metrics_row(name, y_true, y_pred):
    return {
        "split": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }

train_metrics_df = pd.DataFrame([
    metrics_row("train", y_train, y_pred_train)
])

print("===== Train Metrics =====")
print(train_metrics_df.to_string(index=False))

# Train predictions vs actual
def scatter_with_reference(y_true, y_pred, title):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], linewidth=1, color='red')  # reference line
    plt.xlabel("Actual MedHouseVal")
    plt.ylabel("Predicted MedHouseVal")
    plt.title(title)
    plt.tight_layout()

scatter_with_reference(y_train, y_pred_train, "Predicted vs Actual - Train")

# Save scatterplot
plt.savefig("figs/train_actual_vs_pred.png")
plt.show()


# Test predictions
y_pred_test = mlp.predict(X_test_scaled)

# Test predictions metrics
test_metrics_df = pd.DataFrame([
    metrics_row("test", y_test, y_pred_test)
])

print("\n===== Test Metrics =====")
print(test_metrics_df.to_string(index=False))

# Test predictions vs actual plot
scatter_with_reference(
    y_test,
    y_pred_test,
    "Predicted vs Actual – Test"
)

# Save test scatterplot
plt.savefig("figs/test_actual_vs_pred.png")
plt.show()

# Loss curve (adjusting hidden layer size)
plt.figure(figsize=(8,5))
plt.plot(mlp.loss_curve_)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch (sklearn MLP)")
plt.grid(True)
plt.show() 