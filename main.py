# --- 0) Imports ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# --- 1) Load data -------------------------------------------------------------
data = fetch_california_housing(as_frame=True)
df = data.frame.copy()

# Rename target for clarity
# TODO-1: The target column is currently named 'MedHouseVal'. Rename it to 'target'.
# YOUR CODE HERE
df.rename(columns={'MedHouseVal': 'target'}, inplace=True)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# --- 2) Quick EDA -------------------------------------------------------------
# Peek at the data
print("Shape:", df.shape)
print(df.head())


# TODO-2: Print summary stats transposed so features are rows.
# YOUR CODE HERE
print(df.describe())

# TODO-3: Plot correlations with target for the top 8 features
# Then drop 'target' itself and plot a bar chart.
# YOUR CODE HERE
corr = df.corr(numeric_only=True)['target'].sort_values(ascending=False).head(9)
corr.drop(labels=['target']).plot(kind='bar')
plt.title('Correlation with target')
plt.tight_layout()
plt.show()

# --- 3) Train / Test Split ----------------------------------------------------
# Keep a test set for honest evaluation
# TODO-4: Split X,y into train and test with test_size=0.2 and random_state=42
# YOUR CODE HERE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4) Helper: evaluation function ------------------------------------------
def eval_model(model, name):
    """
    Fit model on training data, predict on test, print MAE, RMSE, R2.
    Returns predictions for further plotting.
    """
    # TODO-5: Fit the model on the training data
    # YOUR CODE HERE
    model.fit(X_train, y_train)

    # TODO-6: Predict on the test data
    # YOUR CODE HERE
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    return preds

# --- 5) Models ----------------------------------------------------------------
# 5.1 Linear Regression (with scaling)
lin = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('reg', LinearRegression())
])

# TODO-7: Evaluate the linear model using eval_model and store predictions in p_lin
# YOUR CODE HERE
p_lin = eval_model(lin, "Linear Regression")

# 5.2 Ridge Regression (L2 regularization)
# TODO-8: Create a Ridge pipeline (scaler + Ridge(alpha=1.0))
# YOUR CODE HERE
ridge = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('reg', Ridge(alpha=1.0))
])
p_ridge = eval_model(ridge, "Ridge(alpha=1.0)")

# 5.3 Random Forest (nonlinear baseline)
# NOTE: Tree-based models don't need scaling
rf = RandomForestRegressor(n_estimators=300, random_state=42)

# TODO-9: Evaluate the RF model and store predictions in p_rf
# YOUR CODE HERE
p_rf = eval_model(rf, "RandomForestRegressor")

# --- 6) Visualization ---------------------------------------------------------
# TODO-10: Make a predicted vs actual scatter plot for RandomForest
# YOUR CODE HERE
plt.scatter(y_test,p_rf, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("RF: Pred vs Actual")
plt.tight_layout()
plt.show()

# TODO-11: Plot feature importances for RandomForest
# YOUR CODE HERE
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.plot(kind='bar')
plt.title('Feature Importance (RF)')
plt.tight_layout()
plt.show()
