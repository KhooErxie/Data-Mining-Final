import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset and EDA
# -------------------------------

data_path = 'C:/Users/Erxie/Downloads/ProgressofCOVID19vaccination.csv'
df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData Info:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Identify numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Plot histograms for numeric features
df[num_cols].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Compute and plot a correlation heatmap using matplotlib
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap")
# Add text annotations for correlations
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
plt.show()

# -------------------------------
# 2. Preprocessing and Feature Selection
# -------------------------------

# Fill missing numeric values with the median
for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in {col} with median: {median_val}")

# Use only numeric columns for modeling
df_model = df[num_cols].copy()

# Define target variable (adjust the name if needed)
target_var = 'total_vaccinations'
if target_var not in df_model.columns:
    target_var = num_cols[0]
    print(f"Target variable not found. Defaulting to: {target_var}")

# Compute correlation with the target variable and select features with |corr| > 0.3
cor_target = df_model.corr()[target_var].abs()
print("\nCorrelation with target:")
print(cor_target.sort_values(ascending=False))

selected_features = cor_target[cor_target > 0.3].index.tolist()
if target_var in selected_features:
    selected_features.remove(target_var)
print(f"\nSelected features: {selected_features}")

# -------------------------------
# 3. Train/Test Split (manual)
# -------------------------------

# Convert to NumPy arrays for modeling
X = df_model[selected_features].values
y = df_model[target_var].values

# Set random seed for reproducibility and create a random permutation of indices
np.random.seed(42)
indices = np.random.permutation(len(y))
test_size = int(0.3 * len(y))
test_idx, train_idx = indices[:test_size], indices[test_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# -------------------------------
# 4. Define Evaluation Metrics
# -------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

# -------------------------------
# 5. Linear Regression (Using Normal Equation)
# -------------------------------

class LinearRegressionScratch:
    def __init__(self):
        self.coef_ = None
        
    def fit(self, X, y):
        # Add intercept term
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        # Compute coefficients using the normal equation
        self.coef_ = np.linalg.lstsq(X_b, y, rcond=None)[0]
    
    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b.dot(self.coef_)

# Train and evaluate Linear Regression
lr_model = LinearRegressionScratch()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("\nLinear Regression Performance:")
print("RMSE:", rmse(y_test, y_pred_lr))
print("MAE :", mae(y_test, y_pred_lr))
print("R2  :", r2_score(y_test, y_pred_lr))

# -------------------------------
# 6. Decision Tree Regressor (Basic Implementation)
# -------------------------------

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # Number of features to consider at each split
        self.tree = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
    
    def _sse(self, y):
        # Sum of squared errors
        return np.sum((y - np.mean(y)) ** 2)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth) or (num_samples < self.min_samples_split) or (np.unique(y).shape[0] == 1):
            return {"value": np.mean(y)}
        
        # Determine feature indices to consider
        feature_indices = np.arange(num_features)
        if self.max_features is not None and self.max_features < num_features:
            feature_indices = np.random.choice(num_features, self.max_features, replace=False)
        
        best_mse = np.inf
        best_split = None
        
        # Loop over features and possible thresholds
        for feature in feature_indices:
            values = np.unique(X[:, feature])
            for threshold in values:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                mse_left = self._sse(y[left_idx])
                mse_right = self._sse(y[right_idx])
                mse_total = mse_left + mse_right
                if mse_total < best_mse:
                    best_mse = mse_total
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left_idx": left_idx,
                        "right_idx": right_idx
                    }
        
        if best_split is None:
            return {"value": np.mean(y)}
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[best_split["left_idx"]], y[best_split["left_idx"]], depth + 1)
        right_subtree = self._build_tree(X[best_split["right_idx"]], y[best_split["right_idx"]], depth + 1)
        
        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_subtree,
            "right": right_subtree
        }
    
    def _predict_one(self, x, tree):
        if "value" in tree:
            return tree["value"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        if x[feature] <= threshold:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

# Train and evaluate Decision Tree Regressor
dt_model = DecisionTreeRegressorScratch(max_depth=5, min_samples_split=5, max_features=None)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("\nDecision Tree Regressor Performance:")
print("RMSE:", rmse(y_test, y_pred_dt))
print("MAE :", mae(y_test, y_pred_dt))
print("R2  :", r2_score(y_test, y_pred_dt))

# -------------------------------
# 7. Random Forest Regressor (Simple Implementation)
# -------------------------------

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.trees = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth, 
                                                min_samples_split=self.min_samples_split, 
                                                max_features=self.max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def predict(self, X):
        # Average predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

# Train and evaluate Random Forest Regressor
# Here, we use max_features set to int(sqrt(n_features)) if available
max_features_rf = int(np.sqrt(X_train.shape[1]))
rf_model = RandomForestRegressorScratch(n_estimators=10, max_depth=5, min_samples_split=5, max_features=max_features_rf)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Regressor Performance:")
print("RMSE:", rmse(y_test, y_pred_rf))
print("MAE :", mae(y_test, y_pred_rf))
print("R2  :", r2_score(y_test, y_pred_rf))

# -------------------------------
# 8. Compare Model Performance
# -------------------------------

results = {
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "RMSE": [rmse(y_test, y_pred_lr), rmse(y_test, y_pred_dt), rmse(y_test, y_pred_rf)],
    "MAE": [mae(y_test, y_pred_lr), mae(y_test, y_pred_dt), mae(y_test, y_pred_rf)],
    "R2": [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_dt), r2_score(y_test, y_pred_rf)]
}

results_df = pd.DataFrame(results)
print("\nComparison of Model Performance:")
print(results_df)

# -------------------------------
# 9. Visualization: Actual vs Predicted for the Best Model
# (Here, we use the Random Forest predictions as an example.)
# -------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='b')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest: Actual vs. Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
