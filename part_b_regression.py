
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES


# --------------------------------------------------
# 1. Read modelling table
# --------------------------------------------------

modelling_df = pd.read_csv("data/modelling_table.csv")

X = modelling_df[FEATURES]
y = modelling_df["delivery_hours"]


# --------------------------------------------------
# 2. Split into training, validation and test sets
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)


# --------------------------------------------------
# 3. Scaling without data leakage
# --------------------------------------------------

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test using training scaler
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = np.abs(
    X_train_scaled.mean(axis=0)
).max()

test_scaled_max_abs_mean = np.abs(
    X_test_scaled.mean(axis=0)
).max()


# --------------------------------------------------
# 4. Save split and scaling checks
# --------------------------------------------------

checks = [
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
    ("test_scaled_max_abs_mean", test_scaled_max_abs_mean)
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])

checks_df.to_csv(
    "outputs/part_b_checks.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 5. Helper function for regression metrics
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# 6. Baseline model
# --------------------------------------------------

baseline_prediction = np.full(
    shape=len(y_val),
    fill_value=y_train.mean()
)

baseline_metrics = calculate_metrics(
    y_val,
    baseline_prediction
)

metrics_rows = [
    {
        "model": "mean_baseline",
        "split": "val",
        **baseline_metrics
    }
]


# --------------------------------------------------
# 7. Decision tree depth diagnostic
# --------------------------------------------------

depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_depth_rows = []

for depth in depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_prediction)
    )

    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_prediction)
    )

    tree_depth_rows.append({
        "max_depth": "unlimited" if depth is None else depth,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse
    })


tree_depth_df = pd.DataFrame(tree_depth_rows)

tree_depth_df.to_csv(
    "outputs/part_b_tree_depth.csv",
    index=False,
    float_format="%.4f"
)


# Select the depth with lowest validation RMSE.
# In a tie, prefer the smaller numerical depth.
best_tree_row = min(
    tree_depth_rows,
    key=lambda row: (
        row["val_rmse"],
        float("inf") if row["max_depth"] == "unlimited"
        else row["max_depth"]
    )
)

best_depth = (
    None if best_tree_row["max_depth"] == "unlimited"
    else best_tree_row["max_depth"]
)

print("Best decision tree depth:", best_tree_row["max_depth"])


# --------------------------------------------------
# 8. Define four candidate models
# --------------------------------------------------

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "decision_tree": DecisionTreeRegressor(
        max_depth=best_depth,
        random_state=42
    ),

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
}


# --------------------------------------------------
# 9. Fit candidates and calculate validation metrics
# --------------------------------------------------

validation_rmse = {}

for model_name, model in models.items():

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    model_metrics = calculate_metrics(
        y_val,
        val_prediction
    )

    validation_rmse[model_name] = model_metrics["RMSE"]

    metrics_rows.append({
        "model": model_name,
        "split": "val",
        **model_metrics
    })


# --------------------------------------------------
# 10. Gradient boosting learning-rate experiment
# --------------------------------------------------

learning_rate_rows = []

for lr in [0.01, 0.1, 1.0]:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    gb_prediction = gb_model.predict(X_val)

    gb_rmse = np.sqrt(
        mean_squared_error(y_val, gb_prediction)
    )

    gb_r2 = r2_score(y_val, gb_prediction)

    learning_rate_rows.append({
        "learning_rate": lr,
        "val_rmse": gb_rmse,
        "val_r2": gb_r2
    })


learning_rate_df = pd.DataFrame(learning_rate_rows)

learning_rate_df.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 11. Rank candidates by validation RMSE
# --------------------------------------------------

ranked_models = sorted(
    validation_rmse.items(),
    key=lambda item: item[1]
)

print("\nModels ranked by validation RMSE:")

for name, rmse in ranked_models:
    print(name, ":", round(rmse, 4))


# --------------------------------------------------
# 12. Cross-validation for the two best models
# --------------------------------------------------

X_train_val = pd.concat(
    [X_train, X_val],
    axis=0
)

y_train_val = pd.concat(
    [y_train, y_val],
    axis=0
)

# One shared KFold object for both models
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_rows = []

for model_name, _ in ranked_models[:2]:

    model = models[model_name]

    cv_scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_rows.append({
        "model": model_name,
        "mean_r2": cv_scores.mean(),
        "std_r2": cv_scores.std()
    })


cv_df = pd.DataFrame(cv_rows)

cv_df.to_csv(
    "outputs/part_b_cv.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 13. Select final model and evaluate on test set
# --------------------------------------------------

final_model_name = ranked_models[0][0]
final_model = models[final_model_name]

# This model was already fitted on X_train above.
test_prediction = final_model.predict(X_test)

test_metrics = calculate_metrics(
    y_test,
    test_prediction
)

metrics_rows.append({
    "model": final_model_name,
    "split": "test",
    **test_metrics
})


# --------------------------------------------------
# 14. Save all regression metrics
# --------------------------------------------------

metrics_df = pd.DataFrame(
    metrics_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
)

metrics_df.to_csv(
    "outputs/part_b_metrics.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 15. Display results
# --------------------------------------------------

print("\nPart B completed successfully!")

print("\nSplit and scaling checks:")
print(checks_df.to_string(index=False))

print("\nDecision tree depth comparison:")
print(tree_depth_df.to_string(index=False))

print("\nRegression metrics:")
print(metrics_df.to_string(index=False))

print("\nGradient boosting learning-rate comparison:")
print(learning_rate_df.to_string(index=False))

print("\nCross-validation results:")
print(cv_df.to_string(index=False))

print("\nFinal selected model:", final_model_name)
print("Final test RMSE:", round(test_metrics["RMSE"], 4))



