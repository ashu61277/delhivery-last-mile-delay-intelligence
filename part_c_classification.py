
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

Path("outputs").mkdir(exist_ok=True)

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


# --------------------------------------------------
# 2. Read data
# --------------------------------------------------

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]


# --------------------------------------------------
# 3. Stratified train / validation / test split
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Split sizes:")
print("Train:", len(X_train), "Delayed:", int(y_train.sum()))
print("Validation:", len(X_val), "Delayed:", int(y_val.sum()))
print("Test:", len(X_test), "Delayed:", int(y_test.sum()))


# --------------------------------------------------
# Helper function: calculate classification metrics
# --------------------------------------------------

def calculate_metrics(y_actual, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_actual,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "accuracy": accuracy_score(y_actual, y_pred),
        "precision": precision_score(
            y_actual, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_actual, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_actual, y_pred, zero_division=0
        )
    }


# --------------------------------------------------
# 4. Blind baseline and three classifiers
# --------------------------------------------------

model_results = []

# Blind baseline: predict every parcel as on time
blind_predictions = np.zeros(len(y_val), dtype=int)

blind_metrics = calculate_metrics(y_val, blind_predictions)

model_results.append({
    "model": "blind_all_on_time",
    **blind_metrics
})


# Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

model_results.append({
    "model": "logistic_regression",
    **calculate_metrics(y_val, logistic_predictions)
})


# Decision tree: select depth with highest validation F1
tree_depth_results = []

for depth in [2, 3, 4, 5, 6, 8, 10]:

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    predictions = tree.predict(X_val)

    val_f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    tree_depth_results.append({
        "depth": depth,
        "val_f1": val_f1,
        "model": tree
    })


# Highest F1; ties go to smaller depth
best_tree_result = sorted(
    tree_depth_results,
    key=lambda item: (-item["val_f1"], item["depth"])
)[0]

best_tree_depth = best_tree_result["depth"]
decision_tree_model = best_tree_result["model"]

tree_predictions = decision_tree_model.predict(X_val)

model_results.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **calculate_metrics(y_val, tree_predictions)
})


# Default Random Forest
random_forest_default = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_default.fit(X_train, y_train)

rf_predictions = random_forest_default.predict(X_val)

model_results.append({
    "model": "random_forest_default",
    **calculate_metrics(y_val, rf_predictions)
})


pd.DataFrame(model_results).to_csv(
    "outputs/part_c_models.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 5. Feature importance and F-scores
# --------------------------------------------------

# Random Forest importance
importance_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

importance_model.fit(X_train, y_train)

rf_importances = importance_model.feature_importances_


# F-scores from SelectKBest
f_selector = SelectKBest(
    score_func=f_classif,
    k="all"
)

f_selector.fit(X_train, y_train)

f_scores = f_selector.scores_

feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES]
})

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

feature_scores.to_csv(
    "outputs/part_c_feature_scores.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 6. SelectKBest experiment: k = 5, 7, 10, 12, 17
# --------------------------------------------------

selectk_results = []

for k in [5, 7, 10, 12, 17]:

    model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    val_f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    selectk_results.append({
        "k": k,
        "val_f1": val_f1
    })


pd.DataFrame(selectk_results).to_csv(
    "outputs/part_c_selectk.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 7. Fit SelectKBest with k=7 on training data only
# --------------------------------------------------

selector_7 = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_7 = selector_7.fit_transform(X_train, y_train)
X_val_7 = selector_7.transform(X_val)


# --------------------------------------------------
# 8. Grid Search and Randomized Search
# --------------------------------------------------

parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

base_forest = RandomForestClassifier(random_state=42)


# Grid Search
grid_search = GridSearchCV(
    estimator=base_forest,
    param_grid=parameter_grid,
    cv=5
)

grid_search.fit(X_train_7, y_train)

grid_best_model = grid_search.best_estimator_

grid_predictions = grid_best_model.predict(X_val_7)

grid_val_f1 = f1_score(
    y_val,
    grid_predictions,
    zero_division=0
)


# Randomized Search
random_search = RandomizedSearchCV(
    estimator=base_forest,
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_7, y_train)

random_best_model = random_search.best_estimator_

random_predictions = random_best_model.predict(X_val_7)

random_val_f1 = f1_score(
    y_val,
    random_predictions,
    zero_division=0
)


# Default forest on the same seven selected features
default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest_7.fit(X_train_7, y_train)

default_predictions = default_forest_7.predict(X_val_7)

default_val_f1 = f1_score(
    y_val,
    default_predictions,
    zero_division=0
)


# --------------------------------------------------
# 9. Save tuning results
# --------------------------------------------------

tuning_results = [
    {
        "search": "grid_search",
        "candidates": len(grid_search.cv_results_["params"]),
        "cv_folds": 5,
        "total_fits": len(grid_search.cv_results_["params"]) * 5,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_val_f1
    },
    {
        "search": "randomized_search",
        "candidates": len(random_search.cv_results_["params"]),
        "cv_folds": 5,
        "total_fits": len(random_search.cv_results_["params"]) * 5,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_val_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_val_f1
    }
]

tuning_df = pd.DataFrame(tuning_results)

tuning_df.to_csv(
    "outputs/part_c_tuning.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 10. Select final configuration
# --------------------------------------------------

# Tie priority:
# 1. default_forest_same_7_features
# 2. grid_search
# 3. randomized_search

priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2
}

winner = sorted(
    tuning_results,
    key=lambda row: (
        -row["val_f1"],
        priority[row["search"]]
    )
)[0]

chosen_from = winner["search"]


if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_

elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_

else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }


# Final model pipeline
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_params
    ))
])

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 11. Choose alert threshold using validation recall
# --------------------------------------------------

val_probabilities = final_model.predict_proba(X_val)[:, 1]

thresholds = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_results = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    metrics = calculate_metrics(y_val, predictions)

    threshold_results.append({
        "threshold": threshold,
        **metrics
    })


threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    "outputs/part_c_thresholds.csv",
    index=False,
    float_format="%.4f"
)


# Highest threshold with recall >= 0.80
eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

if len(eligible_thresholds) > 0:
    chosen_threshold = eligible_thresholds["threshold"].max()
else:
    chosen_threshold = 0.1


print("Chosen alert threshold:", chosen_threshold)


# --------------------------------------------------
# 12. One final test evaluation
# --------------------------------------------------

# Do not use test rows for model selection or threshold selection.
# This is the single final evaluation.

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

test_metrics = calculate_metrics(y_test, test_predictions)

final_test_df = pd.DataFrame([{
    "threshold": chosen_threshold,
    **test_metrics
}])

final_test_df.to_csv(
    "outputs/part_c_final_test.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 13. Save final configuration
# --------------------------------------------------

max_depth_value = chosen_params["max_depth"]

if max_depth_value is None:
    max_depth_value = "unlimited"

final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": max_depth_value,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from
}])

final_config.to_csv(
    "outputs/part_c_final_config.csv",
    index=False
)


# --------------------------------------------------
# 14. Display summary
# --------------------------------------------------

print("\nPart C completed successfully.")

print("\nModel comparison:")
print(pd.read_csv("outputs/part_c_models.csv"))

print("\nFeature scores:")
print(pd.read_csv("outputs/part_c_feature_scores.csv"))

print("\nSelectK results:")
print(pd.read_csv("outputs/part_c_selectk.csv"))

print("\nTuning results:")
print(pd.read_csv("outputs/part_c_tuning.csv"))

print("\nThreshold results:")
print(pd.read_csv("outputs/part_c_thresholds.csv"))

print("\nFinal test result:")
print(pd.read_csv("outputs/part_c_final_test.csv"))

print("\nFinal configuration:")
print(pd.read_csv("outputs/part_c_final_config.csv"))
