
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Read Part C final configuration
# --------------------------------------------------

config = pd.read_csv(
    OUTPUT_DIR / "part_c_final_config.csv"
).iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

max_depth_value = config["max_depth"]

if str(max_depth_value).strip().lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(max_depth_value)


# --------------------------------------------------
# 3. Read modelling table and create same split
# --------------------------------------------------

df = pd.read_csv(
    DATA_DIR / "modelling_table.csv"
)

X = df[FEATURES]
y = df["is_delayed"]


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


# --------------------------------------------------
# 4. Rebuild the final Part C pipeline
# --------------------------------------------------

final_pipeline = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])

final_pipeline.fit(X_train, y_train)


# --------------------------------------------------
# 5. Save the trained pipeline
# --------------------------------------------------

model_path = MODEL_DIR / "delay_model.pkl"

joblib.dump(
    final_pipeline,
    model_path
)


# --------------------------------------------------
# 6. Save threshold
# --------------------------------------------------

pd.DataFrame({
    "threshold": [threshold]
}).to_csv(
    MODEL_DIR / "delay_threshold.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 7. Save feature order
# --------------------------------------------------

pd.DataFrame({
    "feature": FEATURES
}).to_csv(
    MODEL_DIR / "feature_order.csv",
    index=False
)


# --------------------------------------------------
# 8. Calculate and save historical imputation medians
# --------------------------------------------------

raw_shipments = pd.read_csv(
    DATA_DIR / "raw_shipments.csv"
)

_, medians = clean_shipments(raw_shipments)

impute_values = pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
})

impute_values.to_csv(
    MODEL_DIR / "impute_values.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 9. Reload model and compare test probabilities
# --------------------------------------------------

reloaded_pipeline = joblib.load(model_path)

in_memory_probabilities = (
    final_pipeline.predict_proba(X_test)[:, 1]
)

reloaded_probabilities = (
    reloaded_pipeline.predict_proba(X_test)[:, 1]
)

max_abs_difference = np.max(
    np.abs(
        in_memory_probabilities
        - reloaded_probabilities
    )
)


# --------------------------------------------------
# 10. Save reload verification
# --------------------------------------------------

reload_check = pd.DataFrame([
    {
        "metric": "test_rows_scored",
        "value": len(X_test)
    },
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_abs_difference
    },
    {
        "metric": "threshold",
        "value": threshold
    }
])

reload_check.to_csv(
    OUTPUT_DIR / "part_d_reload_check.csv",
    index=False,
    float_format="%.10f"
)


# --------------------------------------------------
# 11. Display summary
# --------------------------------------------------

print("\nModel packaging completed.")

print("\nFinal configuration:")
print(config)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\nStored medians:")
print(impute_values)

print("\nReload verification:")
print(reload_check)

print("\nSaved model:", model_path)
