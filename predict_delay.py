
from pathlib import Path

import joblib
import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load new shipments and hub data
# --------------------------------------------------

new_shipments = pd.read_csv(
    DATA_DIR / "new_shipments.csv"
)

hubs = pd.read_csv(
    DATA_DIR / "hubs.csv"
)

# Preserve original file order
new_shipments["_row_order"] = range(
    len(new_shipments)
)


# --------------------------------------------------
# 3. Load stored model and supporting files
# --------------------------------------------------

model = joblib.load(
    MODEL_DIR / "delay_model.pkl"
)

threshold = float(
    pd.read_csv(
        MODEL_DIR / "delay_threshold.csv"
    )["threshold"].iloc[0]
)

feature_order = (
    pd.read_csv(
        MODEL_DIR / "feature_order.csv"
    )["feature"].tolist()
)

impute_df = pd.read_csv(
    MODEL_DIR / "impute_values.csv"
)

stored_medians = dict(
    zip(
        impute_df["column"],
        impute_df["median"]
    )
)

hub_segments = pd.read_csv(
    OUTPUT_DIR / "hub_segments.csv"
)


# --------------------------------------------------
# 4. Clean new shipments using stored medians
# --------------------------------------------------

cleaned_shipments, _ = clean_shipments(
    new_shipments.drop(columns="_row_order"),
    medians=stored_medians
)


# --------------------------------------------------
# 5. Build features using the existing function
# --------------------------------------------------

feature_df = build_features(
    cleaned_shipments,
    hubs
)


# --------------------------------------------------
# 6. Restore original shipment order
# --------------------------------------------------

# Reattach the original order using shipment_id.
# This assumes shipment_id is unique in new_shipments.

order_df = new_shipments[
    ["shipment_id", "_row_order"]
]

feature_df = feature_df.merge(
    order_df,
    on="shipment_id",
    how="left",
    sort=False
)

feature_df = (
    feature_df.sort_values("_row_order")
    .reset_index(drop=True)
)


# --------------------------------------------------
# 7. Arrange features in the saved order
# --------------------------------------------------

X_new = feature_df[feature_order]


# --------------------------------------------------
# 8. Predict delay probabilities
# --------------------------------------------------

probabilities = model.predict_proba(
    X_new
)[:, 1]

# Keep full precision for threshold comparison.
# Round only the probability displayed in the output.

feature_df["delay_probability"] = (
    probabilities.round(4)
)

feature_df["flagged"] = (
    probabilities >= threshold
).astype(int)


# --------------------------------------------------
# 9. Attach hub segments
# --------------------------------------------------

feature_df = feature_df.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left",
    sort=False
)


# --------------------------------------------------
# 10. Assign operational actions
# --------------------------------------------------

def assign_action(row):

    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"

    elif row["flagged"] == 1:
        return "Proactive customer call"

    else:
        return "No action"


feature_df["action"] = feature_df.apply(
    assign_action,
    axis=1
)


# --------------------------------------------------
# 11. Create final action list
# --------------------------------------------------

final_output = feature_df[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
].copy()


# --------------------------------------------------
# 12. Save deterministic output
# --------------------------------------------------

final_output.to_csv(
    OUTPUT_DIR / "new_shipment_actions.csv",
    index=False,
    float_format="%.4f",
    lineterminator="\n"
)


# --------------------------------------------------
# 13. Display summary
# --------------------------------------------------

print("\nNew shipment scoring completed.")

print("\nTotal shipments:", len(final_output))

print(
    "Flagged shipments:",
    int(final_output["flagged"].sum())
)

print("\nAction summary:")
print(
    final_output["action"].value_counts()
)

print("\nFirst 10 scored shipments:")
print(final_output.head(10))

print(
    "\nSaved:",
    OUTPUT_DIR / "new_shipment_actions.csv"
)
