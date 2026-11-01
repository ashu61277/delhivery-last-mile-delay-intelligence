
import pandas as pd
import numpy as np

from src.shipment_prep import (
    clean_shipments,
    build_features,
    FEATURES,
)


# --------------------------------------------------
# 1. Read input files
# --------------------------------------------------

raw_shipments = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")


# --------------------------------------------------
# 2. Remove exact duplicates and capture BEFORE metrics
# --------------------------------------------------

deduplicated = raw_shipments.drop_duplicates().copy()

service_type_spellings_before = deduplicated["service_type"].nunique()

missing_distance_before = deduplicated["distance_km"].isna().sum()

missing_parcel_weight_before = (
    deduplicated["parcel_weight_kg"].isna().sum()
)

missing_declared_value_before = (
    deduplicated["declared_value"].isna().sum()
)

# Count sentinel rows separately
parcel_weight_sentinel_count = (
    deduplicated["parcel_weight_kg"] > 100
).sum()


# --------------------------------------------------
# 3. Clean shipments
# --------------------------------------------------

cleaned_shipments, medians = clean_shipments(deduplicated)

service_type_spellings_after = (
    cleaned_shipments["service_type"].nunique()
)


# --------------------------------------------------
# 4. Build features
# --------------------------------------------------

modelling_df = build_features(cleaned_shipments, hubs)


# --------------------------------------------------
# 5. Create target variable
# --------------------------------------------------

modelling_df["is_delayed"] = (
    modelling_df["delivery_hours"] >
    modelling_df["promised_hours"]
).astype(int)


modelling_df["distance_band"] = pd.cut(
    modelling_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True,
    include_lowest=True
)

# --------------------------------------------------
# 6. Save modelling table
# --------------------------------------------------

modelling_df.to_csv(
    "data/modelling_table.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 7. Calculate modelling table checks
# --------------------------------------------------

# Declared-value metrics use median-filled values
declared_value_mean = modelling_df["declared_value_inr"].mean()
declared_value_median = modelling_df["declared_value_inr"].median()

log_declared_value = np.log(modelling_df["declared_value_inr"])

log_declared_value_mean = log_declared_value.mean()
log_declared_value_median = log_declared_value.median()


# Count nulls across 17 FEATURES + delivery_hours + is_delayed
null_columns = FEATURES + ["delivery_hours", "is_delayed"]

nulls_in_features_and_target = (
    modelling_df[null_columns].isna().sum().sum()
)


# Delivery-hours outlier bounds using 1.5 x IQR
q1 = modelling_df["delivery_hours"].quantile(0.25)
q3 = modelling_df["delivery_hours"].quantile(0.75)

iqr = q3 - q1

hours_lower_bound = q1 - (1.5 * iqr)
hours_upper_bound = q3 + (1.5 * iqr)

hours_outlier_count = (
    (modelling_df["delivery_hours"] < hours_lower_bound) |
    (modelling_df["delivery_hours"] > hours_upper_bound)
).sum()


# --------------------------------------------------
# 8. Create checks DataFrame
# --------------------------------------------------

checks = [
    ("rows_after_deduplication", len(deduplicated)),
    ("duplicate_rows_removed", len(raw_shipments) - len(deduplicated)),
    ("service_type_spellings_before", service_type_spellings_before),
    ("service_type_spellings_after", service_type_spellings_after),
    ("missing_distance_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_parcel_weight_before),
    ("missing_declared_value_before", missing_declared_value_before),
    ("parcel_weight_sentinel_count", parcel_weight_sentinel_count),
    ("nulls_in_features_and_target", nulls_in_features_and_target),
    ("declared_value_mean_inr", declared_value_mean),
    ("declared_value_median_inr", declared_value_median),
    ("log_declared_value_mean", log_declared_value_mean),
    ("log_declared_value_median", log_declared_value_median),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_iqr", iqr),
    ("hours_lower_bound", hours_lower_bound),
    ("hours_upper_bound", hours_upper_bound),
    ("hours_outlier_count", hours_outlier_count),
]

checks_df = pd.DataFrame(checks, columns=["metric", "value"])


# --------------------------------------------------
# 9. Save checks file
# --------------------------------------------------

checks_df.to_csv(
    "outputs/part_a_checks.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 10. Display results
# --------------------------------------------------

print("Part A completed successfully!")

print("\nModelling table:")
print("Rows:", len(modelling_df))
print("Columns:", len(modelling_df.columns))
print("Unique shipments:", modelling_df["shipment_id"].nunique())

print("\nTarget distribution:")
print(modelling_df["is_delayed"].value_counts())

print("\nChecks:")
print(checks_df.to_string(index=False))

print("\nMedians used:")
print(medians)

# Define the three grouping dimensions
dimensions = ["service_type", "hub_tier", "distance_band"]

delay_results = []

for dimension in dimensions:

    grouped = modelling_df.groupby(
        dimension,
        observed=True
    ).agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    grouped = grouped.rename(columns={dimension: "group"})

    grouped.insert(0, "dimension", dimension)

    delay_results.append(grouped)


# Combine all three dimensions
delay_by_group = pd.concat(
    delay_results,
    ignore_index=True
)

# Keep required column order
delay_by_group = delay_by_group[
    [
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
        "mean_hours"
    ]
]

# Save output
delay_by_group.to_csv(
    "outputs/part_a_delay_by_group.csv",
    index=False,
    float_format="%.4f"
)

print("\nDelay evidence table:")
print(delay_by_group.to_string(index=False))


# Select all numeric columns from the modelling table
numeric_df = modelling_df.select_dtypes(include=["number"])

# Calculate Pearson correlation matrix
correlation_matrix = numeric_df.corr()

# Save correlation matrix
correlation_matrix.to_csv(
    "outputs/part_a_correlations.csv",
    index=True,
    index_label="column",
    float_format="%.4f"
)

print("\nCorrelation matrix:")
print(correlation_matrix)