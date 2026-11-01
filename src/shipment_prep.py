
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]




def clean_shipments(raw, medians=None):


    # Work on a copy so that original DataFrame remains unchanged
    clean_df = raw.copy()

    # 1. Remove exact duplicate rows
    clean_df = clean_df.drop_duplicates()

    # 2. Clean service_type
    clean_df["service_type"] = (
        clean_df["service_type"]
        .str.strip()
        .str.title()
    )

    # 3. Fill missing payment_mode
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

    # 4. Convert declared_value to numeric
    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    ).astype("float64")

    # Drop original text column
    clean_df = clean_df.drop(columns=["declared_value"])

    # 5. Convert dispatch_time to datetime
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"],
        errors="coerce"
    )

    # 6. Replace scanner-fault sentinel values with missing
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100,
        "parcel_weight_kg"
    ] = float("nan")

    # 7. Calculate medians or reuse supplied medians
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]

    if medians is None:

        medians = {
            column: float(clean_df[column].median())
            for column in median_columns
        }

    else:

        medians = {
            column: float(medians[column])
            for column in median_columns
        }

    # 8. Fill missing values using medians
    for column in median_columns:
        clean_df[column] = clean_df[column].fillna(medians[column])

    return clean_df, medians

def build_features(clean, hubs):

    # Work on a copy
    feature_df = clean.copy()

    # 1. Attach hub information using LEFT JOIN
    hub_columns = [
        "hub_id",
        "hub_city",
        "region",
        "hub_tier",
        "hub_daily_capacity"
    ]

    feature_df = feature_df.merge(
        hubs[hub_columns],
        on="hub_id",
        how="left"
    )

    # 2. Map service type to promised delivery hours
    feature_df["promised_hours"] = (
        feature_df["service_type"]
        .map(PROMISED_HOURS)
        .astype(int)
    )

    # 3. Identify COD shipments
    feature_df["is_cod"] = (
        feature_df["payment_mode"] == "COD"
    ).astype(int)

    # 4. Identify weekend dispatches
    feature_df["is_weekend"] = (
        feature_df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    # 5. Extract dispatch hour
    feature_df["dispatch_hour"] = (
        feature_df["dispatch_time"].dt.hour
        .astype(int)
    )

    # 6. Calculate natural log of declared value
    import numpy as np

    feature_df["log_declared_value"] = np.log(
        feature_df["declared_value_inr"]
    )

    return feature_df


