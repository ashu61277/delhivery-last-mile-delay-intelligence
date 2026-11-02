
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# --------------------------------------------------
# 1. Paths and input data
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "modelling_table.csv")


# --------------------------------------------------
# 2. Create hub profile
# --------------------------------------------------

hub_profile = (
    df.groupby("hub_id", as_index=False)
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    )
    .sort_values("hub_id")
    .reset_index(drop=True)
)

hub_profile.to_csv(
    OUTPUT_DIR / "hub_profile.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 3. Scale the four clustering columns
# --------------------------------------------------

CLUSTER_COLUMNS = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    hub_profile[CLUSTER_COLUMNS]
)


# --------------------------------------------------
# 4. Separation evidence for K = 2, 3, 4, 5
# --------------------------------------------------

# Pairwise Euclidean distances between all 24 hubs.
# There are 24 * 23 / 2 = 276 unordered pairs.

distance_matrix = pairwise_distances(
    X_scaled,
    metric="euclidean"
)

separation_results = []

for k in [2, 3, 4, 5]:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(hub_profile)), 2):

        distance = distance_matrix[i, j]

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    ratio = mean_across / mean_within

    cluster_sizes = sorted(
        pd.Series(labels).value_counts().tolist(),
        reverse=True
    )

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        )
    })


pd.DataFrame(separation_results).to_csv(
    OUTPUT_DIR / "part_d_separation.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 5. Final K-Means model: K = 3
# --------------------------------------------------

final_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster_id"] = final_kmeans.fit_predict(
    X_scaled
)


# --------------------------------------------------
# 6. Name clusters by their mean delay rate
# --------------------------------------------------

cluster_delay_rates = (
    hub_profile.groupby("cluster_id")["delay_rate"]
    .mean()
    .sort_values()
)

cluster_names = {
    cluster_delay_rates.index[0]: "Standard",
    cluster_delay_rates.index[1]: "Watch",
    cluster_delay_rates.index[2]: "Escalate"
}

hub_profile["segment"] = (
    hub_profile["cluster_id"].map(cluster_names)
)


# --------------------------------------------------
# 7. Create hub_segments.csv
# --------------------------------------------------

hub_details = (
    df[["hub_id", "hub_city", "hub_tier"]]
    .drop_duplicates(subset=["hub_id"])
)

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hub_details,
    on="hub_id",
    how="left",
    sort=False
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values("hub_id").reset_index(drop=True)

hub_segments.to_csv(
    OUTPUT_DIR / "hub_segments.csv",
    index=False
)


# --------------------------------------------------
# 8. Create segment-level profile
# --------------------------------------------------

segment_profile = (
    hub_profile.groupby("segment", as_index=False)
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    )
)

# Display in operational order
segment_order = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2
}

segment_profile["_order"] = (
    segment_profile["segment"].map(segment_order)
)

segment_profile = (
    segment_profile.sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

segment_profile.to_csv(
    OUTPUT_DIR / "segment_profile.csv",
    index=False,
    float_format="%.4f"
)


# --------------------------------------------------
# 9. Identify tier versus segment mismatches
# --------------------------------------------------

def expected_segment(tier):

    tier_text = str(tier).strip().lower()

    if tier_text in ["1", "tier 1"]:
        return "Standard"

    if tier_text in ["2", "tier 2"]:
        return "Watch"

    if tier_text in ["3", "tier 3"]:
        return "Escalate"

    return None


hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].apply(expected_segment)
)

mismatches = hub_segments[
    hub_segments["expected_segment"].notna()
    & (
        hub_segments["segment"]
        != hub_segments["expected_segment"]
    )
].copy()

mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

mismatches.to_csv(
    OUTPUT_DIR / "tier_vs_segment_mismatch.csv",
    index=False
)


# --------------------------------------------------
# 10. Display summary
# --------------------------------------------------

print("\nPart D segmentation completed.")

print("\nHub profile:")
print(hub_profile.drop(columns="cluster_id"))

print("\nSeparation evidence:")
print(pd.read_csv(OUTPUT_DIR / "part_d_separation.csv"))

print("\nHub segments:")
print(hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]])

print("\nSegment profile:")
print(segment_profile)

print("\nTier versus segment mismatches:")
print(mismatches)
