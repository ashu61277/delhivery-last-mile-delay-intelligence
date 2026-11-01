
import pandas as pd

# Read generated output files
delay = pd.read_csv("outputs/part_a_delay_by_group.csv")
corr = pd.read_csv("outputs/part_a_correlations.csv", index_col=0)
checks = pd.read_csv("outputs/part_a_checks.csv")

# Convert checks into a dictionary
check_values = dict(zip(checks["metric"], checks["value"]))


# 1. Service type with highest delay rate
service = delay[delay["dimension"] == "service_type"]
top_service = service.loc[service["delay_rate"].idxmax()]

print("\n1. SERVICE TYPE WITH HIGHEST DELAY RATE")
print(top_service)


# 2. Delay rate by hub tier
tier = delay[delay["dimension"] == "hub_tier"].copy()
tier = tier.sort_values("group")

print("\n2. DELAY RATE BY HUB TIER")
print(tier[["group", "shipments", "delay_rate"]].to_string(index=False))


# 3. Delay rate by distance band
distance = delay[delay["dimension"] == "distance_band"].copy()

distance_order = ["0-200", "201-400", "401-600", "601+"]
distance["group"] = pd.Categorical(
    distance["group"],
    categories=distance_order,
    ordered=True
)
distance = distance.sort_values("group")

print("\n3. DELAY RATE BY DISTANCE BAND")
print(distance[["group", "shipments", "delay_rate"]].to_string(index=False))


# 4. Three strongest correlations with delivery_hours
top_correlations = (
    corr["delivery_hours"]
    .drop(labels=["delivery_hours"])
    .dropna()
    .reindex(
        corr["delivery_hours"]
        .drop(labels=["delivery_hours"])
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .index
    )
    .head(3)
)

print("\n4. TOP THREE CORRELATIONS WITH DELIVERY HOURS")
print(top_correlations.to_string())


# 5. Declared value mean and median before and after log transformation
print("\n5. DECLARED VALUE STATISTICS")

print("Declared value mean:",
      check_values["declared_value_mean_inr"])

print("Declared value median:",
      check_values["declared_value_median_inr"])

print("Log declared value mean:",
      check_values["log_declared_value_mean"])

print("Log declared value median:",
      check_values["log_declared_value_median"])
