# Delhivery Last-Mile Delay Intelligence Pipeline

## Overview
An end-to-end Machine Learning project to predict
delivery time, identify delayed shipments and
segment logistics hubs.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib


## Data findings

### 1. Delay rate by service type

Express shipments have the highest delay rate among the service types, with **48.97%** of shipments delayed (119 delayed out of 243 shipments). This indicates that Express shipments are not consistently meeting their promised delivery times in the dataset.

### 2. Delay rate by hub tier

The delay rate increases significantly as hub tier increases:

- **Tier 1:** 4.00%
- **Tier 2:** 21.50%
- **Tier 3:** 44.00%

Tier 3 has an 40 percentage-point higher delay rate than Tier 1, suggesting a strong association between hub tier and delivery delays.

### 3. Delay rate by distance band

The delay rate increases with shipment distance:

- **0–200 km:** 2.19%
- **201–400 km:** 13.68%
- **401–600 km:** 43.64%
- **601+ km:** 85.23%

Shipments travelling more than 600 km experience the highest delay rate, at 85.23%. This suggests that longer delivery distances are associated with a greater likelihood of delay.

### 4. Strongest correlations with delivery hours

The three numeric columns most strongly correlated with `delivery_hours`, ranked by absolute correlation, are:

| Feature | Correlation with `delivery_hours` |
|---|---:|
| `promised_hours` | +0.8372 |
| `distance_km` | +0.4013 |
| `n_stops` | +0.3247 |

All three correlations are positive, indicating that higher values of these features are associated with higher delivery hours. `promised_hours` has the strongest linear correlation.

### 5. Declared value log transformation

The median-filled `declared_value_inr` column has a mean of **1533.0542** and a median of **1121.5000**. The mean being higher than the median suggests a right-skewed distribution, potentially influenced by high-value shipments.

After applying the natural logarithm, the mean becomes **7.0277** and the median becomes **7.0224**. Their much smaller difference indicates that the transformation compresses the influence of high values and makes the distribution more balanced. This makes `log_declared_value` a useful feature for modelling.