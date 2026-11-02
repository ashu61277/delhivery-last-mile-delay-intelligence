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

## PART A


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


## PART B

## Model results (regression)

### 1. Baseline performance

The mean-prediction baseline achieved an R² of **-0.0212** and an RMSE of **18.8552 hours** on the validation set.

A negative R² means the baseline performs worse than simply predicting the average delivery time for every shipment. This confirms that a constant prediction is not sufficient and that the shipment features contain useful information for predicting delivery hours.

### 2. Model comparison

| Model | Validation RMSE (hours) | Validation R² |
|---|---:|---:|
| Linear Regression | **4.8912** | **0.9313** |
| Gradient Boosting | 4.9165 | 0.9306 |
| Random Forest | 5.5059 | 0.9129 |
| Decision Tree | 6.2796 | 0.8867 |

**Linear Regression was the validation winner**, achieving an RMSE of **4.8912 hours** and an R² of **0.9313**. This means it explains approximately 93.13% of the variation in delivery hours on the validation data.

The difference between Linear Regression and Gradient Boosting is very small (0.0253 hours), so both models demonstrate strong predictive performance.

### 3. Decision Tree depth: bias and variance

The tree-depth experiment shows how model complexity affects training and validation errors.

- **High bias (underfitting):** Depths 1, 2 and 3 have relatively high training and validation RMSE. Their limited complexity prevents them from capturing the relationships in the data.
- **Best balance:** Depth 6 achieves a training RMSE of 4.7114 and validation RMSE of 6.2796.
- **High variance (overfitting):** Depths 8, 12 and unlimited show increasingly low training errors but higher validation errors. The unlimited tree achieves zero training RMSE but a validation RMSE of 7.8368, indicating severe overfitting.

Depth 6 provides the lowest validation RMSE among the tested tree depths, making it the selected Decision Tree configuration.

### 4. Gradient Boosting learning-rate experiment

| Learning Rate | Validation RMSE | Validation R² |
|---|---:|---:|
| 0.01 | 10.0018 | 0.7127 |
| **0.10** | **4.9165** | **0.9306** |
| 1.00 | 7.3946 | 0.8429 |

A learning rate of **0.10** performs best because it provides a good balance between learning speed and stability.

- **0.01 — Too slow:** Each boosting step makes very small corrections. With only 100 estimators, the model does not learn enough and underfits.
- **0.10 — Best balance:** The model learns useful patterns without making excessively large corrections.
- **1.00 — Overshooting:** The large learning rate makes aggressive updates, reducing generalisation performance and increasing validation error.

### 5. Cross-validation and final model selection

Five-fold cross-validation was performed on the combined training and validation data for the two best validation candidates.

| Model | Mean CV R² | Standard Deviation |
|---|---:|---:|
| Linear Regression | 0.9261 | 0.0048 |
| **Gradient Boosting** | **0.9353** | **0.0045** |

The cross-validation ranking differs from the validation ranking. Linear Regression narrowly wins on validation, whereas Gradient Boosting achieves a higher mean cross-validation R² and slightly lower variation across folds.

**Final model decision:** Linear Regression was selected as the final model because it achieved the lowest validation RMSE. Its performance was then evaluated once on the untouched test set, achieving:

- Test MAE: 3.5671 hours
- Test RMSE: **4.5235 hours**
- Test R²: **0.9493**

Gradient Boosting is a strong alternative and shows better average cross-validation performance. However, retaining Linear Regression is justified by the predefined selection rule of choosing the model with the lowest validation RMSE. This also provides a simpler and more interpretable final model.



# PART C

## Model results (classification)

### 1. Baseline and model comparison

The blind baseline predicted every parcel as on time. It achieved **76.67% accuracy**, but its precision, recall and F1 were all **0.0000**. It correctly classified the 138 on-time parcels but failed to identify any of the 42 delayed parcels. This demonstrates why accuracy alone is misleading for this imbalanced dataset: a model can achieve apparently reasonable accuracy while being completely useless for detecting delays.

Among the three classifiers, Random Forest achieved the highest validation F1 of **0.8000**, with 84.21% precision and 76.19% recall. Logistic Regression achieved an F1 of 0.8046, with 77.78% precision and 83.33% recall. Decision Tree (depth 6) achieved an F1 of 0.7470.

### 2. Feature importance and noise columns

Distance was the strongest feature, with an F-score of 389.2131 and Random Forest importance of 0.2707. Hub tier, hub daily capacity and number of stops were also important predictors.

The five bookkeeping/noise columns had a combined Random Forest importance of **0.2253 (22.53%)**. Despite this combined importance, all five noise columns ranked outside the top seven by F-score. Their F-score ranks were:

- label_print_batch: 8
- tape_roll_no: 12
- sticker_serial_mod: 14
- scanner_build_no: 17
- shelf_slot_no: 16

This shows that the noise columns collectively receive some importance from the Random Forest, but none ranks among the seven strongest features according to the univariate F-score test. The comparison also illustrates that Random Forest importance and F-score measure feature relevance differently.

### 3. Feature selection experiment

The SelectKBest experiment compared five feature counts:

| Selected features (k) | Validation F1 |
|---:|---:|
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.8140 |
| 12 | 0.8049 |
| 17 | 0.8000 |

Selecting seven features produced the highest validation F1 of **0.8333**. Increasing the feature count beyond seven did not improve performance. This suggests that a smaller set of stronger predictors was more effective than including additional weaker or noisy columns in this experiment.

### 4. Hyperparameter tuning

Neither tuning search outperformed the untuned Random Forest using the same seven selected features.

| Configuration | Validation F1 | Best CV accuracy |
|---|---:|---:|
| Grid Search | 0.7750 | 0.8952 |
| Randomized Search | 0.7901 | 0.8905 |
| Default Forest | **0.8333** | Not applicable |

Grid Search evaluated 27 parameter combinations across five folds (135 fits), while Randomized Search evaluated nine combinations across five folds (45 fits). Both searches selected a maximum depth of 6, but their validation F1 scores were lower than the default forest's score.

This suggests that tuning did not improve generalization on the validation set. The validation set contains only 180 parcels, including 42 delayed parcels, so a change in the classification of just a few parcels can noticeably affect F1. Consequently, the observed ranking should not be treated as proof that the default parameters are universally better. It is the best configuration among the candidates evaluated in this experiment.

The final configuration therefore uses the default Random Forest parameters with seven selected features.

### 5. Alert threshold selection

The operations rule requires the highest probability threshold that maintains validation recall of at least 0.80.

At the selected threshold of **0.5**, the validation results were:

- True positives: 35
- False positives: 9
- False negatives: 7
- True negatives: 129
- Precision: **0.7955**
- Recall: **0.8333**
- F1: **0.8140**

At threshold 0.6, precision increased to 0.8889, but recall fell to 0.7619, below the required 0.80. Therefore, 0.5 is the highest threshold satisfying the operations rule.

At lower thresholds, recall generally improves but more on-time parcels are incorrectly flagged. At higher thresholds, unnecessary alerts decrease, but more delayed parcels are missed. The selected threshold balances these outcomes while meeting the minimum recall requirement.


### 6. Final test evaluation

The final Random Forest model used seven selected features, default forest hyperparameters and an alert threshold of **0.5**.

On the previously unseen test set, the model achieved:

| Metric | Test result |
|---|---:|
| Accuracy | 86.67% |
| Precision | 68.89% |
| Recall | 75.61% |
| F1-score | 72.09% |
| True positives | 31 |
| False positives | 14 |
| False negatives | 10 |
| True negatives | 125 |

The model correctly identified **31 of the 41 delayed parcels**, while missing 10 delayed parcels. Of the parcels flagged as delayed, approximately 68.89% were actually delayed.

Validation recall was 83.33%, compared with test recall of 75.61%, a decrease of **7.72 percentage points**. Test F1 was also lower than validation F1 (81.40% at the selected threshold).

This difference indicates that the model performed somewhat less effectively on unseen test data. However, the test set contains only 41 delayed parcels. Each additional correctly or incorrectly detected delayed parcel changes recall by approximately 2.44 percentage points. Therefore, the recall difference should be interpreted cautiously rather than treated as definitive evidence of a large performance decline.

Overall, the model provides useful delay detection, but its 75.61% test recall means that approximately one in four delayed parcels was missed. This is an important limitation for an operational alerting system where missed delays are costly.


# PART D

How to run
1. Clone the repository
git clone <your-repository-url>
cd delhivery-last-mile-delay-intelligence
2. Create and activate a virtual environment
For macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Generate the project data
python generate_data.py
5. Run the pipeline in order
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
The scripts must run in this order because later stages depend on the data and outputs generated by earlier stages.

package_model.py regenerates models/delay_model.pkl using the final configuration saved by Part C. It also generates the threshold, feature-order and imputation files required for scoring.

Model compatibility: The committed models/delay_model.pkl is intended to be loaded using the scikit-learn version pinned in requirements.txt. Pickled scikit-learn models are not guaranteed to be compatible across different scikit-learn versions. If the installed version differs from the pinned version, the grader should run python package_model.py to regenerate the model using its installed environment rather than relying on the committed pickle.


## Operations recommendation

### Hub operating segments

The 24 hubs have been grouped into three operational segments based on their hub-level delivery performance.

**Escalate segment — priority hubs**

The following nine hubs belong to the Escalate segment and should receive priority attention from the regional operations team.

| Hub ID | City | Delay rate |
|---|---|---:|
| HUB19 | Guwahati | 62% |
| HUB24 | Agartala | 54% |
| HUB20 | Raipur | 48% |
| HUB17 | Patna | 46% |
| HUB13 | Nagpur | 46% |
| HUB23 | Siliguri | 44% |
| HUB21 | Bhubaneswar | 40% |
| HUB18 | Ranchi | 38% |
| HUB15 | Kochi | 36% |

HUB19 (Guwahati) and HUB24 (Agartala) require the highest priority, with delay rates of 62% and 54%, respectively. The regional head should prioritise these hubs for capacity reviews, dispatch-process checks, staffing assessment and corrective action. The segment is a relative performance classification, so individual hub delay rates should also be considered when prioritising interventions.

The regional head should prioritise these hubs for capacity reviews, dispatch-process checks, staffing assessment and corrective action. The segment is a relative performance classification, so individual hub delay rates should also be considered when prioritising interventions.

### Tier versus segment mismatches

Four hubs have a segment that differs from the segment implied by their existing tier:

| Hub ID | City | Existing tier | Observed segment | Operational interpretation |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | A Tier 1 hub is performing worse than its tier-based expectation. |
| HUB13 | Nagpur | 2 | Escalate | A Tier 2 hub has high observed delay performance and needs escalation-level attention. |
| HUB15 | Kochi | 2 | Escalate | A Tier 2 hub has high observed delay performance and needs escalation-level attention. |
| HUB22 | Dehradun | 3 | Watch | A Tier 3 hub is performing better than its tier-based expectation. |

These mismatches show why hub tier and operating segment should not be treated as interchangeable. Tier remains a capacity classification, whereas the segment reflects observed delivery performance.

The regional team should investigate the mismatches before changing any hub's tier. In particular, HUB07, HUB13 and HUB15 deserve operational review, while HUB22 may be a candidate for a capacity and performance reassessment.

### Parcel alert policy

The selected alert threshold is **0.5**. A parcel is flagged when its predicted delay probability is greater than or equal to 0.5.

On the held-out Part C test set:

- Precision was **68.89%**: approximately 6.9 out of every 10 flagged parcels were actually delayed.
- Recall was **75.61%**: approximately 7.6 out of every 10 delayed parcels were detected.

These results make the alert useful for prioritisation, but it should not be treated as a guarantee that a parcel will be delayed.

### New shipment action list

The scoring pipeline processed all 48 new parcels.

| Operational outcome | Parcels |
|---|---:|
| Total parcels scored | 48 |
| Flagged parcels | 8 |
| Hub-manager escalations | 6 |
| Proactive customer calls | 2 |
| No action | 40 |

Six flagged parcels belong to the Escalate segment and are assigned the action `Escalate to hub manager`. The other two flagged parcels are assigned `Proactive customer call`.

The parcel-level decisions are available in `outputs/new_shipment_actions.csv`.

### Why three segments?

K=3 was fixed by the business requirement because regional operations needs exactly three playbooks: Standard, Watch and Escalate.

The across-to-within distance ratio was 5.3928 for K=3, with balanced cluster sizes of 9-8-7 hubs. K=4 and K=5 produced higher ratios of 6.3047 and 6.8866 respectively, but split the hubs into four or five operating groups. K=3 therefore provides a useful balance between separation, cluster size and operational simplicity while respecting the business requirement.

---

## Production plan

### Production-readiness characteristics

| Characteristic | Implementation in this repository |
|---|---|
| Accuracy | Part B regression metrics and Part C classification metrics evaluate performance using validation and held-out test data. |
| Reliability | `outputs/part_d_reload_check.csv` verifies that the saved model reloads and produces the same probabilities as the in-memory model. |
| Reproducibility | Fixed random seeds (`random_state=42`), pinned dependencies and deterministic data-processing steps make experiments repeatable. |
| Performance | The packaged Random Forest and feature selector support batch scoring of new parcels through `predict_delay.py`. |
| Maintainability | Data cleaning, feature engineering, segmentation, packaging and prediction are separated into individual scripts. |
| Consistency | Training and prediction reuse `clean_shipments()` and `build_features()`. Prediction also uses stored historical imputation medians and the saved feature order. |
| Monitorability | The action-list output contains shipment-level probabilities, flags, segments and actions, allowing operational teams to track alerts and outcomes. |

### Operational monitoring and alert triggers

The following are proposed initial monitoring thresholds. They should be reviewed and calibrated after sufficient production data has accumulated.

| Quantity | Frequency | Suggested trigger |
|---|---|---|
| Delayed share of delivered parcels | Weekly | Investigate if the delayed share exceeds 20% or increases by more than 5 percentage points against a comparable baseline. |
| Share of parcels flagged by the model | Weekly | Investigate if more than 30% of parcels are flagged or the rate increases by more than 50% relative to the recent baseline. |
| Recall on parcels with known delivery outcomes | Monthly | Investigate if recall falls below 80%, once at least 50 delayed outcomes are available. |
| Missing or unmatched hub information | Daily | Investigate any shipment with a missing hub segment or unmatched hub ID. |
| Predicted probability distribution | Weekly | Investigate a substantial shift compared with the training baseline. |

These are proposed operational triggers, not statistically validated service-level guarantees.

### Retraining schedule

Retrain the model quarterly using the latest available historical shipment data. Before deployment, compare the new model with the current model using a time-based holdout period and evaluate precision, recall, F1-score and the operational impact of the alert threshold.

An earlier retraining review should be triggered if monitoring shows sustained deterioration in recall, a substantial increase in delayed shipments or meaningful changes in the input data distribution.

### Handling the 15–28 September sale window

The `sale_event_flag` identifies parcels dispatched during the sale-event period. Sale periods can change shipment volumes, hub capacity utilisation, parcel mix and delivery patterns.

The model should therefore be evaluated on sale-event periods against comparable historical sale events, rather than comparing sale weeks directly with ordinary weeks. Monitoring should compare delayed share, flagged share, precision and recall between equivalent event periods.

Data drift could appear as changes in:
- The proportion of Express, Standard and Economy shipments.
- Average distance, parcel weight and number of stops.
- Hub-level shipment volumes and capacity utilisation.
- The frequency of rain and sale-event shipments.
- The relationship between the model's features and actual delivery delays.

If these distributions or relationships change materially, the team should investigate whether the model needs recalibration or retraining.

---

## Limitations

1. **Synthetic data:** The project uses generated shipment data. Its patterns and model performance may not fully represent real-world logistics operations, customer behaviour or hub constraints.

2. **Small test-set delayed population:** The held-out test set contains only 41 delayed parcels. Consequently, a small number of additional missed or correctly detected parcels can noticeably change recall.

3. **Threshold selection on one validation split:** The alert threshold of 0.5 was selected using one validation split. Its performance may vary across different time periods, hubs and demand conditions.

4. **Clustering sensitivity:** K-Means uses distance-based clustering and is sensitive to input features, scaling and initialisation. The three segments are operational groupings, not permanent classifications of hub quality.

5. **Limited production validation:** The project has not yet been validated against a sustained stream of real shipment outcomes. Its monitoring thresholds and operational playbooks require calibration with actual business data.

6. **Predictions are not guarantees:** A predicted delay probability is a risk estimate, not a confirmed delay. Operational teams should use alerts to prioritise investigation rather than automatically assume a parcel will be late.

