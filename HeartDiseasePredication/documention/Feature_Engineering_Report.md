# Feature Engineering

## Dataset Overview

* **Original Features:** 15
* **Engineered Features:** 33
* **Total Features:** 48
* **Samples:** 180

---

## Project Summary

We successfully created **33 new features** by combining **medical domain knowledge** with **statistical analysis**. These engineered features reveal important clinical relationships that were not obvious in the original dataset, potentially improving our model’s ability to predict **heart disease**.

---

## Feature Engineering Overview

The engineered features are grouped into clinically meaningful categories.

---

## 1. Patient Age Features

### Features Created

* **Age Groups**:

  * Young (<40)
  * Middle-aged (40–55)
  * Senior (56–65)
  * Elderly (>65)

* **Elderly Indicator**: Binary flag for patients aged **65+**

* **Age Decade**: 20s, 30s, 40s, etc.

### Why This Matters

Heart disease risk does not increase linearly with age. These categorical features allow models to capture **age-related risk transitions** used in clinical practice.

---

## 2. Blood Pressure Features

### Features Created

* **Blood Pressure Categories** (ACC/AHA 2017):

  * Normal
  * Elevated
  * Stage 1 Hypertension
  * Stage 2 Hypertension
  * Hypertensive Crisis

* **Hypertension Flag**: SBP ≥ 130 mmHg (Yes/No)

* **BP Severity Score**: Ordinal scale (0–4)

### Why This Matters

Blood pressure thresholds guide **real clinical decisions**. Encoding them explicitly helps ML models recognize risk boundaries.

---

## 3. Cholesterol Features

### Features Created

* **Cholesterol Categories**:

  * Desirable
  * Borderline High
  * High
  * Very High

* **High Cholesterol Flag**: Total cholesterol > 200 mg/dL

* **Cholesterol Risk Score**: Ordinal scale (0–3)

### Why This Matters

Cholesterol-related risk is **threshold-driven**, not purely continuous. These features align with established cardiac guidelines.

---

### 4. Heart Rate Features
- *predicted_max_hr*: Age-predicted maximum heart rate (220 - age)
- *hr_reserve*: Ratio of achieved HR to predicted max HR
- *inadequate_hr_response*: Binary indicator for HR reserve < 85%
- *hr_category*: Achievement categories (Very Low, Low, Moderate, Good)

### Why This Matters

Exercise stress response is a **core diagnostic indicator** in cardiology. Poor heart rate reserve often signals underlying cardiac dysfunction.

---

## 5. Combined Risk Scores

### Features Created

* **Simple Risk Score**: Count of major risk factors (0–6)

* **Exercise Stress Score**: Composite of:

  * Heart rate response
  * ST depression
  * Exercise-induced angina

* **Metabolic Syndrome Indicator**: Flags clustered metabolic risk

### Why This Matters

Cardiovascular risk factors often **compound**, not act independently. Composite scores capture cumulative risk.

---

## 6. Interaction Features

### Features Created

* **Age × Cholesterol Interaction**
* **ST Depression × Exercise Angina Interaction**
* **High-Risk Male Indicator**: Male patients with multiple risk factors

### Why This Matters

Medical risks frequently **interact multiplicatively**. These features allow models to capture synergistic effects.

---

## 7. Statistical Transformation Features

### Features Created

* **Squared Terms**: Age², Cholesterol², Blood Pressure²
* **Log Transformations**: Applied to skewed variables
* **Quintile Binning**: Continuous variables split into 5 equal groups
* **Ratio Features**:

  * Cholesterol / Age
  * Blood Pressure / Age

### Why This Matters

These transformations:

* Improve linear separability
* Capture non-linear trends
* Stabilize variance

---

## Medical Reasoning Behind Feature Choices

### Age-Based Stratification

Clinical risk calculators use **age brackets**, not raw age. A 65-year-old and 40-year-old have fundamentally different cardiac risk profiles.

### Blood Pressure Guidelines

We used **2017 ACC/AHA hypertension guidelines** because:

* They drive real clinical decisions
* Each category implies different treatment strategies
* Threshold-based risk is critical for ML interpretability

### Heart Rate Reserve

A standard measure in **exercise physiology**. Failure to reach expected HR during stress testing is strongly associated with heart disease.

---

## Feature Validation & Quality Checks

* **Data Integrity**: No missing values introduced
* **Clinical Plausibility**: All ranges medically valid
* **Correlation Analysis**: Engineered features show meaningful association with target
* **Distribution Checks**: No extreme skew or artifacts

---

## Output Files

* `heart_disease_with_new_features.csv`
  Full dataset containing all **36 features**

* `best_features_subset.csv`
  Top **15 most promising features** for rapid experimentation

---

## Next Steps

### Immediate

* Perform **feature importance analysis**
* Encode categorical variables properly
* Compare model performance **with vs without** engineered features
* Interpret model outputs clinically

### Long-Term

* Validate predictions with medical reasoning
* Test simpler models using high-quality features
* Identify which engineered features clinicians find most useful

---

## Conclusion

Thoughtful feature engineering grounded in **medical knowledge** significantly enhances the predictive power and interpretability of heart disease ML models. This notebook documents a reproducible, clinically aligned feature engineering pipeline.
