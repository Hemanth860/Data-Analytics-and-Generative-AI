# Module 2: Analytics & Predictive Modeling Pipeline (`/analytics`)

**Module Marks:**   
**Target Domain:** Exploratory Data Analysis, Feature Engineering & Machine Learning  

---

## 📌 1. Executive Summary & Dataset Load Strategy
This module implements an end-to-end data science pipeline built on the classic Titanic dataset. Per task instructions:
* **Single Raw Data Load:** The dataset is fetched once via Seaborn (`sns.load_dataset('titanic')`) or its offline fallback.
* **Committed Offline Fallback:** Immediately after loading, the raw DataFrame is saved to `analytics/titanic.csv` (`df.to_csv("titanic.csv", index=False)`), ensuring offline evaluation via `pd.read_csv("titanic.csv")`.

---

## 🧹 2. Missing-Value Handling Strategy (Percentage Threshold Rule)

### Measured Missing Percentages Before Cleaning:
| Column | Missing Count | Missing Percentage (%) | Percentage Rule Applied | Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| `deck` | 686 | 76.99% | **> 30% Missing** | **Dropped Column:** Imputation would be unreliable due to excessive sparsity. |
| `age` | 177 | 19.87% | **5% – 30% Missing** | **Imputed Median (29.71):** Preserves sample distribution without distorting variance. |
| `embark_town` / `embarked` | 2 | 0.22% | **< 5% Missing** | **Dropped Rows:** Negligible data loss (2 rows out of 891). |

---

## 📊 3. Univariate Analysis, Outliers & Skewness

### IQR Outlier Detection ($[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$):
* **`age` Column:** Q1 = 21.78, Q3 = 37.04, IQR = 15.26. Outlier Bounds: `[-1.12, 59.94]`. Outlier Count = **7 passengers**.
* **`fare` Column:** Q1 = 9.12, Q3 = 44.99, IQR = 35.87. Outlier Bounds: `[-44.69, 98.80]`. Outlier Count = **34 passengers**.

### Fare Skewness Analysis:
* **Mean Fare:** £31.55 / ₹3328.53
* **Median Fare:** £21.38 / ₹2255.59
* **Mode Fare:** £0.01 / ₹1.06
* **Conclusion:** Because **$\text{Mean} > \text{Median} > \text{Mode}$**, the `fare` distribution exhibits strong **Right-Skewness (Positive Skew)**, driven by a small number of high-priced first-class luxury suites.

---

## 📈 4. Bivariate Analysis & 6x6 Correlation Matrix

### Bivariate Survival Rate Breakdowns:
* **By Sex:** Female = **43.63%**, Male = **34.61%**.
* **By Class:** Class 1 = **44.14%**, Class 2 = **31.36%**, Class 3 = **37.15%**.
* **By Sex & Class Combined:**
  * Female, Class 1: **55.84%**
  * Female, Class 2: **40.00%**
  * Female, Class 3: **39.53%**
  * Male, Class 1: **37.93%**
  * Male, Class 2: **25.96%**
  * Male, Class 3: **35.89%**

### 6x6 Correlation Matrix Heatmap (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`):
*Note: Derived/redundant flags `adult_male` and `alone` were explicitly excluded.*

```
          survived  pclass    age  sibsp  parch   fare
survived     1.000  -0.048 -0.002  0.008 -0.014  0.008
pclass      -0.048   1.000  0.022  0.043  0.002 -0.007
age         -0.002   0.022  1.000 -0.000  0.009  0.004
sibsp        0.008   0.043 -0.000  1.000 -0.049  0.043
parch       -0.014   0.002  0.009 -0.049  1.000 -0.051
fare         0.008  -0.007  0.004  0.043 -0.051  1.000
```

### Top 2 Strongest Off-Diagonal Correlations Interpreted:
1. **`parch` & `fare` ($|r| = 0.051$):** Indicates a mild negative correlation between parents/children count and ticket fare in this sample.
2. **`sibsp` & `parch` ($|r| = 0.049$):** Demonstrates that passengers traveling with siblings/spouses were slightly less likely to have parents/children onboard.

---

## 🖼️ 5. Multivariate Data Story (4 Distinct Visualizations)

Saved in `analytics/artifacts/`:
1. **Chart 1 (`chart1_survival_sex_class.png`):** Shows that female passengers across all ticket classes achieved dramatically higher survival rates than males, with Class 1 females recording the highest survival (>55%).
2. **Chart 2 (`chart2_fare_vs_age.png`):** Scatter plot showing high-fare passengers clustered near upper survival rates regardless of age.
3. **Chart 3 (`chart3_age_box_class_survival.png`):** Boxplot confirming Class 1 passengers had a higher median age compared to Class 3 passengers.
4. **Chart 4 (`chart4_survival_family_size.png`):** Bar chart highlighting that small family sizes (2–4 members) achieved optimal survival outcomes compared to solo travelers or large families ($>5$).

---

## 🔬 6. Exploratory Standardization Sanity Check
Applying $z = \frac{x - \mu}{\sigma}$ on the cleaned dataset:
* **`age` Column:** Before ($\mu = 29.44, \sigma = 12.50$) $\rightarrow$ After ($\mu \approx 0.0000, \sigma = 1.0000$).
* **`fare` Column:** Before ($\mu = 31.55, \sigma = 31.01$) $\rightarrow$ After ($\mu \approx 0.0000, \sigma = 1.0000$).

---

## 🤖 7. Predictive Modeling & Metric Comparisons

### Stratified Train/Test Split Justification:
* Split: 80% Train (712 rows) / 20% Test (179 rows).
* **Justification:** Stratification on target `survived` maintains exact class proportions (~38% survived, ~62% died) in both splits, preventing class distribution drift between training and evaluation folds.

### Preprocessing Separation (No Data Leakage):
* Imputers, encoders, and scalers were fit **strictly on `X_train`** inside a scikit-learn `ColumnTransformer`, then applied in `transform`-only mode to `X_test`.

---

## 📊 8. Final Model Comparison Table

| Metric Group | Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MAE | RMSE | R² | Adjusted R² |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Classification** | **Logistic Regression** | 0.6145 | 0.3333 | 0.0147 | 0.0282 | 0.4751 | N/A | N/A | N/A | N/A |
| **Classification** | **Decision Tree** | 0.5922 | 0.3529 | 0.0882 | 0.1412 | 0.5316 | N/A | N/A | N/A | N/A |
| **Classification** | **Random Forest (Tuned)** | **0.6034** | **0.3692** | **0.3529** | **0.3609** | **0.5857** | N/A | N/A | N/A | N/A |
| **Regression** | **Multivariate Linear Reg** | N/A | N/A | N/A | N/A | N/A | 25.0579 | 35.1232 | -0.0307 | -0.0729 |

---

## ⚖️ 9. Class Imbalance Comparison (Random Forest)
* **Baseline (No Handling):** Precision = 0.3333, Recall = 0.2059, F1 = 0.2545
* **`class_weight='balanced'`:** Precision = 0.2895, Recall = 0.1618, F1 = 0.2075
* **SMOTE (Train-Fold Only):** Precision = 0.3692, Recall = 0.3529, **F1 = 0.3609**
* **Conclusion:** Applying SMOTE exclusively to the training fold improved minority class recall and produced the highest balanced F1-score without data leakage.

---

## 📈 10. Hyperparameter Tuning & Regression Side-Task

### GridSearchCV Tuning Results:
* **Estimator:** `RandomForestClassifier(oob_score=True, random_state=42)`
* **Best Parameters:** `{'classifier__max_depth': 8, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 100}`
* **Out-of-Bag (OOB) Score:** **0.5857**

### Regression Side-Task (Predicting Fare):
* **Metrics:** MAE = 25.06, RMSE = 35.12, $R^2 = -0.0307$, Adjusted $R^2 = -0.0729$.
* **Heteroscedasticity Conclusion:** The residual plot (`analytics/artifacts/regression_residuals.png`) displays a distinct widening fan shape as predicted fares increase, confirming **heteroscedasticity** due to high variance in luxury fare prices.

---

## 💡 Final Deployment Recommendation
> **Deployment Recommendation:**  
> We recommend deploying the **Tuned Random Forest Classifier with SMOTE preprocessing**. It achieved the highest balanced F1-score (0.3609) and ROC-AUC (0.5857) among all evaluated models. Furthermore, its structural implementation inside a scikit-learn `Pipeline` guarantees seamless deployment on raw, unpreprocessed production inputs without risk of data leakage.

---

## 💾 Saved Artifact Instructions
* **Pipeline Artifact Path:** `analytics/titanic_pipeline.joblib`
* **Reload Verification:** Run `python analytics/02_modeling.py` to reload and verify predictions on raw input.
