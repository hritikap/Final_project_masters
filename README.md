# Exploratory Analysis, Visualization, and Machine Learning–Based Prediction of Healthcare Insurance Costs in the United States

**Master's Final Project **

---

## Project Overview

This project investigates the key factors driving healthcare insurance costs in the United States using exploratory data analysis, statistical visualization, and machine learning. Using the Kaggle Medical Cost Personal Dataset, the study builds and compares three predictive models — Linear Regression, Random Forest, and Gradient Boosting — to forecast individual insurance charges based on demographic and lifestyle variables.

## Dataset

- **Source:** [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size:** 1,338 records, 7 features
- **Target Variable:** `charges` (individual medical costs billed by insurance)

| Feature    | Type        | Description                                |
| ---------- | ----------- | ------------------------------------------ |
| `age`      | Numerical   | Age of the primary beneficiary (18–64)     |
| `sex`      | Categorical | Gender (male / female)                     |
| `bmi`      | Numerical   | Body mass index                            |
| `children` | Numerical   | Number of dependents covered (0–5)         |
| `smoker`   | Categorical | Smoking status (yes / no)                  |
| `region`   | Categorical | U.S. residential area (NE, NW, SE, SW)     |
| `charges`  | Numerical   | Annual individual medical costs billed ($) |

## Project Structure

Project_main/
├── data/ # Raw dataset (insurance.csv)
├── notebook/ # Jupyter notebook with full analysis
│ └── healthcare_analysis.ipynb
├── figures/ # All saved visualizations (PNG, 300 DPI) ├
── venv/ # Python virtual environment
└── README.md

## Key Findings

### 1. Smoking is the Dominant Cost Driver

Smokers pay approximately **3–4x more** in insurance charges than non-smokers. The average charge for smokers is ~$32,050 compared to ~$8,434 for non-smokers. Smoking status alone explains ~62% of the correlation with charges (r ≈ 0.79).

### 2. Age and BMI are Secondary Predictors

- **Age:** Charges increase consistently with age across all groups. Older beneficiaries (51–64) pay significantly more than younger ones (18–30).
- **BMI:** BMI has a moderate direct effect, but its interaction with smoking status is critical — obese smokers face the highest charges in the dataset.

### 3. Gender, Region, and Children Have Minimal Impact

- Gender differences in charges are statistically insignificant when controlling for other factors.
- Regional variation is minimal, with the Southeast being slightly higher on average.
- Number of children (0–5) does not meaningfully predict charges.

### 4. Non-Linear Models Outperform Linear Regression

Tree-based ensemble methods (Random Forest, Gradient Boosting) significantly outperform Linear Regression because the relationship between features and charges is non-linear — particularly the smoker × BMI interaction.

## Model Performance

### Initial Model Comparison (80/20 Train-Test Split)

| Model             | MAE ($)  | RMSE ($) | R²     |
| ----------------- | -------- | -------- | ------ |
| Linear Regression | 4,181.19 | 5,796.28 | 0.7836 |
| Random Forest     | 2,550.67 | 4,577.74 | 0.8650 |
| Gradient Boosting | 2,664.18 | 4,776.92 | 0.8530 |

### After Hyperparameter Tuning (Gradient Boosting via GridSearchCV)

| Metric          | Value                                                              |
| --------------- | ------------------------------------------------------------------ |
| Best Parameters | `learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8` |
| Best CV R²      | 0.8486                                                             |
| Test R²         | **0.8782**                                                         |
| Test MAE        | **$2,520.75**                                                      |

The tuned Gradient Boosting model achieved the best overall performance, predicting insurance charges within an average error of ~$2,521.

### 5-Fold Cross-Validation

| Model             | Mean R² | Std Dev |
| ----------------- | ------- | ------- |
| Linear Regression | 0.7469  | ±0.0249 |
| Random Forest     | 0.8361  | ±0.0324 |
| Gradient Boosting | 0.8314  | ±0.0423 |

Low standard deviations confirm that all models generalize consistently across different data splits.

## Visualizations

The `figures/` directory contains all generated plots:

- `charges_distribution.png` — Distribution of charges (raw and log-transformed)
- `charges_by_smoker.png` — Box and violin plots of charges by smoking status
- `scatter_age_bmi_charges.png` — Scatter plots of age/BMI vs charges (colored by smoker)
- `charges_by_region_gender.png` — Box plots of charges by region and gender
- `correlation_heatmap.png` — Correlation heatmap of all encoded features
- `charges_children.png` — Box plot of charges by number of children
- `model_comparison.png` — Bar chart comparing MAE, RMSE, R² across models
- `actual_vs_predicted.png` — Actual vs predicted charges (best model)
- `residual_analysis.png` — Residual scatter plot and histogram
- `shap_summary.png` — SHAP summary plot (feature impact on predictions)
- `shap_bar.png` — SHAP bar plot (mean absolute feature importance)
- `rf_feature_importance.png` — Random Forest built-in feature importance

## Methodology

1. **Data Inspection & Cleaning:** Verified no missing values; removed 1 duplicate row.
2. **Exploratory Data Analysis:** Generated 8+ visualizations analyzing cost distributions across all features.
3. **Feature Engineering:** Encoded categorical variables; applied one-hot encoding for region; standardized features for Linear Regression.
4. **Model Training:** Trained Linear Regression, Random Forest (100 trees), and Gradient Boosting (200 trees, lr=0.1, depth=4).
5. **Model Evaluation:** Compared models using MAE, RMSE, and R²; analyzed residuals and actual-vs-predicted plots.
6. **Hyperparameter Tuning:** Applied GridSearchCV (5-fold, 72 combinations) to optimize Gradient Boosting.
7. **Explainability:** Used SHAP (TreeExplainer) for feature importance analysis.
8. **Cross-Validation:** Validated all models with 5-fold cross-validation for robustness.

## Insights & Recommendations

1. **Smoking Cessation Programs:** Given that smoking is the strongest predictor of high costs, investing in cessation programs could yield significant cost reductions for insurers and public health systems.

2. **BMI Management for Smokers:** The smoker-BMI interaction suggests that weight management programs targeting smokers specifically could have outsized impact on reducing extreme insurance claims.

3. **Age-Adjusted Pricing Transparency:** Age is a consistent cost driver. Transparent, actuarially fair pricing models should account for this while maintaining equitable access.

4. **Limitations:**
   - Small dataset (1,338 records) limits the power of complex models.
   - Only 7 features — real insurance data includes plan type, deductibles, claim history, pre-existing conditions, income, etc.
   - No temporal dimension — cannot analyze cost trends over time.
5. **Future Work:** Incorporate larger datasets (e.g., MEPS), add engineered features (BMI category, age group, smoker × BMI interaction), explore deep learning approaches, and investigate model fairness/bias.

## Technologies Used

- **Python 3.11**
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, shap
- **Environment:** Jupyter Notebook, Visual Studio Code

## How to Run

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
4. Launch the notebook:
   jupyter notebook notebook/healthcare_analysis.ipynb
