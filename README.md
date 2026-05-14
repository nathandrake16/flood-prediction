# Machine Learning-Based Flood Prediction for Disaster Preparedness and Response

This repository contains an academic machine learning project focused on flood-related disaster preparedness. The work compares multiple classification models to predict whether evacuation is required based on environmental and flood-risk indicators.

The project is not a production flood-warning system. It is a supervised machine learning experiment that performs data exploration, preprocessing, model training, evaluation, and result comparison on a flood prediction dataset.

## Project Overview

Floods can cause severe damage to lives, infrastructure, and emergency response systems. Early prediction helps authorities prepare evacuation plans, allocate resources, and reduce risk. This project explores whether machine learning models can learn useful patterns from flood-related features and classify evacuation requirement.

The main objective is to compare classical machine learning algorithms on the same processed dataset and identify which model performs better for this classification task.

## What This Project Does

The notebook performs the following workflow:

1. Loads the flood dataset from a public Google Drive CSV link.
2. Performs exploratory data analysis on the dataset.
3. Checks dataset shape, data types, missing values, class distribution, and feature distribution.
4. Visualizes relationships between evacuation requirement and environmental variables.
5. Removes less useful or redundant columns.
6. Handles missing values using mean imputation and row removal where necessary.
7. Encodes categorical flood-risk labels using one-hot encoding.
8. Splits the cleaned dataset into training and testing sets.
9. Trains and evaluates multiple machine learning models.
10. Compares model performance using accuracy, F1-score, classification reports, and confusion matrices.

## Dataset Description

The original dataset contains 5,050 records and 6 columns:

| Column | Description |
|---|---|
| `Rainfall_mm` | Rainfall amount measured in millimeters |
| `River_Level_m` | River water level measured in meters |
| `Soil_Moisture_%` | Soil moisture percentage |
| `City` | Area type such as rural, urban, or suburban |
| `Flood_Risk` | Categorical flood risk level: low, moderate, or high |
| `Evacuation_Required` | Target variable indicating whether evacuation is required |

After preprocessing, the final dataset contains 4,800 records and 7 features after one-hot encoding the `Flood_Risk` column.

## Preprocessing Steps

The following preprocessing operations were applied:

- Removed duplicate rows.
- Dropped the `City` column because it was treated as less useful for the final prediction task.
- Filled missing values in `Soil_Moisture_%` using the column mean.
- Filled missing values in `River_Level_m` using the column mean.
- Dropped rows where `Rainfall_mm` was missing because rainfall was considered a critical feature.
- Converted `Flood_Risk` into numerical form using one-hot encoding.
- Used `Evacuation_Required` as the target variable.

Final encoded columns include:

```text
Rainfall_mm
River_Level_m
Soil_Moisture_%
Flood_Risk_High
Flood_Risk_Low
Flood_Risk_Moderate
Evacuation_Required
```

## Models Used

The project evaluates the following models:

| Model | Purpose |
|---|---|
| Decision Tree | Baseline tree-based classifier |
| Random Forest | Ensemble model using multiple decision trees |
| AdaBoost | Boosting-based ensemble classifier |
| K-Nearest Neighbors | Distance-based non-parametric classifier |
| Support Vector Machine | Margin-based classifier using a linear kernel |
| Artificial Neural Network | Multi-layer perceptron classifier tested as an additional model |

## Model Results

The notebook and report show that all models performed close to baseline level, with accuracy mostly around 50% to 54%. This means the dataset and selected features did not allow any model to dominate strongly.

Approximate model performance from the implementation:

| Model | Accuracy | Notes |
|---|---:|---|
| Decision Tree | ~50.6% | Captured some patterns but showed limited generalization |
| Random Forest | ~52.2% | Slightly better than Decision Tree |
| AdaBoost | ~52% to ~54% | Best or near-best accuracy depending on parameter setting |
| KNN | ~51.3% | Balanced but still weak predictive performance |
| SVM | ~50.2% | Collapsed toward predicting the majority/no-flood class |
| ANN / MLP | ~51.0% | Tested as an additional neural-network baseline |

The report identifies AdaBoost as the best-performing model by accuracy, while KNN shows comparatively stronger F1-score behavior. However, the overall results indicate that more feature engineering, better data quality, and stronger imbalance handling are needed before this can become a reliable flood prediction model.

## Key Findings

- The dataset is almost balanced for the target variable, but the models still struggle to separate evacuation and non-evacuation cases clearly.
- Basic environmental features alone were not enough to achieve high predictive performance.
- Random Forest and AdaBoost performed slightly better than single Decision Tree models.
- SVM performed poorly for the flood class because it predicted almost all samples as no flood in the tested configuration.
- The project is useful as a comparative ML baseline, not as a deployable emergency-response tool.

## Repository Structure

```text
.
├── Sec_01_Group_No_04_22101174_22101452_22101821.ipynb   # Main notebook with EDA, preprocessing, model training, and evaluation
├── Sec 01_Group No 04_22101174_22101452_22101821.pdf     # Project report / paper
└── README.md                                             # Project documentation
```

## Requirements

Install the following Python libraries before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. Open the Jupyter notebook:

```bash
jupyter notebook Sec_01_Group_No_04_22101174_22101452_22101821.ipynb
```

4. Run all cells from top to bottom.

The notebook loads the dataset directly from a Google Drive CSV link, so internet access is required unless the dataset is downloaded and loaded locally.

## Evaluation Metrics

The project uses the following metrics:

- Accuracy
- F1-score
- Confusion matrix
- Classification report

Accuracy measures the overall number of correct predictions, while F1-score gives a better view of class-wise performance when false positives and false negatives matter.

## Limitations

This project has several important limitations:

- Model performance is low and close to random baseline.
- The feature set is limited.
- The SVM model predicts only one class in the tested setup.
- Hyperparameter tuning is basic and can be improved.
- Cross-validation and stronger validation strategies were not fully developed.
- No deployment pipeline or real-time flood data integration is included.
- The project should not be used for real disaster decision-making without major improvement and validation.

## Future Improvements

Possible improvements include:

- Add stronger feature engineering using rainfall trends, river-level change rate, seasonality, and geographic indicators.
- Apply proper scaling for distance-based and margin-based models such as KNN and SVM.
- Use cross-validation instead of relying only on a single train-test split.
- Tune hyperparameters using GridSearchCV or RandomizedSearchCV.
- Test imbalance-handling methods such as SMOTE or class weighting.
- Compare more advanced models such as XGBoost, LightGBM, CatBoost, and LSTM-based temporal models.
- Add model explainability using SHAP or feature importance analysis.
- Convert the best model into a simple API or web dashboard for demonstration.

## Conclusion

This project demonstrates a complete beginner-to-intermediate machine learning workflow for flood-related evacuation prediction. The strongest value of the project is not the raw accuracy, but the full experimentation pipeline: data loading, cleaning, visualization, encoding, model comparison, and critical evaluation.

The result shows that flood prediction is a difficult problem when using limited tabular features. Better real-world performance would require richer environmental data, temporal patterns, geospatial context, and more rigorous model validation.
