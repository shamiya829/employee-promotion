# Employee Promotion Prediction

## Overview

This project aims to predict employee promotions within an organization using advanced machine learning techniques. The goal is to create a fair, objective, and data-driven system to assist Human Resources (HR) teams in making promotion decisions, minimizing biases, and improving overall organizational efficiency. 

The solution employs techniques such as data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model training, hyperparameter optimization, and threshold adjustment. Evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC were used to assess the effectiveness of the models.

## Background

Promotion decisions in organizations often rely on subjective evaluations, which can be inconsistent and prone to cognitive biases. These biases undermine diversity, equity, and employee morale. The project addresses this problem by developing a machine-learning-based predictive model that ensures fairness and efficiency.

Inspired by the study *“Employee Promotion Prediction Using Improved AdaBoost Machine Learning Approach”* by Jafor et al. (2023), this project explores and optimizes various machine learning algorithms, including Gradient Boosting, AdaBoost, and LightGBM.

---

## Project Features

- **Data Preprocessing**:
  - Imputation of missing values.
  - Handling outliers.
  - Encoding categorical features.
  - Standardizing numerical features.

- **Exploratory Data Analysis (EDA)**:
  - Distribution and correlation analysis.
  - Insights on feature importance.

- **Machine Learning Models**:
  - AdaBoost
  - Gradient Boosting
  - LightGBM

- **Optimization**:
  - Hyperparameter tuning using RandomizedSearchCV.
  - Class balancing with SMOTE (Synthetic Minority Oversampling Technique).
  - Threshold adjustments using Youden’s J statistic, Top-Left Corner, and F1-score maximization.

- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score, and ROC-AUC.
  - Confusion matrices and ROC curve visualizations.

---
## Dataset

The dataset, **Employees Evaluation for Promotion**, was sourced from [Kaggle](https://www.kaggle.com/datasets/muhammadimran112233/employees-evaluation-for-promotion). It contains 13 features across 54,808 rows.

### Key Features:
| **Feature**               | **Description**                                           |
|---------------------------|-----------------------------------------------------------|
| `employee_id`             | Unique identifier for each employee                       |
| `department`              | Department of the employee                                |
| `region`                  | Geographic region of the employee                         |
| `education`               | Highest education level                                   |
| `gender`                  | Gender (Male/Female)                                      |
| `recruitment_channel`     | Channel through which the employee was recruited          |
| `no_of_trainings`         | Number of training programs completed                     |
| `age`                     | Age of the employee                                       |
| `previous_year_rating`    | Performance rating in the previous year                   |
| `length_of_service`       | Number of years in the company                            |
| `awards_won`              | Binary indicator for awards won (1: Yes, 0: No)          |
| `avg_training_score`      | Average score in recent training                          |
| `is_promoted`             | Target variable: 1 (Promoted), 0 (Not Promoted)           |

---

## Workflow

1. **Data Preprocessing**:
   - Imputed missing values (median for numerical, mode for categorical).
   - Scaled numerical features and encoded categorical variables.
   - Split data into training and testing sets.

2. **Exploratory Data Analysis**:
   - Visualized feature distributions and relationships.
   - Highlighted key insights, such as the correlation between `avg_training_score` and promotion likelihood.

3. **Feature Engineering**:
   - Selected features based on importance using ablation testing.
   - Applied LabelEncoding for categorical features.
   - Applied StandardScaler to standardize numerical features.

4. **Model Training**:
   - Trained AdaBoost, Gradient Boosting, and LightGBM models.
   - Evaluated using baseline metrics and ROC-AUC.

5. **Optimization**:
   - Conducted hyperparameter tuning with RandomizedSearchCV.
   - Used SMOTE to balance the dataset.
   - Adjusted thresholds using multiple optimization strategies.

6. **Evaluation**:
   - Compared models based on accuracy, precision, recall, and F1-score.
   - Visualized performance with ROC curves and confusion matrices.

---

## Key Findings

1. Employees with high `avg_training_score` and longer `length_of_service` are more likely to be promoted.
2. LightGBM provided the best balance of performance metrics.
3. Threshold optimization and SMOTE effectively addressed class imbalance.

---

## Contributors

- **Naomi Ichiriu**
- **Shamiya Lin**: 
---

## License

This project is for educational purposes. Dataset usage falls under fair use guidelines.

--- 
