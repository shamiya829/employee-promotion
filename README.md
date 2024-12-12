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

## References

- Belyadi, H., & Haghighat, A. (2021). Chapter 5 - Supervised learning. In Machine Learning Guide for Oil and Gas Using Python (pp. 169–295). Gulf Professional Publishing. https://doi.org/10.1016/B978-0-12-821929-4.00004-4 
- de Giorgio, A., Cola, G., & Wang, L. (2023). Systematic review of class imbalance problems in manufacturing. Journal of Manufacturing Systems, 71, 620–644. https://doi.org/10.1016/j.jmsy.2023.10.014 
- Jafor, M., Wadud, M. A., Nur, K., & Rahman, M. M. (2023). Employee promotion prediction using improved AdaBoost machine learning approach. AIUB Journal of Science and Engineering (AJSE), 22(3), 258–266. https://doi.org/10.53799/ajse.v22i3.781 
- Kallner, A. (2018). Formulas. In A. Kallner (Ed.), Laboratory Statistics (2nd ed., pp. 1–140). Elsevier. https://doi.org/10.1016/B978-0-12-814348-3.00001-0 
- Lemons, M. A., & Jones, C. A. (2001). Procedural justice in promotion decisions: using perceptions of fairness to build employee commitment. Journal of Managerial Psychology, 16(4), 268–281. https://doi.org/10.1108/02683940110391517
- PennGuides: Text analysis: Topic modeling. Topic Modelling - Text Analysis - Guides at Penn Libraries. (2024, June 3). https://guides.library.upenn.edu/penntdm/methods/topic_modeling. 
- Ruderman, Marian N., et al. Managerial Promotion: The Dynamics for Men and Women : The Dynamics for Men and Women, Center for Creative Leadership, 1996. ProQuest Ebook Central, http://ebookcentral.proquest.com/lib/utxa/detail.action?docID=2097903. 
- Sashkin, M., & Williams, R. L. (1990). Does fairness make a difference? Organizational Dynamics, 19(3), 56–71.
- Scikit-Learn Developers. (n.d.). 6.3. Preprocessing Data. scikit. https://scikit-learn.org/1.5/modules/preprocessing.html 
- Soomro, A. A., Mokhtar, A. A., Hussin, H. B., Lashari, N., Oladosu, T. L., Jameel, S. M., & Inayat, M. (2024). Analysis of machine learning models and data sources to forecast burst pressure of petroleum corroded pipelines: A comprehensive review. Engineering Failure Analysis, 155, 107747. https://doi.org/10.1016/j.engfailanal.2023.107747 
- Stim, R., & Law, R. S. A. at. (2021, November 25). What is fair use?. Stanford Copyright and Fair Use Center. https://fairuse.stanford.edu/overview/fair-use/what-is-fair-use/ 
- Zaman, M. I. (2021, September 25). Employees Evaluation for Promotion. Kaggle. https://www.kaggle.com/datasets/muhammadimran112233/employees-evaluation-for-promotion/data. 


--- 
