# Loan-approval-ml-model

# Machine Learning Model for Loan Approval

## Project Overview

FinTech Innovations currently relies heavily on manual loan application reviews, which can be time-consuming and may lead to inconsistent decisions. This project develops a machine learning classification model to support faster and more consistent loan approval decisions while considering the financial consequences of incorrect predictions.

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework, covering business understanding, data understanding, data preparation, modeling, evaluation, and recommendations.

The dataset contains **20,000 loan applications** with applicant financial, credit, employment, asset, liability, and loan-related information. The target variable is `LoanApproved`, indicating whether an application was approved.

## Business Problem

The business has different financial consequences for incorrect predictions:

* **False Positive:** An incorrectly approved loan is assumed to cost **$50,000**.
* **False Negative:** An incorrectly denied creditworthy applicant is assumed to represent **$8,000** in lost profit.

A false positive is therefore **6.25 times more costly** than a false negative.

Because of these asymmetric costs and the imbalance in the target variable, accuracy alone is not sufficient for evaluating the model.

## Objectives

The project aims to:

* Explore and understand the loan application data.
* Identify important patterns and relationships associated with loan approval.
* Prepare numerical, categorical, and ordinal features for machine learning.
* Compare multiple classification algorithms.
* Tune the selected model using cross-validation.
* Evaluate model performance using multiple metrics.
* Calculate the estimated financial cost of classification errors.
* Evaluate model performance across applicant segments.
* Identify potential data leakage and model limitations.
* Provide recommendations for potential business implementation.

## Dataset

The dataset contains **20,000 observations**.

The target variable is:

`LoanApproved`

The target distribution is approximately:

* **76% Not Approved**
* **24% Approved**

The dataset includes features relating to:

* Applicant demographics
* Employment
* Education
* Income
* Credit history
* Debt
* Assets and liabilities
* Loan characteristics
* Interest rates
* Risk measures

During data preparation, missing values were identified in `MaritalStatus`, `EducationLevel`, and `SavingsAccountBalance`. `AnnualIncome` also required formatting because it was initially stored as text containing currency symbols and commas.

## Methodology

### 1. Exploratory Data Analysis

The exploratory analysis included:

* Dataset structure and summary statistics
* Duplicate checks
* Missing-value analysis
* Target distribution
* Numerical feature distributions
* Categorical feature distributions
* Group-level comparisons
* Correlation analysis

The correlation analysis identified several notable relationships with loan approval. `RiskScore` had the strongest correlation with `LoanApproved` at **-0.766**, while `MonthlyIncome` and `AnnualIncome` had correlations of **0.604** and **0.598**, respectively.

These correlations represent associations and should not be interpreted as evidence of causation.

### 2. Data Preparation

A preprocessing pipeline was created using `ColumnTransformer` and `Pipeline`.

**Numerical features:**

* Median imputation
* Standard scaling

**Categorical and ordinal features:**

* Most-frequent imputation
* One-hot encoding

`EducationLevel` was treated as categorical rather than assigning numerical values to the education levels. This avoids assuming that the difference between education categories is equally spaced.

The preprocessing steps were placed inside the modeling pipeline to help prevent data leakage during training and testing.

### 3. Model Selection

Three classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

These models provided a comparison between an interpretable linear model and tree-based models capable of capturing nonlinear relationships and interactions.

### 4. Hyperparameter Tuning

The Random Forest model was tuned using **GridSearchCV with 5-fold cross-validation**.

The selected configuration was:

| Parameter           | Value |
| ------------------- | ----: |
| `n_estimators`      |   200 |
| `max_depth`         |    20 |
| `min_samples_split` |     5 |
| `min_samples_leaf`  |     1 |

The best cross-validation ROC-AUC was approximately **0.9994**.

## Model Performance

The tuned Random Forest was evaluated on a held-out test set.

| Metric          |       Result |
| --------------- | -----------: |
| Accuracy        |   **99.05%** |
| Precision       |   **98.52%** |
| Recall          |   **97.49%** |
| ROC-AUC         |   **99.94%** |
| Business Cost   | **$892,000** |
| False Positives |       **14** |
| False Negatives |       **24** |

The business cost was calculated using:

```text
Business Cost =
(False Positives × $50,000) +
(False Negatives × $8,000)
```

For the final model:

```text
(14 × $50,000) + (24 × $8,000)
= $700,000 + $192,000
= $892,000
```

The $892,000 represents an **estimated cost under the assumptions provided for this project**, rather than an observed financial loss.

## Feature Importance

The Random Forest feature importance analysis identified the following major predictors:

| Feature                | Importance |
| ---------------------- | ---------: |
| RiskScore              |     45.75% |
| TotalDebtToIncomeRatio |     12.93% |
| MonthlyIncome          |     10.03% |
| AnnualIncome           |      9.81% |
| InterestRate           |      3.81% |
| LoanAmount             |      2.38% |

`RiskScore` was by far the most influential feature in the model.

This result requires additional investigation because `RiskScore` also had the strongest correlation with the target. Before deployment, it would be necessary to confirm that the score is calculated only from information available at the time of application and does not incorporate the approval decision or subsequent loan information.

Feature importance represents predictive contribution and does not establish causation.

## Segment Analysis

Model performance was also examined across selected applicant segments.

### Employment Status

| Segment       | Precision | Recall |
| ------------- | --------: | -----: |
| Employed      |    98.30% | 98.18% |
| Unemployed    |   100.00% | 98.00% |
| Self-Employed |   100.00% | 90.24% |

### Education Level

Recall ranged from approximately **96.31% to 98.80%** across education groups.

### Bankruptcy History

| Segment            | Precision | Recall |
| ------------------ | --------: | -----: |
| No Bankruptcy      |    98.48% | 97.95% |
| Bankruptcy History |   100.00% | 82.14% |

The lower recall observed for self-employed applicants and applicants with a bankruptcy history should be monitored. These differences do not by themselves establish model bias and may be influenced by sample size, feature distributions, or other characteristics of the data.

## Key Findings

1. The tuned Random Forest achieved very strong predictive performance on the held-out test set.
2. The model achieved **99.94% ROC-AUC** and **99.05% accuracy**.
3. The model produced 14 false positives and 24 false negatives.
4. Under the project's cost assumptions, these errors resulted in an estimated business cost of **$892,000**.
5. `RiskScore` was the dominant predictive feature, contributing approximately **45.75%** of total feature importance.
6. Income and debt-related features were also among the most important predictors.
7. Some differences in recall were observed across employment and bankruptcy-history segments.
8. The extremely high model performance and dominant role of `RiskScore` warrant further investigation for potential data leakage.

## Business Recommendations

The model could be considered as a **decision-support tool** for loan officers rather than an unconditional replacement for human review.

Before production deployment, the following steps are recommended:

* Validate how `RiskScore` is calculated and confirm that it contains only information available at application time.
* Validate the assumed false-positive and false-negative costs using actual financial outcomes.
* Evaluate different probability thresholds because false positives have substantially higher assumed costs.
* Monitor model performance across applicant segments.
* Conduct formal fairness analysis where appropriate.
* Perform temporal or external validation using new loan applications.
* Monitor for changes in data distributions and model performance over time.
* Compare the machine learning system against the existing manual approval process.

## Limitations

Several limitations should be considered:

* The target represents historical loan approval decisions rather than actual loan default outcomes.
* The financial costs used in the analysis are assumptions provided for the project.
* The extremely high predictive performance may indicate that some features contain information very closely related to the target.
* `RiskScore` requires additional investigation for potential target leakage.
* Segment-level differences were examined, but a formal fairness analysis was not performed.
* The model was evaluated using a held-out test set from the available dataset; external or temporal validation would provide stronger evidence of real-world performance.

## Tools and Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Jupyter Notebook
* Git/GitHub

## Project Structure

```text
loan-approval-machine-learning/
│
├── data/
│   └── financial_loan_data.csv
│
├── notebooks/
│   └── loan_approval_analysis.ipynb
│
├── images/
│   └── loan.png
│
├── environment.yml
├── README.md
└── .gitignore
```

## Conclusion

The project demonstrates how machine learning can be applied to a financial decision-support problem while incorporating both predictive performance and business considerations.

The Random Forest model achieved strong performance across the selected evaluation metrics. However, the results should not be interpreted as sufficient evidence for immediate production deployment. Validating feature availability, investigating the dominant RiskScore variable, optimizing the decision threshold, validating financial assumptions, and monitoring segment-level performance are important next steps before using the model in a real loan approval environment.

