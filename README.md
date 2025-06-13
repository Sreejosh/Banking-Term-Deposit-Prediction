# Banking-Term-Deposit-Prediction

📌 Project Overview
This project focuses on predicting whether a customer will subscribe to a term deposit product based on various features collected from a direct marketing campaign by a Portuguese banking institution. It is a binary classification problem tackled using machine learning models like KNN, Decision Trees, Logistic Regression, and more.

The goal is to assist banks in identifying potential clients who are likely to subscribe to term deposit products, thereby increasing marketing efficiency and reducing costs.

🧠 Problem Statement
Financial institutions run marketing campaigns to promote long-term deposit subscriptions. However, contacting every customer is expensive and often ineffective. The objective is to build a predictive model that will:

✅ Predict if a client will subscribe to a term deposit (yes or no) based on historical client and campaign data.

🎯 Business Use Case
Improve marketing ROI by targeting likely subscribers.

Personalize campaign strategies using predicted customer behavior.

Reduce telemarketing costs by minimizing unnecessary calls.

Optimize resource allocation across departments.

📁 Dataset Information
The dataset used is the Bank Marketing Data Set from the UCI Machine Learning Repository.

Attributes Include:

Client Data: age, job, marital status, education, etc.

Campaign Info: contact method, number of contacts, previous outcomes.

Economic Context: employment variation rate, consumer price index, etc.

Target Variable: y (yes/no – client subscribed a term deposit)

⚙️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Jupyter Notebook

📊 Exploratory Data Analysis (EDA)
Analyzed class imbalance.

Univariate and bivariate visualizations to identify trends.

Checked correlations between features.

Handled missing and categorical data.

🔍 Data Preprocessing
Label encoding and one-hot encoding for categorical variables.

Feature scaling for continuous variables using StandardScaler.

Train-test split for validation.

Balanced dataset handling if required (e.g., SMOTE or under-sampling).

🤖 Models Used
Common ML models trained and evaluated:

K-Nearest Neighbors (KNN)

Logistic Regression

Decision Tree Classifier

Random Forest

Support Vector Machine (SVM)

📌 Metrics Used:

Accuracy

Precision, Recall, F1 Score

Confusion Matrix

ROC-AUC Curve

🧪 Model Evaluation
Each model was evaluated and compared based on:

Classification Report

ROC-AUC Scores

Precision vs Recall Trade-off

Feature Importance (for tree-based models)

✅ Results
[Insert best-performing model here, e.g., Random Forest gave the best accuracy of ~X%]

Important features influencing the prediction were: duration, previous outcome, contact method, and campaign.

