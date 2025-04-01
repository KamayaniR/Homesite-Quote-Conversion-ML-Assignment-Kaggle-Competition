# Homesite Quote Conversion – ML Assignment & Kaggle Competition

## Predicting Home Insurance Conversion Rates
# Overview
This project is focused on building a machine learning model to predict the likelihood of a customer purchasing a given home insurance quote. The dataset and problem statement were sourced from a Kaggle competition hosted by Homesite Insurance.

# Objective
The goal is to predict the QuoteConversion_Flag (1 for purchase, 0 otherwise) for given insurance quotes based on customer, property, and quote details. Accurate predictions can help Homesite optimize pricing and target customer segments effectively.

# Evaluation Metric
Submissions are evaluated using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), measuring the model's ability to distinguish between classes.

# Key Steps
1) Exploratory Data Analysis (EDA)
Investigated class imbalance and feature correlations.
Visualized target distribution and feature relationships.
Preprocessing

2)Managed missing values.
3)Performed one-hot encoding for categorical variables.
4) Applied SMOTE to handle class imbalance.

# Modeling
Experimented with various classifiers:
Decision Tree
Random Forest
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Multilayer Perceptron (MLP)
Implemented ensemble stacking with Gradient Boosting as the meta-model.

# Hyperparameter Tuning
Used RandomizedSearchCV for optimizing model parameters.

# Evaluation
Measured accuracy and AUC-ROC for model performance on training and validation sets.

# Submission
Generated predictions and saved them in the required submission format for Kaggle.

# Results
Achieved a mean AUC-ROC of 0.955 using the ensemble stacked model.
Saved predictions from various models (DecisionTree, RandomForest, SVM, etc.) for comparison.
