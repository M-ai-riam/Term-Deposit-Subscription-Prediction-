Term Deposit Subscription Prediction
This project implements a complete machine learning pipeline to predict whether a banking client will subscribe to a term deposit based on historical campaign data (bank-additional-full.csv).

■ Task objective
The primary goal of this project is to develop a predictive classification model that identifies potential customers likely to subscribe to a term deposit. By leveraging historical marketing campaign data, the model aims to:

Optimize Marketing Efficiency: Assist financial institutions in targeting the right audience, reducing operational costs from cold-calling.

Benchmark Performance: Compare a traditional linear baseline model against a non-linear ensemble method.

Provide Explainable AI (XAI): Uncover the underlying features driving model predictions to give stakeholders actionable insights into customer behavior.

■ Your approach
The project utilizes a structured, production-grade data science workflow broken down into the following stages:

Exploratory Data Analysis (EDA): Evaluates dataset dimensions, data types, missing values, and statistical distributions to ensure data integrity.

Feature Engineering & Preprocessing:

Separates numeric and categorical attributes automatically.

Encodes categorical text features using LabelEncoder.

Splits the dataset into an 80/20 train-test split, implementing stratification to preserve the balance of the target class (y).

Feature scales the data using StandardScaler to prevent feature dominance and data leakage.

Model Training & Comparison:

Logistic Regression: Serves as a fast, interpretable linear baseline (configured with max_iter=1000).

Random Forest Classifier: Serves as a robust non-linear ensemble model (configured with n_estimators=100 and max_depth=10 to control overfitting).

Model Evaluation: Evaluates both architectures across three standard classification metrics: Accuracy, F1-Score, and ROC-AUC Score.

Visual Analytics: Generates side-by-side Confusion Matrices and overlapping Receiver Operating Characteristic (ROC) curves for clear visual performance diagnostics.

Explainable AI (SHAP): Deploys LinearExplainer and TreeExplainer to extract and plot global feature importance plots, visualizing exactly how the models weight different customer variables.

■ Results and findings
Upon running the pipeline, the codebase provides comparative metrics and interpretable plots:

1. Performance Summary
The script outputs a direct statistical comparison between the two trained models:

Plaintext
Logistic Regression:
  Accuracy: <score>, F1: <score>, AUC: <score>

Random Forest:
  Accuracy: <score>, F1: <score>, AUC: <score>
2. Key Findings & Interpretability
Model Trade-offs: The side-by-side Confusion Matrix and ROC Curves allow you to see whether the Random Forest model's non-linear boundaries outperform Logistic Regression in handling class imbalance.

Feature Importance (SHAP): The SHAP summary bar charts highlight the top features driving predictions. Typically, in this dataset, campaign factors like call duration and macroeconomic indicators play a massive role in whether a client says "Yes".

Granular Predictions: The script finishes by printing exact case-by-case probability distributions against the actual ground truth, proving the models' reliability on unseen data:

Plaintext
Prediction #1:
  Actual: NO
  LR: NO (14.20%)
  RF: NO (8.54%)
⚙️ How to Run
Clone the repository:

Bash
git clone https://github.com/your-username/term-deposit-subscription-prediction.git
cd term-deposit-subscription-prediction
Install requirements:

Bash
pip install pandas numpy scikit-learn matplotlib shap
Data Placement: Place your bank_additional_full.csv file directly into the project root folder.

Run: Run the code script or .ipynb notebook file to train the models and generate the diagnostic visualizations.
