INFS 7203 Project Submission

Student ID: s4980297

Final Choices:

1] Preprocessing Methods: 

  a] Missing Value Handling:
  - Numerical features (Num_Col1 to Num_Col25): Mean imputation
  - Categorical features (Nom_Col26 to Nom_Col43): Mode imputation with 'MISSING' placeholder encoding
  - Mean imputation preserved data distribution better than median for this dataset, leading to improved F1 performance

  b] Feature Encoding:
  - Categorical features: Label encoding
  - Missing categorical values: Encoded as 'MISSING' before label transformation

  c] Feature Scaling:
  - Numerical features: StandardScaler (zero mean, unit variance)
  - Categorical features: No scaling applied after label encoding

  d] Feature Selection: All 43 original features retained.

2] Classification Model

- Final Model: RandomForestClassifier
- Ensemble Method: Soft Voting Ensemble (Random Forest + Decision Tree + k-NN)
- Random Forest demonstrated the best F1 score (0.655) while maintaining good accuracy (0.78)

3] Hyperparameters:

RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42
)

Environment Description:

1] System Requirements: 

- Operating System: Windows 10/11, macOS, or Linux
- Python Version: 3.8+
- RAM: Minimum 8GB
- Storage: Minimum 1GB free space

2] Required Packages:
- pandas==1.5.0
- numpy==1.23.0
- scikit-learn==1.2.0

3] Installation 
pip install pandas==1.5.0 numpy==1.23.0 scikit-learn==1.2.0

Reproduction Instructions: 
1] File structure: 

project/<br>
├── main.py<br>
├── train.csv<br>
├── test_data.csv<br>
├── requirements.txt<br>
└── README.md<br>

2] Environment setup: 
pip install -r requirements.txt

3] Execution: python main.py 

4] Output: 

The script will generate:
- Terminal output showing model performance metrics
- Submission file: s4980297.infs4203 with 2714 rows

Additional Justifications: 

1] Preprocessing Choices:

- Mean over Median Imputation: Experimental results showed mean imputation provided better F1 performance, likely due to preserving the original data distribution more effectively.
- Label Encoding over One-Hot: Chosen to avoid high dimensionality with 18 categorical features.
- StandardScaler: Essential for distance-based algorithms (k-NN) and improves tree-based model performance.

2] Model Selection Rationale: 

- Random Forest: Selected for its robustness to overfitting and ability to handle mixed data types.
- Class Weight Balancing: Used class_weight='balanced' to address class imbalance (7200 class 0 vs 3653 class 1).
- Soft Voting Ensemble: Explored but Random Forest individual performance was superior.

3] Hyperparameter Tuning Strategy: 

- Comprehensive Grid Search: Tested multiple parameter combinations for each classifier.
- F1 Optimization: All tuning focused on maximizing F1 score as per assignment requirements.
- Stratified Cross-Validation: Used to maintain class distribution in validation folds.

References: 

- scikit-learn documentation for implementation details

| AI Tool | Purpose | Prompt Used | Section Affected in Project |
|----------|----------|--------------|-----------------------------|
| ChatGPT | Data Preprocessing Help | "How to handle missing values in a dataset with mixed data types?" | Data Cleaning Pipeline |
| ChatGPT | Model Selection Guidance | "Compare Random Forest vs XGBoost for classification with imbalanced data" | Model Implementation Section |
| ChatGPT | Hyperparameter Tuning | "Best hyperparameter optimization in scikit-learn" | Model Training Script |
| ChatGPT | Performance Metrics | "Which evaluation metrics are best for multi-class classification problems?" | Model Evaluation Module |


