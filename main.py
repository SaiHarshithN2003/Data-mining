import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.label_encoders = {}
        
    def fit_transform(self, data):
        target_column = 'Target (Col44)'
        
        # Separate features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        numeric_cols = [f'Num_Col{i}' for i in range(1, 26)]
        categorical_cols = [f'Nom_Col{i}' for i in range(26, 44)]
        
        # MEAN imputation for numerical features
        self.numeric_imputer = SimpleImputer(strategy='mean')
        X_numeric = self.numeric_imputer.fit_transform(X[numeric_cols])
        
        # Handle categorical features
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_categorical = self.categorical_imputer.fit_transform(X[categorical_cols])
        
        # Label encode categorical features
        X_categorical_encoded = np.zeros_like(X_categorical, dtype=float)
        for i in range(X_categorical.shape[1]):
            le = LabelEncoder()
            col_data = X_categorical[:, i]
            mask = pd.isna(col_data)
            if mask.any():
                col_data = col_data.astype(str)
                col_data[mask] = 'MISSING'
            else:
                col_data = col_data.astype(str)
            
            X_categorical_encoded[:, i] = le.fit_transform(col_data)
            self.label_encoders[i] = le
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        
        # Combine features
        X_processed = np.hstack([X_numeric_scaled, X_categorical_encoded])
        
        return X_processed, y
    
    def transform(self, data):
        numeric_cols = [f'Num_Col{i}' for i in range(1, 26)]
        categorical_cols = [f'Nom_Col{i}' for i in range(26, 44)]
        
        X_numeric = self.numeric_imputer.transform(data[numeric_cols])
        X_categorical = self.categorical_imputer.transform(data[categorical_cols])
        
        X_categorical_encoded = np.zeros_like(X_categorical, dtype=float)
        for i in range(X_categorical.shape[1]):
            le = self.label_encoders[i]
            col_data = X_categorical[:, i]
            mask = pd.isna(col_data)
            if mask.any():
                col_data = col_data.astype(str)
                col_data[mask] = 'MISSING'
            else:
                col_data = col_data.astype(str)
            
            try:
                X_categorical_encoded[:, i] = le.transform(col_data)
            except ValueError:
                X_categorical_encoded[:, i] = 0
        
        X_numeric_scaled = self.scaler.transform(X_numeric)
        X_processed = np.hstack([X_numeric_scaled, X_categorical_encoded])
        
        return X_processed

class F1Optimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_f1 = 0
        self.model_scores = {}
        
    def find_best_model(self, X, y):
        """Find the best model using F1-focused optimization"""
        print("Testing classifiers...")
        
        # Test all individual models with comprehensive parameter grids
        rf_model, rf_f1 = self._optimize_random_forest(X, y)
        dt_model, dt_f1 = self._optimize_decision_tree(X, y)
        knn_model, knn_f1 = self._optimize_knn(X, y)
        nb_model, nb_f1 = self._test_naive_bayes(X, y)
        
        # Try ensemble with best models
        ensemble_model, ensemble_f1 = self._create_ensemble(X, y, [rf_model, dt_model, knn_model])
        
        # Print all scores
        print("\nClassifier F1 Scores:")
        for model_name, score in self.model_scores.items():
            print(f"  {model_name}: {score:.3f}")
        
        print(f"\nBest model: {type(self.best_model).__name__}")
        print(f"Best CV F1: {self.best_f1:.3f}")
        
        return self.best_model
    
    def _optimize_random_forest(self, X, y):
        """Optimize Random Forest for F1 score - Comprehensive grid from code 1"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=skf, scoring='f1', 
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        self.model_scores['Random Forest'] = grid_search.best_score_
        
        if grid_search.best_score_ > self.best_f1:
            self.best_f1 = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            
        return grid_search.best_estimator_, grid_search.best_score_
    
    def _optimize_decision_tree(self, X, y):
        """Optimize Decision Tree for F1 score - Comprehensive grid from code 1"""
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'class_weight': ['balanced', None]
        }
        
        dt = DecisionTreeClassifier(random_state=self.random_state)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            dt, param_grid, cv=skf, scoring='f1',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        self.model_scores['Decision Tree'] = grid_search.best_score_
        
        if grid_search.best_score_ > self.best_f1:
            self.best_f1 = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            
        return grid_search.best_estimator_, grid_search.best_score_
    
    def _optimize_knn(self, X, y):
        """Optimize k-NN for F1 score - Comprehensive grid from code 1"""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance']
        }
        
        knn = KNeighborsClassifier()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            knn, param_grid, cv=skf, scoring='f1',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        
        self.model_scores['k-NN'] = grid_search.best_score_
        
        if grid_search.best_score_ > self.best_f1:
            self.best_f1 = grid_search.best_score_
            self.best_model = grid_search.best_estimator_
            
        return grid_search.best_estimator_, grid_search.best_score_
    
    def _test_naive_bayes(self, X, y):
        """Test Naive Bayes classifier"""
        nb = GaussianNB()
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        f1_scores = cross_val_score(nb, X, y, cv=skf, scoring='f1')
        nb_f1 = np.mean(f1_scores)
        
        self.model_scores['Naive Bayes'] = nb_f1
        
        if nb_f1 > self.best_f1:
            self.best_f1 = nb_f1
            nb.fit(X, y)
            self.best_model = nb
            
        return nb, nb_f1
    
    def _create_ensemble(self, X, y, models):
        """Create voting ensemble - Using soft voting """
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models[0]), 
                ('dt', models[1]), 
                ('knn', models[2])
            ],
            voting='soft'
        )
        
        # Evaluate ensemble with more folds
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        f1_scores = cross_val_score(ensemble, X, y, cv=skf, scoring='f1')
        ensemble_f1 = np.mean(f1_scores)
        
        self.model_scores['Ensemble'] = ensemble_f1
        
        if ensemble_f1 > self.best_f1:
            self.best_f1 = ensemble_f1
            ensemble.fit(X, y)
            self.best_model = ensemble
            
        return ensemble, ensemble_f1

class CrossValidationEvaluator:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def evaluate(self, model, X, y):
        """Perform cross-validation and return accuracy and F1"""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        accuracy_scores = []
        f1_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            accuracy_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average='binary', pos_label=1))
        
        mean_accuracy = np.mean(accuracy_scores)
        mean_f1 = np.mean(f1_scores)
        
        # Round to three decimal places as required
        mean_accuracy = round(mean_accuracy, 3)
        mean_f1 = round(mean_f1, 3)
        
        print(f"Final Cross-validation:")
        print(f"  Accuracy: {mean_accuracy}")
        print(f"  F1 Score: {mean_f1}")
        print(f"  F1 Std: {np.std(f1_scores):.3f}")
        
        return mean_accuracy, mean_f1

class ResultReportGenerator:
    def generate_report(self, test_predictions, accuracy, f1, student_id="s1234567"):
        filename = f"{student_id}.infs4203"
        
        with open(filename, 'w') as f:
            # Rows 1-2713: Test predictions
            for pred in test_predictions:
                f.write(f"{int(pred)},\n")
            
            # Row 2714: Accuracy and F1
            f.write(f"{accuracy:.3f},{f1:.3f},")
        
        # Verify file format
        with open(filename, 'r') as f:
            lines = f.readlines()
            print(f"Report generated: {filename}")
            print(f"Total rows: {len(lines)} (2713 predictions + 1 evaluation)")
            print(f"Last row: {lines[-1].strip()}")
        
        return filename

def main():
    np.random.seed(42)
    
    # 1. Load data
    print("\n1. Loading data...")
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test_data.csv')
    print(f"   Training: {train_data.shape}, Test: {test_data.shape}")
    print("\n")

    # 2. Preprocessing with MEAN imputation
    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_processed, y_train = preprocessor.fit_transform(train_data)
    X_test_processed = preprocessor.transform(test_data)
    print(f"   Processed features: {X_train_processed.shape}")
    
    print("\n")

    # 3. Find best model for F1 score
    print("3. Model optimization...")
    optimizer = F1Optimizer()
    best_model = optimizer.find_best_model(X_train_processed, y_train)
    print("\n")
    
    # 4. Final cross-validation evaluation
    print("4. Final evaluation...")
    cv_evaluator = CrossValidationEvaluator(n_splits=5)
    accuracy, f1 = cv_evaluator.evaluate(best_model, X_train_processed, y_train)
    print("\n")

    # 5. Train final model and predict
    print("5. Generating predictions...")
    best_model.fit(X_train_processed, y_train)
    test_predictions = best_model.predict(X_test_processed)
    print("\n")

    # 6. Generate the result report file
    print("6. Creating submission file...")
    report_generator = ResultReportGenerator()
    student_id = "s4980297"  # Your student ID
    
    report_filename = report_generator.generate_report(test_predictions, accuracy, f1, student_id)
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Best Model: {type(best_model).__name__}")
    print(f"Final CV - Accuracy: {accuracy}, F1: {f1}")
    print(f"Submission File: {report_filename}")
    print(f"Test Predictions: {len(test_predictions)} rows")

if __name__ == "__main__":
    main()