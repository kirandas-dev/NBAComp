import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
from data_loader import Loader 

loader_object = Loader()
X_train_selected, X_val_selected, X_test_selected, y_train_resampled, y_val_resampled, y_test = loader_object.passdata()


# Use the best parameters obtained from grid search
best_params = {'learning_rate': 0.0145, 'max_depth': 2, 'n_estimators': 400, 'reg_lambda': 0.5}

# Create XGBoost model with the best parameters
best_model = XGBClassifier(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    reg_lambda=best_params['reg_lambda']
)

# Train the model on the training data with selected features
best_model.fit(X_train_selected, y_train_resampled)

# Evaluate the model on the validation set
train_predictions = best_model.predict(X_train_selected)
train_accuracy = accuracy_score(y_train_resampled, train_predictions)
train_f1 = f1_score(y_train_resampled, train_predictions)
print("Train Accuracy:", train_accuracy)
print("Train F1 Score:", train_f1)

# Evaluate the model on the validation set
val_predictions = best_model.predict(X_val_selected)
val_accuracy = accuracy_score(y_val_resampled, val_predictions)
val_f1 = f1_score(y_val_resampled, val_predictions)
print("Validation Accuracy:", val_accuracy)
print("Validation F1 Score:", val_f1)

# Evaluate the model on the test set
test_predictions = best_model.predict(X_test_selected)
test_accuracy = accuracy_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)
print("Test F1 Score:", test_f1)



dump(best_model, 'models/xgb_model_best.joblib') 