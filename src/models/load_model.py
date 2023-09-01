import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from joblib import load
import pandas as pd
from data_loader import Loader 
# Load the best model from the joblib file
best_model = load('models/xgb_model_best.joblib')

print ("Model Successfully Loaded")

loader_object = Loader()
X_train_selected, X_val_selected, X_test_selected, y_train_resampled, y_val_resampled, y_test = loader_object.passdata()



loaded_model = load('models/xgb_model_best.joblib')

# Now you can use the loaded model for predictions
val_probabilities = loaded_model.predict_proba(X_val_selected)[:, 1]
test_probabilities = loaded_model.predict_proba(X_test_selected)[:, 1]

# Calculate AUROC for validation set
val_auroc = roc_auc_score(y_val_resampled, val_probabilities)
print("Validation AUROC:", val_auroc)

# Calculate AUROC for test set
test_auroc = roc_auc_score(y_test, test_probabilities)
print("Test AUROC:", test_auroc)

# Plot ROC curve for validation set
fpr_val, tpr_val, _ = roc_curve(y_val_resampled, val_probabilities)
plt.figure()
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label='Validation ROC curve (area = %0.2f)' % val_auroc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the validation ROC curve figure
plt.savefig('reports/figures/validation_roc_curve.png')


# Plot ROC curve for test set
fpr_test, tpr_test, _ = roc_curve(y_test, test_probabilities)
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='Test ROC curve (area = %0.2f)' % test_auroc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
# Save the test ROC curve figure
plt.savefig('reports/figures/test_roc_curve.png')

plt.close('all')


print("Validation ROC Curve Image: reports/figures/validation_roc_curve.png")
print("Test ROC Curve Image: reports/figures/test_roc_curve.png")