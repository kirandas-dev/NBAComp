from joblib import load
import pandas as pd

# Load the best model from the joblib file
best_model = load('models/xgb_model_best.joblib')




selected_features = [
    'GP', 'Min_per', 'Ortg', 'usg', 'eFG', 'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per',
    'FTM', 'FTA', 'FT_per', 'twoPM', 'twoPA', 'twoP_per', 'TPM', 'TPA', 'TP_per', 'blk_per',
    'stl_per', 'ftr', 'porpag', 'adjoe', 'pfr', 'Rec_Rank', 'ast_tov', 'rimmade', 'rimmade_rimmiss',
    'midmade', 'midmade_midmiss', 'rim_ratio', 'mid_ratio', 'dunksmade', 'dunksmiss_dunksmade',
    'dunks_ratio', 'pick', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm', 'gbpm',
    'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts'
]

#By taking the above coefficient value for each feature, we are going to try use the same features with a more robust model, XGBoost. 

coefficients = pd.DataFrame({
    'Feature': ['adrtg', 'Min_per', 'AST_per', 'twoPM', 'obpm', 'twoPA', 'stl_per', 'dunksmiss_dunksmade', 'ast',
                'usg', 'midmade_midmiss', 'TS_per', 'adjoe', 'rimmade', 'midmade', 'porpag', 'drtg', 'dgbpm',
                'dunksmade', 'stops', 'ogbpm', 'eFG', 'dbpm', 'TO_per', 'pfr', 'mp', 'rimmade_rimmiss', 'TPM',
                'dunks_ratio', 'GP', 'bpm', 'Rec_Rank', 'twoP_per', 'FT_per', 'FTA', 'mid_ratio', 'dreb', 'DRB_per',
                'dporpag', 'blk_per', 'gbpm', 'stl', 'oreb', 'ast_tov', 'FTM', 'ftr', 'rim_ratio', 'blk', 'treb',
                'TPA', 'TP_per', 'pick', 'ORB_per', 'pts', 'Ortg'],
    'Absolute Coefficient': [3.340078, 3.542694, 2.779246, 2.558382, 2.441382, 2.395607, 2.205659, 2.131396,
                             2.092174, 2.086758, 2.043614, 1.923499, 1.795148, 1.754931, 1.675424, 1.623246,
                             1.584561, 1.505753, 1.505267, 1.457823, 1.264243, 1.249943, 1.243752, 1.079634,
                             1.048334, 0.984514, 0.927019, 0.904347, 0.875047, 0.861829, 0.839320, 0.789912,
                             0.597381, 0.575458, 0.528011, 0.525412, 0.498164, 0.436940, 0.426927, 0.421211,
                             0.377454, 0.369609, 0.337747, 0.287269, 0.248238, 0.246318, 0.204145, 0.192189,
                             0.168635, 0.151730, 0.151492, 0.139234, 0.088484, 0.086800, 0.042992]
})

test = pd.read_csv('data/raw/test.csv')
# Keeping only the selected features in the test dataframe
test_selected = test[selected_features]

loaded_scaler = load('models/scaler.joblib')
X_tt = loaded_scaler.transform(test_selected)


# Converting scaled array to DataFrame
X_tt_df = pd.DataFrame(X_tt, columns=test_selected.columns)


import pandas as pd


# Load the test data
test = pd.read_csv('data/raw/test.csv')

# Setting a threshold for absolute coefficient values
threshold = 0.04

# Selecting features with absolute coefficient values above the threshold
selected_features = coefficients[coefficients['Absolute Coefficient'] > threshold]['Feature'].tolist()


# Keeping only the selected features in the test dataframe
test_selected = X_tt_df[selected_features]

test_selected.to_csv('data/processed/final_test_set.csv', index=False)


# Load the saved model
xgb_loaded_model = load('models/xgb_model_best.joblib')

predicted_probabilities_xgb = xgb_loaded_model.predict_proba(test_selected)

testing = pd.read_csv('data/raw/test.csv')


xgb_result_df = pd.DataFrame({
    'player_id': testing['player_id'] ,
    'drafted_probability': predicted_probabilities_xgb[:, 1]
})

# Save the predicted probabilities to a CSV file
xgb_result_df.to_csv('data/interim/week_3_best_model_predicted_probabilities.csv', index=False)
xgb_result_df.to_csv('reports/figures/final_model_outputs_week3_31stAug.csv', index=False)


print ("Model Successfully Predicted. Please check reports/figures/final_model_outputs_week3_31stAug.csv ")
