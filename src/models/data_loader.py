import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from joblib import load


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


class Loader:
    def __init__(self):
        self.X_train = pd.read_csv('data/processed/X_train.csv')
        self.X_val = pd.read_csv('data/processed/X_val.csv')
        self.X_test = pd.read_csv('data/processed/X_test.csv')
        self.y_train_resampled = pd.read_csv('data/processed/y_train.csv').iloc[:, 0] 
        self.y_val_resampled = pd.read_csv('data/processed/y_val.csv').iloc[:, 0] 
        self.y_test =  pd.read_csv('data/processed/y_test.csv' )

    def passdata(self):
        threshold = 0.04
        selected_features = coefficients[coefficients['Absolute Coefficient'] > threshold]['Feature'].tolist()
        X_train_selected = self.X_train[selected_features]
        X_val_selected = self.X_val[selected_features]
        X_test_selected = self.X_test[selected_features]
        y_train_resampled = pd.read_csv('data/processed/y_train.csv').iloc[:, 0] 
        y_val_resampled = pd.read_csv('data/processed/y_val.csv').iloc[:, 0] 
        y_test =  pd.read_csv('data/processed/y_test.csv' )
        return X_train_selected, X_val_selected, X_test_selected, y_train_resampled, y_val_resampled, y_test
      