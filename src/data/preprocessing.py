import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        pass

    def read_data(self, raw_data_path):
        """Read raw data into DataProcessor."""
        self.df = pd.read_csv(raw_data_path, low_memory=False)

    def process_data(self, stable=True):
        """Process raw data into useful files for the model."""
        self._filter_columns()
        self._encode_columns()
        self._impute_missing_values()
        self._drop_uninformative_columns()
        self._split_data()
        self._resample_data()

   

    def _filter_columns(self):
        threshold = 100
        self.filtered_data = self.df.dropna(axis=1, thresh=threshold)

    def _encode_columns(self):
        # Frequency encoding the 'yr' categorical feature
        frequency_encoded = self.filtered_data['yr'].value_counts().to_dict()
        self.filtered_data['yr'] = self.filtered_data['yr'].map(frequency_encoded)

    
    def _impute_missing_values(self):
        
        # List of features with normal distribution
        normal_distribution_features = ['Ortg', 'eFG', 'TS_per', 'adjoe']

        # Create a DataFrame containing only the selected features
        selected_features_df = self.filtered_data[normal_distribution_features]

        # Initialize KNNImputer with the desired number of neighbors
        knn_imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors (k)

        # Impute missing values using KNN
        imputed_data = knn_imputer.fit_transform(selected_features_df)

        # Convert the imputed data back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=normal_distribution_features)

        # Replace the original columns with the imputed values
        self.filtered_data[normal_distribution_features] = imputed_df

        # Impute dunks_ratio, dunksmiss_dunksmade, rim_ratio, mid_ratio with the median
        median_dunks_ratio = 0.259
        median_dunksmiss_dunksmade = 0.363
        median_rim_ratio = 0.638
        median_mid_ratio = 0.384

        self.filtered_data['dunks_ratio'].fillna(median_dunks_ratio, inplace=True)
        self.filtered_data['dunksmiss_dunksmade'].fillna(median_dunksmiss_dunksmade, inplace=True)
        self.filtered_data['rim_ratio'].fillna(median_rim_ratio, inplace=True)
        self.filtered_data['mid_ratio'].fillna(median_mid_ratio, inplace=True)


        # Impute Rec_Rank with a value outside the range of existing ranks (e.g., -1)
        self.filtered_data['Rec_Rank'].fillna(-1, inplace=True)

        # Impute pick with the mode
        mode_pick = self.filtered_data['pick'].mode().iloc[0]
        self.filtered_data['pick'].fillna(mode_pick, inplace=True)

        # Impute rimmade_rimmiss with the median
        median_rimmade_rimmiss = 1.571
        self.filtered_data['rimmade_rimmiss'].fillna(median_rimmade_rimmiss, inplace=True)

        # Impute dunksmade, midmade, rimmade with zero
        self.filtered_data['dunksmade'].fillna(0, inplace=True)
        self.filtered_data['midmade'].fillna(0, inplace=True)
        self.filtered_data['rimmade'].fillna(0, inplace=True)
        # List of categorical features
        categorical_features = ['yr', 'ht', 'num']

        # List of features with non-normal distribution
        skewed_features = ['GP', 'Min_per', 'usg', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per',
                        'ftr', 'porpag', 'pfr', 'ast_tov', 'rimmade_rimmiss', 'midmade_midmiss',
                        'dunksmiss_dunksmade', 'drtg', 'adrtg', 'dporpag', 'stops', 'bpm', 'obpm', 'dbpm',
                        'gbpm', 'mp', 'ogbpm', 'dgbpm', 'oreb', 'dreb', 'treb', 'ast', 'stl', 'blk', 'pts']

        # Impute categorical features with mode
        self.filtered_data[categorical_features] = self.filtered_data[categorical_features].fillna(self.filtered_data[categorical_features].mode().iloc[0])

        # Impute skewed features with median
        self.filtered_data[skewed_features] = self.filtered_data[skewed_features].fillna(self.filtered_data[skewed_features].median())

        # Impute percentage features with a reasonable central value (e.g., median)
        percentage_features = ['FT_per', 'TP_per', 'blk_per']
        central_value = self.filtered_data[percentage_features].median()
        self.filtered_data[percentage_features] = self.filtered_data[percentage_features].fillna(central_value)

    def _drop_uninformative_columns(self):
        columns_to_drop = ['year', 'yr', 'ht', 'num', 'team', 'conf', 'type', 'player_id']
        self.filtered_data.drop(columns_to_drop, axis=1, inplace=True)
        

    def _split_data(self):
        # Splitting data code here
        self.y = self.filtered_data.pop('drafted')
        self.X = self.filtered_data
        

        # Split the data into train and test sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Split the train set into train and validation sets (75% train, 25% validation)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.25, random_state=42)
       


    def _resample_data(self):
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)

        # Instantiate the StandardScaler and SMOTE
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)

        # Fit and transform the scaler on the training set
        X_train_scaled = scaler.fit_transform(self.X_train)

        # Resample the scaled training set using SMOTE
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(X_train_scaled, self.y_train)
       
        # Saving Scaler and smote to the model folder
        dump(scaler,  'models/scaler.joblib')
        dump(smote,  'models/smote.joblib')
        # Resample the validation set
        self.X_val_scaled = scaler.transform(self.X_val)
        self.X_val_resampled, self.y_val_resampled = smote.fit_resample(self.X_val_scaled, self.y_val)

        # Resample the test set
        self.X_test_scaled = scaler.transform(self.X_test)

        self.X_test_resampled, self.y_test_resampled = smote.fit_resample(self.X_test_scaled, self.y_test)

    
    def write_data(self, processed_data_path):
        """Write processed data to the directory."""
       
        self.X_train_resampled = pd.DataFrame(self.X_train_resampled, columns=self.X_train.columns)
        self.X_val_resampled = pd.DataFrame(self.X_val_resampled, columns=self.X_val.columns)
        self.X_test_resampled = pd.DataFrame(self.X_test_resampled, columns=self.X_test.columns)

        self.y_train_resampled = pd.Series(self.y_train_resampled)
        self.y_val_resampled = pd.Series(self.y_val_resampled)
        self.y_test_resampled = pd.Series(self.y_test_resampled)
        
        self.X_train_resampled.to_csv(processed_data_path + '/X_train.csv', index=False)
        self.X_val_resampled.to_csv(processed_data_path + '/X_val.csv', index=False)
        self.X_test_resampled.to_csv(processed_data_path + '/X_test.csv', index=False)
        self.y_train_resampled.to_csv(processed_data_path + '/y_train.csv', index=False)
        self.y_val_resampled.to_csv(processed_data_path + '/y_val.csv', index=False)
        self.y_test_resampled.to_csv(processed_data_path + '/y_test.csv', index=False)


if __name__ == "__main__":
    data_processor = DataProcessor()
    raw_data_path = './data/raw/train.csv'
    processed_data_path = './data/processed'

    data_processor.read_data(raw_data_path)
    data_processor.process_data()
    data_processor.write_data(processed_data_path)
