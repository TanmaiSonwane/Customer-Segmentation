import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from imblearn.combine import SMOTETomek
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_clustering import CreateClusters
from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import SimpleImputerConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
       
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self.utils = MainUtils()
        self.imputer_config = SimpleImputerConfig()

        # Define expected base columns (before feature engineering)
        self.expected_base_columns = [
            'Year_Birth', 'Education', 'Marital_Status', 'Income',
            'Kidhome', 'Teenhome', 'Dt_Customer', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts',
            'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases',
            'NumDealsPurchases', 'NumWebVisitsMonth', 'AcceptedCmp1',
            'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
            'AcceptedCmp5', 'Response', 'Recency'
        ]
        
        # Define expected output columns (after feature engineering)
        self.expected_output_columns = [
            "Age", "Education", "Marital Status", "Parental Status",
            "Children", "Income", "Total_Spending", "Days_as_Customer",
            "Recency", "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold",
            "Web", "Catalog", "Store", "Discount Purchases", "Total Promo",
            "NumWebVisitsMonth"
        ]

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            # Convert column names to uniform case (optional)
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            return df
        except Exception as e:
            raise CustomerException(e, sys)

    def _validate_input_data(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Validate input data has all required columns"""
        missing_cols = set(col.lower() for col in self.expected_base_columns) - set(col.lower() for col in df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in {dataset_name}: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}"
            )

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        return df

    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Create new features and transform existing ones"""
        try:
            # Validate and clean input data
            self._validate_input_data(train_set, "training set")
            self._validate_input_data(test_set, "test set")
            
            train_set = self._clean_column_names(train_set)
            test_set = self._clean_column_names(test_set)

            train_set_with_new_features = pd.DataFrame()
            test_set_with_new_features = pd.DataFrame()
            
            datasets = {"train_set": train_set, "test_set": test_set}
            
            for key, dataset in datasets.items():
                # Feature engineering
                dataset['Age'] = 2022 - dataset['Year_Birth']
                
                education_mapping = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
                dataset["Education"] = dataset["Education"].map(education_mapping).fillna(2)  # Default to Graduation
                
                marital_mapping = {
                    "Married": 1, "Together": 1, "Absurd": 0, "Widow": 0,
                    "YOLO": 0, "Divorced": 0, "Single": 0, "Alone": 0
                }
                dataset['Marital_Status'] = dataset['Marital_Status'].map(marital_mapping).fillna(0)
                
                dataset['Children'] = dataset['Kidhome'] + dataset['Teenhome']
                dataset['Family_Size'] = dataset['Marital_Status'] + dataset['Children'] + 1
                dataset['Total_Spending'] = (dataset["MntWines"] + dataset["MntFruits"] + 
                                           dataset["MntMeatProducts"] + dataset["MntFishProducts"] + 
                                           dataset["MntSweetProducts"] + dataset["MntGoldProds"])
                dataset["Total_Promo"] = (dataset["AcceptedCmp1"] + dataset["AcceptedCmp2"] + 
                                         dataset["AcceptedCmp3"] + dataset["AcceptedCmp4"] + 
                                         dataset["AcceptedCmp5"])
                
                dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'], errors='coerce')
                dataset['Days_as_Customer'] = (datetime.today() - dataset['Dt_Customer']).dt.days
                dataset['Offers_Responded_To'] = (dataset['AcceptedCmp1'] + dataset['AcceptedCmp2'] + 
                                                dataset['AcceptedCmp3'] + dataset['AcceptedCmp4'] + 
                                                dataset['AcceptedCmp5'] + dataset['Response'])
                dataset["Parental_Status"] = np.where(dataset["Children"] > 0, 1, 0)
                
                # Drop and rename columns
                columns_to_drop = ['Year_Birth', 'Kidhome', 'Teenhome']
                dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], 
                           axis=1, inplace=True, errors='ignore')
                
                rename_map = {
                    "Marital_Status": "Marital_Status",
                    "MntWines": "Wines", "MntFruits": "Fruits",
                    "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
                    "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
                    "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
                    "NumStorePurchases": "Store", "NumDealsPurchases": "Discount_Purchases",
                    "Total_Promo": "Total_Promo"
                }
                dataset.rename(columns=rename_map, inplace=True)
                
                # Select final columns
                final_columns = [col for col in self.expected_output_columns if col in dataset.columns]
                dataset = dataset[final_columns]
                
                if key == 'train_set':
                    train_set_with_new_features = dataset.copy()
                else:
                    test_set_with_new_features = dataset.copy()
            
            logging.info("New features created successfully")
            return train_set_with_new_features, test_set_with_new_features
            
        except Exception as e:
            raise CustomerException(e, sys)

    def transform_data(self, train_set: DataFrame, test_set: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Apply preprocessing transformations"""
        try:
            # Verify columns match between train and test
            if not set(train_set.columns) == set(test_set.columns):
                mismatch = set(train_set.columns).symmetric_difference(set(test_set.columns))
                raise ValueError(f"Column mismatch between train and test sets: {mismatch}")
            
            numeric_features = [feat for feat in train_set.columns if train_set[feat].dtype != 'O']
            outlier_features = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"]
            numeric_features = [x for x in numeric_features if x not in outlier_features]

            numeric_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(**self.imputer_config.__dict__)), 
                ("StandardScaler", StandardScaler())
            ])
            
            outlier_features_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                ("transformer", PowerTransformer(standardize=True))
            ])

            preprocessor = ColumnTransformer([
                ("numeric_pipeline", numeric_pipeline, numeric_features),
                ("outlier_features_pipeline", outlier_features_pipeline, outlier_features)
            ])
            
            preprocessed_train_set = preprocessor.fit_transform(train_set)
            preprocessed_test_set = preprocessor.transform(test_set)
            
            # Ensure we maintain all columns (including any that weren't transformed)
            preprocessed_train_set = pd.DataFrame(preprocessed_train_set, 
                                                columns=numeric_features + outlier_features)
            preprocessed_test_set = pd.DataFrame(preprocessed_test_set, 
                                               columns=numeric_features + outlier_features)
            
            # Save preprocessing object
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), 
                       exist_ok=True)
            self.utils.save_object(self.data_transformation_config.transformed_object_file_path, 
                                 preprocessor)
            
            return preprocessed_train_set, preprocessed_test_set

        except Exception as e:
            raise CustomerException(e, sys)
    

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Execute complete data transformation pipeline"""
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data validation failed - cannot proceed with transformation")
            
            logging.info("Starting data transformation")
            
            # Read and validate data
            train_set = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_set = self.read_data(self.data_ingestion_artifact.test_file_path)
            
            logging.info(f"Original train columns: {train_set.columns.tolist()}")
            logging.info(f"Original test columns: {test_set.columns.tolist()}")
            
            # Feature engineering
            train_set, test_set = self.get_new_features(train_set, test_set)
            
            logging.info(f"Post-feature engineering train columns: {train_set.columns.tolist()}")
            logging.info(f"Post-feature engineering test columns: {test_set.columns.tolist()}")
            
            # Data transformation
            preprocessed_train_set, preprocessed_test_set = self.transform_data(train_set, test_set)
            
            # Clustering
            cluster_creator = CreateClusters()
            labelled_train_set = cluster_creator.initialize_clustering(preprocessed_data=preprocessed_train_set)
            labelled_test_set = cluster_creator.initialize_clustering(preprocessed_data=preprocessed_test_set)
            
            # Prepare final datasets
            X_train = labelled_train_set.drop(columns=[TARGET_COLUMN], axis=1, errors='ignore')
            y_train = labelled_train_set[TARGET_COLUMN]
            X_test = labelled_test_set.drop(columns=[TARGET_COLUMN], axis=1, errors='ignore')
            y_test = labelled_test_set[TARGET_COLUMN]
            
            train_arr = np.c_[np.array(X_train), np.array(y_train)]
            test_arr = np.c_[np.array(X_test), np.array(y_test)]
            
            # Save results
            self.utils.save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, 
                array=train_arr
            )
            self.utils.save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, 
                array=test_arr
            )
            
            logging.info("Data transformation completed successfully")
            
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomerException(e, sys)