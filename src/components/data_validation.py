import json
import sys
from typing import Tuple, Union
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pandas import DataFrame
from pprint import pprint
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, write_yaml_file


class DataValidation:
    def __init__(self, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig ):
        
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        
        self.utils = MainUtils()

        self._schema_config = self.utils.read_schema_config_file()

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_schema_columns
        Description :   This method validates the schema columns for the particular dataframe 
        
        Output      :   True or False value is returned based on the schema 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        try:
            
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info("Is required column present[{status}]")

            return status

        except Exception as e:
            raise CustomerException(e, sys) from e

   

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
      """
       Validates that dataframe columns exactly match schema columns.
      """
      try:
        expected_columns = list(self._schema_config["columns"].keys())
        actual_columns = list(dataframe.columns)

        status = expected_columns == actual_columns
        if not status:
            logging.warning(f"Expected columns: {expected_columns}")
            logging.warning(f"Actual columns: {actual_columns}")

        return status

      except Exception as e:
        raise CustomerException(e, sys) from e

    
    
    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
      try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report_dict = report.as_dict()

        # Extract the correct metric result from Evidently 0.4.1
        drift_result = None
        for metric in report_dict.get("metrics", []):
            if metric.get("metric") == "DataDriftTable":
                drift_result = metric.get("result")
                break

        if not drift_result:
            raise ValueError("Could not find 'DataDriftTable' result in the report.")

        # Use `.get()` with defaults to avoid KeyErrors
        n_features = drift_result.get("n_features", "Unknown")
        n_drifted_features = drift_result.get("n_drifted_features", "Unknown")
        drift_status = drift_result.get("dataset_drift", False)

        logging.info(f"{n_drifted_features}/{n_features} features show drift.")
        
        # Save the entire report safely
        write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=report_dict)

        return drift_status

      except Exception as e:
        raise CustomerException(e, sys) from e

        
        
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.2
        Revisions   :   moved setup to cloud
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")

        try:
            logging.info("Initiated data validation for the dataset")

            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            
            
            drift = self.detect_dataset_drift(train_df, test_df)

            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns(train_set=train_df, test_set=test_df)

            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )

            logging.info("Validated dataset schema columns")

            

            if (
                schema_train_col_status is True
                and schema_test_col_status is True
                and drift is False
            ):
                logging.info("Dataset schema validation completed")

                validation_status = True
            else:
                validation_status = False
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
