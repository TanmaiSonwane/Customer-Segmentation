import sys
import os
from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.constant.database import DATABASE_NAME, COLLECTION_NAME
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.customer_data import CustomerData
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

    def split_data_as_train_test(self, dataframe: DataFrame) -> Tuple[DataFrame, DataFrame]:
        logging.info("Entered split_data_as_train_test method of DataIngestion class")
        try:
            if dataframe.empty:
                raise CustomerException("The dataframe is empty. Cannot split.", sys)

            if len(dataframe) < 2:
                raise CustomerException("Not enough data to split into train and test sets.", sys)

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            logging.info("Performed train-test split on the dataframe")

            ingested_data_dir = self.data_ingestion_config.ingested_data_dir
            os.makedirs(ingested_data_dir, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Saved train and test datasets")

        except Exception as e:
            raise CustomerException(e, sys) from e

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            customer_data = CustomerData()
            customer_dataframe = customer_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)

            if customer_dataframe is None or customer_dataframe.empty:
                raise CustomerException("Exported dataframe from MongoDB is empty", sys)

            logging.info(f"Shape of exported dataframe: {customer_dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)

            customer_dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Saved exported data to: {feature_store_file_path}")
            return customer_dataframe

        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Initiating data ingestion process")

        try:
            dataframe = self.export_data_into_feature_store()

            _schema_config = self.utils.read_schema_config_file()
            drop_columns = _schema_config.get("drop_columns", [])

            # Drop only columns that actually exist
            columns_to_drop = [col for col in drop_columns if col in dataframe.columns]
            if columns_to_drop:
                dataframe.drop(columns=columns_to_drop, axis=1, inplace=True)
                logging.info(f"Dropped columns: {columns_to_drop}")
            else:
                logging.warning("No matching drop columns found in dataframe.")

            if dataframe.empty:
                raise CustomerException("Dataframe is empty after dropping columns", sys)

            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
