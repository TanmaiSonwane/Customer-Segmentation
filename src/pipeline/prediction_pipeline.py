from src.ml.model.s3_estimator import CustomerClusterEstimator
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.constant.training_pipeline import *
from src.entity.config_entity import training_pipeline_config
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig

from src.entity.config_entity import DataTransformationConfig , ModelTrainerConfig
from src.logger import logging
from src.utils.main_utils import MainUtils

from src.exception import CustomerException
import pandas as pd
import numpy as np
import sys

import logging
import sys
from pandas import DataFrame
import pandas as pd


from pymongo import MongoClient
from src.constant.database import DATABASE_NAME
from src.logger import logging


class CustomerData:
    def __init__(self):
        try:
            # ✅ Replace <username>, <password>, and <cluster-url> with your actual Atlas credentials
            mongo_uri = "mongodb+srv://tanmaisonwane:(60W#3LRrw'@cluster0.kkubrdd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

            self.mongo_client = MongoClient(mongo_uri)
            self.database = self.mongo_client[DATABASE_NAME]
        except Exception as e:
            raise Exception(f"Error connecting to MongoDB Atlas: {e}")

    def export_collection_as_dataframe(self, collection_name: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from MongoDB Atlas: {DATABASE_NAME}.{collection_name}")
            collection = self.database[collection_name]
            data = list(collection.find())

            if not data:
                raise Exception("MongoDB collection is empty")

            dataframe = pd.DataFrame(data)

            if "_id" in dataframe.columns:
                dataframe.drop(columns=["_id"], inplace=True)

            logging.info(f"Exported {len(dataframe)} rows from MongoDB Atlas")
            return dataframe

        except Exception as e:
            raise Exception(f"Failed to export data from MongoDB Atlas: {e}")

        
        
    


class PredictionPipeline:
    def __init__(self):
        self.utils = MainUtils()
        
    def prepare_input_data(self, input_data:list) -> pd.DataFrame:
        """ 
        method: prepare_input_data 
        
        objective: This method creates a dataframe taking the column names from prediction schema file
                       with the input values for prediction and returns it

        Args:
            input_data (list): input data 

        Raises:
            CustomerException

        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """
        try:
        
            
            customerDataframe = CustomerData.form_input_dataframe(data = input_data)
            logging.info("customerDatafram has been created")
            return customerDataframe
        except Exception as e:
            raise CustomerException(e,sys)
        
   
        
    
        
    def get_trained_model(self, ModelTrainerConfig = ModelTrainerConfig):
        """
        method: get_trained_model
        
        objective: this method returns the model

        Args:
            ModelTrainerConfig

        Raises:
            CustomerException: 

        Returns:
            model: latest trained model
        """
        try:
            prediction_config = PredictionPipelineConfig()
            model = CustomerClusterEstimator(
                bucket_name= prediction_config.model_bucket_name,
                model_path= prediction_config.model_file_name
            )
                
            return model
                
        except Exception as e:
            raise CustomerException(e, sys) from e
        
    def run_pipeline(self, input_data:list):
        
        """
        method: run_pipeline
        
        objective: run_pipeline method runs the whole prediction pipeline.

        Raises:
            CustomerException: 
        """
        try:
            input_dataframe =  self.prepare_input_data(input_data) 
            model = self.get_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction
            
        except Exception as e:
            raise CustomerException(e, sys)
            
            
        
            
        

 
        

        