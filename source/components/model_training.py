import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import os, sys
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import warnings
from logger import logger
from exception import CustomException
from dataclasses import dataclass
from source.components.data_ingestion import DataIngestion
from source.components.data_transformation import DataTransformation
from source.utils.common import evaluate_models, save_object


@dataclass
class ModelTrainingConfig:
    training_model_file_path = os.path.join('artifact', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def model_training(self, train_arr, test_arr, preprocessor_path):
        try:
            logger.info("split train and test input data")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                        "Linear Regression": LinearRegression(),
                        "Lasso": Lasso(),
                        "Ridge": Ridge(),
                        "K-Neighbors Regressor": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "XGBRegressor": XGBRegressor(),
                        "AdaBoost Regressor": AdaBoostRegressor()
                    } 
            
            model_report = evaluate_models(x_train, y_train, x_test, y_test, models)

             ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logger.info(f"Best found model on both training and testing dataset is {best_model}")

            save_object(
                file_path=self.model_trainer_config.training_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            logger.info(e)
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    ingest_data = DataIngestion()
    _, train_data, test_data = ingest_data.data_ingestion() 

    data_transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformer.data_transformation(train_data, test_data)

    model = ModelTrainer()
    print(model.model_training(train_arr, test_arr, preprocessor_path))

        
