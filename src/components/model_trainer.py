from src.constants import MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
from src.entity.estimator import MyModel
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils import load_numpy_array_data, load_object, save_object, read_yaml_file
from src.logger import logging
from src.exception import MyException
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import sys
from typing import Tuple
import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.model_parameters = read_yaml_file(
            file_path=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)

        # 🚀 Set tracking URI and experiment
        mlflow.set_tracking_uri(
            "https://dagshub.com/yashmasane68/InsurePredict-Customer-Conversion-Insights.mlflow")
        mlflow.set_experiment("Insurance_Model_Training")

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        try:
            logging.info(
                "Training GradientBoostingClassifier with specified parameters")

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            # 🚀 Start MLflow run
            run_name = f"GradientBoosting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name):
                model = GradientBoostingClassifier(
                    **self.model_parameters['gradient_boosting'])

                logging.info("Model training going on...")
                model.fit(x_train, y_train)
                logging.info("Model training done.")

                # Predictions and evaluation metrics
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)

                signature = infer_signature(x_train, y_pred)

                # 🚀 Log parameters and metrics
                mlflow.log_params(self.model_parameters['gradient_boosting'])
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall
                })

                # Save report locally
                model_report_dir = os.path.dirname(
                    self.model_trainer_config.model_report_file_path)
                os.makedirs(model_report_dir, exist_ok=True)

                model_report = {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall
                }
                with open(self.model_trainer_config.model_report_file_path, "w") as report_file:
                    json.dump(model_report, report_file, indent=4)

                # 🚀 Log report file as artifact
                mlflow.log_artifact(
                    self.model_trainer_config.model_report_file_path)

                # 🚀 Log model (temp path – final object logged later)
                mlflow.sklearn.log_model(model, artifact_path="trained_model")

            # Return model and metric object
            metric_artifact = ClassificationMetricArtifact(
                accuracy=accuracy,
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )
            return model, metric_artifact, signature

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info(
            "Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")

            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")

            trained_model, metric_artifact, signature = self.get_model_object_and_report(
                train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")

            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check accuracy threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception(
                    "No model found with score above the base score")

            # Save final model object (includes preprocessing)
            logging.info(
                "Saving new model as performance is better than previous one.")
            my_model = MyModel(
                preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(
                self.model_trainer_config.trained_model_file_path, my_model)

            logging.info(
                "Saved final model object that includes both preprocessing and the trained model")

            # 🚀 Register final model with MLflow
            with mlflow.start_run(run_name="Model_Registration", nested=True):
                mlflow.log_metric("train_accuracy", accuracy_score(
                    train_arr[:, -1], trained_model.predict(train_arr[:, :-1])))
                mlflow.log_param("model_type", "RandomForestClassifier")

                mlflow.sklearn.log_model(
                    sk_model=my_model,
                    artifact_path="insurance_model",
                    registered_model_name="RegisteredInsuranceModel",
                    signature=signature
                )

            # Return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e


if __name__ == "__main__":
    try:
        # Create required artifacts
        data_transformation_artifact = DataTransformationArtifact(
            transformed_object_file_path="artifact/data_transformation/transformed_object/preprocessing.pkl",
            transformed_train_file_path="artifact/data_transformation/transformed/train.npy",
            transformed_test_file_path="artifact/data_transformation/transformed/test.npy"
        )

        model_trainer_config = ModelTrainerConfig()

        # Initialize model trainer
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=model_trainer_config
        )

        # Train model and get artifacts
        model_trainer_artifact = model_trainer.initiate_model_trainer()

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise MyException(e, sys)
