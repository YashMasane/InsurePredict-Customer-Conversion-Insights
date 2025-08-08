import pytest
import numpy as np
import os
import json
from unittest.mock import patch, MagicMock

from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException


@pytest.fixture
def sample_data():
    """Create sample training and testing data"""
    # Create synthetic data with features and target
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # Binary target
    # Combine features and target
    train_data = np.column_stack((X[:80], y[:80]))  # 80% for training
    test_data = np.column_stack((X[80:], y[80:]))   # 20% for testing
    return train_data, test_data


@pytest.fixture
def model_trainer_setup(tmp_path):
    """Setup artifacts and configurations for testing"""
    # Create temporary directories
    artifact_dir = tmp_path / "artifacts"
    model_dir = tmp_path / "model"
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create paths for required files
    transform_obj_path = artifact_dir / "preprocessing.pkl"
    train_path = artifact_dir / "train.npy"
    test_path = artifact_dir / "test.npy"
    model_path = model_dir / "model.pkl"
    report_path = model_dir / "report.json"

    # Create artifacts
    data_transformation_artifact = DataTransformationArtifact(
        transformed_object_file_path=str(transform_obj_path),
        transformed_train_file_path=str(train_path),
        transformed_test_file_path=str(test_path)
    )

    model_trainer_config = ModelTrainerConfig(
        trained_model_file_path=str(model_path),
        model_report_file_path=str(report_path),
        expected_accuracy=0.5
    )

    return {
        'artifact_dir': artifact_dir,
        'model_dir': model_dir,
        'data_transformation_artifact': data_transformation_artifact,
        'model_trainer_config': model_trainer_config
    }


class TestModelTrainer:
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_model_trainer_initialization(self, mock_set_experiment, mock_set_tracking_uri, model_trainer_setup):
        """Test ModelTrainer initialization"""
        model_trainer = ModelTrainer(
            model_trainer_setup['data_transformation_artifact'],
            model_trainer_setup['model_trainer_config']
        )
        assert mock_set_tracking_uri.called
        assert mock_set_experiment.called
        assert model_trainer.model_parameters is not None

    @patch('mlflow.start_run')
    def test_get_model_object_and_report(self, mock_start_run, model_trainer_setup, sample_data):
        """Test model training and report generation"""
        train_data, test_data = sample_data

        # Save numpy arrays
        np.save(
            model_trainer_setup['data_transformation_artifact'].transformed_train_file_path, train_data)
        np.save(
            model_trainer_setup['data_transformation_artifact'].transformed_test_file_path, test_data)

        model_trainer = ModelTrainer(
            model_trainer_setup['data_transformation_artifact'],
            model_trainer_setup['model_trainer_config']
        )

        # Mock MLflow context manager
        mock_start_run.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock())

        model, metric_artifact, signature = model_trainer.get_model_object_and_report(
            train_data, test_data)

        assert model is not None
        assert isinstance(metric_artifact, ClassificationMetricArtifact)
        assert signature is not None
        assert os.path.exists(
            model_trainer_setup['model_trainer_config'].model_report_file_path)

    @patch('src.components.model_trainer.load_numpy_array_data')
    @patch('src.components.model_trainer.load_object')
    @patch('mlflow.start_run')
    def test_initiate_model_trainer(self, mock_start_run, mock_load_object, mock_load_data,
                                    model_trainer_setup, sample_data):
        """Test complete model training pipeline"""

        train_data, test_data = sample_data

        # Setup mocks
        mock_load_data.side_effect = [train_data, test_data]
        mock_load_object.return_value = object()

        # Mock mlflow.start_run context manager
        mock_start_run.return_value.__enter__.return_value = None
        mock_start_run.return_value.__exit__.return_value = None

        model_trainer = ModelTrainer(
            model_trainer_setup['data_transformation_artifact'],
            model_trainer_setup['model_trainer_config']
        )

        artifact = model_trainer.initiate_model_trainer()

        assert isinstance(artifact, ModelTrainerArtifact)
        assert os.path.exists(artifact.trained_model_file_path)
        assert artifact.metric_artifact is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
