import os
import pytest
import pandas as pd
from pandas import DataFrame
from unittest.mock import patch, MagicMock

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException


@pytest.fixture
def sample_dataframe() -> DataFrame:
    """Fixture to create sample DataFrame for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Age': [25, 30, 35, 40],
        'Driving_License': [1, 0, 1, 0],
        'Region_Code': [1, 2, 3, 4],
        'Previously_Insured': [0, 1, 0, 1],
        'Vehicle_Age': ['<1 Year', '1-2 Year', '>2 Years', '1-2 Year'],
        'Vehicle_Damage': ['Yes', 'No', 'Yes', 'No'],
        'Annual_Premium': [25000, 30000, 35000, 40000],
        'Policy_Sales_Channel': [100, 101, 102, 103],
        'Vintage': [10, 20, 30, 40],
        'Response': [1, 0, 1, 0]
    })


@pytest.fixture
def data_ingestion_config():
    """Fixture for DataIngestionConfig"""
    return DataIngestionConfig(
        collection_name="insurance_collection",
        feature_store_file_path="tests/artifacts/data_ingestion/feature_store/insurance.csv",
        training_file_path="tests/artifacts/data_ingestion/ingested/train.csv",
        testing_file_path="tests/artifacts/data_ingestion/ingested/test.csv",
        train_test_split_ratio=0.25
    )


class TestDataIngestion:

    def test_data_ingestion_initialization(self, data_ingestion_config):
        """Test if DataIngestion class initializes correctly"""
        data_ingestion = DataIngestion(data_ingestion_config)
        assert data_ingestion.data_ingestion_config == data_ingestion_config

    @patch('src.components.data_ingestion.Proj1Data')
    def test_export_data_into_feature_store(self, mock_proj1_data, data_ingestion_config, sample_dataframe):
        """Test if export_data_into_feature_store method works correctly"""
        # Setup mock
        mock_instance = mock_proj1_data.return_value
        mock_instance.export_collection_as_dataframe.return_value = sample_dataframe

        # Initialize data ingestion
        data_ingestion = DataIngestion(data_ingestion_config)

        # Execute method
        result_df = data_ingestion.export_data_into_feature_store()

        # Assertions
        assert isinstance(result_df, DataFrame)
        assert result_df.equals(sample_dataframe)
        assert os.path.exists(data_ingestion_config.feature_store_file_path)
        assert os.path.exists('data/raw/data.csv')

    def test_split_data_as_train_test(self, data_ingestion_config, sample_dataframe):
        """Test if split_data_as_train_test method works correctly"""
        data_ingestion = DataIngestion(data_ingestion_config)

        # Execute split
        data_ingestion.split_data_as_train_test(sample_dataframe)

        # Verify files are created
        assert os.path.exists(data_ingestion_config.training_file_path)
        assert os.path.exists(data_ingestion_config.testing_file_path)

        # Load and verify split files
        train_df = pd.read_csv(data_ingestion_config.training_file_path)
        test_df = pd.read_csv(data_ingestion_config.testing_file_path)

        # Check split ratio
        expected_test_size = int(
            len(sample_dataframe) * data_ingestion_config.train_test_split_ratio)
        assert len(test_df) == expected_test_size
        assert len(train_df) == len(sample_dataframe) - expected_test_size

    @patch('src.components.data_ingestion.Proj1Data')
    def test_initiate_data_ingestion(self, mock_proj1_data, data_ingestion_config, sample_dataframe):
        """Test if initiate_data_ingestion method works end-to-end"""
        # Setup mock
        mock_instance = mock_proj1_data.return_value
        mock_instance.export_collection_as_dataframe.return_value = sample_dataframe

        # Initialize and execute
        data_ingestion = DataIngestion(data_ingestion_config)
        artifact = data_ingestion.initiate_data_ingestion()

        # Verify artifact
        assert isinstance(artifact, DataIngestionArtifact)
        assert os.path.exists(artifact.trained_file_path)
        assert os.path.exists(artifact.test_file_path)

    def test_invalid_split_ratio(self, data_ingestion_config, sample_dataframe):
        """Test if invalid split ratio raises exception"""
        data_ingestion_config.train_test_split_ratio = 1.5
        data_ingestion = DataIngestion(data_ingestion_config)

        with pytest.raises(MyException):
            data_ingestion.split_data_as_train_test(sample_dataframe)

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Cleanup test artifacts after each test"""
        yield
        test_dirs = [
            'tests/artifacts/data_ingestion/feature_store',
            'tests/artifacts/data_ingestion/ingested',
            'data/raw'
        ]
        for dir in test_dirs:
            if os.path.exists(dir):
                for file in os.listdir(dir):
                    file_path = os.path.join(dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                try:
                    os.rmdir(dir)
                except OSError:
                    pass  # Directory might not be empty or might be already removed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
