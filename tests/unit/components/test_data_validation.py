import pytest
import yaml
import pandas as pd
import os
import json
from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import MyException


@pytest.fixture
def sample_valid_data():
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
def validation_setup(tmp_path):
    """Setup test paths and artifacts"""
    # Create temporary test directories and files
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    report_path = tmp_path / "validation" / "validation_report.yaml"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=str(train_path),
        test_file_path=str(test_path)
    )

    validation_config = DataValidationConfig(
        validation_report_file_path=str(report_path)
    )

    return {
        'train_path': train_path,
        'test_path': test_path,
        'report_path': report_path,
        'ingestion_artifact': ingestion_artifact,
        'validation_config': validation_config
    }


class TestDataValidation:
    def test_validate_number_of_columns(self, sample_valid_data, validation_setup):
        """Test column count validation"""
        validator = DataValidation(
            validation_setup['ingestion_artifact'],
            validation_setup['validation_config']
        )

        # Test with valid dataframe
        assert validator.validate_number_of_columns(sample_valid_data) == True

        # Test with invalid dataframe (missing column)
        invalid_df = sample_valid_data.drop('Age', axis=1)
        assert validator.validate_number_of_columns(invalid_df) == False

    def test_is_column_exist(self, sample_valid_data, validation_setup):
        """Test column existence validation"""
        validator = DataValidation(
            validation_setup['ingestion_artifact'],
            validation_setup['validation_config']
        )

        # Test with valid dataframe
        assert validator.is_column_exist(sample_valid_data) == True

        # Test with missing numerical column
        df_missing_numerical = sample_valid_data.drop('Age', axis=1)
        assert validator.is_column_exist(df_missing_numerical) == False

        # Test with missing categorical column
        df_missing_categorical = sample_valid_data.drop('Gender', axis=1)
        assert validator.is_column_exist(df_missing_categorical) == False

    def test_read_data(self, sample_valid_data, validation_setup):
        """Test data reading functionality"""
        # Save test data
        sample_valid_data.to_csv(validation_setup['train_path'], index=False)

        # Test reading
        df = DataValidation.read_data(validation_setup['train_path'])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_valid_data)

        # Test with non-existent file
        with pytest.raises(MyException):
            DataValidation.read_data("non_existent_file.csv")

    def test_initiate_data_validation(self, sample_valid_data, validation_setup):
        """Test complete validation process"""
        # Save test data
        sample_valid_data.to_csv(validation_setup['train_path'], index=False)
        sample_valid_data.to_csv(validation_setup['test_path'], index=False)

        validator = DataValidation(
            validation_setup['ingestion_artifact'],
            validation_setup['validation_config']
        )

        # Test validation process
        validation_artifact = validator.initiate_data_validation()

        # Verify artifact
        assert isinstance(validation_artifact, DataValidationArtifact)
        assert validation_artifact.validation_status == True
        assert os.path.exists(validation_artifact.validation_report_file_path)

        # Verify report content
        with open(validation_artifact.validation_report_file_path) as f:
            report = yaml.safe_load(f)
            assert report['validation_status'] == True
            assert report['message'] == ""

    def test_validation_with_invalid_data(self, sample_valid_data, validation_setup):
        """Test validation with invalid data"""
        # Create invalid data
        invalid_data = sample_valid_data.drop(['Age', 'Gender'], axis=1)

        # Save invalid data
        invalid_data.to_csv(validation_setup['train_path'], index=False)
        invalid_data.to_csv(validation_setup['test_path'], index=False)

        validator = DataValidation(
            validation_setup['ingestion_artifact'],
            validation_setup['validation_config']
        )

        validation_artifact = validator.initiate_data_validation()

        assert validation_artifact.validation_status == False
        assert "Columns are missing" in validation_artifact.message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
