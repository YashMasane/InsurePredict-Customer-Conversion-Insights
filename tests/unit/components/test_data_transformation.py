import pytest
import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline

from src.components.data_transformation import DataTransformation
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.constants import TARGET_COLUMN


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Age': [25, 30, 35, 40, 45, 50],
        'Region_Code': [1, 2, 3, 4, 5, 6],
        'Previously_Insured': [0, 1, 0, 1, 0, 1],
        'Vehicle_Age': ['< 1 Year', '1-2 Year', '> 2 Years', '1-2 Year', '> 2 Years', '< 1 Year'],
        'Vehicle_Damage': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Annual_Premium': [25000, 30000, 65000, 40000, 50000, 45000],
        'Policy_Sales_Channel': [100, 101, 102, 103, 104, 105],
        'Vintage': [10, 20, 30, 40, 50, 60],
        'Response': [0, 1, 0, 1, 0, 1]
    })


@pytest.fixture
def transformation_setup(tmp_path):
    """Setup artifacts and configurations for testing"""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    transform_path = tmp_path / "transform"
    os.makedirs(transform_path, exist_ok=True)

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=str(train_path),
        test_file_path=str(test_path)
    )

    validation_artifact = DataValidationArtifact(
        validation_status=True,
        message="Validation Successful",
        validation_report_file_path=str(tmp_path / "validation_report.yaml")
    )

    transformation_config = DataTransformationConfig(
        transformed_train_file_path=str(transform_path / "train.npz"),
        transformed_test_file_path=str(transform_path / "test.npz"),
        transformed_object_file_path=str(transform_path / "transformer.pkl")
    )

    return {
        'train_path': train_path,
        'test_path': test_path,
        'transform_path': transform_path,
        'ingestion_artifact': ingestion_artifact,
        'validation_artifact': validation_artifact,
        'transformation_config': transformation_config
    }


class TestDataTransformation:
    def test_get_data_transformer_object(self, transformation_setup):
        """Test if transformer object is created correctly"""
        transformer = DataTransformation(
            transformation_setup['ingestion_artifact'],
            transformation_setup['transformation_config'],
            transformation_setup['validation_artifact']
        )

        pipeline = transformer.get_data_transformer_object()
        assert isinstance(pipeline, Pipeline)

    def test_capping_outliers(self, sample_data, transformation_setup):
        """Test outlier capping functionality"""
        transformer = DataTransformation(
            transformation_setup['ingestion_artifact'],
            transformation_setup['transformation_config'],
            transformation_setup['validation_artifact']
        )

        df_transformed = transformer._capping_outliers(sample_data.copy())
        assert df_transformed['Annual_Premium'].max() <= 61000
        assert (df_transformed['Annual_Premium'] <= 61000).all()

    def test_map_columns(self, sample_data, transformation_setup):
        """Test vehicle age mapping functionality"""
        transformer = DataTransformation(
            transformation_setup['ingestion_artifact'],
            transformation_setup['transformation_config'],
            transformation_setup['validation_artifact']
        )

        df_transformed = transformer._map_columns(sample_data.copy())
        assert set(df_transformed['Vehicle_Age'].unique()) <= {0, 1, 2}
        assert df_transformed['Vehicle_Age'].dtype == np.int64

    def test_drop_unnecessary_columns(self, sample_data, transformation_setup):
        """Test column dropping functionality"""
        transformer = DataTransformation(
            transformation_setup['ingestion_artifact'],
            transformation_setup['transformation_config'],
            transformation_setup['validation_artifact']
        )

        df_transformed = transformer._drop_unnecessary_columns(
            sample_data.copy())
        schema_config = transformer._schema_config
        for col in schema_config['drop_columns']:
            assert col not in df_transformed.columns

    def test_initiate_data_transformation(self, sample_data, transformation_setup):
        """Test complete transformation process"""
        # Save sample data as train and test
        train_data = sample_data.copy()
        test_data = sample_data.copy()
        train_data.to_csv(transformation_setup['train_path'], index=False)
        test_data.to_csv(transformation_setup['test_path'], index=False)

        transformer = DataTransformation(
            transformation_setup['ingestion_artifact'],
            transformation_setup['transformation_config'],
            transformation_setup['validation_artifact']
        )

        transformation_artifact = transformer.initiate_data_transformation()

        # Verify artifact paths exist
        assert os.path.exists(
            transformation_artifact.transformed_train_file_path)
        assert os.path.exists(
            transformation_artifact.transformed_test_file_path)
        assert os.path.exists(
            transformation_artifact.transformed_object_file_path)

    def test_read_data(self, sample_data, transformation_setup):
        """Test data reading functionality"""
        # Save sample data
        sample_data.to_csv(transformation_setup['train_path'], index=False)

        df = DataTransformation.read_data(transformation_setup['train_path'])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_data)
        assert all(df.columns == sample_data.columns)

    @pytest.mark.parametrize(
        "invalid_data",
        [
            pd.DataFrame({'Gender': ['Male'], 'Age': [25]}),  # Missing columns
            pd.DataFrame(),  # Empty DataFrame
        ]
    )
    def test_transformation_with_invalid_data(self, invalid_data, transformation_setup):
        """Test transformation with invalid data"""
        invalid_data.to_csv(transformation_setup['train_path'], index=False)
        invalid_data.to_csv(transformation_setup['test_path'], index=False)

        transformer = DataTransformation(
            transformation_setup['ingestion_artifact'],
            transformation_setup['transformation_config'],
            transformation_setup['validation_artifact']
        )

        with pytest.raises(Exception):
            transformer.initiate_data_transformation()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
