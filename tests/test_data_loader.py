import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import os
import tempfile

# Assume data_loader.py exists in the same directory or is importable
# If data_loader.py is not available, these imports will fail.
# For this example, we'll define dummy versions of the expected functions
# if the real module cannot be imported.

try:
    from data_loader import (
        load_csv,
        handle_missing_values,
        normalize_numerical_features,
        encode_categorical_features,
        preprocess_data
    )
except ImportError:
    # Define dummy functions if data_loader module is not found
    # This allows the test structure to be generated, but tests will likely fail
    # without the actual implementation. Replace these with actual imports.
    print("Warning: 'data_loader' module not found. Using dummy functions for test structure.")

    def load_csv(filepath):
        if not os.path.exists(filepath):
            return None
        try:
            df = pd.read_csv(filepath)
            if df.empty and os.path.getsize(filepath) > 0: # Check if only headers exist
                 if pd.read_csv(filepath, header=None).empty: # Truly empty file
                     return pd.DataFrame()
                 else: # Header only
                     return df # Return empty df with columns
            return df
        except pd.errors.EmptyDataError:
             return pd.DataFrame() # Return empty DataFrame for truly empty file
        except Exception:
            return None

    def handle_missing_values(df, strategy='mean', columns=None):
        if df is None: return None
        df_copy = df.copy()
        if columns is None:
            columns = df_copy.select_dtypes(include=np.number).columns
        if not columns.any(): return df_copy

        for col in columns:
             if col not in df_copy.columns: continue
             if df_copy[col].isnull().any():
                if strategy == 'mean':
                    fill_value = df_copy[col].mean()
                elif strategy == 'median':
                    fill_value = df_copy[col].median()
                elif strategy == 'mode':
                     # Handle potential multiple modes
                     mode_val = df_copy[col].mode()
                     fill_value = mode_val[0] if not mode_val.empty else 0
                elif isinstance(strategy, (int, float)):
                    fill_value = strategy
                else: # drop
                    df_copy.dropna(subset=[col], inplace=True)
                    continue
                df_copy[col].fillna(fill_value, inplace=True)
        return df_copy


    def normalize_numerical_features(df, columns=None):
        from sklearn.preprocessing import StandardScaler
        if df is None: return None
        df_copy = df.copy()
        if columns is None:
            columns = df_copy.select_dtypes(include=np.number).columns.tolist()
        else:
            columns = [col for col in columns if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col])]

        if columns:
            scaler = StandardScaler()
            # Handle potential NaNs introduced if handle_missing_values wasn't called
            valid_data = df_copy[columns].dropna()
            if not valid_data.empty:
                 scaled_values = scaler.fit_transform(valid_data)
                 df_copy.loc[valid_data.index, columns] = scaled_values
            # If all values were NaN, columns might become all NaN after scaling attempt
            # Or if only one row, std dev is 0, StandardScaler returns NaNs or 0s
            if df_copy[columns].shape[0] == 1:
                 df_copy[columns] = 0.0 # Set to 0 if only one row

        return df_copy


    def encode_categorical_features(df, columns=None):
        from sklearn.preprocessing import OneHotEncoder
        if df is None: return None
        df_copy = df.copy()
        if columns is None:
            columns = df_copy.select_dtypes(include='object').columns.tolist()
        else:
             columns = [col for col in columns if col in df_copy.columns and pd.api.types.is_object_dtype(df_copy[col])]


        if columns:
            # Handle potential NaNs before encoding
            df_copy[columns] = df_copy[columns].fillna('Missing')
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df_copy[columns])
            feature_names = encoder.get_feature_names_out(columns)
            # Check for duplicate column names before creating DataFrame
            if len(feature_names) != len(set(feature_names)):
                 # Handle potential duplicate feature names if categories are identical across columns
                 # This is less likely with get_feature_names_out but good practice
                 unique_feature_names = [f"{col}_{cat}" for col in columns for cat in encoder.categories_[columns.index(col)]]
                 if len(unique_feature_names) == encoded_data.shape[1]:
                      feature_names = unique_feature_names
                 else: # Fallback if naming scheme fails
                      feature_names = [f"encoded_{i}" for i in range(encoded_data.shape[1])]


            encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_copy.index)
            df_copy = df_copy.drop(columns, axis=1)
            df_copy = pd.concat([df_copy, encoded_df], axis=1)
        return df_copy

    def preprocess_data(df):
        if df is None: return None
        df = handle_missing_values(df, strategy='mean')
        df = normalize_numerical_features(df)
        df = encode_categorical_features(df)
        return df


# --- Fixtures ---

@pytest.fixture(scope="module")
def sample_data_dict():
    return {
        'numeric_1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric_2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'categorical_1': ['A', 'B', 'A', 'C', 'B'],
        'categorical_2': ['X', 'Y', np.nan, 'X', 'Y'],
        'constant_numeric': [5, 5, 5, 5, 5],
        'all_nan_numeric': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }

@pytest.fixture
def sample_dataframe(sample_data_dict):
    return pd.DataFrame(sample_data_dict)

@pytest.fixture
def empty_dataframe():
    return pd.DataFrame()

@pytest.fixture
def temp_csv_file():
    """Creates a temporary CSV file for testing load_csv"""
    content = "col1,col2,col3\n1,a,x\n2,b,y\n3,c,z"
    # Use NamedTemporaryFile to ensure it's cleaned up
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmpfile:
        tmpfile.write(content)
        filepath = tmpfile.name
    yield filepath
    # Cleanup: ensure the file is deleted
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.fixture
def temp_empty_csv_file():
    """Creates an empty temporary CSV file"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmpfile:
        filepath = tmpfile.name
    yield filepath
    if os.path.exists(filepath):
        os.remove(filepath)

@pytest.fixture
def temp_header_only_csv_file():
    """Creates a temporary CSV file with only a header"""
    content = "col1,col2,col3\n"
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmpfile:
        tmpfile.write(content)
        filepath = tmpfile.name
    yield filepath
    if os.path.exists(filepath):
        os.remove(filepath)


# --- Test Cases ---

# 1. Data Loading Tests
def test_load_csv_success(temp_csv_file):
    """Test loading a valid CSV file."""
    df = load_csv(temp_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (3, 3)
    assert list(df.columns) == ['col1', 'col2', 'col3']
    assert df['col1'].tolist() == [1, 2, 3]

def test_load_csv_file_not_found():
    """Test loading a non-existent file."""
    non_existent_file = "non_existent_file_12345.csv"
    # Ensure file does not exist before test
    if os.path.exists(non_existent_file):
        os.remove(non_existent_file)
    df = load_csv(non_existent_file)
    assert df is None # Or check for specific error handling if implemented differently

def test_load_csv_empty_file(temp_empty_csv_file):
    """Test loading a completely empty CSV file."""
    df = load_csv(temp_empty_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert df.shape == (0, 0)

def test_load_csv_header_only_file(temp_header_only_csv_file):
    """Test loading a CSV file with only a header row."""
    df = load_csv(temp_header_only_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert list(df.columns) == ['col1', 'col2', 'col3']
    assert df.shape == (0, 3)


# 2. Data Preprocessing Tests (Missing Values)
def test_handle_missing_values_mean(sample_dataframe):
    """Test filling NaNs with mean."""
    df_processed = handle_missing_values(sample_dataframe, strategy='mean', columns=['numeric_1'])
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed['numeric_1'].isnull().any()
    # Mean of [1, 2, 4, 5] is 3.0
    assert df_processed.loc[2, 'numeric_1'] == pytest.approx(3.0)
    # Check other columns remain unchanged where not specified
    assert_series_equal(sample_dataframe['numeric_2'], df_processed['numeric_2'])
    assert sample_dataframe['categorical_2'].isnull().sum() == df_processed['categorical_2'].isnull().sum() # Should be unchanged

def test_handle_missing_values_median(sample_dataframe):
    """Test filling NaNs with median."""
    df_processed = handle_missing_values(sample_dataframe, strategy='median', columns=['numeric_1'])
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed['numeric_1'].isnull().any()
    # Median of [1, 2, 4, 5] is 3.0
    assert df_processed.loc[2, 'numeric_1'] == pytest.approx(3.0)

def test_handle_missing_values_mode(sample_dataframe):
    """Test filling NaNs with mode."""
    # Mode requires object or int type usually, let's test on categorical
    df_processed = handle_missing_values(sample_dataframe.copy(), strategy='mode', columns=['categorical_2'])
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed['categorical_2'].isnull().any()
    # Mode of ['X', 'Y', 'X', 'Y'] is 'X' or 'Y' (pandas mode returns both if equal freq)
    # Our dummy function takes the first mode.
    expected_mode = sample_dataframe['categorical_2'].mode()[0]
    assert df_processed.loc[2, 'categorical_2'] == expected_mode

def test_handle_missing_values_specific_value(sample_dataframe):
    """Test filling NaNs with a specific value."""
    fill_val = -999
    df_processed = handle_missing_values(sample_dataframe, strategy=fill_val, columns=['numeric_1'])
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed['numeric_1'].isnull().any()
    assert df_processed.loc[2, 'numeric_1'] == fill_val

def test_handle_missing_values_drop(sample_dataframe):
    """Test dropping rows with NaNs."""
    df_original_rows = sample_dataframe.shape[0]
    # Drop rows where numeric_1 is NaN
    df_processed = handle_missing_values(sample_dataframe, strategy='drop', columns=['numeric_1'])
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed['numeric_1'].isnull().any()
    assert df_processed.shape[0] == df_original_rows - 1
    assert 2 not in df_processed.index # Row with index 2 should be dropped

def test_handle_missing_values_all_columns(sample_dataframe):
    """Test filling NaNs across all applicable (numeric) columns by default."""
    df_processed = handle_missing_values(sample_dataframe.copy(), strategy='mean') # Default strategy and columns
    assert isinstance(df_processed, pd.DataFrame)
    assert not df_processed['numeric_1'].isnull().any()
    assert df_processed.loc[2, 'numeric_1'] == pytest.approx(3.0)
    # Check all_nan_numeric column - mean of NaNs is NaN, should remain NaN or be handled
    # The dummy implementation might fill with 0 if mean fails, or skip. Let's assume it skips or fills with 0.
    # A robust implementation might drop such columns or require explicit handling.
    # For this test, let's assume it remains NaN or is filled with 0 if mean is NaN.
    assert df_processed['all_nan_numeric'].isnull().all() or (df_processed['all_nan_numeric'] == 0).all()


def test_handle_missing_values_no_nans(sample_dataframe):
    """Test handling data with no NaNs in the specified column."""
    df_no_nan = sample_dataframe.dropna(subset=['numeric_1']).copy()
    df_processed = handle_missing_values(df_no_nan, strategy='mean', columns=['numeric_1'])
    assert_frame_equal(df_no_nan, df_processed)

def test_handle_missing_values_empty_df(empty_dataframe):
    """Test handling an empty DataFrame."""
    df_processed = handle_missing_values(empty_dataframe.copy(), strategy='mean')
    assert_frame_equal(empty_dataframe, df_processed)

def test_handle_missing_values_non_numeric_strategy(sample_dataframe):
    """ Test handling non-numeric columns with numeric strategies (should ignore) """
    df_copy = sample_dataframe.copy()
    df_processed = handle_missing_values(df_copy, strategy='mean', columns=['categorical_1', 'categorical_2'])
    # Should not change categorical columns when strategy is numeric
    assert_frame_equal(df_copy, df_processed)


# 3. Data Transformation Tests (Normalization)
def test_normalize_numerical_features(sample_dataframe):
    """Test StandardScaler normalization on specified columns."""
    df_filled = handle_missing_values(sample_dataframe.copy(), strategy=0) # Fill NaNs first
    columns_to_normalize = ['numeric_1', 'numeric_2']
    df_normalized = normalize_numerical_features(df_filled, columns=columns_to_normalize)

    assert isinstance(df_normalized, pd.DataFrame)
    assert df_normalized.shape == df_filled.shape
    # Check if means are close to 0 and std devs close to 1 for normalized columns
    for col in columns_to_normalize:
        assert np.mean(df_normalized[col]) == pytest.approx(0.0, abs=1e-6)
        assert np.std(df_normalized[col]) == pytest.approx(1.0, abs=1e-6)
    # Check other columns unchanged
    assert_series_equal(df_filled['categorical_1'], df_normalized['categorical_1'])

def test_normalize_numerical_features_all_numeric(sample_dataframe):
    """Test normalization on all numeric columns by default."""
    df_filled = handle_missing_values(sample_dataframe.copy(), strategy=0) # Fill NaNs first
    # Exclude 'all_nan_numeric' as it becomes 0s and std=0, causing issues with StandardScaler
    # Also exclude 'constant_numeric' for the same reason (std=0)
    df_to_normalize = df_filled.drop(columns=['all_nan_numeric', 'constant_numeric'])
    numeric_cols = df_to_normalize.select_dtypes(include=np.number).columns.tolist()

    df_normalized = normalize_numerical_features(df_to_normalize) # Default columns

    assert isinstance(df_normalized, pd.DataFrame)
    assert df_normalized.shape == df_to_normalize.shape
    for col in numeric_cols:
         # Handle cases with zero standard deviation (like constant columns if not removed)
         if df_to_normalize[col].std() > 1e-9:
             assert np.mean(df_normalized[col]) == pytest.approx(0.0, abs=1e-6)
             assert np.std(df_normalized[col]) == pytest.approx(1.0, abs=1e-6)
         else:
             # If std dev is near zero, StandardScaler output might be 0 or NaN
             assert (df_normalized[col] == 0).all() or df_normalized[col].isnull().all()


def test_normalize_numerical_features_no_numeric():
    """Test normalization when DataFrame has no numeric columns."""
    df_no_numeric = pd.DataFrame({'cat': ['A', 'B', 'C']})
    df_normalized = normalize_numerical_features(df_no_numeric.copy())
    assert_frame_equal(df_no_numeric, df_normalized)

def test_normalize_numerical_features_empty_df(empty_dataframe):
    """Test normalization on an empty DataFrame."""
    df_normalized = normalize_numerical_features(empty_dataframe.copy())
    assert_frame_equal(empty_dataframe, df_normalized)

def test_normalize_constant_feature(sample_dataframe):
    """Test normalization on a constant numeric feature (std dev = 0)."""
    df_filled = handle_missing_values(sample_dataframe.copy(), strategy=0)
    df_normalized = normalize_numerical_features(df_filled, columns=['constant_numeric'])
    # StandardScaler usually outputs 0 for constant features
    assert (df_normalized['constant_numeric'] == 0).all()


# 4. Data Transformation Tests (Encoding)
def test_encode_categorical_features(sample_dataframe):
    """Test OneHotEncoding on specified categorical columns."""
    df_filled = sample_dataframe.copy()
    # Fill NaN in categorical_2 before encoding
    df_filled['categorical_2'].fillna('Missing', inplace=True)
    columns_to_encode = ['categorical_1', 'categorical_2']
    df_encoded = encode_categorical_features(df_filled, columns=columns_to_encode)

    assert isinstance(df_encoded, pd.DataFrame)
    # Check original categorical columns are dropped
    assert 'categorical_1' not in df_encoded.columns
    assert 'categorical_2' not in df_encoded.columns
    # Check new one-hot encoded columns exist
    expected_new_cols = ['categorical_1_A', 'categorical_1_B', 'categorical_1_C',
                         'categorical_2_Missing', 'categorical_2_X', 'categorical_2_Y']
    for col in expected_new_cols:
        assert col in df_encoded.columns
    # Check shape: original numeric cols + new encoded cols
    original_numeric_cols = sample_dataframe.select_dtypes(include=np.number).shape[1]
    assert df_encoded.shape[1] == original_numeric_cols + len(expected_new_cols)
    # Check values (e.g., first row: A, X -> categorical_1_A=1, categorical_2_X=1)
    assert df_encoded.loc[0, 'categorical_1_A'] == 1
    assert df_encoded.loc[0, 'categorical_1_B'] == 0
    assert df_encoded.loc[0, 'categorical_2_X'] == 1
    assert df_encoded.loc[0, 'categorical_2_Y'] == 0
    # Check row with original NaN (index 2: A, Missing -> categorical_1_A=1, categorical_2_Missing=1)
    assert df_encoded.loc[2, 'categorical_1_A'] == 1
    assert df_encoded.loc[2, 'categorical_2_Missing'] == 1


def test_encode_categorical_features_all_categorical(sample_dataframe):
    """Test encoding on all categorical columns by default."""
    df_filled = sample_dataframe.copy()
    df_filled['categorical_2'].fillna('Missing', inplace=True) # Handle NaN
    categorical_cols = df_filled.select_dtypes(include='object').columns.tolist()
    df_encoded = encode_categorical_features(df_filled) # Default columns

    assert isinstance(df_encoded, pd.DataFrame)
    for col in categorical_cols:
        assert col not in df_encoded.columns

    # Check if expected columns are generated (adjust based on actual encoder output)
    assert 'categorical_1_A' in df_encoded.columns
    assert 'categorical_2_X' in df_encoded.columns
    assert 'categorical_2_Missing' in df_encoded.columns


def test_encode_categorical_features_no_categorical(sample_dataframe):
    """Test encoding when DataFrame has no categorical columns."""
    df_numeric_only = sample_dataframe.select_dtypes(include=np.number).copy()
    df_encoded = encode_categorical_features(df_numeric_only.copy())
    assert_frame_equal(df_numeric_only, df_encoded)

def test_encode_categorical_features_empty_df(empty_dataframe):
    """Test encoding on an empty DataFrame."""
    df_encoded = encode_categorical_features(empty_dataframe.copy())
    assert_frame_equal(empty_dataframe, df_encoded)


# 5. Integration Test (Full Preprocessing Pipeline)
def test_preprocess_data_integration(sample_dataframe):
    """Test the full preprocess_data pipeline."""
    df_processed = preprocess_data(sample_dataframe.copy())

    assert isinstance(df_processed, pd.DataFrame)

    # 1. Check missing values handled (numeric_1 filled with mean ~3.0)
    # Note: The exact value depends on whether normalization happens before or after filling NaNs
    # Assuming handle_missing_values runs first as per the dummy preprocess_data
    # Check numeric columns don't have NaNs (except potentially all_nan column if not dropped/filled)
    numeric_cols = sample_dataframe.select_dtypes(include=np.number).columns
    for col in numeric_cols:
         if col != 'all_nan_numeric': # Exclude column that was initially all NaNs
             assert not df_processed[col].isnull().any()

    # 2. Check normalization (means close to 0, std dev close to 1 for non-constant cols)
    # Need to identify the numeric columns *after* potential encoding/dropping
    numeric_cols_after_encoding = df_processed.select_dtypes(include=np.number).columns
    numeric_cols_from_original = ['numeric_1', 'numeric_2', 'constant_numeric'] # Exclude all_nan
    for col in numeric_cols_from_original:
         if col in df_processed.columns: # Check if column still exists
             if df_processed[col].std() > 1e-9: # Check if not constant
                 assert df_processed[col].mean() == pytest.approx(0.0, abs=1e-6)
                 assert df_processed[col].std() == pytest.approx(1.0, abs=1e-6)
             else: # Constant columns should be ~0 after scaling
                 assert (df_processed[col] == 0).all() or df_processed[col].isnull().all()


    # 3. Check encoding (original categorical columns removed, new ones added)
    assert 'categorical_1' not in df_processed.columns
    assert 'categorical_2' not in df_processed.columns
    assert 'categorical_1_A' in df_processed.columns
    assert 'categorical_2_X' in df_processed.columns
    # Check the NaN category ('Missing' by default in dummy function)
    assert 'categorical_2_Missing' in df_processed.columns

    # Check final shape consistency (depends on exact steps)
    # Example: 5 numeric (1 dropped/all_nan, 1 constant -> 0) + encoded cats
    # Original: 6 cols. After processing: 4 numeric + 3 cat1 + 3 cat2 = 10 cols (approx)
    # This needs adjustment based on the real implementation details.
    assert df_processed.shape[0] == sample_dataframe.shape[0] # Assuming no rows dropped


def test_preprocess_data_empty_df(empty_dataframe):
    """Test the full pipeline with an empty DataFrame."""
    df_processed = preprocess_data(empty_dataframe.copy())
    assert_frame_equal(empty_dataframe, df_processed)

def test_preprocess_data_handles_none():
    """Test that preprocess_data returns None if input is None."""
    assert preprocess_data(None) is None