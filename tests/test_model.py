import numpy as np
import pytest

# Assuming the model module exists and contains a class like DummyModel
# For self-contained testing, we define a DummyModel here.
# In a real scenario, you would import this from your actual model module:
# from model import Model as DummyModel # Or whatever your model class is named

class DummyModel:
    def __init__(self, hyperparameter1=0.1, hyperparameter2='default'):
        self.hyperparameter1 = hyperparameter1
        self.hyperparameter2 = hyperparameter2
        self._is_trained = False
        self._internal_param = None
        self.classes_ = None

    def train(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
             raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self._internal_param = np.mean(X, axis=0)
        self.classes_ = np.unique(y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        if not isinstance(X, np.ndarray):
             raise TypeError("X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self._internal_param.shape[0]:
             raise ValueError("X has incorrect number of features.")

        if self.classes_ is not None and len(self.classes_) > 0:
            # Simple prediction: return the first class learned
            predictions = np.full(X.shape[0], self.classes_[0], dtype=self.classes_.dtype)
        else:
             # Fallback unlikely case
             predictions = np.zeros(X.shape[0])
        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
             raise TypeError("X and y must be numpy arrays.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[1] != self._internal_param.shape[0]:
             raise ValueError("X has incorrect number of features.")

        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return float(accuracy)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

# --- Pytest Fixtures ---

@pytest.fixture
def sample_data():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 1, 0, 1])
    return X, y

@pytest.fixture
def dummy_model_instance():
    return DummyModel(hyperparameter1=0.5, hyperparameter2='test')

# --- Test Functions ---

def test_model_initialization(dummy_model_instance):
    model = dummy_model_instance
    assert model.hyperparameter1 == 0.5
    assert model.hyperparameter2 == 'test'
    assert not model.is_trained
    assert model._internal_param is None
    assert model.classes_ is None

def test_model_training(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    assert not model.is_trained
    assert model._internal_param is None
    assert model.classes_ is None

    model.train(X_train, y_train)

    assert model.is_trained
    assert model._internal_param is not None
    assert isinstance(model._internal_param, np.ndarray)
    assert model._internal_param.shape == (X_train.shape[1],)
    assert model.classes_ is not None
    assert isinstance(model.classes_, np.ndarray)
    np.testing.assert_array_equal(np.sort(model.classes_), np.unique(y_train))

def test_model_training_raises_error_on_mismatch_samples(dummy_model_instance):
    model = dummy_model_instance
    X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_train_wrong = np.array([0])
    with pytest.raises(ValueError, match="same number of samples"):
        model.train(X_train, y_train_wrong)

def test_model_training_raises_error_on_wrong_dims(dummy_model_instance):
    model = dummy_model_instance
    X_train_1d = np.array([1.0, 2.0, 3.0, 4.0])
    X_train_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_train_1d = np.array([0, 1])
    y_train_2d = np.array([[0], [1]])

    with pytest.raises(ValueError, match="X must be a 2D array"):
        model.train(X_train_1d, y_train_1d)
    with pytest.raises(ValueError, match="y must be a 1D array"):
        model.train(X_train_2d, y_train_2d)

def test_model_prediction_before_training(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_test, _ = sample_data
    with pytest.raises(RuntimeError, match="Model must be trained"):
        model.predict(X_test)

def test_model_prediction_after_training(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    X_test = np.array([[9.0, 10.0], [11.0, 12.0]])

    model.train(X_train, y_train)
    assert model.is_trained

    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_test.shape[0],)
    assert predictions.dtype == y_train.dtype
    assert np.all(np.isin(predictions, model.classes_))
    # Check specific dummy prediction logic
    expected_predictions = np.full(X_test.shape[0], model.classes_[0], dtype=model.classes_.dtype)
    np.testing.assert_array_equal(predictions, expected_predictions)

def test_model_prediction_raises_error_on_wrong_features(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    model.train(X_train, y_train) # Train with 2 features

    X_test_wrong_features = np.array([[9.0, 10.0, 11.0]]) # 3 features
    with pytest.raises(ValueError, match="incorrect number of features"):
        model.predict(X_test_wrong_features)

def test_model_prediction_raises_error_on_wrong_dims(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    model.train(X_train, y_train)

    X_test_1d = np.array([9.0, 10.0])
    with pytest.raises(ValueError, match="X must be a 2D array"):
        model.predict(X_test_1d)

def test_model_evaluation_before_training(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_test, y_test = sample_data
    with pytest.raises(RuntimeError, match="Model must be trained"):
        model.evaluate(X_test, y_test)

def test_model_evaluation_after_training(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    X_test, y_test = sample_data # Use same data for simplicity

    model.train(X_train, y_train)
    assert model.is_trained

    score = model.evaluate(X_test, y_test)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    # Check specific dummy evaluation logic (predicts class 0, y=[0,1,0,1] -> acc=0.5)
    expected_score = 0.5
    assert score == pytest.approx(expected_score)

def test_model_evaluation_raises_error_on_mismatch_samples(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    model.train(X_train, y_train)

    X_test = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_test_wrong = np.array([0])
    with pytest.raises(ValueError, match="same number of samples"):
        model.evaluate(X_test, y_test_wrong)

def test_model_evaluation_raises_error_on_wrong_features(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    model.train(X_train, y_train) # Train with 2 features

    X_test_wrong_features = np.array([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]) # 3 features
    y_test = np.array([0, 1])
    with pytest.raises(ValueError, match="incorrect number of features"):
        model.evaluate(X_test_wrong_features, y_test)

def test_model_evaluation_raises_error_on_wrong_dims(dummy_model_instance, sample_data):
    model = dummy_model_instance
    X_train, y_train = sample_data
    model.train(X_train, y_train)

    X_test_1d = np.array([1.0, 2.0, 3.0, 4.0])
    X_test_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_test_1d = np.array([0, 1])
    y_test_2d = np.array([[0], [1]])

    with pytest.raises(ValueError, match="X must be a 2D array"):
        model.evaluate(X_test_1d, y_test_1d)
    with pytest.raises(ValueError, match="y must be a 1D array"):
        model.evaluate(X_test_2d, y_test_2d)