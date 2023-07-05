import pandas as pd
import pytest
import pickle
import os
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture
def clean_data():
    # Returns list Holding data as a tuple of (X_train, y_train, X_test, y_test)
    # as well as a tuple of the resulting onehot encoder and binarizer

    # read data from csv
    data = pd.read_csv('data/census.csv')

    #Split into train and test
    train, test = train_test_split(data, test_size=0.20)

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    return (X_train, y_train, X_test, y_test), (encoder, lb)

@pytest.fixture
def trained_model(clean_data):
    X_train, y_train, X_test, y_test = clean_data[0]
    encoder, _ = clean_data[1]

    # Train Model
    return train_model(X_train, y_train)

def test_process_data(clean_data):

    X_train, y_train, X_test, y_test = clean_data[0]
    encoder, lb = clean_data[1]

    # Make sure cleaned dataframes have rows and columns
    assert X_train.shape[0] > 100
    assert X_train.shape[1] > 5

    assert y_train.shape[0] > 100

    assert X_test.shape[0] > 100
    assert X_test.shape[1] > 5

    assert y_test.shape[0] > 100

    # X_test and X_train should have same number of features

    assert X_train.shape[1] == X_test.shape[1]

    # Check types, including for encoder and binarizer, are appropriate
    assert 'numpy.ndarray' in str(type(X_train))
    assert 'numpy.ndarray' in str(type(y_train))
    assert 'numpy.ndarray' in str(type(X_test))
    assert 'numpy.ndarray' in str(type(y_test))
    assert 'sklearn.preprocessing._encoders.OneHotEncoder' in str(type(encoder))
    assert 'sklearn.preprocessing._label.LabelBinarizer' in str(type(lb))

def test_model_training(trained_model):

    '''
    # Save Model and trained encoder
    with open('src/test_results_pickle/mlp_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('src/test_results_pickle/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    '''
    assert 'sklearn.neural_network._multilayer_perceptron.MLPClassifier' in str(type(trained_model))


def test_inference_and_performance(clean_data, trained_model):

    X_train, y_train, X_test, y_test = clean_data[0]

    # Perform inference
    preds = inference(trained_model, X_test)

    # Test testing data predictions type and shape
    assert 'numpy.ndarray' in str(type(preds))
    assert preds.shape[0] > 5

    # Compute model metric
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Perform type asserts
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    # Make sure results are not alarmingly low
    assert precision > 0
    assert recall > 0
    assert fbeta > 0

