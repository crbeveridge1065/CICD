# Add the necessary imports for the starter code.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import pandas as pd

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

def slice_metric(dataframe, feature):
    '''

    prints performance on each slice of the original data based on feature

    :param dataframe: dataframe to slice
    :param feature: feature to slice on

    '''

    # Only write class once, keeps track of first time
    first = True

    with open('src/slice_output.txt', 'a') as f:
        for cls in dataframe[feature].unique():
            df_temp = dataframe[dataframe[feature] == cls]

            # Optional enhancement, use K-fold cross validation instead of a train-test split.
            train, test = train_test_split(df_temp, test_size=0.20)

            X_train, y_train, encoder, lb = process_data(
                train, categorical_features=cat_features, label="salary", training=True
            )

            # Proces the test data with the process_data function.
            X_test, y_test, encoder, lb = process_data(
                test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )

            # Train the model.
            model = train_model(X_train, y_train)

            precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))


            if first:
                f.write(f"Class: {feature}")
                first = False

            f.write('\n')
            f.write(f"{cls} precision: {precision:.4f}")
            f.write('\n')
            f.write(f"{cls} recall: {recall:.4f}")
            f.write('\n')
            f.write(f"{cls} fbeta: {fbeta:.4f}")
            f.write('\n')
            f.write('\n')

        f.close()


with open('src/slice_output.txt', 'w') as f:
    f.write('\n')
    f.close()

# Add code to load in the data.
data = pd.read_csv('data/census.csv')

slice_metric(data, cat_features[0])