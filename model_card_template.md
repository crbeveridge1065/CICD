# Model Card

## Model Details

Connor R Beveridge created this model. It is a Multi-Layer Perceptron using the following hyperparameters in scikit-learn 1.2.2:

- hidden_layer_sizes=(64, 64)
- activation='relu'
- solver='adam'
- max_iter=10000


## Intended Use

The model is to be used to predict whether a person's income exceeds $50K/yr. Users are likely governmental agencies.

## Training Data

The training data is census data originally acquired from the 1994 Census database. The data was obtained for training from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income).

The target class consists of a binary of greater than or less than $50K/yr.

The data has 48842 instances. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

Evaluation data used was split from the original data at 20% split. No stratification was done. 

## Metrics

On evaluation, the model showed a precision of .63, a recall of .36, and a fbeta score of .46.

## Ethical Considerations

This model has not shown performance standards and should not be used for official duties.

The model as also shown very poor performance for those with unknown workclass.

## Caveats and Recommendations

No hyperparameter tuning was done. Hyperparameter tuning is recommended to improve the performance of the model.