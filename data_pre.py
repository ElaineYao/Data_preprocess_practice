import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

data_path = './dataset/cars_data.csv'

# Set pandas output display to have one digit for decimal places
# and limit it to printing 15 rows
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 15

# Provide the names for the columns since the CSV file with the data
# doesn't have a header row

feature_names = ['symboling', 'normalized-losses', 'make', 'fuel-type',
        'aspiration', 'num-doors', 'body-style', 'drive-wheels',
        'engine-location', 'wheel-base', 'length', 'width', 'height', 'weight',
        'engine-type', 'num-cylinders', 'engine-size', 'fuel-system', 'bore',
        'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg',
        'highway-mpg', 'price']

LABEL = 'price'
numeric_feature_names = ['symboling', 'normalized-losses', 'wheel-base',
            'length', 'width', 'height', 'weight', 'engine-size', 'horsepower',
            'peak-rpm', 'city-mpg', 'highway-mpg', 'bore', 'stroke',
             'compression-ratio']

batch_size = 16


def preprocessing(feature_names):
    # Load in the data from a CSV
    car_data = pd.read_csv(data_path, sep=',', names=feature_names, header=None, encoding='latin-1')

    # We'll randomize the data, just to be sure not to get any pathological ordering effects that
    # might harm the performance of Stochastic Gradient Descent
    car_data = car_data.reindex(np.random.permutation(car_data.index))

    print("***********")
    print("Data set loaded. Num examples: ", len(car_data))

    # Split the original training set into a reduced training set, a validation set and a training set
    test_split = 0.2
    validation_split=0.2
    x_train, x_test = train_test_split(car_data, test_size=test_split, random_state=1)
    x_trin, x_val = train_test_split(x_train, test_size=validation_split, random_state=1)

    print('***********')
    print('Dataset is divided in training set with length {}, validation set with length {}'
          ' and test set with length {}'.format(len(x_train), len(x_val), len(x_test)))

    # Manually curate a list of numeric_feature_names and categorical_feature_names
    categorical_feature_names = list(set(feature_names) - set(numeric_feature_names) - set([LABEL]))

    print('****')
    print(x_train[numeric_feature_names])

    # Coerce the numeric features to numbers
    for feature_names in numeric_feature_names + [LABEL]:
        x_train[feature_names] = pd.to_numeric(x_train[feature_names], errors='coerce')

    # Fill the missing value with 0
    x_train.fillna(0, inplace=True)

    # train the model with numeric features and categorical features
    x_df = x_train[numeric_feature_names + categorical_feature_names]
    y_series = x_train['price']

    # feed panda data into the model
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
                    x=x_df,
                    y=y_series,
                    batch_size=batch_size,
                    num_epochs=None,
                    shuffle=True
    )
    eval_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=x_df,
        y=y_series,
        batch_size=batch_size,
        shuffle=False
    )
    predict_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=x_df,
        batch_size=batch_size,
        shuffle=False
    )

    # Feature columns allow the model to parse the data, perform common
    # preprocessing, and automatically generate an input layer for the tf.Estimator
    # Epsilon prevents divide by zero
    # For categorical data
    # We will have an in-memory vocabulary mapping each value to an integer ID
    epsilon = 0.000001
    model_feature_columns = [tf.feature_column.numeric_column(feature_name,
                            normalizer_fn=lambda val: (val - x_df.mean()[feature_name]) / (epsilon + x_df.std()[feature_name])) for feature_name in numeric_feature_names]


    model_feature_columns = [tf.feature_column.indicator_column(
                                tf.feature_column.categorical_column_with_vocabulary_list(
                                        feature_name, vocabulary_list=x_train[feature_name].unique()))
                                    for feature_name in categorical_feature_names
    ] + [tf.feature_column.numeric_column(feature_name,
                            normalizer_fn=lambda val: (val - x_df.mean()[feature_name]) / (epsilon + x_df.std()[feature_name])) for feature_name in numeric_feature_names]

    return train_input_fn, eval_input_fn, model_feature_columns


if __name__ == '__main__':
    preprocessing()

