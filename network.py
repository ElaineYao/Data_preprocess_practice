from data_pre import *

# Hyperparameters
batch_size = 16

def network(model_feature_columns):
    est = tf.estimator.DNNRegressor(
        feature_columns=model_feature_columns,
        hidden_units=[64],
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.01)
    )
    return est
