from data_pre import *
from network import *

# Hyperparameters
num_print_statements = 10
num_training_steps = 10000

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

def train():
    [train_input_fn, eval_input_fn, feature_columns] = preprocessing(feature_names)
    est = network(feature_columns)
    for _ in range(num_print_statements):
        est.train(train_input_fn, steps=num_training_steps // num_print_statements)
        scores = est.evaluate(eval_input_fn)
        print('scores', scores)

if __name__ == '__main__':
    train()