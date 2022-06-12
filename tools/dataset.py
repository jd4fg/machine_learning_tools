import numpy as np


class Dataset:
    def __init__(self):
        self.raw_input_val = np.array([
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6]
        ])
        self.true_val = np.array([
            0.04, 1.52, 2.111, 2.81, 4.44, 5.1, 6.56
        ])

        # optimizes data
        self.std = np.std(self.raw_input_val, axis=0, keepdims=True)
        self.mean = np.mean(self.raw_input_val, axis=0, keepdims=True)
        self.input_val = self.optimize_value(self.raw_input_val)
        self.input_val = self.add_intercept()

        self.feature_count = self.input_val.shape[-1]
        self.input_count = self.input_val.shape[0]

    # scales data
    def feature_scaling(self, data):
        return data / (self.std + 1e-8)

    # mean-normalizes data
    def mean_normalisation(self, data):
        return data - self.mean

    # applies feature scaling and mean normalization for gradient descent optimization
    def optimize_value(self, data):
        return self.feature_scaling(self.mean_normalisation(data))

    def add_intercept(self, data=None):
        if data is None:
            data = self.input_val

        sample_size = data.shape[0]
        new_data = np.concatenate(
            (np.ones(shape=(sample_size, 1)), data),
            axis=-1
        )
        return new_data

