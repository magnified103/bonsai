import os
from abc import ABC

import numpy as np
import ray.tune
from sklearn.model_selection import train_test_split

from network_generator.models import SimpleFFNNModel, load_model
from training import loader
from training.train_ffnn import train_network


class Trainable(ray.tune.Trainable):
    def setup(self, config):
        train, test = loader.load_all_data(None, "/home/akos/bonsai/training/data/small_sample_data.csv.gz")
        trainset = np.concatenate((train[0], train[1]), axis=1)
        train_subset, val_subset = train_test_split(
            trainset, test_size=0.2, random_state=42)

        self.train = np.split(train_subset, [train_subset.shape[1] - 1], axis=1)
        self.val = np.split(val_subset, [val_subset.shape[1] - 1], axis=1)

        self.config = config
        del self.config["layers"]
        self.model = SimpleFFNNModel(self.config)

    def step(self):
        training_loss, loss, test_r2, test_rmse, test_mae, test_mape = train_network(self.model, self.train, self.val)
        return {"training_loss": training_loss, "loss": loss, "mape": test_mape, "r2": test_r2, "rmse": test_rmse,
                "mae": test_mae}

    def reset_config(self, new_config):
        self.config = new_config
        del self.config["layers"]
        self.model = SimpleFFNNModel(self.config)
        return True

    def save_checkpoint(self, checkpoint_dir):
        return self.model.save(os.path.join(checkpoint_dir, 'model.pth'))

    def load_checkpoint(self, checkpoint_path):
        self.model = load_model(os.path.join(checkpoint_path, 'model.pth'))
        return True