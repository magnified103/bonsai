import os
import unittest

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

from training.loader import load_data

class LoaderTestCase(unittest.TestCase):
    def test_load_data(self):
        os.chdir(os.path.dirname(__file__))
        self.assertEqual(os.getcwd(), os.path.dirname(__file__))
        train, test = load_data('../training/data')
        print(train[0])

    def test_split_and_merge(self):
        os.chdir(os.path.dirname(__file__))
        self.assertEqual(os.getcwd(), os.path.dirname(__file__))
        train, _ = load_data('../training/data')
        trainset = np.concatenate((train[0], train[1]), axis=1)
        train_subset, val_subset = train_test_split(
            trainset, test_size=0.2, random_state=42)

        train_X, train_y = np.split(train_subset, [train_subset.shape[1]-1], axis=1)
        self.assertEqual(train_X.shape[1], train[0].shape[1])
        self.assertEqual(train_y.shape[1], train[1].shape[1])

if __name__ == '__main__':
    unittest.main()
