import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import network_generator.models.base as models
from training.loader import load_all_data, load_data

import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer

class BaseTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # load datasets and prepare data
        #self.train, self.test = load_all_data('', '/Users/akos/Documents/Projects/bonsai/training/data/all_data.csv.gz', 1000)
        self.train, self.test = load_data('/Users/akos/Documents/Projects/bonsai/training/data',
                                              1000)
        self.X = self.train[0]
        self.y = self.train[1]
        self.X_test = self.test[0]
        self.y_test = self.test[1]


    def test_load_model(self):
        linear = models.load_model('test_models/simpleLinearModel.pkl')
        y_pred = linear.predict(self.X_test)
        self.assertEqual(y_pred.shape, (250,))
        self.assertNotIn(-1, y_pred)
        svr = models.load_model('test_models/simpleSVRModel.pkl')
        y_pred = svr.predict(self.X_test)
        self.assertEqual(y_pred.shape, (250,))
        self.assertNotIn(-1, y_pred)
        ffnn = models.load_model('test_models/simpleFFNNModel.pkl')
        y_pred = ffnn.predict(self.X_test)
        self.assertEqual(y_pred.shape, (250,))
        self.assertNotIn(-1, y_pred)

    def test_simpleFFNNModel(self):

        hidden_size = [64, 256, 16]
        model = models.SimpleFFNNModel({'input_size': self.X.shape[1], 'output_size': self.y.shape[1], 'hidden_size': hidden_size,
                                        'network': 'relu', 'batch_size': 32,
                                        'transform': 'standard_and_power',
                                        #'transform': 'power',
                                        #'transform': 'standard',
                                        'init': 'xavier',
                                        'epochs': 1,
                                        'to_scale_features': [2,3,4,5],
                                        'to_transform_features': [0, 1, 6, 7, 8, 9, 10, 11],
                                        'optimizer': 'rmsprop'})
        model.fit(self.X, self.y)
        model.score(self.X_test, self.y_test)
        y_pred = model.predict(self.X_test)
        negative_values = np.sum(y_pred < 0)
        #self.assertEqual(negative_values, 0)
        _, loss, r2, rmse, mae, mape = model.score(self.X_test, self.y_test)
        print(f'loss: {loss}, r2: {r2}, rmse: {rmse}, mae: {mae}, mape: {mape}')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.y_test, y_pred, s=2, alpha=0.5)
        method = 'none'
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set(xlabel='True', ylabel='Predicted',
               title='Predicted vs true values \n'
                     f'{hidden_size}, mape={mape:.2f}')
        plt.tight_layout()
        #plt.savefig(f'predicted_{method}.png')

        plt.show()
        #model.save('test_models/simpleFFNNModel.pkl')

    def test_Unet(self):
        model = models.SimpleFFNNModel(
            {'input_size': self.X.shape[1], 'output_size': self.y.shape[1], 'hidden_size': [256, 16, 256],
             'network': 'skiprelu', 'batch_size': 32,
              'transform': 'standard',
                'init': 'xavier',
             'skip_connections': 'add',
             # 'transform': 'power',
             'epochs': 100,
             # "'to_scale_features': [2,3,4,5],
             # 'to_transform_features': [0, 1, 6, 7, 8, 9, 10, 11],
             'optimizer': 'adam'})
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X_test)
        negative_values = np.sum(y_pred < 0)
        # self.assertEqual(negative_values, 0)
        loss, r2, rmse, mae, mape = model.score(self.X_test, self.y_test)
        print(f'loss: {loss}, r2: {r2}, rmse: {rmse}, mae: {mae}, mape: {mape}')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.y_test, y_pred, s=2, alpha=0.5)
        method = 'none'
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set(xlabel='True', ylabel='Predicted',
               title='Predicted vs true values \n'
                     f'{method}, mape={mape:.2f}')
        plt.tight_layout()
        # plt.savefig(f'predicted_{method}.png')

        plt.show()

    def test_simpleLinearModel(self):
        model = models.SimpleLinearModel({})
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X_test)
        self.assertNotIn(-1, y_pred)
        model.save('test_models/simpleLinearModel.pkl')

    def test_simpleSVRModel(self):
        model = models.SimpleSVRModel({})
        model.fit(self.X, self.y)
        y_pred = model.predict(self.X_test)
        self.assertNotIn(-1, y_pred)
        model.save('test_models/simpleSVRModel.pkl')






if __name__ == '__main__':
    unittest.main()
