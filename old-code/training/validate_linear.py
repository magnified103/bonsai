import argparse
import os

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer

from bonsai import models
from training.loader import load, Dataset
from training.train import train_kfold

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='data/small_sample_data.csv.gz')
parser.add_argument('--sample_size', type=int, default=10000)
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--folds', type=int, default=5)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the data
    data = load(args.data_file, sample_size=args.sample_size, max_std=1)
    featureChooser = data.choose_features()
    features = featureChooser.choose_features([
        'source_country',
        'dest_country',
        'source_asn',
        'dest_asn',
        'source_lat',
        'source_lon',
        'dest_lat',
        'dest_lon',
        'distance',
    ])
    target = featureChooser.choose_targets(['rtt_p50'])
    X, y = data.get_features(features), data.get_target(target)
    dataset = Dataset(X, y)

    transformer = ColumnTransformer([
        # ('standard', StandardScaler(), features.get_feature_indices(['source_lat',
        #                                                              'source_lon',
        #                                                              'dest_lat',
        #                                                              'dest_lon'])
        #  ),
        ('power', PowerTransformer(), features.get_feature_indices(['distance',
                                                                    'source_lat',
                                                                    'source_lon',
                                                                    'dest_lat',
                                                                    'dest_lon'
                                                                    ])
         ),
        ('target', TargetEncoder(), features.get_feature_indices(['source_country',
                                                                  'dest_country',
                                                                  'source_asn',
                                                                  'dest_asn'])
         )
    ])
    out_transformer = PowerTransformer()
    model = models.SimpleLinearModel({}, 'test_linear', transformer, out_transformer)
    scores = train_kfold(model, dataset, k=args.folds)
    print(scores)
    print(f'Average MAE: {sum(scores["mae"]) / len(scores["mae"])}')
    print(f'Average RMSE: {sum(scores["rmse"]) / len(scores["rmse"])}')
    print(f'Average R2: {sum(scores["r2"]) / len(scores["r2"])}')
    print(f'Average MAPE: {sum(scores["mape"]) / len(scores["mape"])}')


    os.makedirs(args.output_dir, exist_ok=True)
    scores.to_csv(os.path.join(args.output_dir, 'scores.csv'))



