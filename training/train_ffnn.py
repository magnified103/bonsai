import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

import training.loader as loader
from network_generator.models.base import SimpleFFNNModel, load_model


def train_network(net, train, validation):
    """
    Train the network and return the loss, r2, rmse, mae, mape
    :param net: the network to train
    :param train: the training data
    :param validation: the validation data
    :return: the loss, r2, rmse, mae, mape
    """
    net.fit(train[0], train[1])
    training_loss, loss, test_r2, test_rmse, test_mae, test_mape = net.score(validation[0], validation[1])

    return training_loss, loss, test_r2, test_rmse, test_mae, test_mape


# python3 -m training.train_ffnn --epochs 60 --current_best_config_path results/best_config_ffnn_layers_2.json --out train_results_ffnn_2 --data_file training/data/all_data.csv.gz --resume True
# python3 -m training.train_ffnn --epochs 60 --current_best_config_path results/best_config_ffnn_layers_3.json --out train_results_ffnn_3 --data_file training/data/all_data.csv.gz --resume True
# python3 -m training.train_ffnn --epochs 60 --current_best_config_path results/best_config_ffnn_layers_4.json --out train_results_ffnn_4 --data_file training/data/all_data.csv.gz --resume True
# python3 -m training.train_ffnn --epochs 60 --current_best_config_path results/best_config_ffnn_layers_5.json --out train_results_ffnn_5 --data_file training/data/all_data.csv.gz --resume True
# python3 -m training.train_ffnn --epochs 60 --current_best_config_path results/best_config_ffnn_layers_6.json --out train_results_ffnn_6 --data_file training/data/all_data.csv.gz --resume True

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--data_file', type=str, default=None)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--current_best_config_path', type=str, default=None)
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--rerun', nargs='+', type=int, default=None)
parser.add_argument('--kfolds', type=int, default=5)
parser.add_argument('--max_tries', type=int, default=1)
parser.add_argument('--sample_size', type=int, default=-1)


def checkpoint(net, epoch, rows, checkpoint_dir):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(checkpoint_dir, 'train_scores_{}.csv'.format(epoch)))
    net.save(os.path.join(checkpoint_dir, "model_{}.pth".format(epoch)))


def get_model_to_train(config, out, resume):
    start_epoch = 0
    if resume:  # resume training
        if os.path.isdir(out):  # if the directory exists
            if os.path.isfile(
                    os.path.join(out, 'model.pth')):  # if the model exists, then we have already trained this config
                return None, None, 0, None
            else:  # else we need to resume training
                for f in os.listdir(out):
                    if f.startswith('model_'):  # find the last checkpoint
                        epoch = int(f.split('_')[1].split('.')[0]) + 1  # start from the next epoch
                        if epoch > start_epoch:
                            start_epoch = epoch
            if start_epoch > 0:  # if we have a checkpoint, then load the model
                print("Resuming from checkpoint")
                net = load_model(os.path.join(out, f'model_{start_epoch - 1}.pth'))  # load the model
                df = pd.read_csv(os.path.join(out, 'train_scores_{}.csv'.format(start_epoch - 1)))  # load the scores
                rows = df.to_dict('records')  # convert to list of dicts
                print(f"Resuming from epoch {start_epoch - 1} with config: {config}")
                print(
                    f"Last epoch: loss={df['loss'].iloc[-1]}, r2={df['r2'].iloc[-1]}, rmse={df['rmse'].iloc[-1]}, mae={df['mae'].iloc[-1]}, mape={df['mape'].iloc[-1]}, time={df['time'].iloc[-1]}")
                resume_time = df['time'].iloc[-1]
                return net, rows, start_epoch, resume_time
            else:  # if we don't have a checkpoint, then raise an error
                raise ModelNotFoundException("Cannot resume training because there is no checkpoint")
        else:  # if the directory doesn't exist, then can't resume, raise an error
            raise ModelNotFoundException("Cannot resume training because the directory doesn't exist")

            # if not resuming, then create a new model
    return SimpleFFNNModel(config), [], 0, 0


def train_algorithm(X_train, y_train, X_val, y_val, net, rows, out, start_epoch, resume_time):
    start = time.time()
    # for each training epoch
    for epoch in range(start_epoch, args.epochs):
        # train the network
        training_loss, loss, test_r2, test_rmse, test_mae, test_mape = train_network(net, (X_train, y_train), (X_val, y_val))
        end = time.time()
        print(
            f"Epoch {epoch}: training_loss={training_loss} loss={loss}, r2={test_r2}, rmse={test_rmse}, mae={test_mae}, mape={test_mape}, time={(end - start) + resume_time}")
        # save the scores
        rows.append({'training loss': training_loss, 'loss': loss, 'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape,
                     'time': end - start})
        # checkpoint the model
        checkpoint(net, epoch, rows, checkpoint_dir=out)
        # if the loss has not improved for 2 epochs, then stop training
        if len(rows) > 1:
            if rows[-1]['loss'] == rows[-2]['loss']:
                print("Stopping early because the loss has not improved for 2 epochs")
                return False
    return True






def validate_algorithm(X_val, y_val, net, rows, out):
    """
    Validate the model
    :param X_val: Input validation data
    :param y_val: Output validation data
    :param net: The model
    :param rows: scores
    :param out:  output directory
    :return: r2, rmse, mae, mape
    """

    # score with the validation set
    _, loss, test_r2, test_rmse, test_mae, test_mape = net.score(X_val, y_val)
    start = time.time()
    pred = net.predict(test[0])
    end = time.time()

    print(
        f"Test: loss={loss}, r2={test_r2}, rmse={test_rmse}, mae={test_mae}, mape={test_mape}, time={end - start}")
    rows.append(
        {'training loss': pd.NA, 'loss': loss, 'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape, 'time': end - start})

    # save scores
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'train_scores.csv'))

    # save the predictions
    results = np.column_stack((pred, test[1]))
    df = pd.DataFrame(results, columns=['pred', 'actual'])
    df.to_csv(os.path.join(out, 'results.csv'))

    # save the model
    net.save(os.path.join(out, 'model.pth'))

    return loss, test_r2, test_rmse, test_mae, test_mape


def train_kfolds(train_X, train_y, config, args, config_number):
    best_net = None
    best_score = None
    best_fold = None

    # need to set the random state to get the same folds each time
    kf = KFold(n_splits=args.kfolds, shuffle=True, random_state=42)

    print(f"Shape of training data: {train_X.shape}")
    # For each fold
    for j, (train_index, test_index) in enumerate(kf.split(train_X)):
        out = os.path.join(args.out, f'config-{config_number}', f'kfold-{j}')
        os.makedirs(out, exist_ok=True)

        # Get the model to train
        net, rows, start_epoch, resume_time = get_model_to_train(config, out, args.resume)
        if net is None:  # if the model has already been trained, then skip this fold
            print("Skipping Fold {} because it has already been trained".format(j))
            continue
        print(f' -------- Fold {j} -------')
        print(f"Training on {len(train_index)} samples, validating on {len(test_index)} samples")
        print(f"Train index: {train_index}")
        print(f"Test index: {test_index}")
        #X_train, X_val = np.take(train_X, train_index), np.take(train_X, test_index)
        #y_train, y_val = np.take(train_y, train_index), np.take(train_y, test_index)

        X_train, X_val = train_X[train_index], train_X[test_index]
        y_train, y_val = train_y[train_index], train_y[test_index]

        # train the model
        tries = 0
        while not train_algorithm(X_train, y_train, X_val, y_val, net, rows, out, start_epoch, resume_time) and tries < args.max_tries:
            print("Training failed, trying again, try #{}".format(tries))
            tries += 1
            net, rows, start_epoch, resume_time = get_model_to_train(config, out, args.resume)


        # validate the model
        loss, test_r2, test_rmse, test_mae, test_mape = validate_algorithm(X_val, y_val, net, rows, out)

        # if the mape is better than the best, then remember this model
        if best_score is None or loss < best_score:
            best_score = loss
            best_net = net
            best_fold = j
    return best_net, best_fold


class ModelNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)


def load_best_model(out):
    """
    Load the best model from the output directory
    :param out: The output directory
    :return: The model
    """
    # find the best model
    best_fold_model = None
    best_score = None
    best_fold = None
    for f in os.listdir(out):
        # find the kfold model with the best score
        if f.startswith('kfold-'):
            # get the score
            fold_number = float(f.split('-')[1])
            res = pd.read_csv(os.path.join(out, f, 'train_scores.csv'))
            score = res['loss'].iloc[-1]
            print(f"Fold {fold_number} score: {score}")
            if best_score is None or score < best_score:
                best_score = score
                best_fold_model = f
                best_fold = fold_number
    if best_fold_model is None:
        raise ModelNotFoundException(f"Could not find a model in {out}")
    # load the model
    net = load_model(os.path.join(out, best_fold_model, 'model.pth'))
    return net, best_fold


def get_best_model(train_X, train_y, config, args, config_number):
    rows, start_epoch, resume_time = [], 0, 0
    best, fold = train_kfolds(train_X, train_y, config, args, config_number)
    if best is None:  # if kfold didn't return a model, probably because it was already trained
        # let's try to resume the best model
        try:
            best, rows, start_epoch, resume_time = get_model_to_train(config, out, args.resume)
            if best is None:
                # skip this model.
                print(f"Skipping model {config_number} because it was already trained")
                return None, None, None, None, None
        except ModelNotFoundException:
            # we need to load the score to and load the correct model
            best, fold = load_best_model(out)

    return best, fold, rows, start_epoch, resume_time


if __name__ == "__main__":
    args = parser.parse_args()

    # Load the current best config
    if args.current_best_config_path is not None:
        with open(args.current_best_config_path, 'r') as f:
            current_best_config = json.load(f)

    # Load the data
    data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
    print(f"Loading data from {data_dir}...")
    if args.data_file is None:
        train, test = loader.load_data(data_dir, args.sample_size)
    else:
        train, test = loader.load_all_data(data_dir, args.data_file)
    trainset = np.concatenate((train[0], train[1]), axis=1)
    train_subset, val_subset = train_test_split(
        trainset, test_size=0.2, random_state=42)

    # Split the data into Features (X) and target (y)
    train_X, train_y = np.split(train_subset, [train_subset.shape[1] - 1], axis=1)
    val_X, val_y = np.split(val_subset, [val_subset.shape[1] - 1], axis=1)

    print(
        f"Training with {train_X.shape[0]} samples, validating with {val_X.shape[0]} samples, testing with {test[0].shape[0]} samples...")

    # For each config, train the model
    config_number = 0
    for config in current_best_config:
        config_number += 1
        start_epoch = 0

        # If we are only running a subset of configs, skip the ones we don't want
        if args.rerun is not None and config_number not in args.rerun:
            continue

        out = os.path.join(args.out, f'config-{config_number}')
        hidden_layers = 0
        for k in config.keys():
            if k.startswith('hidden_size_'):
                hidden_layers += 1

        os.makedirs(out, exist_ok=True)
        config['hidden_size'] = [config[f'hidden_size_{x}'] for x in range(hidden_layers)]
        for i in range(hidden_layers):
            del config[f'hidden_size_{i}']

        # write the config to the trial directory
        json_obj = json.dumps(config, indent=4)
        with open(os.path.join(out, 'config.json'), 'w') as f:
            f.write(json_obj)

        print(f" -------- Training config {config_number} ------- ")
        print(f"Config: {config}")

        # find best model in kfold
        best, fold, rows, start_epoch, resume_time = get_best_model(train_X, train_y, config, args, config_number)
        if best is None:
            print(f"Skipping model {config_number} because it was already trained")
            continue

        print(f' -------- Training best model config {config_number} fold {fold} -------')
        # train the best model on the full training set
        tries = 0
        while not train_algorithm(train_X, train_y, val_X, val_y, best, rows, out, start_epoch, resume_time) and tries < args.max_tries:
            print("Training failed, trying again, try #{}".format(tries))
            tries += 1
            best, fold, rows, start_epoch, resume_time = get_best_model(train_X, train_y, config, args, config_number)

        # validate the model against the test set
        loss, r2, rmse, mae, mape = validate_algorithm(val_X, val_y, best, rows, out)
        print(f"Loss: {loss}, Test R2: {r2}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

## do relu at the end with offset to zero.
