import argparse
import os
from functools import partial

import numpy as np
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from bonsai.models import load_model, SimpleSVRModel
from training import loader


def train_svr(config, max_epochs, data_dir=None):
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as checkpoint_dir:
            net = load_model(os.path.join(checkpoint_dir, "model"))
    else:
        nu_svr = config['nu_svr']
        del config['nu_svr']
        if nu_svr:
            del config['epsilon']
        else:
            del config['nu']
        net = SimpleSVRModel(config, nu_svr)

    for epoch in range(max_epochs):
        train, _ = loader.load_data(data_dir)
        trainset = np.concatenate((train[0], train[1]), axis=1)
        train_subset, val_subset = train_test_split(
            trainset, test_size=0.2, random_state=42)

        train_X, train_y = np.split(train_subset, [train_subset.shape[1] - 1], axis=1)

        net.fit(train_X, train_y)

        val_X, val_y = np.split(val_subset, [val_subset.shape[1]-1], axis=1)

        test_r2, test_rmse = net.score(val_X, val_y)
        checkpoint_dir = f"checkpoint-{epoch}"
        net.save(os.path.join(checkpoint_dir, "model"))
        session.report(
            metrics={"r2": test_r2, "rmse": test_rmse},
            checkpoint=Checkpoint.from_directory(checkpoint_dir),
        )


def test_svr(best_trained_model, test):
    return best_trained_model.score(test[0], test[1])


def explore(num_samples=10, max_num_epochs=10, cpus_per_trial=2, gpus_per_trial=2):
    print("Starting exploration with {} samples {} epochs {} cpus per trial {} gpus per_trial"
          .format(num_samples, max_num_epochs, cpus_per_trial, gpus_per_trial))
    data_dir = os.path.join(os.path.dirname(__file__), "./data")
    train, test = loader.load_data(data_dir)
    config = {
        'C': tune.loguniform(1e-4, 1e4),
        'epsilon': tune.loguniform(1e-4, 1e4),
        'degree': tune.choice([1, 2, 3, 4, 5]),
        'gamma': tune.choice(['scale', 'auto', tune.loguniform(1e-4, 1e4)]),
        'kernel': tune.choice(['linear', 'poly', 'rbf']),
        'nu_svr': tune.choice([True, False]),
        'nu': tune.uniform(0.1, 0.9),
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_svr, max_epochs=max_num_epochs, data_dir=data_dir),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="r2",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()

    # the best trial is the one thas has the lowest loss and was trained for the longest
    best_trial = results.get_best_trial("r2", "max", "last")
    print("Best trial config: {}".format(best_trial.__config))
    print("Best trial final validation r2: {}".format(
        best_trial.last_result["r2"]))
    print("Best trial final validation rmse: {}".format(
        best_trial.last_result["rmse"]))

    #best_trained_model = SimpleFFNNModel(best_trial.config)

    best_checkpoint_dir = best_trial.checkpoint.to_directory()
    print("Loading best model from checkpoint {}...".format(best_checkpoint_dir))
    best_trained_model = load_model(os.path.join(
        best_checkpoint_dir, "model"))
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.model.load_state_dict(model_state)

    loss, r2, rmse = test_svr(best_trained_model, test)
    print("Best trial test set r2: {}, rmse: {}".format(r2, rmse))
    df_results = results.dataframe()
    df_sorted = df_results.sort_values(by=['r2'], ascending=False)
    print(df_sorted)
    df_sorted.to_csv('results_svr.csv', index=False)

parser = argparse.ArgumentParser()
parser.add_argument('--cpus_per_trial', type=int, default=2)
parser.add_argument('--gpus_per_trial', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--max_num_epochs', type=int, default=10)



if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    args = parser.parse_args()
    explore(num_samples=args.num_samples, max_num_epochs=args.max_num_epochs, cpus_per_trial=args.cpus_per_trial, gpus_per_trial=args.gpus_per_trial)
