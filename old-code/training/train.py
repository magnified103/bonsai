import pandas as pd
from sklearn.model_selection import KFold


def train_model(model, train, validation):
    """
        Train the model and return the training_loss, loss, r2, rmse, mae, mape
        :param net: the network to train
        :param train: the training data
        :param validation: the validation data
        :return: the training_loss, loss, r2, rmse, mae, mape
        """
    model.fit(train[0], train[1])
    training_loss, _, _, _, _ = model.score(train[0], train[1])
    loss, test_r2, test_rmse, test_mae, test_mape = model.score(validation[0], validation[1])

    return training_loss, loss, test_r2, test_rmse, test_mae, test_mape


def train_kfold(model, data, model_trainer=train_model, k=10, random_state=42):
    """
    Train the model using k-fold cross validation

    :param model: the model to train
    :type: BaseModel
    :param data: the data to train on
    :type: Dataset
    :param model_trainer: the function to train the model
    :type: function
    :param k: the number of folds
    :type: int
    :param random_state: the random state
    :type: int
    :return: scores
    :type: pd.DataFrame
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []
    for i, (train_idx, val_idx) in enumerate(kf.split(data.X)):
        print(f"------ Training fold {i} ------")
        train = data.X[train_idx], data.y[train_idx]
        validation = data.X[val_idx], data.y[val_idx]

        training_loss, loss, r2, rmse, mae, mape = model_trainer(model, train, validation)

        scores.append({'fold': i, 'loss': loss, 'training loss': training_loss, 'r2': r2, 'rmse': rmse, 'mae': mae,
                       'mape': mape})
        print(
            f'fold: {i}, loss: {loss}, training_loss: {training_loss}, r2: {r2}, rmse: {rmse}, mae: {mae}, mape: {mape}')

    return pd.DataFrame(scores)
