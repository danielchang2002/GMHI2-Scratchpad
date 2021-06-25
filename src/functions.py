import pandas as pd
import config

def load_train(boolean = False):
    """
        Returns X_train, y_train from the data directory
        If boolean, then returns X_train as a boolean matrix,
        where each nonzero value is replaced with 1
    """
    species = pd.read_csv(config.TRAIN_DIR + "taxonomy.csv", index_col = 0)
    species[species < 0.00001] = 0
    X_train = species.T
    isHealthy = pd.read_csv(config.TRAIN_DIR + "isHealthy.csv", index_col=0)
    y_train = isHealthy.T
    if boolean:
        X_train = (X_train > 0) * 1
        y_train = y_train > 0
    return X_train, y_train

def load_val(boolean = False):
    """
        Returns X_val, y_val from the data directory
        If boolean, then returns X_val as a boolean matrix,
        where each nonzero value is replaced with 1
    """
    species_val = pd.read_csv(config.VAL_DIR + "taxonomy679.csv", index_col=0)
    species_val[species_val < 0.00001] = 0
    X_val = species_val.T
    isHealthy_val = pd.read_csv(config.VAL_DIR + "isHealthy679.csv", index_col=0)
    y_val = isHealthy_val.T
    if boolean:
        X_val = (X_val > 0) * 1
        y_val = y_val > 0
    return X_val, y_val


