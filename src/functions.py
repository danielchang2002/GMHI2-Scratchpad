import pandas as pd
import config

def load_train(boolean = False):
    """
        Returns X_train, y_train from the data directory
        If boolean, then returns X_train as a boolean matrix,
        where each nonzero value is replaced with 1
    """
    return load(config.TRAIN_DIR + "taxonomy.csv", 
            config.TRAIN_DIR + "isHealthy.csv", boolean)

def load_val(boolean = False):
    """
        Returns X_val, y_val from the data directory
        If boolean, then returns X_val as a boolean matrix,
        where each nonzero value is replaced with 1
    """
    return load(config.VAL_DIR + "taxonomy679.csv", 
            config.VAL_DIR + "isHealthy679.csv", boolean)

def load(x_dir, y_dir, boolean):
    species = pd.read_csv(x_dir, index_col=0)
    species[species < 0.00001] = 0
    X = species.T
    isHealthy = pd.read_csv(y_dir, index_col=0)
    y = isHealthy.T
    if boolean:
        X = (X > 0) * 1
    return X, y


